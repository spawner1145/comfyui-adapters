import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import node_helpers
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torchvision import transforms

import comfy.conds
import comfy.ldm.anima.model as comfy_anima_model
import comfy.ldm.common_dit
import comfy.model_base
import comfy.model_management
import comfy.utils
from einops import rearrange


logger = logging.getLogger(__name__)
NODE_DIR = Path(__file__).resolve().parent


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def _to_pil_list(image: Optional[torch.Tensor]) -> List[Image.Image]:
    if image is None:
        return []
    if image.ndim == 3:
        image = image.unsqueeze(0)
    out = []
    for item in image.detach().cpu().float().clamp(0, 1):
        arr = (item.numpy() * 255.0).round().astype(np.uint8)
        out.append(Image.fromarray(arr).convert("RGB"))
    return out


def _resize_for_vae(image: torch.Tensor, max_area: int) -> torch.Tensor:
    samples = image.movedim(-1, 1)
    align = 16
    if max_area and max_area > 0:
        scale_by = math.sqrt(max_area / (samples.shape[3] * samples.shape[2]))
        width = max(align, round(samples.shape[3] * scale_by / align) * align)
        height = max(align, round(samples.shape[2] * scale_by / align) * align)
        samples = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    else:
        width = max(align, samples.shape[3] - samples.shape[3] % align)
        height = max(align, samples.shape[2] - samples.shape[2] % align)
        if width != samples.shape[3] or height != samples.shape[2]:
            samples = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return samples.movedim(1, -1)[:, :, :, :3]


def _generate_rope_from_ids(pos_embedder, ids_l_3: torch.Tensor, fps: Optional[torch.Tensor] = None) -> torch.Tensor:
    h_ntk_factor = getattr(pos_embedder, "h_ntk_factor", 1.0)
    w_ntk_factor = getattr(pos_embedder, "w_ntk_factor", 1.0)
    t_ntk_factor = getattr(pos_embedder, "t_ntk_factor", 1.0)
    h_theta = 10000.0 * h_ntk_factor
    w_theta = 10000.0 * w_ntk_factor
    t_theta = 10000.0 * t_ntk_factor
    h_spatial_freqs = 1.0 / (h_theta ** pos_embedder.dim_spatial_range.to(ids_l_3.device))
    w_spatial_freqs = 1.0 / (w_theta ** pos_embedder.dim_spatial_range.to(ids_l_3.device))
    temporal_freqs = 1.0 / (t_theta ** pos_embedder.dim_temporal_range.to(ids_l_3.device))

    ids_l_3 = ids_l_3.to(dtype=torch.float32)
    t = ids_l_3[:, 0]
    h = ids_l_3[:, 1]
    w = ids_l_3[:, 2]
    if getattr(pos_embedder, "enable_fps_modulation", False) and fps is not None:
        t = t / fps[:1].to(device=t.device, dtype=t.dtype) * pos_embedder.base_fps

    half_emb_t = t[:, None] * temporal_freqs[None]
    half_emb_h = h[:, None] * h_spatial_freqs[None]
    half_emb_w = w[:, None] * w_spatial_freqs[None]
    half_emb_t = torch.stack([torch.cos(half_emb_t), -torch.sin(half_emb_t), torch.sin(half_emb_t), torch.cos(half_emb_t)], dim=-1)
    half_emb_h = torch.stack([torch.cos(half_emb_h), -torch.sin(half_emb_h), torch.sin(half_emb_h), torch.cos(half_emb_h)], dim=-1)
    half_emb_w = torch.stack([torch.cos(half_emb_w), -torch.sin(half_emb_w), torch.sin(half_emb_w), torch.cos(half_emb_w)], dim=-1)
    emb_l_d_4 = torch.cat([half_emb_t, half_emb_h, half_emb_w], dim=-2)
    return rearrange(emb_l_d_4, "l d (i j) -> l d i j", i=2, j=2).float()


def _prepare_flat_tokens(model, x_b_c_t_h_w: torch.Tensor, t_offset: int = 0, padding_mask: Optional[torch.Tensor] = None):
    if model.concat_padding_mask:
        if padding_mask is None:
            padding_mask = torch.zeros(
                x_b_c_t_h_w.shape[0],
                1,
                x_b_c_t_h_w.shape[-2],
                x_b_c_t_h_w.shape[-1],
                dtype=x_b_c_t_h_w.dtype,
                device=x_b_c_t_h_w.device,
            )
        padding_mask = transforms.functional.resize(
            padding_mask, list(x_b_c_t_h_w.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
        )
        x_b_c_t_h_w = torch.cat(
            [x_b_c_t_h_w, padding_mask.unsqueeze(1).repeat(1, 1, x_b_c_t_h_w.shape[2], 1, 1)],
            dim=1,
        )
    x_b_t_h_w_d = model.x_embedder(x_b_c_t_h_w)
    _, t, h, w, _ = x_b_t_h_w_d.shape
    ids = torch.cartesian_prod(
        torch.arange(t_offset, t_offset + t, device=x_b_t_h_w_d.device),
        torch.arange(h, device=x_b_t_h_w_d.device),
        torch.arange(w, device=x_b_t_h_w_d.device),
    )
    return rearrange(x_b_t_h_w_d, "b t h w d -> b (t h w) d"), ids, (t, h, w)


def _block_forward_flat(
    block,
    x_b_l_d: torch.Tensor,
    emb_b_1_d: torch.Tensor,
    crossattn_emb: torch.Tensor,
    rope_emb_l_1_1_d: Optional[torch.Tensor],
    adaln_lora_b_1_3d: Optional[torch.Tensor],
    transformer_options: Optional[dict],
    use_fp32: bool,
) -> torch.Tensor:
    if use_fp32:
        x_b_l_d = x_b_l_d.float()
    residual_dtype = x_b_l_d.dtype
    compute_dtype = emb_b_1_d.dtype
    with torch.autocast(device_type=x_b_l_d.device.type, dtype=torch.float32, enabled=use_fp32):
        if block.use_adaln_lora:
            shift_self, scale_self, gate_self = (block.adaln_modulation_self_attn(emb_b_1_d) + adaln_lora_b_1_3d).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = (block.adaln_modulation_cross_attn(emb_b_1_d) + adaln_lora_b_1_3d).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (block.adaln_modulation_mlp(emb_b_1_d) + adaln_lora_b_1_3d).chunk(3, dim=-1)
        else:
            shift_self, scale_self, gate_self = block.adaln_modulation_self_attn(emb_b_1_d).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = block.adaln_modulation_cross_attn(emb_b_1_d).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = block.adaln_modulation_mlp(emb_b_1_d).chunk(3, dim=-1)

    def expand_param(x):
        return x.expand(-1, x_b_l_d.shape[1], -1)

    def adaln(x, norm, scale, shift):
        return norm(x) * (1 + expand_param(scale)) + expand_param(shift)

    normalized = adaln(x_b_l_d, block.layer_norm_self_attn, scale_self, shift_self)
    result = block.self_attn(normalized.to(compute_dtype), None, rope_emb=rope_emb_l_1_1_d, transformer_options=transformer_options or {})
    x_b_l_d = x_b_l_d + expand_param(gate_self).to(residual_dtype) * result.to(residual_dtype)

    normalized = adaln(x_b_l_d, block.layer_norm_cross_attn, scale_cross, shift_cross)
    result = block.cross_attn(normalized.to(compute_dtype), crossattn_emb, rope_emb=rope_emb_l_1_1_d, transformer_options=transformer_options or {})
    x_b_l_d = x_b_l_d + expand_param(gate_cross).to(residual_dtype) * result.to(residual_dtype)

    normalized = adaln(x_b_l_d, block.layer_norm_mlp, scale_mlp, shift_mlp)
    result = block.mlp(normalized.to(compute_dtype))
    x_b_l_d = x_b_l_d + expand_param(gate_mlp).to(residual_dtype) * result.to(residual_dtype)
    return x_b_l_d


class AnimaReferenceImageAppend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "references": ("ANIMA_REFERENCE_IMAGES",),
            },
        }

    RETURN_TYPES = ("ANIMA_REFERENCE_IMAGES",)
    RETURN_NAMES = ("references",)
    FUNCTION = "append"
    CATEGORY = "adapters/Anima/conditioning"

    def append(self, image, references=None):
        refs = list(references or [])
        refs.append(image)
        return (refs,)


class AnimaReferenceImagesConnector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "references1": ("ANIMA_REFERENCE_IMAGES",),
                "references2": ("ANIMA_REFERENCE_IMAGES",),
                "references3": ("ANIMA_REFERENCE_IMAGES",),
                "references4": ("ANIMA_REFERENCE_IMAGES",),
            }
        }

    RETURN_TYPES = ("ANIMA_REFERENCE_IMAGES",)
    RETURN_NAMES = ("references",)
    FUNCTION = "connect"
    CATEGORY = "adapters/Anima/conditioning"

    def connect(self, references1=None, references2=None, references3=None, references4=None):
        refs = []
        for part in (references1, references2, references3, references4):
            if part:
                refs.extend(part)
        return (refs,)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return out * self.weight


class AdapterRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq = self.inv_freq[None, :, None].float().to(x.device)
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class AdapterImageRotaryEmbedding(AdapterRotaryEmbedding):
    @torch.no_grad()
    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int], num_images: int = 1):
        h, w = grid_hw
        y = torch.arange(h, device=x.device)
        z = torch.arange(w, device=x.device)
        ids = torch.cartesian_prod(y, z).sum(dim=-1).unsqueeze(0)
        if num_images > 1:
            ids = ids.repeat(1, num_images)
        return super().forward(x, ids)


class CCIPAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query_dim = dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-5)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-5)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, position_embeddings=None):
        b, s, d = x.shape
        q = self.q_norm(self.q_proj(x).view(b, s, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, s, self.num_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.o_proj(out)


class CCIPFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class CCIPRefinerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 8.0 / 3.0, norm_eps: float = 1e-5):
        super().__init__()
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention = CCIPAttention(dim, num_heads)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        hidden = int(dim * mlp_ratio)
        self.feed_forward = CCIPFeedForward(dim, hidden)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, position_embeddings=None):
        attn_out = self.attention(self.attention_norm1(x), position_embeddings=position_embeddings)
        x = x + self.attention_norm2(attn_out)
        x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class AnimaCCIPVisualAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 2,
        num_feature_tokens: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_feature_tokens = num_feature_tokens
        self.feature_norm = RMSNorm(feature_dim, eps=1e-5)
        self.feature_proj = nn.Linear(feature_dim, hidden_size, bias=True)
        self.feature_expand = nn.Linear(feature_dim, hidden_size * num_feature_tokens, bias=True)
        self.rotary_emb = AdapterRotaryEmbedding(hidden_size // num_heads)
        self.image_rotary_emb = AdapterImageRotaryEmbedding(hidden_size // num_heads, rope_theta=256.0)
        self.refiner = nn.ModuleList([CCIPRefinerBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim == 4:
            b, h, w, d = features.shape
            features = features.reshape(b, h * w, d)
        b, seq_len, _ = features.shape
        normed = self.feature_norm(features).to(self.feature_proj.weight.dtype)
        if seq_len == 1:
            tokens = self.feature_expand(normed[:, 0]).reshape(b, self.num_feature_tokens, self.hidden_size)
        else:
            tokens = self.feature_proj(normed)
        pos = self._position_embeddings(tokens, seq_len)
        for layer in self.refiner:
            tokens = layer(tokens, position_embeddings=pos)
        return self.out_proj(tokens)

    def _position_embeddings(self, tokens: torch.Tensor, seq_len: int):
        root = int(math.isqrt(seq_len))
        if root * root == seq_len:
            return self.image_rotary_emb(tokens, (root, root), num_images=1)
        if seq_len % 144 == 0:
            return self.image_rotary_emb(tokens, (12, 12), num_images=seq_len // 144)
        position_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        return self.rotary_emb(tokens, position_ids)


class CCIPTokenExtractor:
    def __init__(self, checkpoint: str, device: torch.device):
        self.checkpoint = checkpoint
        self.device = device
        self.model, self.transform, self.feature_dim = self._load(checkpoint, device)

    @staticmethod
    def _prepare_ccip_models():
        try:
            import timm.layers.helpers as timm_layer_helpers
            sys.modules.setdefault("timm.models.layers.helpers", timm_layer_helpers)
        except Exception:
            pass

    @classmethod
    def _load(cls, checkpoint: str, device: torch.device):
        cls._prepare_ccip_models()
        from ccip_lib.models.caformer import get_caformer

        path = Path(checkpoint)
        if path.is_dir():
            candidates = sorted(path.glob("*.ckpt")) + sorted(path.glob("*.pth")) + sorted(path.glob("*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No CCIP checkpoint found in {checkpoint}")
            path = candidates[0]
        logger.info(f"Loading CCIP token checkpoint: {path}")
        sd = cls._load_checkpoint_state(path)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.removeprefix("module._orig_mod.").removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
        first_conv = sd.get("feature.backbone.caformer.downsample_layers.0.conv.weight")
        arch = "caformer_b36_384_in21ft1k" if first_conv is not None and first_conv.shape[0] == 128 else "caformer_s36_384_in21ft1k"
        backbone, transform = get_caformer(arch=arch, pretrained=False)
        backbone_sd = {k.removeprefix("feature.backbone."): v for k, v in sd.items() if k.startswith("feature.backbone.")}
        missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
        logger.info(f"Loaded CCIP token backbone arch={arch}, missing={len(missing)}, unexpected={len(unexpected)}")
        backbone.to(device).eval().requires_grad_(False)
        return backbone, transform, backbone.caformer.output_dim

    @staticmethod
    def _load_checkpoint_state(path: Path):
        if path.suffix.lower() == ".safetensors":
            return load_file(str(path), device="cpu")
        try:
            try:
                return torch.load(str(path), map_location="cpu")
            except Exception:
                return torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as e:
            try:
                with open(path, "rb") as f:
                    head = f.read(96)
                head_text = head.decode("utf-8", errors="replace")
            except Exception:
                head_text = "<could not read file header>"
            raise RuntimeError(
                f"Failed to load CCIP checkpoint as a PyTorch checkpoint: {path}\n"
                f"File header: {head_text!r}\n"
                "Please make sure this is the real ccip-caformer checkpoint file, not a download pointer, HTML/text file, "
                "or unsupported archive. If it is safetensors, use a .safetensors suffix."
            ) from e

    def extract(self, images: List[Image.Image], dtype: torch.dtype) -> torch.Tensor:
        device = comfy.model_management.get_torch_device()
        if device != self.device:
            self.device = device
            self.model.to(device)
        tensors = []
        for image in images:
            image = image.convert("RGB").resize((384, 384), Image.Resampling.BILINEAR)
            tensor = transforms.ToTensor()(image)
            for transform in self.transform:
                tensor = transform(tensor)
            tensors.append(tensor)
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            fmap = self.model._get_cnn_result(batch)
            feats = fmap.flatten(2).transpose(1, 2).contiguous()
        return feats.reshape(1, -1, feats.shape[-1]).to(dtype=dtype)


def _patch_anima_extra_conds():
    if getattr(comfy.model_base.Anima, "_comfy_adapters_anima_extra_conds", False):
        return
    original = comfy.model_base.Anima.extra_conds

    def extra_conds(self, **kwargs):
        out = original(self, **kwargs)
        reference_latents = kwargs.get("reference_latents", None)
        if reference_latents is not None:
            out["reference_latents"] = comfy.conds.CONDList([self.process_latent_in(lat) for lat in reference_latents])
        reference_t_offset_scale = kwargs.get("reference_t_offset_scale", None)
        if reference_t_offset_scale is not None:
            out["reference_t_offset_scale"] = comfy.conds.CONDConstant(reference_t_offset_scale)
        ip_adapter_embeds = kwargs.get("anima_ip_adapter_embeds", None)
        if ip_adapter_embeds is not None:
            out["anima_ip_adapter_embeds"] = comfy.conds.CONDRegular(ip_adapter_embeds)
        return out

    comfy.model_base.Anima.extra_conds = extra_conds
    comfy.model_base.Anima._comfy_adapters_anima_extra_conds = True


def _patch_anima_forward():
    if getattr(comfy_anima_model.Anima, "_comfy_adapters_anima_forward", False):
        return
    original_forward = comfy_anima_model.Anima.forward

    def forward(self, x, timesteps, context, **kwargs):
        transformer_options = kwargs.get("transformer_options", {})
        visual_condition_adapter = transformer_options.get("anima_visual_condition_adapter", None)

        t5xxl_ids = kwargs.pop("t5xxl_ids", None)
        if t5xxl_ids is not None:
            context = self.preprocess_text_embeds(context, t5xxl_ids, t5xxl_weights=kwargs.pop("t5xxl_weights", None))

        refs = kwargs.pop("reference_latents", None)
        reference_t_offset_scale = kwargs.pop("reference_t_offset_scale", 10)
        ip_embeds = kwargs.pop("anima_ip_adapter_embeds", None)

        if refs is None and ip_embeds is None:
            return original_forward(self, x, timesteps, context, **kwargs)

        if refs is None:
            if ip_embeds is not None and visual_condition_adapter is not None:
                visual_condition_adapter = visual_condition_adapter.to(device=context.device, dtype=context.dtype)
                visual_tokens = visual_condition_adapter(ip_embeds.to(device=context.device, dtype=context.dtype))
                context = torch.cat([context, visual_tokens.to(context)], dim=1)
            return super(comfy_anima_model.Anima, self).forward(x, timesteps, context, **kwargs)

        if self.extra_per_block_abs_pos_emb:
            raise NotImplementedError("Anima multi-image reference conditioning does not support extra_per_block_abs_pos_emb.")
        if x.shape[2] != 1:
            raise NotImplementedError("Anima multi-image reference conditioning currently supports image latents (T=1) only.")

        orig_shape = list(x.shape)
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_temporal, self.patch_spatial, self.patch_spatial))
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)

        outputs = []
        for batch_index in range(x.shape[0]):
            x_i = x[batch_index : batch_index + 1]
            target_tokens, target_ids, target_grid = _prepare_flat_tokens(self, x_i, t_offset=0, padding_mask=None)

            ref_tokens = []
            ref_ids = []
            sample_refs = refs if isinstance(refs, list) else [refs]
            for ref_index, ref_latent in enumerate(sample_refs):
                if ref_latent.ndim == 4:
                    ref_latent = ref_latent.unsqueeze(2)
                ref_latent = ref_latent[batch_index : batch_index + 1] if ref_latent.shape[0] == x.shape[0] else ref_latent[:1]
                ref_latent = ref_latent.to(device=x_i.device, dtype=x_i.dtype)
                tokens, ids, _ = _prepare_flat_tokens(
                    self,
                    ref_latent,
                    t_offset=int(reference_t_offset_scale) * (ref_index + 1),
                    padding_mask=None,
                )
                ref_tokens.append(tokens)
                ref_ids.append(ids)

            if ref_tokens:
                x_tokens = torch.cat([target_tokens] + ref_tokens, dim=1)
                ids = torch.cat([target_ids] + ref_ids, dim=0)
            else:
                x_tokens = target_tokens
                ids = target_ids

            context_i = context[batch_index : batch_index + 1]
            if ip_embeds is not None and visual_condition_adapter is not None:
                sample_ip = ip_embeds[batch_index : batch_index + 1] if ip_embeds.shape[0] == x.shape[0] else ip_embeds[:1]
                visual_condition_adapter = visual_condition_adapter.to(device=x_i.device, dtype=context_i.dtype)
                visual_tokens = visual_condition_adapter(sample_ip.to(device=x_i.device, dtype=context_i.dtype))
                context_i = torch.cat([context_i, visual_tokens.to(context_i)], dim=1)

            rope_emb = _generate_rope_from_ids(self.pos_embedder, ids, fps=None).unsqueeze(1).unsqueeze(0)
            timestep_i = timesteps[batch_index : batch_index + 1, :1]
            t_embedding, adaln_lora = self.t_embedder[1](self.t_embedder[0](timestep_i).to(x_tokens.dtype))
            t_embedding = self.t_embedding_norm(t_embedding)
            use_fp32 = x_tokens.dtype == torch.float16
            if use_fp32:
                x_tokens = x_tokens.float()

            for block in self.blocks:
                x_tokens = _block_forward_flat(
                    block,
                    x_tokens,
                    t_embedding,
                    context_i,
                    rope_emb,
                    adaln_lora,
                    transformer_options,
                    use_fp32,
                )

            target_len = target_tokens.shape[1]
            x_target = x_tokens[:, :target_len]
            t, h, w = target_grid
            x_target = rearrange(x_target, "b (t h w) d -> b t h w d", t=t, h=h, w=w)
            x_out = self.final_layer(x_target.to(context_i.dtype), t_embedding, adaln_lora_B_T_3D=adaln_lora)
            outputs.append(self.unpatchify(x_out))

        return torch.cat(outputs, dim=0)[:, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]]

    comfy_anima_model.Anima.forward = forward
    comfy_anima_model.Anima._comfy_adapters_anima_forward = True


_patch_anima_extra_conds()
_patch_anima_forward()


class AnimaMultiImageEditConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
            },
            "optional": {
                "references": ("ANIMA_REFERENCE_IMAGES",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "reference_max_area": ("INT", {"default": 1048576, "min": 0, "max": 16777216, "step": 1024}),
                "reference_t_offset_scale": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "adapters/Anima/conditioning"

    def apply(
        self,
        conditioning,
        vae,
        references=None,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        reference_max_area=1024 * 1024,
        reference_t_offset_scale=10,
    ):
        latents = []
        images = list(references or [])
        images.extend([image for image in (image1, image2, image3, image4) if image is not None])
        if not images:
            raise ValueError("Anima Multi Image Edit Conditioning requires at least one reference image.")
        for image in images:
            if image is not None:
                latents.append(vae.encode(_resize_for_vae(image, reference_max_area)))
        conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": latents}, append=True)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_t_offset_scale": reference_t_offset_scale})
        return (conditioning,)


class AnimaApplyCCIPAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ip_adapter_weight": ("STRING", {"default": ""}),
                "ccip_checkpoint": ("STRING", {"default": ""}),
            },
            "optional": {
                "num_layers": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "num_feature_tokens": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("MODEL", "ANIMA_CCIP_EXTRACTOR")
    FUNCTION = "apply"
    CATEGORY = "adapters/Anima/model"

    def apply(self, model, ip_adapter_weight, ccip_checkpoint, num_layers=2, num_feature_tokens=4):
        m = model.clone()
        device = comfy.model_management.get_torch_device()
        extractor = CCIPTokenExtractor(ccip_checkpoint, device)
        diffusion_model = m.model.diffusion_model
        hidden_size = getattr(diffusion_model, "crossattn_emb_channels", 1024)
        num_heads = getattr(diffusion_model, "num_heads", 16)
        adapter_sd = None
        if ip_adapter_weight:
            sd = load_file(ip_adapter_weight)
            adapter_sd = {k.removeprefix("visual_condition_adapter."): v for k, v in sd.items() if k.startswith("visual_condition_adapter.")}
            inferred = adapter_sd.get("feature_expand.weight", None)
            if inferred is None:
                inferred = sd.get("feature_expand.weight", None)
            if inferred is not None and inferred.shape[0] % hidden_size == 0:
                inferred_tokens = inferred.shape[0] // hidden_size
                if inferred_tokens != num_feature_tokens:
                    logger.info(
                        f"Inferred Anima CCIP adapter num_feature_tokens={inferred_tokens} from weights "
                        f"(overriding UI value {num_feature_tokens})."
                    )
                num_feature_tokens = inferred_tokens
        adapter = AnimaCCIPVisualAdapter(
            extractor.feature_dim,
            hidden_size,
            num_heads,
            num_layers=num_layers,
            num_feature_tokens=num_feature_tokens,
        )
        if ip_adapter_weight:
            if not adapter_sd:
                adapter_keys = set(adapter.state_dict().keys())
                adapter_sd = {k: v for k, v in sd.items() if k in adapter_keys}
            if not adapter_sd:
                raise ValueError("No Anima CCIP adapter weights found. Expected keys with `visual_condition_adapter.` prefix.")
            missing, unexpected = adapter.load_state_dict(adapter_sd, strict=False)
            logger.info(f"Loaded Anima CCIP adapter weights: missing={len(missing)}, unexpected={len(unexpected)}")
        adapter.to(device=device, dtype=diffusion_model.dtype if hasattr(diffusion_model, "dtype") else torch.bfloat16).eval().requires_grad_(False)
        m.model_options.setdefault("transformer_options", {})["anima_visual_condition_adapter"] = adapter
        return (m, extractor)


class AnimaCCIPAdapterConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "extractor": ("ANIMA_CCIP_EXTRACTOR",),
            },
            "optional": {
                "references": ("ANIMA_REFERENCE_IMAGES",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "adapters/Anima/conditioning"

    def apply(self, conditioning, extractor, references=None, image1=None, image2=None, image3=None, image4=None):
        images = []
        tensors = list(references or [])
        tensors.extend([image for image in (image1, image2, image3, image4) if image is not None])
        for image in tensors:
            images.extend(_to_pil_list(image))
        if not images:
            raise ValueError("Anima CCIP Adapter Conditioning requires at least one reference image.")
        embeds = extractor.extract(images, dtype=torch.bfloat16)
        return (node_helpers.conditioning_set_values(conditioning, {"anima_ip_adapter_embeds": embeds}),)


NODE_CLASS_MAPPINGS = {
    "AnimaReferenceImageAppend": AnimaReferenceImageAppend,
    "AnimaReferenceImagesConnector": AnimaReferenceImagesConnector,
    "AnimaMultiImageEditConditioning": AnimaMultiImageEditConditioning,
    "AnimaApplyCCIPAdapter": AnimaApplyCCIPAdapter,
    "AnimaCCIPAdapterConditioning": AnimaCCIPAdapterConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaReferenceImageAppend": "Anima Reference Image Append",
    "AnimaReferenceImagesConnector": "Anima Reference Images Connector",
    "AnimaMultiImageEditConditioning": "Anima Multi Image Edit Conditioning",
    "AnimaApplyCCIPAdapter": "Anima Apply CCIP Adapter",
    "AnimaCCIPAdapterConditioning": "Anima CCIP Adapter Conditioning",
}
