import json
from typing import Literal, Union, List, Optional, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from sklearn.cluster import DBSCAN, OPTICS
from tqdm.auto import tqdm

from ccip_lib.data import MultiImagesTyping, load_images, ImageTyping
from ccip_lib.utils.onnxruntime import open_onnx_model
from ccip_lib.utils.cache import ts_lru_cache
import os

__all__ = [
    'ccip_extract_feature',
    'ccip_batch_extract_features',

    'ccip_default_threshold',
    'ccip_difference',
    'ccip_same',
    'ccip_batch_differences',
    'ccip_batch_same',

    'ccip_default_clustering_params',
    'ccip_clustering',

    'ccip_merge',
]


def _normalize(data, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    mean, std = np.asarray(mean), np.asarray(std)
    return (data - mean[:, None, None]) / std[:, None, None]


def _preprocess_image(image: Image.Image, size: int = 384):
    image = image.resize((size, size), resample=Image.BILINEAR)
    data = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
    data = _normalize(data)

    return data


@ts_lru_cache()
def _open_feat_model(model):
    if isinstance(model, str) and os.path.isdir(model):
        path = os.path.join(model, 'model_feat.onnx')
        return open_onnx_model(path)

    return open_onnx_model(hf_hub_download(
        f'deepghs/ccip_onnx',
        f'{model}/model_feat.onnx',
    ))


@ts_lru_cache()
def _open_metric_model(model):
    if isinstance(model, str) and os.path.isdir(model):
        path = os.path.join(model, 'model_metrics.onnx')
        return open_onnx_model(path)

    return open_onnx_model(hf_hub_download(
        f'deepghs/ccip_onnx',
        f'{model}/model_metrics.onnx',
    ))


@ts_lru_cache()
def _open_metrics(model):
    if isinstance(model, str) and os.path.isdir(model):
        path = os.path.join(model, 'metrics.json')
        with open(path, 'r') as f:
            return json.load(f)

    with open(hf_hub_download(f'deepghs/ccip_onnx', f'{model}/metrics.json'), 'r') as f:
        return json.load(f)


@ts_lru_cache()
def _open_cluster_metrics(model):
    if isinstance(model, str) and os.path.isdir(model):
        path = os.path.join(model, 'cluster.json')
        with open(path, 'r') as f:
            return json.load(f)

    with open(hf_hub_download(f'deepghs/ccip_onnx', f'{model}/cluster.json'), 'r') as f:
        return json.load(f)


_VALID_MODEL_NAMES = [
    'ccip-caformer-24-randaug-pruned',
    'ccip-caformer-6-randaug-pruned_fp32',
    'ccip-caformer-5_fp32',
]
_DEFAULT_MODEL_NAMES = 'ccip-caformer-24-randaug-pruned'


def ccip_extract_feature(image: ImageTyping, size: int = 384, model: str = _DEFAULT_MODEL_NAMES):
    return ccip_batch_extract_features([image], size, model)[0]


def ccip_batch_extract_features(images: MultiImagesTyping, size: int = 384, model: str = _DEFAULT_MODEL_NAMES):
    feat_model = _open_feat_model(model)
    model_inputs = feat_model.get_inputs()
    if model_inputs:
        input_shape = model_inputs[0].shape
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 4:
            expected_h = input_shape[2]
            expected_w = input_shape[3]
            if isinstance(expected_h, int) and isinstance(expected_w, int) and expected_h == expected_w:
                if size != expected_h:
                    size = expected_h

    images = load_images(images, mode='RGB')
    data = np.stack([_preprocess_image(item, size=size) for item in images]).astype(np.float32)
    output, = feat_model.run(['output'], {'input': data})
    return output


_FeatureOrImage = Union[ImageTyping, np.ndarray]


def _p_feature(x: _FeatureOrImage, size: int = 384, model: str = _DEFAULT_MODEL_NAMES):
    if isinstance(x, np.ndarray):  # if feature
        return x
    else:  # is image or path
        return ccip_extract_feature(x, size, model)


def ccip_default_threshold(model: str = _DEFAULT_MODEL_NAMES) -> float:
    return _open_metrics(model)['threshold']


def ccip_difference(x: _FeatureOrImage, y: _FeatureOrImage,
                    size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> float:
    return ccip_batch_differences([x, y], size, model)[0, 1].item()


def ccip_same(x: _FeatureOrImage, y: _FeatureOrImage, threshold: Optional[float] = None,
              size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> float:
    diff = ccip_difference(x, y, size, model)
    threshold = threshold if threshold is not None else ccip_default_threshold(model)
    return diff <= threshold


def ccip_batch_differences(images: List[_FeatureOrImage],
                           size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> np.ndarray:
    input_ = np.stack([_p_feature(img, size, model) for img in images]).astype(np.float32)
    output, = _open_metric_model(model).run(['output'], {'input': input_})
    return output


def ccip_batch_same(images: List[_FeatureOrImage], threshold: Optional[float] = None,
                    size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> np.ndarray:
    batch_diff = ccip_batch_differences(images, size, model)
    threshold = threshold if threshold is not None else ccip_default_threshold(model)
    return batch_diff <= threshold


CCIPClusterMethodTyping = Literal['dbscan', 'dbscan_2', 'dbscan_free', 'optics', 'optics_best']
_METHOD_MAPPING = {'optics_best': 'optics'}


def ccip_default_clustering_params(model: str = _DEFAULT_MODEL_NAMES,
                                   method: CCIPClusterMethodTyping = 'optics') -> Tuple[float, int]:
    if method == 'dbscan':
        return ccip_default_threshold(model), 2
    if method == 'optics':
        return 0.5, 5
    else:
        _info = _open_cluster_metrics(model)[_METHOD_MAPPING.get(method, method)]
        return _info['eps'], _info['min_samples']


def ccip_clustering(images: List[_FeatureOrImage], method: CCIPClusterMethodTyping = 'optics',
                    eps: Optional[float] = None, min_samples: Optional[int] = None,
                    size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> np.ndarray:
    _default_eps, _default_min_samples = ccip_default_clustering_params(model, method)
    eps = eps or _default_eps
    min_samples = min_samples or _default_min_samples

    images = [_p_feature(img, size, model) for img in tqdm(images, desc='Extract features')]
    batch_diff = ccip_batch_differences(images, size, model)

    def _metric(x, y):
        return batch_diff[int(x), int(y)].item()

    samples = np.arange(len(images)).reshape(-1, 1)
    if 'dbscan' in method:
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=_metric).fit(samples)
    elif 'optics' in method:
        clustering = OPTICS(max_eps=eps, min_samples=min_samples, metric=_metric).fit(samples)
    else:
        assert False, f'Unknown mode for CCIP clustering - {method!r}.'  # pragma: no cover

    return clustering.labels_.tolist()


def ccip_merge(images: Union[List[_FeatureOrImage], np.ndarray],
               size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> np.ndarray:
    embs = np.stack([_p_feature(img, size, model) for img in images]).astype(np.float32)
    lengths = np.linalg.norm(embs, axis=-1)
    embs = embs / lengths.reshape(-1, 1)
    ret_embedding = embs.mean(axis=0)
    return ret_embedding / np.linalg.norm(ret_embedding) * lengths.mean()
