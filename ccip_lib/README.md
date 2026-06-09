CCIP 本地模型文件夹结构要求

必需文件:
- model_feat.onnx        : ONNX 模型，用于提取特征（对应代码中 `_open_feat_model`）
- model_metrics.onnx     : ONNX 模型，用于计算成对差异（对应 `_open_metric_model`）
- metrics.json           : 包含模型指标（必须包含 `threshold` 字段），例如用于默认阈值

可选文件:
- cluster.json           : 包含聚类相关的默认参数（eps, min_samples），用于 `ccip_clustering`

示例目录结构:
```
C:\models\ccip-caformer_b36-24\
├─ model_feat.onnx
├─ model_metrics.onnx
└─ metrics.json
```

使用方法:
- 在 CLI 中传入 `--model-dir C:\models\ccip-caformer_b36-24`。
- 若 `model` 参数不是本地目录，代码将尝试从 Hugging Face 仓库 `deepghs/ccip_onnx` 下载对应模型。

注意:
- `metrics.json` 中应包含 `threshold` 字段（数值），以便 `ccip_default_threshold` 使用。
- 如果要离线使用，请先将上述文件放到本地目录后再传入 `--model-dir`。
