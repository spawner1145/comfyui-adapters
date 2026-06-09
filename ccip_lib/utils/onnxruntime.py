import os
import logging
from typing import Optional

try:
    from onnxruntime import get_available_providers, get_all_providers, InferenceSession, SessionOptions, GraphOptimizationLevel
except Exception:
    InferenceSession = None


def get_onnx_provider(provider: Optional[str] = None):
    if not provider:
        try:
            if "CUDAExecutionProvider" in get_available_providers():
                return "CUDAExecutionProvider"
            else:
                return "CPUExecutionProvider"
        except Exception:
            return "CPUExecutionProvider"
    else:
        return provider


def _open_onnx_model(ckpt: str, provider: str, use_cpu: bool = True):
    if InferenceSession is None:
        raise ImportError('onnxruntime is required to open ONNX models. Please install onnxruntime.')

    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [provider]
    if use_cpu and "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    logging.info(f'Model {ckpt!r} loaded with provider {provider!r}')
    return InferenceSession(ckpt, options, providers=providers)


def open_onnx_model(ckpt: str, mode: Optional[str] = None):
    return _open_onnx_model(ckpt, get_onnx_provider(mode or os.environ.get('ONNX_MODE', None)))
