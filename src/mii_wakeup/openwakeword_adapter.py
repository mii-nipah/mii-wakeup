from __future__ import annotations

import inspect
import sys
import types
import warnings
from pathlib import Path


class ModelLoadError(RuntimeError):
    """Raised when openWakeWord cannot be initialized."""


def load_openwakeword_model(
    model_paths: list[Path],
    *,
    vad_threshold: float,
    speex_noise_suppression: bool,
):
    for model_path in model_paths:
        if not model_path.is_file():
            raise ModelLoadError(f"Wake-word model does not exist: {model_path}")

    _install_openwakeword_training_stub()

    try:
        from openwakeword.model import Model
    except Exception as exc:
        raise ModelLoadError(f"Could not import openWakeWord: {exc}") from exc

    kwargs: dict[str, object] = {
        "vad_threshold": vad_threshold,
        "enable_speex_noise_suppression": speex_noise_suppression,
    }

    signature = inspect.signature(Model)
    parameters = signature.parameters
    model_path_strings = [str(path) for path in model_paths]

    if model_path_strings:
        if "wakeword_model_paths" in parameters:
            kwargs["wakeword_model_paths"] = model_path_strings
        elif "wakeword_models" in parameters:
            kwargs["wakeword_models"] = model_path_strings
        else:
            raise ModelLoadError(
                "This openWakeWord version does not expose a supported custom "
                "model-path parameter."
            )

    # Newer openWakeWord versions default to tflite on Linux. mii-wakeup is
    # intentionally ONNX-first because users pass custom .onnx wake-word models.
    if "inference_framework" in parameters:
        kwargs["inference_framework"] = "onnx"

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Specified provider 'CUDAExecutionProvider'.*",
                category=UserWarning,
                module=r"onnxruntime\..*",
            )
            return Model(**kwargs)
    except Exception as exc:
        raise ModelLoadError(f"Could not initialize openWakeWord: {exc}") from exc


def _install_openwakeword_training_stub() -> None:
    if "openwakeword.custom_verifier_model" in sys.modules:
        return

    stub = types.ModuleType("openwakeword.custom_verifier_model")

    def train_custom_verifier(*args: object, **kwargs: object) -> None:
        raise RuntimeError("mii-wakeup does not package openWakeWord training helpers")

    stub.train_custom_verifier = train_custom_verifier
    sys.modules["openwakeword.custom_verifier_model"] = stub
