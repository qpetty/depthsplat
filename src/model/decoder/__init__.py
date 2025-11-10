from typing import TYPE_CHECKING

from ...dataset import DatasetCfg
from .decoder import Decoder

# Lazy import decoder implementations to avoid requiring diff-gaussian-rasterization
# at module import time. This allows the encoder to be loaded without the decoder.
_decoder_impls = None
_decoder_cfg_type = None


def _get_decoder_impls():
    """Lazy import decoder implementations."""
    global _decoder_impls, _decoder_cfg_type
    if _decoder_impls is None:
        from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
        _decoder_impls = {
            "splatting_cuda": DecoderSplattingCUDA,
        }
        _decoder_cfg_type = DecoderSplattingCUDACfg
    return _decoder_impls, _decoder_cfg_type


# Use __getattr__ for module-level lazy imports (Python 3.7+)
def __getattr__(name: str):
    if name == "DecoderCfg":
        if TYPE_CHECKING:
            # For type checking, import directly
            from .decoder_splatting_cuda import DecoderSplattingCUDACfg
            return DecoderSplattingCUDACfg
        else:
            # At runtime, lazy load
            _, cfg_type = _get_decoder_impls()
            return cfg_type
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_decoder(decoder_cfg, dataset_cfg: DatasetCfg) -> Decoder:
    """Get a decoder instance. Lazy loads decoder implementations."""
    decoders, _ = _get_decoder_impls()
    return decoders[decoder_cfg.name](decoder_cfg, dataset_cfg)
