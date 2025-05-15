# only import if running as a custom node
from .nodes.lora import NunchakuFluxLoraLoader
from .nodes.indexlora import NunchakuFluxLoraLoaderIndex
from .nodes.models import NunchakuFluxDiTLoader, NunchakuTextEncoderLoader
from .nodes.preprocessors import FluxDepthPreprocessor

NODE_CLASS_MAPPINGS = {
    "NunchakuFluxDiTLoader": NunchakuFluxDiTLoader,
    "NunchakuTextEncoderLoader": NunchakuTextEncoderLoader,
    "NunchakuFluxLoraLoader": NunchakuFluxLoraLoader,
    "NunchakuFluxLoraLoaderIndex": NunchakuFluxLoraLoaderIndex,
    "NunchakuDepthPreprocessor": FluxDepthPreprocessor,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
