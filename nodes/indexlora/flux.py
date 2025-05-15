import copy
import logging
import os

import folder_paths

from nunchaku.lora.flux import to_diffusers

from ..models.flux import ComfyFluxWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NunchakuFluxLoraLoaderIndex")


class NunchakuFluxLoraLoaderIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The diffusion model the LoRA will be applied to."},
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                    },
                ),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING",)
    RETURN_NAMES = ("MODEL", "lora_name",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_index_lora"
    TITLE = "Nunchaku FLUX.1 LoRA Index Loader"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "LoRAs are used to modify the diffusion model, "
        "altering the way in which latents are denoised such as applying styles. "
        "You can link multiple LoRA nodes."
    )

    def load_index_lora(self, model, index: int, lora_strength: float):
        model_wrapper = model.model.diffusion_model
        assert isinstance(model_wrapper, ComfyFluxWrapper)

        transformer = model_wrapper.model
        model_wrapper.model = None
        ret_model = copy.deepcopy(model)  # copy everything except the model
        ret_model_wrapper = ret_model.model.diffusion_model
        assert isinstance(ret_model_wrapper, ComfyFluxWrapper)

        model_wrapper.model = transformer
        ret_model_wrapper.model = transformer
        #print(folder_paths.get_folder_paths("loras"))
        lora_name = os.listdir(folder_paths.get_folder_paths("loras")[0])[index]

        print(lora_name)
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        ret_model_wrapper.loras.append((lora_path, lora_strength))

        sd = to_diffusers(lora_path)

        if "transformer.x_embedder.lora_A.weight" in sd:
            new_in_channels = sd["transformer.x_embedder.lora_A.weight"].shape[1]
            assert new_in_channels % 4 == 0
            new_in_channels = new_in_channels // 4

            old_in_channels = ret_model.model.model_config.unet_config["in_channels"]
            if old_in_channels < new_in_channels:
                ret_model.model.model_config.unet_config["in_channels"] = new_in_channels

        return (ret_model, lora_name,)
