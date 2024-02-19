# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
import sys

sys.path.extend(["/IP-Adapter"])
import torch
import shutil
from PIL import Image
from typing import List
from ip_adapter import IPAdapterXL
from diffusers import StableDiffusionXLPipeline


base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "/IP-Adapter/sdxl_models/image_encoder"

## TODO: which one is a better solution?
ip_ckpt = "/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
# ip_ckpt = "/IP-Adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin"
# ip_ckpt = "/IP-Adapter/sdxl_models/ip-adapter-plus-_sdxl_vit-h.bin"


device = "cuda"
MODEL_CACHE = "model-cache"

## InstantID
## https://github.com/zsxkib/InstantID/blob/main/cog.yaml


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # Setup any one-off operations like loading trained models
        # Many models use this method to download weights

        # Alternatively, we can store weights directly in the image
        # alongside cog.yaml, update .dockerignore file

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
            cache_dir=MODEL_CACHE,
        )

    def predict(
        self,
        image: Path = Input(description="Ip adapter image", default=None),
        prompt: str = Input(
            description="Prompt (leave blank for image variations)", default=""
        ),
        negative_prompt: str = Input(description="Negative Prompt", default=""),
        scale: float = Input(
            description="Scale (influence of input image on generation)",
            ge=0.0,
            le=1.0,
            default=0.4,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=30
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        """ 1. pre-processing """

        # Create a random seed if not provided
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # IP-Adapter works best for square images
        # But you can just resize to 224x224 for non-square images
        image = Image.open(image)
        image.resize((512, 512))

        # load ip-adapter
        ip_model = IPAdapterXL(self.pipe, image_encoder_path, ip_ckpt, device)

        ## TODO: load the correct weights

        """ 2. run prediction """

        images = ip_model.generate(
            pil_image=image,
            num_samples=num_outputs,
            num_inference_steps=num_inference_steps,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            scale=scale,
        )

        """ 3. post-processing """

        ## TODO: check for NSFW content
        output_paths = []
        for i, output_image in enumerate(images):
            output_path = f"/tmp/out_{i}.png"
            output_image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
