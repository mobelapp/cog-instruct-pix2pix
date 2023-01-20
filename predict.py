import os
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from PIL import Image

MODEL_ID = "timbrooks/instruct-pix2pix"
MODEL_CACHE = "diffusers-cache"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            revision='fp16',
            torch_dtype=torch.float16,
        ).to("cuda")
        # self.pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.enable_vae_slicing()

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="The prompt or prompts to guide the image generation.",
            default="A fantasy landscape, trending on artstation",
        ),
        negative_prompt: str = Input(
            description="The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if guidance_scale is less than 1).",
            default=None,
        ),
        image: Path = Input(
            description="Image which will be repainted according to prompt.",
        ),
        guidance_scale: float = Input(
            description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.", ge=1, le=20, default=1.5
        ),
        image_guidance_scale: float = Input(
            description="Image guidance scale is to push the generated image towards the inital image image. Higher image guidance scale encourages to generate images that are closely linked to the source image `image`, usually at the expense of lower image quality.", ge=1, le=20, default=1.5
        ),
        num_images_per_prompt: int = Input(
            description="The number of images to generate per prompt.",
            ge=1,
            le=8,
            default=1,
        ),
        eta: float = Input(
            description="Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to [schedulers.DDIMScheduler], will be ignored for others.",
            default=0.0,
        ),
        num_inference_steps: int = Input(
            description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.", ge=1, le=500, default=50
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Seed. Leave blank to randomize the seed.", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)

        extra_kwargs = {
            "image": Image.open(image).convert("RGB"),
        }

        output = self.pipe(
            prompt=[prompt] * num_images_per_prompt if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_images_per_prompt
            if negative_prompt is not None
            else None,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator,
            eta=eta,
            num_inference_steps=num_inference_steps,
            **extra_kwargs,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]

