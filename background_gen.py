import torch
import argparse
from diffusion_gen import DiffusionGeneration



from diffusers import StableDiffusionXLInpaintPipeline
from diffusers import DiffusionPipeline

# Argument 
parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument("-p", "--prompt", type=str, help="Prompt to generate image")
parser.add_argument("-n", "--negative_prompt", type=str, help="Negative prompt")

args = parser.parse_args()

# Setup hyper parameters
hyper_params = {
    "seed": -305,
    "kernel_size": (5, 5),
    "kernel_iterations": 15,
    "num_inference_steps": 70,
    "denoising_start": 0.70,
    "guidance_scale": 7.5,
    "prompt": args.prompt,
    "negative_prompt": args.negative_prompt,
}

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup pipelines
inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

refine_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=inpaint_pipe.text_encoder_2,
    vae=inpaint_pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

# Execute
diffusion_gen = DiffusionGeneration(inpaint_pipe, refine_pipe, hyper_params, device)

# TODO 



