import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

model_id = "aldenn13l/geo-finetuned"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
   model_id, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

url = "https://datasets-server.huggingface.co/assets/aldenn13l/182-fine-tune/--/1014744dd1c828c7d7a4837b8b32a176b1daec13/--/default/train/76/original_image/image.jpg"


def download_image(url):
   image = PIL.Image.open(requests.get(url, stream=True).raw)
   image = PIL.ImageOps.exif_transpose(image)
   image = image.convert("RGB")
   return image


image = download_image(url)
prompt = "Remove the power lines on the top of the bridge"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

edited_image = pipe(
   prompt,
   image=image,
   num_inference_steps=num_inference_steps,
   image_guidance_scale=image_guidance_scale,
   guidance_scale=guidance_scale,
   generator=generator,
).images[0]
edited_image.save("edited_image.png")