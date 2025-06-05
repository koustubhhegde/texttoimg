!pip install diffusers transformers accelerate

from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch

model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id2 = "stabilityai/stable-diffusion-x1-base-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")

def generate_image(pipe, prompt, params):
  img = pipe(prompt, **params).images

  if isinstance(img, list):
    num_images = len(img)
    if num_images > 1:
      fig, axes = plt.subplots(nrows=1, ncols=num_images)
      for i in range(num_images):
        axes[i].imshow(img[i]);
        axes[i].axis('off');
    else:
      fig = plt.figure()
      plt.imshow(img[0]);
      plt.axis('off');
  else:
    fig = plt.figure()
    plt.imshow(img);
    plt.axis('off');

  plt.tight_layout()

prompt = """ a happy man sitting on the chair """

params = {}

generate_image(pipe, prompt, params)
