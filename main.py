from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import pipeline
from controlnet_aux import HEDdetector

from PIL import Image
import cv2
import numpy as np
import torch
from io import BytesIO
import uuid
from pathlib import Path
import os

# from fastapi.background import BackgroundTasks
app = FastAPI(title="Photo to Line Art Converter",
              description="Convert photos to line art using Stable Diffusion ControlNet",
              version="1.0.0")

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the ControlNet model
all_controlnets = {
    0: "lllyasviel/sd-controlnet-canny",
    1: "lllyasviel/sd-controlnet-depth",
    2: "lllyasviel/sd-controlnet-hed",
}
# 默认选择 lllyasviel/sd-controlnet-canny
def build_pipeline(controlnet_index=0):
    if controlnet_index not in all_controlnets:
        raise ValueError(f"Invalid controlnet_index: {controlnet_index}. Available options: {list(all_controlnets.keys())}")
    
    controlnet = ControlNetModel.from_pretrained(
        all_controlnets[controlnet_index], torch_dtype=dtype
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")
    
    return pipe

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

@app.post("/convert-to-line-art/")

async def convert_to_line_art(
    controlnet_index: int = 0,  # ControlNet 模型编号, 0: lllyasviel/sd-controlnet-canny, 1: lllyasviel/sd-controlnet-depth, 2: lllyasviel/sd-controlnet-hed
    file: UploadFile = File(...), 
    low_threshold: int = 100, 
    high_threshold: int = 200,
    prompt: str = "clean cartoon-style line art, black and white, clean lines",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    ):
    try:
        # build the pipeline
        pipe = build_pipeline(controlnet_index)
        # Read the uploaded image
        if controlnet_index == 0:
            image_bytes = await file.read()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            # limit the image size to 1024x1024
            image.thumbnail((1024, 1024))

            image = np.array(image)
            # Apply Canny edge detection
            edges = cv2.Canny(image, low_threshold, high_threshold)
            edges = edges[:, :, None]
            edges = np.concatenate([edges, edges, edges], axis=2)
            edges_image = Image.fromarray(edges)
        elif controlnet_index == 1:
            image_bytes = await file.read()
            depth_estimator = pipeline('depth-estimation')
            image = depth_estimator(image_bytes)['depth']
            image = np.array(image)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            edges_image = Image.fromarray(image)
        elif controlnet_index == 2:
            hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
            image_bytes = await file.read()
            edges_image = hed(image_bytes)
        else:
            print(f"Invalid controlnet_index: {controlnet_index}. Available options: {list(all_controlnets.keys())}")
            return {"error": "Invalid controlnet_index. Available options: 0, 1, 2."}

        # Generate line art using the pipeline
        result = pipe(
                    prompt,  # 优化提示词
                    edges_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt="blurry, messy, noisy, low quality, grayscale, shaded, low contrast, detailed texture, color",  # 添加负向提示
                    # negative_prompt="blurry, messy, noisy, low quality"  # 添加负向提示
                    
                ).images[0]

        # 生成唯一文件名
        filename = f"lineart_{uuid.uuid4().hex}.png"
        output_path = OUTPUT_DIR / filename
        result.save(output_path)

        return FileResponse(
            output_path,
            media_type="image/png",
            filename=filename,
            background=BackgroundTask(lambda: os.remove(output_path))  # ❌ 自动删除
        )
    except Exception as e:
        return {"error": str(e)}