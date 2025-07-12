import streamlit as st
import subprocess
import time
import requests
from datetime import datetime
from pathlib import Path
import uuid
import os

API_URL = "http://127.0.0.1:8000/convert-to-line-art/"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

st.title("Photo-to-Line-Art Generator")

# 启动 FastAPI 服务
def start_fastapi():
    status_msg = st.empty()
    try:
        res = requests.get("http://127.0.0.1:8000/docs")
        if res.status_code == 200:
            status_msg.success("✅ FastAPI is running")
            return
    except:
        status_msg.warning("⚠️ FastAPI is not running, trying to start...")
        subprocess.Popen(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
        time.sleep(5)

    # 启动后再次确认
    for _ in range(10):
        try:
            res = requests.get("http://127.0.0.1:8000/docs")
            if res.status_code == 200:
                status_msg.success("✅ FastAPI started successfully")
                return
        except:
            time.sleep(1)
    status_msg.error("❌ Failed to start FastAPI")

start_fastapi()

# 上传图片
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

# 模型和参数选择
controlnet_index = st.selectbox("Select Model", ["sd-controlnet-canny", "sd-controlnet-depth", "sd-controlnet-hed"])
index_map = {"sd-controlnet-canny": 0, "sd-controlnet-depth": 1, "sd-controlnet-hed": 2}
prompt = st.text_input("Prompt", value="clean cartoon-style line art, black and white, clean lines")
low_threshold = st.slider("Low Threshold", 0, 255, 100)
high_threshold = st.slider("High Threshold", 0, 255, 200)

# 生成线稿
if st.button("Generate Line Art"):
    if uploaded_file is None:
        st.error("Please upload an image first.")
    else:
        # 读取图片数据
        image_bytes = uploaded_file.read()
        files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
        data = {
            "controlnet_index": index_map[controlnet_index],
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "prompt": prompt
        }

        with st.spinner("Generating..."):
            try:
                response = requests.post(API_URL, files=files, data=data)
                if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
                    timestamp = datetime.now().strftime("%Y%m%d")
                    serial = len(list(OUTPUT_DIR.glob(f"line_art_{timestamp}_*.png"))) + 1
                    serial_str = f"{serial:04d}"
                    output_name = f"line_art_{timestamp}_{serial_str}.png"
                    output_path = OUTPUT_DIR / output_name

                    # 保存图片
                    with open(output_path, "wb") as f:
                        f.write(response.content)

                    st.image(str(output_path), caption=output_name, use_container_width=True)
                    st.success("✅ Line art generated successfully!")
                else:
                    st.error("❌ Failed to generate image.")
                    try:
                        st.json(response.json())
                    except:
                        st.text(response.text)
            except Exception as e:
                st.error("❌ Error during request.")
                st.exception(e)
