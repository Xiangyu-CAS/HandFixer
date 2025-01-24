import streamlit as st
import numpy as np
from PIL import Image
from diffusers_pipeline import HandFixerPipeline

@st.cache_resource
def load_pipeline():
    return HandFixerPipeline('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-fincv/zhuxiangyu04/pretrain/stable-diffusion/huggingface.co/black-forest-labs/FLUX.1-dev')

def process_image(pipe, input_image):
    # 确保输入是 PIL Image
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    
    # 处理图像
    output_image = pipe(input_image)
    
    # 确保输出是 PIL Image
    if not isinstance(output_image, Image.Image):
        output_image = Image.fromarray(output_image)
    
    return output_image

def main():
    st.set_page_config(page_title="HandFixer: 手部修复")
    
    st.title("HandFixer: 手部修复")
    st.markdown("上传一张图片，查看处理前后的对比结果。")

    # 加载pipeline
    pipe = load_pipeline()

    # 文件上传
    uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 读取上传的图片
        input_image = Image.open(uploaded_file)

        # 处理图片
        output_image = process_image(pipe, input_image)

        # 创建两列来并排显示图片
        col1, col2 = st.columns(2)

        # 在第一列显示原始图片
        with col1:
            st.subheader("原始图片")
            st.image(input_image, use_container_width=True)

        # 在第二列显示处理后的图片
        with col2:
            st.subheader("处理后的图片")
            st.image(output_image, use_container_width=True)

if __name__ == "__main__":
    main()
