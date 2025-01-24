import gradio as gr
import numpy as np
from PIL import Image
from diffusers_pipeline import HandFixerPipeline

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

def create_interface():
    # 创建pipeline
    pipe = HandFixerPipeline('black-forest-labs/FLUX.1-dev')

    # 创建Gradio界面
    with gr.Blocks() as iface:
        gr.Markdown("#HandFixer:手部修复")
        gr.Markdown("左侧上传一张图片,右侧显示处理后的图片。")
        with gr.Row():
            input_image = gr.Image(type="numpy")
            output_image = gr.Image(type="numpy")
        input_image.change(fn=lambda img: process_image(pipe, img), inputs=input_image, outputs=output_image)
    
    return iface

if __name__ == "__main__":
    iface = create_interface()
    # 启动Gradio应用
    iface.launch(server_name="0.0.0.0", server_port=8502)