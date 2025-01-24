import torch
import argparse
import glob, os
from diffusers import FluxInpaintPipeline, FluxFillPipeline
from diffusers.utils import load_image

from utils import MediapipeEngine, ImageCaptioner

# inpainting pipeline
class HandFixerPipeline:
    def __init__(self,
                flux_model_path="black-forest-labs/FLUX.1-dev"):
        self.engine = MediapipeEngine()
        self.captioner = ImageCaptioner()
        self.pipe = FluxInpaintPipeline.from_pretrained(
                flux_model_path,
                torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()
    
    def __call__(self, image_path, 
                    prompt='hand',
                    strength=0.8,
                     **kwargs):
        # prepare image and mask
        image, mask = self.engine(load_image(image_path))
        width, height = image.size
        # prepare prompt
        prompt = self.captioner.generate_caption(image, "")
        fixed_image = self.pipe(prompt = prompt,
                image = image,
                mask_image = mask,
                width = width,
                height = height,
                strength=strength, **kwargs,
                ).images[0]
        return fixed_image

# fill pipeline
class HandFixerFillPipeline:
    def __init__(self,
                flux_model_path="black-forest-labs/FLUX.1-fill-dev"):
        self.engine = MediapipeEngine()
        self.captioner = ImageCaptioner()
        self.pipe = FluxFillPipeline.from_pretrained(
                flux_model_path,
                torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()
    
    def __call__(self, image_path, 
                    prompt='hand',
                    strength=0.8,
                     **kwargs):
        # prepare image and mask
        image, mask = self.engine(load_image(image_path))
        width, height = image.size
        # prepare prompt
        prompt = self.captioner.generate_caption(image, "")
        fixed_image = self.pipe(prompt = prompt,
                image = image,
                mask_image = mask,
                width = width,
                height = height,
                # strength=strength, 
                **kwargs,
                ).images[0]
        return fixed_image

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='black-forest-labs/FLUX.1-dev',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=['inpaint', 'fill'],
        default='inpaint',
        help="Choose the pipeline to use: 'inpaint' or 'fill'",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        default=None,
    )  
    parser.add_argument(
        "--output_dir",
        type=str,
        default='outputs',
    )
    return parser.parse_args(input_args)

if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    
    # 定义常见的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
    
    # 使用列表推导式和glob来获取所有图片路径
    image_paths = [
        path for ext in image_extensions
        for path in glob.glob(os.path.join(input_dir, ext))
    ]
    
    # 打印找到的图片路径（可选）
    print(f"Found {len(image_paths)} images in {input_dir}:")

    # 根据参数选择pipeline
    if args.pipeline == 'inpaint':
        hand_fixer = HandFixerPipeline(args.pretrained_model_name_or_path)
    elif args.pipeline == 'fill':  # 'fill'
        hand_fixer = HandFixerFillPipeline(args.pretrained_model_name_or_path)
    else:
        raise ValueError("Invalid pipeline choice. Must be 'inpaint' or 'fill'.")
        
    os.makedirs(args.output_dir, exist_ok=True)
    for path in image_paths:
        fixed_image = hand_fixer(path, strength=0.8)
        output_path = os.path.join(args.output_dir, os.path.basename(path))
        # 保存图像
        fixed_image.save(output_path)