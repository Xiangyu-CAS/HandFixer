import torch
import numpy as np
import cv2
from .utils import MediapipeEngine

class MediapipeHandNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                # "confidence": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.1, "tooltip": "threshold to detect hands"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "preview")

    FUNCTION = "process_image"
    CATEGORY = "mediapipe_hand"

    def __init__(self):
        self.engine = MediapipeEngine()

    def process_image(self, image):
        np_image = image.numpy()[0] * 255
        np_image = np_image.astype(np.uint8)
        pil_image, pil_mask = self.engine(np_image)
        np_image, np_mask = np.array(pil_image), np.array(pil_mask)
        image, mask = torch.from_numpy(np_image.astype(np.float32) / 255.0).unsqueeze(dim=0), torch.from_numpy(np_mask.astype(np.float32) / 255.0).unsqueeze(dim=0)
        preview = image *  (1 - mask).unsqueeze(-1)
        return image, mask, preview

# This line is necessary for ComfyUI to recognize and load your custom node
NODE_CLASS_MAPPINGS = {
    "MediapipeHandNode": MediapipeHandNode
}