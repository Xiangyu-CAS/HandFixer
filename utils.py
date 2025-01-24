import warnings
import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

warnings.filterwarnings("ignore")


class MediapipeEngine:
    def __init__(self, model_asset_path='assets/gesture_recognizer.task'):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=10, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def __call__(self, image):
        image = np.array(image)
        # image = self.resize_image(image)

        annotations = self.detect(image.copy())
        mask = self.prepare_mask(image, annotations)
        return Image.fromarray(image), Image.fromarray(mask)

    def detect(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        # 进行手部检测
        results = self.hands.process(image)
        annotations = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
            #     self.mp_drawing.draw_landmarks(
            #         image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                # 获取所有关键点的坐标
                coords = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
    
                # 计算边界框
                x_min, y_min, _ = np.min(coords, axis=0)
                x_max, y_max, _ = np.max(coords, axis=0)
    
                # 转换为图像坐标
                H, W, _ = image.shape
                x_min, y_min = int(x_min * W), int(y_min * H)
                x_max, y_max = int(x_max * W), int(y_max * H)

                # loosen bbox
                dynamic_resize = 0.15
                padding = 30

                bb_xpad = max(int((x_max - x_min + 1) * dynamic_resize), padding)
                bb_ypad = max(int((y_max - y_min + 1) * dynamic_resize), padding)
                bbx_min = max(int(x_min - bb_xpad), 0)
                bbx_max = min(int(x_max + bb_xpad), W-1)
                bby_min = max(int(y_min - bb_ypad), 0)
                bby_max = min(int(y_max + bb_ypad), H-1)

                annotations.append({
                    'bbox': [bbx_min, bby_min, bbx_max, bby_max],
                    'landmarks': coords
                })
        return annotations
    
    def prepare_mask(self, image, annotations):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for annotation in annotations:
            bbox = annotation['bbox']
            cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
        return mask

    def resize_image(self, image, target_size=1024):
        h, w = image.shape[:2]
    
        # 计算长边
        long_side = max(h, w)
    
        # 计算缩放比例
        scale = target_size / long_side
    
        # 计算新的尺寸
        new_h = int(h * scale)
        new_w = int(w * scale)
    
        # 使用cv2.resize进行缩放
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
        return resized_image


class ImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

    def load_image(self, image_path):
        if image_path.startswith('http'):
            return Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
        else:
            return Image.open(image_path).convert('RGB')

    def generate_caption(self, image, conditional_text="a photography of"):
        if conditional_text:
            inputs = self.processor(image, conditional_text, return_tensors="pt").to("cuda", torch.float16)
        else:
            inputs = self.processor(image, return_tensors="pt").to("cuda", torch.float16)

        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
