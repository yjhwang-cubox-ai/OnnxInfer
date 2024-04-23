import argparse
import os
import numpy as np
import onnxruntime as ort
import torch
import math
import cv2
import time

from PIL import Image, ImageFont, ImageDraw

def parse_args():
    parser = argparse.ArgumentParser(description="Test onnx")
    parser.add_argument('--onnx', default='models/svtr-base_20e_st_mj_vn_20240404.onnx')
    parser.add_argument('--onnx_ops', default="")
    parser.add_argument('--images', default='images')
    parser.add_argument('--outputs', default="outputs")
    parser.add_argument('--dict_path', default="dicts/vietnamese_unicode.txt")
    parser.add_argument('--font_size', default=14, type=int)
    
    args = parser.parse_args()
    return args

class TextRecognition:
    def __init__(self, model_path, ops_path, font_path, font_size, dict_path) -> None:
        self.font = ImageFont.truetype(font_path, font_size)
        self.impl = SVTR(model_path, ops_path, dict_path)
    
    def recognition(self, image, draw_prediction=False, out_path="", file_name=""):
        #todo...
        pass
        
    
def main():
    args = parse_args()    
    


if __name__ == "__main__":
    main()