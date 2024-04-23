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
    parser = argparse.ArgumentParser(description='Test onnx')
    parser.add_argument('--onnx', default="models/svtr-base_20e_st_mj_vn_20240404.onnx")
    parser.add_argument('--onnx_ops', default="")
    parser.add_argument('--images', default="images/test2.png")
    parser.add_argument('--outputs', default="outputs")
    parser.add_argument('--dict_path', default="dicts/vietnamese_unicode.txt")
    parser.add_argument('--font_path', default="arial.ttf")
    parser.add_argument('--font_size', default=14, type=int)
    
    args = parser.parse_args()
    return args

class OrtBase:
    def __init__(self, model_path, ort_custom_op_path ="") -> None:
        session_options = ort.SessionOptions()
        if ort_custom_op_path:
            session_options.register_custom_ops_library(ort_custom_op_path)

        # TODO - GPU지원
        self.device_type = "cpu"
        self.device_id = -1
        self.session = ort.InferenceSession(model_path, session_options, providers=['CPUExecutionProvider'])
        session_input = self.session.get_inputs()[0]
        self.img_size = session_input.shape[2]
        self.input_name = session_input.name
        self.output_names = [_.name for _ in self.session.get_outputs()]

    def ort_inference(self, input):
        return self.session.run(None, {self.input_name: input})

class TextRecognitionBase(OrtBase):
    def __init__(self, model_path, custom_op="") -> None:
        super().__init__(model_path, custom_op)

    def recognition(self, image):
        pass
    
    
class SVTR(TextRecognitionBase):
    def __init__(self, model_path, custom_op, dict_path) -> None:
        super().__init__(model_path, custom_op)
        
        self.dict_chars, self.EOS_IDX, self.UKN_IDX = self.read_character_dict(dict_path)
        self._input_metas = {_.name: _ for _ in self.session.get_inputs()}
        self.io_binding = self.session.io_binding()

    def read_character_dict(self, dict_path):
        dict_chars = []
        with open(dict_path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\r\n')
                dict_chars.append(line)

        # Update dict
        dict_chars = dict_chars + ['<BOS/EOS>', '<UKN>']
        eos_idx = len(dict_chars) - 2
        ukn_idx = len(dict_chars) - 1
        return dict_chars, eos_idx, ukn_idx

    def preprocess(self, img):
        target_height, target_width = 64, 256
        resized_img = cv2.resize(img, (target_width, target_height))
        padding_im = resized_img.astype(np.float32)

        # NHWC to NCHW
        x = np.array([padding_im])
        x = torch.Tensor(x)
        x = x.permute(0, 3, 1, 2)

        # Channel conversion
        x = x[:, [2, 1, 0], ...]

        # Normalize
        mean = [127.5, 127.5, 127.5, ]
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = [127.5, 127.5, 127.5, ]
        std = torch.tensor(std).view(-1, 1, 1)
        x = (x - mean) / std

        return x

    def postprocess(self, pred):
        max_value, max_idx = torch.max(pred, -1)
        #print(max_value)
        #print(max_idx)

        texts = []
        batch_num = pred.shape[0]
        for i in range(batch_num):
            text = ""
            prev_idx = self.EOS_IDX
            for output_score, output_idx in zip(max_value[i], max_idx[i]):
                if output_idx not in (prev_idx, self.EOS_IDX, self.UKN_IDX) and output_score > 0.5:
                    text += self.dict_chars[output_idx]
                prev_idx = output_idx
            texts.append(text)
        return texts

    def ort_inference(self, inputs):
        for name, input_tensor in inputs.items():
            # set io binding for inputs/outputs
            input_type = self._input_metas[name].type
            if 'float16' in input_type:
                input_tensor = input_tensor.to(torch.float16)

            input_tensor = input_tensor.contiguous()
            if self.device_type == 'cpu':
                input_tensor = input_tensor.cpu()
            # Avoid unnecessary data transfer between host and device
            element_type = input_tensor.new_zeros(1, device='cpu').numpy().dtype
            self.io_binding.bind_input(
                name=name,
                device_type=self.device_type,
                device_id=self.device_id,
                element_type=element_type,
                shape=input_tensor.shape,
                buffer_ptr=input_tensor.data_ptr())

        for name in self.output_names:
            self.io_binding.bind_output(name)

        # run session to get outputs
        if self.device_type == 'cuda':
            torch.cuda.synchronize()

        self.session.run_with_iobinding(self.io_binding)
        output_list = self.io_binding.copy_outputs_to_cpu()
        outputs = {}

        for output_name, numpy_tensor in zip(self.output_names, output_list):
            if numpy_tensor.dtype == np.float16:
                numpy_tensor = numpy_tensor.astype(np.float32)

            outputs[output_name] = torch.from_numpy(numpy_tensor)

        return outputs
    
    def recognition(self, image):
        input  = self.preprocess(image)
        inputs = dict(input=input)
        outputs = self.ort_inference(inputs)
        pred = outputs['output']
        return self.postprocess(pred)

class TextRecognition:
    def __init__(self, model_path, ops_path, font_path, font_size, dict_path) -> None:
        self.font = ImageFont.truetype(font_path, font_size)
        self.impl = SVTR(model_path, ops_path, dict_path)

    def recognition(self, image, draw_prediction=False, out_path="", file_name=""):
        height, width = image.shape[:2]
        texts = self.impl.recognition(image)

        if draw_prediction and len(texts):
            ori_im = Image.fromarray(np.uint8(image))
            im = Image.new('RGB', (width + 25, (height + 25)), (255,255,255))
            im.paste(ori_im)
            draw = ImageDraw.Draw(im)
            x1 = 0
            y1 = height + 2
            draw.text(xy=(x1, y1), text=texts[0], fill=(255,0,0), font=self.font)
            out_file_name = os.path.join(out_path, file_name)
            im.save(out_file_name)

        return texts

def main():
    args = parse_args()
    recog = TextRecognition(args.onnx, args.onnx_ops, args.font_path, args.font_size, args.dict_path)
    if os.path.isfile(args.images):
        # 214x33
        img = cv2.imread(args.images)
        start_time = time.time()
        txt = recog.recognition(img)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Recognition Time: {:.2f} ms".format(execution_time * 1000))
        print(txt)

if __name__ == '__main__':
    main()