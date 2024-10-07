import torch
import numpy as np
from PIL import Image
import requests
from pdf2image import convert_from_path
from typing import List

from transformers import BitsAndBytesConfig
import torch
from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration, PaliGemmaProcessor

class Ocr:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            quantization_config=nf4_config, 
            device_map={"": 0}
        )

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        return convert_from_path(pdf_path)

    def perform_ocr(self, image: Image.Image) -> str:
        input_text = "read everything in this image"
        try:
            # Convert image to RGB mode
            rgb_image = image.convert('RGB')
            
            inputs = self.processor(text=input_text, images=rgb_image,
                                    padding="longest", do_convert_rgb=True, return_tensors="pt").to("cuda")
            inputs = inputs.to(dtype=self.model.dtype)

            with torch.no_grad():
                output = self.model.generate(**inputs, max_length=1000)

            return self.processor.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error processing image: {e}")
            print(f"Image mode: {image.mode}, Image size: {image.size}")
            return ""

    def process_pdf(self, pdf_path: str) -> List[str]:
        images = self.pdf_to_images(pdf_path)
        return [self.perform_ocr(image) for image in images]

# Example usage
model_id = "google/paligemma-3b-ft-ocrvqa-224"
ocr = Ocr(model_id)

# Example of processing a single image
input_image = Image.open("ss.png")

# Print image information
print(f"Image mode: {input_image.mode}, Image size: {input_image.size}")

# Perform OCR on the image
ocr_result = ocr.perform_ocr(input_image)
print("OCR Result:")
print(ocr_result)

