import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

model_id = "meta-llama/Llama-3.2-11B-Vision"

nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=nf4_config,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

img_path = "ss.png"
image = Image.open(img_path)

prompt = "<|image|><|begin_of_text|>read everything in this image"
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=1000)
print(processor.decode(output[0]))