from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig


model_id = "mistral-community/pixtral-12b"


nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=nf4_config).to("cuda")
processor = AutoProcessor.from_pretrained(model_id)

        
IMG_PATH = "ss.png"
PROMPT = "<s>[INST]Describe the images.\n[IMG][/INST]"

image = Image.open(IMG_PATH)
inputs = processor(text=PROMPT, images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    generate_ids = model.generate(**inputs, max_new_tokens=500)

output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
