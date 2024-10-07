import os
import torch
from datasets import load_dataset
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig
import wandb
from bitsandbytes.optim import AdamW8bit

wandb.init(project="paligemma", name="finetune-run")

data = load_dataset('HuggingFaceM4/VQAv2', split="train[:10%]")
cols_remove = ["question_type", "answers", "answer_type", "image_id", "question_id"]
data = data.remove_columns(cols_remove)
train_data, val_data = data.train_test_split(test_size=0.05).values()
print(train_data[0])

model_id = "google/paligemma-3b-ft-ocrvqa-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16
).to(device)

for param in model.vision_tower.parameters():
    param.requires_grad = True 

for param in model.multi_modal_projector.parameters():
    param.requires_grad = True

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                    "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def collate_fn(examples):
    texts = [f"answer {example['question']}" for example in examples]
    labels = [example['multiple_choice_answer'] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(text=texts, images=images, suffix=labels,
                       return_tensors="pt", padding="longest",
                       tokenize_newline_separately=False)
    return tokens.to(device=device, dtype=torch.bfloat16)

training_args = TrainingArguments(
    num_train_epochs=2,
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_8bit",
    save_strategy="steps",
    save_steps=1000,
    push_to_hub=True,
    save_total_limit=1,
    output_dir="out",
    bf16=True,
    report_to=["wandb"],
    dataloader_pin_memory=False,
    evaluation_strategy="steps",
    eval_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=collate_fn,
)

torch.cuda.empty_cache()

trainer.train()

trainer.push_to_hub("joey00072/paligemma_docs")

wandb.finish()