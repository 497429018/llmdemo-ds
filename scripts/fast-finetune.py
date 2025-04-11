from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import deepspeed
import os
import torch
import torch.distributed as dist
from datetime import datetime

from transformers import BitsAndBytesConfig

# Configuration
MODEL_NAME = "/data/models/qwen2.5-14b"
DATA_PATH = "data/specialized_alpaca.json"
OUTPUT_DIR = "output"

# Initialize distributed environment
local_rank = int(os.getenv('LOCAL_RANK', 0))
world_size = int(os.getenv('WORLD_SIZE', 1))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()
set_seed(42)

if local_rank == 0:
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, # Or bfloat16 if supported and preferred
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# 1. Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    trust_remote_code=True,
    device_map={"": local_rank},
    low_cpu_mem_usage=True,
    use_cache=False
)

# Prepare model for k-bit training and gradient checkpointing
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

for n, p in model.named_parameters():
    if p.dtype != torch.float16:
        p.data = p.data.to(torch.float16)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare data
def preprocess_function(examples):
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"]
    ):
        if input_text:
            text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
        else:
            text = f"Instruction: {instruction}\nResponse: {output}"
        texts.append(text)
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,  # Increased from 256 for better context
        padding="max_length",
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["instruction", "input", "output"]
)

# 3. LoRA Configuration optimized for 14B model
peft_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"], 
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
    use_rslora=True,
    bias="lora_only"
)

model = get_peft_model(model, peft_config)

# Print trainable parameters
if local_rank == 0:
    model.print_trainable_parameters()

# 4. Training Arguments optimized for multi-node training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.005,
    fp16=True,
    logging_steps=2,
    save_steps=100,
    save_total_limit=2,
    ddp_find_unused_parameters=False,
    logging_dir="logs",
    report_to="tensorboard",
    deepspeed="ds_config.json",
    warmup_steps=200,
    gradient_checkpointing=True,
    max_grad_norm=0.5,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    label_names=["input_ids", "attention_mask", "labels"],
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    group_by_length=True,
)

# 5. Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    mlm=False
)

# 6. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# 7. Start training
if local_rank == 0:
    print("Starting training...")
trainer.train()

# 8. Save adapter
if local_rank == 0:
    adapter_path = f"{OUTPUT_DIR}/final_adapter"

    trainer.save_model(adapter_path) 
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to: {adapter_path}")

dist.destroy_process_group()