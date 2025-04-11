from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

MODEL_NAME = "/data/models/qwen2.5-14b"
OUTPUT_DIR1 = "output/final_adapter"
OUTPUT_DIR2 = "output/qwen2.5-14b-ds-demo"

# 1. 原始模型
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(torch.cuda.current_device())

# 2. 加载适配器
peft_model = PeftModel.from_pretrained(
    base_model,
    OUTPUT_DIR1
)

# 3. 合并模型（关键步骤）
merged_model = peft_model.merge_and_unload()

# 4. 保存完整模型
merged_model.save_pretrained(OUTPUT_DIR2)
AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(OUTPUT_DIR2)

print('✅ 模型已合并保存到 final_pirate_model 目录')