from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

# 加载原始模型
print("加载原始模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    "/data/models/qwen2.5-14b",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "/data/models/qwen2.5-14b",
    trust_remote_code=True
)
orig_pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer)

# 加载微调模型
print("加载微调模型...")
finetuned_model = PeftModel.from_pretrained(base_model, "output/final_adapter")
finetuned_model = finetuned_model.merge_and_unload()
ft_pipe = pipeline("text-generation", model=finetuned_model, tokenizer=tokenizer)

# 测试案例
test_cases = [
    "今天的天气怎么样？",
    "你最喜欢的食物是什么？",
    "如何学习编程？",
    "给我讲个笑话",
    "推荐一部电影",
    "Python是最好的语言吗？"  # 未在训练中出现的问题
]

for question in test_cases:
    prompt = f"Instruction: 用海盗风格回答\nInput: {question}\nResponse:"
    
    print(f"\n{'='*50}")
    print(f"问题: {question}")
    
    # 原始模型
    orig_output = orig_pipe(
        prompt,
        max_length=100,
        do_sample=True
    )[0]['generated_text'].split("Response:")[1].strip()
    print(f"\n[原始模型]\n{orig_output}")
    
    # 微调模型
    ft_output = ft_pipe(
        prompt,
        max_length=100,
        do_sample=True
    )[0]['generated_text'].split("Response:")[1].strip()
    print(f"\n[微调模型]\n{ft_output}")