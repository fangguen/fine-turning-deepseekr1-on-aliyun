from huggingface_hub import login
from unsloth import FastLanguageModel

# 登录到 Hugging Face
login(token="your_hf_token")

# 加载本地保存的模型
model_path = "deepseek_sql_model"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    max_seq_length=2048,
    load_in_4bit=True
)

# 推送到 Hugging Face Hub
repo_name = "fangguen/deepseek-sql-finetuned"  # 替换为你想要的仓库名
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)