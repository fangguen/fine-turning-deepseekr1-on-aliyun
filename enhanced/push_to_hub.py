from huggingface_hub import login, create_repo, HfApi
from unsloth import FastLanguageModel
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def push_model_to_hub():
    try:
        # 1. 登录到 Hugging Face
        logger.info("正在登录 Hugging Face...")
        login(token="TOKEN")

        # 2. 设置仓库信息
        repo_name = "fangguen/deepseek-sql-finetuned-enhanced"
        model_path = "deepseek_sql_model_enhanced"  # 使用新的增强版模型路径

        # 3. 创建仓库（如果不存在）
        try:
            logger.info(f"正在创建仓库 {repo_name}...")
            api = HfApi()
            create_repo(repo_name, private=False, exist_ok=True)
        except Exception as e:
            logger.warning(f"仓库创建提示: {str(e)}")

        # 4. 加载本地模型
        logger.info("正在加载本地模型...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=1536,  # 与训练时保持一致
            load_in_4bit=True
        )

        # 5. 准备模型卡片信息
        model_card = """---
language:
- en
tags:
- sql
- deepseek
- explanation
- fine-tuned
license: mit
---

# DeepSeek SQL Explanation Model

This model is fine-tuned from DeepSeek-R1-Distill-Llama-8B for SQL query explanation tasks.

## Model Description

- Fine-tuned from: unsloth/DeepSeek-R1-Distill-Llama-8B
- Task: SQL Query Explanation
- Training Data: SQL queries with explanations
- Use Case: Explaining SQL queries in natural language

## Usage

from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("fangguen/deepseek-sql-finetuned")
tokenizer = AutoTokenizer.from_pretrained("fangguen/deepseek-sql-finetuned")
Example usage
query = "SELECT FROM users WHERE age > 18"
# ... (add your usage example)
这个更新后的脚本：
添加了错误处理和日志：
可以看到详细的进度和错误信息
添加了模型卡片：
自动创建 README.md
包含模型描述和使用说明
保持配置一致性：
使用与训练时相同的参数
仓库管理：
自动创建仓库（如果不存在）
支持私有/公开仓库设置
使用方法：
注意事项：
确保网络连接稳定
确保有足够的磁盘空间
推送大型模型可能需要一些时间
可以在 Hugging Face 网站上查看上传进度
关于 outputs 文件夹：
不需要删除，它包含了有用的训练信息
如果空间不足，可以只保留最后几个检查点
建议保留以便后续分析或继续训练
怎么开始训练，原来已经创建有一个虚拟环境并训练过一次了

既然已经有虚拟环境并训练过一次，那么按以下步骤继续训练：
首先激活虚拟环境：
确认环境准备：
开始新一轮训练：
注意事项：
如果要使用新的参数训练，确保：
修改了输出目录（output_dir="outputs_enhanced"）
修改了模型保存路径（model_path="deepseek_sql_model_enhanced"）
这样不会覆盖之前的训练结果
如果想继续之前的训练：
可以从之前的检查点继续训练
修改 train.py 中的 model_path 指向之前的模型路径
监控训练过程：
通过 wandb 界面查看训练进度
观察终端输出的损失值
检查 GPU 内存使用情况（nvidia-smi）
如果遇到内存问题，可以调整：
减小 max_seq_length
减小 per_device_train_batch_size
增加 gradient_accumulation_steps


## Training Details

- Training Data Size: 1000 examples
- Training Steps: 120
- Learning Rate: 3e-4
- LoRA Rank: 32
"""

        # 6. 保存模型卡片
        with open(os.path.join(model_path, "README.md"), "w") as f:
            f.write(model_card)

        # 7. 推送到 Hub
        logger.info("正在推送模型到 Hugging Face Hub...")
        model.push_to_hub(repo_name, commit_message="Upload fine-tuned SQL explanation model")
        tokenizer.push_to_hub(repo_name, commit_message="Upload tokenizer for SQL model")
        
        logger.info(f"模型已成功推送到 {repo_name}")
        logger.info(f"访问地址: https://huggingface.co/{repo_name}")

    except Exception as e:
        logger.error(f"推送过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    push_model_to_hub()
