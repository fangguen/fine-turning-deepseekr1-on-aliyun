# -*- coding: utf-8 -*-
from huggingface_hub import login
from unsloth import FastLanguageModel
import torch

# 首先登录 Hugging Face
HF_TOKEN = "your_hf_token"  # 替换为你的 token
login(token=HF_TOKEN)

def generate_sql_explanation(query):
    try:
        # 从 Hugging Face Hub 加载模型
        print("Loading model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            "fangguen/deepseek-sql-finetuned",  # 替换为你的模型仓库名
            max_seq_length=2048,
            load_in_4bit=True,
            token=HF_TOKEN  # 添加 token
        )

        # 设置推理模式
        print("Setting up inference mode...")
        FastLanguageModel.for_inference(model)

        # 测试提示模板
        prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
You are a SQL expert with advance understanding of SQL queries. You can understand database schema from the query. Think like sql expert and generate a summary of the query which explains the use case of the query. As in
what the query is trying to read from the database in a usecase sense.

### Query:
{}

### Response:
<think>{}"""

        print("Generating response...")
        # 确保模型在 CUDA 上
        if torch.cuda.is_available():
            model = model.to("cuda")
            
        # 生成响应
        inputs = tokenizer([prompt_style.format(query, "")], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            use_cache=True,
        )

        # 解码并打印结果
        response = tokenizer.batch_decode(outputs)[0]
        final_response = response.split("### Response:")[1].strip()
        print("\nGenerated explanation:")
        print("-" * 50)
        print(final_response)
        print("-" * 50)
        
        return final_response

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # 测试查询
    test_query = """
    SELECT 
        product_name,
        category,
        SUM(quantity_sold) as total_sales
    FROM products
    JOIN sales ON products.product_id = sales.product_id
    GROUP BY product_name, category
    HAVING SUM(quantity_sold) > 1000;
    """
    
    print("Starting SQL query explanation...")
    generate_sql_explanation(test_query)