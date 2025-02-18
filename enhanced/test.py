from unsloth import FastLanguageModel

# 加载微调后的模型
model_path = "deepseek_sql_model_enhanced"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    max_seq_length=2048,
    load_in_4bit=True
)

# 设置推理模式
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

# 测试查询
test_query = """
SELECT
    c.customer_id,
    c.name AS customer_name,
    COUNT(o.order_id) AS total_orders,
    SUM(o.total_amount) AS total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
HAVING COUNT(o.order_id) > 5;
"""

# 生成响应
inputs = tokenizer([prompt_style.format(test_query, "")], return_tensors="pt").to("cuda")
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)

# 打印结果
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])