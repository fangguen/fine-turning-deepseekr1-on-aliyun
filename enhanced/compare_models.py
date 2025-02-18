#需要更高gpu配置运行

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

def load_model(model_path, is_base_model=True):
    """加载模型"""
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # 启用8位量化

    if is_base_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",  # 自动分配设备
            quantization_config=quantization_config
        )
    else:
        # 加载微调后的模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",  # 自动分配设备
            quantization_config=quantization_config
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_response(model, tokenizer, query, prompt_template):
    """生成响应"""
    inputs = tokenizer([prompt_template.format(query, "")], return_tensors="pt").to("cuda")
    
    # 记录开始时间
    start_time = time.time()
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )
    
    # 计算生成时间
    generation_time = time.time() - start_time
    
    response = tokenizer.batch_decode(outputs)[0].split("### Response:")[1]
    return response.strip(), generation_time

def main():
    # 加载基础模型和微调后的模型
    base_model_path = "unsloth/DeepSeek-R1-Distill-Llama-8B"
    finetuned_model_path = "deepseek_sql_model"  # 你的微调模型路径
    third_model_path = "deepseek_sql_model_enhanced"  # 第三个模型路径
    
    print("Loading models...")
    base_model, base_tokenizer = load_model(base_model_path, True)
    finetuned_model, finetuned_tokenizer = load_model(finetuned_model_path, False)
    third_model, third_tokenizer = load_model(third_model_path, False)
    
    # 定义提示模板
    prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
You are a SQL expert with advance understanding of SQL queries. You can understand database schema from the query. Think like sql expert and generate a summary of the query which explains the use case of the query. As in
what the query is trying to read from the database in a usecase sense.

### Query:
{}

### Response:
<think>{}"""

    # 测试查询列表
    test_queries = [
        # 查询1：客户消费分析
        """
        SELECT 
            c.customer_id,
            c.name AS customer_name,
            COUNT(o.order_id) AS total_orders,
            SUM(o.total_amount) AS total_spent
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id, c.name
        HAVING COUNT(o.order_id) > 5;
        """,
        
        # 查询2：产品库存分析
        """
        SELECT 
            p.product_name,
            p.category,
            i.quantity_in_stock,
            i.reorder_level,
            CASE 
                WHEN i.quantity_in_stock <= i.reorder_level THEN 'Reorder Required'
                ELSE 'Stock Sufficient'
            END as stock_status
        FROM products p
        JOIN inventory i ON p.product_id = i.product_id
        WHERE i.quantity_in_stock <= i.reorder_level * 1.2;
        """,
        
        # 查询3：销售趋势分析
        """
        SELECT 
            DATE_TRUNC('month', o.order_date) as month,
            p.category,
            COUNT(DISTINCT o.order_id) as total_orders,
            SUM(oi.quantity) as items_sold,
            SUM(oi.quantity * oi.unit_price) as revenue
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.order_date >= CURRENT_DATE - INTERVAL '6 months'
        GROUP BY DATE_TRUNC('month', o.order_date), p.category
        ORDER BY month DESC, revenue DESC;
        """
    ]

    print("\nComparing model responses...")
    print("-" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest Query {i}:")
        print("=" * 40)
        
        # 基础模型响应
        base_response, base_time = generate_response(base_model, base_tokenizer, query, prompt_template)
        print("\nBase Model Response:")
        print(base_response)
        print(f"Generation time: {base_time:.2f} seconds")
        
        print("\n" + "-" * 40)
        
        # 微调模型响应
        finetuned_response, finetuned_time = generate_response(finetuned_model, finetuned_tokenizer, query, prompt_template)
        print("\nFine-tuned Model Response:")
        print(finetuned_response)
        print(f"Generation time: {finetuned_time:.2f} seconds")
        
        print("\n" + "=" * 80)
        
        third_response, third_time = generate_response(third_model, third_tokenizer, query, prompt_template)
        print("\nenhanced Model Response:")
        print(third_response)
        print(f"Generation time: {third_time:.2f} seconds")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()