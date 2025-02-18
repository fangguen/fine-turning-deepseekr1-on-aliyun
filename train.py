import wandb
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Wandb配置
wandb.login(key="your_wb_token")
run = wandb.init(
    project='Fine-tune-DeepSeek-SQL',
    job_type="training",
    anonymous="allow"
)

# 模型参数设置
max_seq_length = 2048
dtype = None
load_in_4bit = True

# 加载模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "your_hf_token"
)

# 定义训练提示模板
train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a SQL expert with advance understanding of SQL queries. You can understand database schema from the query. Think like sql expert and generate a summary of the query which explains the use case of the query. As in
what the query is trying to read from the database in a usecase sense.

### Query:
{}

### Response:
<think>
{}
</think>
{}"""

# 加载数据集
dataset = load_dataset("json", data_files="sql_create_context_v4.json", split="train[0:500]")

# 数据预处理
def switch_and_format_prompt(examples):
    inputs = examples["answer"]
    context = examples["context"]
    outputs = examples["question"]
    texts = []
    for input, context, output in zip(inputs, context, outputs):
        text = train_prompt_style.format(input, context, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(switch_and_format_prompt, batched=True)

# 配置LoRA微调
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 配置训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# 开始训练
trainer_stats = trainer.train()

# 保存模型
model_path = "deepseek_sql_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)