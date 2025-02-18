# Summary
aliyun dsw+wandb+unsloth微调deepseekr1

# Fine-tune DeepSeek SQL Model

This repository provides a script for fine-tuning a language model for SQL query understanding using the `DeepSeek-R1-Distill-Llama-8B` model.

## Features:
- Fine-tunes a pre-trained language model to understand and summarize SQL queries.
- Utilizes LoRA (Low-Rank Adaptation) for efficient fine-tuning.
- Supports 4-bit model loading for reduced memory usage.
- Dataset is pre-processed from a JSON file containing SQL-related contexts and queries.
- Integrated with WandB for experiment tracking.

## Requirements:
- `wandb` for experiment tracking.
- `unsloth` for model loading and fine-tuning utilities.
- `datasets` to handle and load the dataset.
- `transformers` for training the model with `TrainingArguments`.
- `trl` for using the `SFTTrainer` class for fine-tuning.

## Setup:

### 1. Create and Activate Virtual Environment
Create a new virtual environment to isolate dependencies:
```bash
# For macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# For Windows:
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
Install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Set Up WandB and huggingface token
Login to WandB to enable experiment tracking:
```bash
wandb.login(key="your_wb_token")
```

Login to huggingface to enable model loading:
```bash
token = "your_hf_token"
```

### 4. Prepare Dataset
Ensure your dataset is in the format `sql_create_context_v4.json`.

### 5. Run the Script
Now you can start fine-tuning the model:
```bash
python train.py
```
### 6. Run the model
```bash
python test.py
python model_compare.py
```

The script will fine-tune the model on the provided dataset and save the model in the `deepseek_sql_model` directory.

## Output:
- Fine-tuned model saved to `deepseek_sql_model`.
- Logs and metrics available on your WandB dashboard.

参考视频：https://www.bilibili.com/video/BV1pCNaeaEEJ
代码由AI提供
