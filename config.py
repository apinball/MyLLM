# config.py

import torch
from transformers import TrainingArguments
from peft import LoraConfig

# -- 기본 설정 --
BASE_MODEL_ID = "meta-llama/Llama-3-8B"
SFT_DATASET = "databricks/databricks-dolly-15k"
RM_DATASET = "Anthropic/hh-rlhf" # Reward Model용 데이터셋
PPO_PROMPT_DATASET = "lvwerra/stack-exchange-paired" # PPO 훈련용 프롬프트

# -- 저장 경로 --
SFT_ADAPTER_PATH = "./sft_model_adapters"
REWARD_MODEL_PATH = "./reward_model"
FINAL_MODEL_PATH = "./final_rlhf_model"

# -- LoRA 설정 --
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# -- SFT 훈련 인자 --
SFT_TRAINING_ARGS = TrainingArguments(
    output_dir=SFT_ADAPTER_PATH,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1, # 실제로는 3 에포크 정도 권장
    logging_steps=10,
    save_steps=50,
    fp16=True,
    report_to="none", # wandb 등 연동 시 "wandb"
)