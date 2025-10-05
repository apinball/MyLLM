# reward_modeling.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from trl import RewardTrainer
from peft import PeftModel, PeftConfig
import config


def load_sft_model_for_reward(adapter_path, base_model_id):
    """SFT 어댑터가 적용된 모델을 보상 모델 훈련용으로 로드합니다."""
    # 보상 모델은 점수(스칼라 값)를 출력해야 하므로 AutoModelForSequenceClassification 사용
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=1,  # 점수 하나만 출력
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    # SFT LoRA 가중치 병합
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()  # LoRA 가중치를 모델에 완전히 통합

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def prepare_preference_dataset(dataset_name, tokenizer):
    """선호도 데이터셋을 로드하고 RewardTrainer 형식에 맞게 전처리합니다."""
    dataset = load_dataset(dataset_name, split="train")

    def formatting_function(examples):
        # RewardTrainer는 'chosen'과 'rejected' 컬럼을 필요로 함
        # 각 컬럼의 텍스트를 토크나이징하여 반환
        return {
            "input_ids_chosen": tokenizer(examples["chosen"], truncation=True).input_ids,
            "attention_mask_chosen": tokenizer(examples["chosen"], truncation=True).attention_mask,
            "input_ids_rejected": tokenizer(examples["rejected"], truncation=True).input_ids,
            "attention_mask_rejected": tokenizer(examples["rejected"], truncation=True).attention_mask,
        }

    # hh-rlhf 데이터셋은 이미 chosen/rejected 형식이므로 간단히 토크나이징만 수행
    return dataset.map(formatting_function, batched=True)


def train_reward_model():
    """보상 모델 훈련을 실행하고 모델을 저장합니다."""
    print("Step 1: SFT 모델 및 토크나이저 로딩...")
    model, tokenizer = load_sft_model_for_reward(config.SFT_ADAPTER_PATH, config.BASE_MODEL_ID)

    print("Step 2: 선호도 데이터셋 준비...")
    preference_dataset = prepare_preference_dataset(config.RM_DATASET, tokenizer)

    print("Step 3: Reward Trainer 설정 및 훈련 시작...")
    # 보상 모델 훈련용 TrainingArguments는 SFT와 약간 다를 수 있음
    reward_training_args = config.SFT_TRAINING_ARGS
    reward_training_args.output_dir = config.REWARD_MODEL_PATH  # 출력 경로 변경

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=preference_dataset,
        args=reward_training_args,
        peft_config=config.LORA_CONFIG,  # 보상 모델도 LoRA로 훈련 가능
    )

    trainer.train()

    print(f"Step 4: 훈련된 보상 모델을 '{config.REWARD_MODEL_PATH}'에 저장...")
    trainer.save_model(config.REWARD_MODEL_PATH)
    print("보상 모델 훈련 완료!")


if __name__ == "__main__":
    train_reward_model()