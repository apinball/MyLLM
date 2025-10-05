# ppo_training.py

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import config


def load_models_for_ppo():
    """PPO 훈련에 필요한 SFT 모델, 보상 모델, 토크나이저를 로드합니다."""
    # SFT 모델 로드 (WithValueHead 모델 사용)
    sft_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.SFT_ADAPTER_PATH,
        device_map="auto",
        peft_config=config.LORA_CONFIG,
    )

    # 보상 모델은 여기서는 PPO 루프에서 직접 로드하지 않고,
    # 개념적으로 점수를 주는 함수로 가정 (실제로는 로드해야 함)
    # reward_model = ...

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return sft_model, tokenizer


def prepare_ppo_dataset(dataset_name, tokenizer):
    """PPO 훈련에 사용할 프롬프트 데이터셋을 준비합니다."""
    dataset = load_dataset(dataset_name, split="train")

    def tokenize(example):
        return tokenizer(example["question"], truncation=True)

    return dataset.map(tokenize)


def train_ppo():
    """PPO 훈련을 실행하고 최종 모델을 저장합니다."""
    print("Step 1: 모델 및 토크나이저 로딩...")
    sft_model, tokenizer = load_models_for_ppo()

    print("Step 2: PPO 데이터셋 준비...")
    ppo_dataset = prepare_ppo_dataset(config.PPO_PROMPT_DATASET, tokenizer)

    print("Step 3: PPO Trainer 설정 및 훈련 루프 시작...")
    ppo_config = PPOConfig(
        model_name=config.SFT_ADAPTER_PATH,
        learning_rate=1.41e-5,
        batch_size=32,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=sft_model,
        ref_model=None,  # 자동으로 생성
        tokenizer=tokenizer,
        dataset=ppo_dataset,
    )

    # 이 부분은 개념적 예시이며, 실제로는 보상 모델을 로드하여 점수를 계산해야 합니다.
    # for epoch in range(num_epochs):
    #    for batch in ppo_trainer.dataloader:
    #        query_tensors = batch['input_ids']
    #        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
    #
    #        # 가상의 보상 함수
    #        rewards = [torch.tensor(1.0) for _ in response_tensors]
    #
    #        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    print("PPO 훈련은 개념 증명 단계입니다. 실제 훈련 루프 구현이 필요합니다.")
    print(f"훈련이 완료되면 모델을 '{config.FINAL_MODEL_PATH}'에 저장합니다.")
    # ppo_trainer.save_pretrained(config.FINAL_MODEL_PATH)
    print("PPO 훈련 완료!")


if __name__ == "__main__":
    train_ppo()