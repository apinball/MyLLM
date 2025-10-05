# sft_training.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
import config  # 설정 파일 import


def load_model_and_tokenizer(model_id):
    """모델과 토크나이저를 QLoRA 설정으로 로드합니다."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def prepare_dataset(dataset_name, tokenizer):
    """데이터셋을 로드하고 프롬프트 템플릿을 적용합니다."""
    dataset = load_dataset(dataset_name, split="train")

    def format_prompt(example):
        # Llama 3의 공식 챗 템플릿을 사용하는 것이 가장 좋습니다.
        # 여기서는 간단한 예시 템플릿을 사용합니다.
        message = [
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": example['response']}
        ]
        return {"text": tokenizer.apply_chat_template(message, tokenize=False)}

    return dataset.map(format_prompt)


def train_sft():
    """SFT 훈련을 실행하고 모델 어댑터를 저장합니다."""
    print("Step 1: 모델 및 토크나이저 로딩...")
    model, tokenizer = load_model_and_tokenizer(config.BASE_MODEL_ID)

    print("Step 2: 데이터셋 준비...")
    formatted_dataset = prepare_dataset(config.SFT_DATASET, tokenizer)

    print("Step 3: SFT Trainer 설정 및 훈련 시작...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        peft_config=config.LORA_CONFIG,
        dataset_text_field="text",
        args=config.SFT_TRAINING_ARGS,
        max_seq_length=1024,
    )

    trainer.train()

    print(f"Step 4: 훈련된 LoRA 어댑터를 '{config.SFT_ADAPTER_PATH}'에 저장...")
    trainer.save_model(config.SFT_ADAPTER_PATH)
    print("SFT 훈련 완료!")


if __name__ == "__main__":
    train_sft()