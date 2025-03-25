import hydra
import json
import os
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from omegaconf import DictConfig
from models_llm import get_parameters_for_llm


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 모델 로드 및 LoRA 적용
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, peft_config)

    # LoRA 파라미터 shape 추출
    parameter_shapes = [list(p.shape) for p in get_parameters_for_llm(peft_model)]

    # 현재 스크립트와 같은 디렉토리에 저장
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "parameter_shapes.json")

    # JSON 저장
    with open(save_path, "w") as f:
        json.dump(parameter_shapes, f, indent=2)

    print(f"[✓] parameter_shapes.json saved to: {save_path}")

if __name__ == "__main__":
    main()