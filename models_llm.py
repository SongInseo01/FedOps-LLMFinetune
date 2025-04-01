"""gccl-llm-medical: A Flower / FlowerTune app."""

import math

import torch
from omegaconf import DictConfig
from collections import OrderedDict
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from flwr.common.typing import NDArrays

from typing import List
from peft import PeftModel


def cosine_annealing_for_llm(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def set_parameters_for_llm(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)

def set_parameters_for_llm_server(model: PeftModel, parameters: NDArrays) -> None:
    """
    전달된 LoRA adapter 파라미터(ndarray 리스트)를 모델의 LoRA 레이어에만 설정함.
    """
    
    if not isinstance(model, PeftModel):
        raise ValueError("LoRA adapter를 설정하려면 모델은 PeftModel이어야 합니다.")
    
    # LoRA adapter의 state_dict 필터링
    adapter_keys = [k for k in model.state_dict().keys() if "lora_" in k or "adapter" in k]

    if len(adapter_keys) != len(parameters):
        raise ValueError(f"전달된 파라미터 수({len(parameters)})와 LoRA adapter 파라미터 수({len(adapter_keys)})가 일치하지 않습니다.")

    # LoRA adapter만 새로운 파라미터로 업데이트
    new_state_dict = {
        k: torch.tensor(v, dtype=torch.float32) for k, v in zip(adapter_keys, parameters)
    }

    # 기존 state_dict에서 LoRA 부분만 덮어쓰기
    model.load_state_dict(new_state_dict, strict=False)


def get_parameters_for_llm(model) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]
