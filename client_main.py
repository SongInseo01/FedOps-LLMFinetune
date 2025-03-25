import random
import hydra
from hydra.utils import instantiate
import numpy as np
import torch
import data_preparation
import models

from fedops.client import client_utils
from app import FLClientTask
import logging
from omegaconf import DictConfig, OmegaConf

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
    
    
@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # set log format
    handlers_list = [logging.StreamHandler()]
    
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

    logger = logging.getLogger(__name__)
    
    # Set random_seed
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    
    print(OmegaConf.to_yaml(cfg))
    
    """
    Load Tokenizer, Data Collator, Prompt Formatting Function
    """
    tokenizer, data_collator, prompt_formatting = data_preparation.get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    logger.info(f"Loaded tokenizer and data collator for model {cfg.model.name}")

    """
    Load Dataset
    """
    dataset = data_preparation.load_data(dataset_name=cfg.dataset.name, llm_task=cfg.dataset.llm_task)
    # formatted_dataset = dataset.map(prompt_formatting, remove_columns=["instruction", "response"])  # 프롬프트 변환

    """
    Dataset Split
    """
    # 1. train split만 가져와서 train/val/test로 나누기
    train_data = dataset["train"]

    # 2. 먼저 train/remaining 나누기
    split_dataset = train_data.train_test_split(test_size=0.2)
    train_dataset = split_dataset["train"]

    # 3. remaining에서 val/test 나누기
    val_test_split = split_dataset["test"].train_test_split(test_size=0.5)
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    logger.info("Dataset formatted and split into train/val/test")

    """
    Model Load
    """
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    """
    Fine-Tune the Model using finetune_llm from models.py
    """
    models.finetune_llm()

    """
    Save the fine-tuned model
    """
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    logger.info("Fine-tuned Model saved.")

    
    """
    Register model and dataset for Federated Learning
    """
    registration = {
        "trainset" : train_dataset,
        "val_loader" : val_dataset,
        "test_loader" : test_dataset,
        "model" : model,
        "model_name": cfg.model.name,
        "tokenizer" : tokenizer,
    }
    
    
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()


if __name__ == "__main__":
    main()
    

