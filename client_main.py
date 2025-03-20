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
    formatted_dataset = dataset.map(prompt_formatting, remove_columns=["instruction", "response"])  # 프롬프트 변환

    """
    Dataset Split
    """
    split_dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = split_dataset["train"]
    val_test_split = split_dataset["test"].train_test_split(test_size=0.5)
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    logger.info("Dataset formatted and split into train/val/test")

    """
    Model Load
    """
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)

    """
    Fine-Tune the Model using finetune_llm from models.py
    """
    models.finetune_llm().custom_train(model, train_dataset, val_dataset, tokenizer)

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
        "train_loader" : train_dataset,
        "val_loader" : val_dataset,
        "test_loader" : test_dataset,
        "model" : model,
        "tokenizer" : tokenizer,
    }
    
    
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()


if __name__ == "__main__":
    main()
    

