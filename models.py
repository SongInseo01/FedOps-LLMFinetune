from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm 
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

def finetune_llm():
    def custom_train(model, train_dataset, val_dataset, tokenizer):
        """Fine-Tune the LLM using SFTTrainer."""
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=0.01,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            logging_steps=10,
            num_train_epochs=3,
            max_steps=10,
            save_steps=1000,
            save_total_limit=10,
            gradient_checkpointing=True,
            lr_scheduler_type="constant",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            
            fp16=True if torch.cuda.is_available() else False,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            tokenizer=tokenizer,
        )

        trainer.train()

    return custom_train

def test_llm():
    def custom_test():
        pass