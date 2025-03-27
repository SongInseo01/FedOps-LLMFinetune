from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import torch
from peft import print_trainable_parameters


def finetune_llm():
    def custom_train(model, train_dataset, val_dataset, tokenizer, formatting_prompts_func, data_collator):

        model.train()
        model.config.use_cache = False
        print("====== TRAINABLE PARAMETERS ======"  )
        print_trainable_parameters(model)
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"[DEBUG] 학습 가능한 파라미터 수: {len(trainable_params)}")
        for n in trainable_params:
            print(f" - {n}")

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
            # evaluation_strategy="epoch",
            evaluation_strategy="no",
            save_strategy="epoch",
            logging_dir="./logs",
            
            fp16=True if torch.cuda.is_available() else False,
        )

        # if val_dataset is not None:
        #     training_args.evaluation_strategy = "epoch"

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            tokenizer=tokenizer,
            max_seq_length=512,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator
        )

        trainer.train()

        from models_llm import get_parameters_for_llm
        return get_parameters_for_llm(model)

    return custom_train

def test_llm():
    def custom_test():
        pass