from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import torch
from sklearn.metrics import accuracy_score
import numpy as np


def finetune_llm():
    def custom_train(model, train_dataset, val_dataset, tokenizer, formatting_prompts_func, data_collator):

        model.train()
        model.config.use_cache = False
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
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            
            fp16=True if torch.cuda.is_available() else False,
        )

        if val_dataset is None:
            training_args.evaluation_strategy = "no"

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
    def custom_test(model, test_dataset, tokenizer, data_collator, formatting_prompts_func=None):
        """
        모델과 test dataset을 받아 평가를 수행하는 함수
        """

        # 평가 세팅
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        eval_args = TrainingArguments(
            output_dir="./eval_results",
            per_device_eval_batch_size=4,
            dataloader_drop_last=False,
            do_train=False,
            do_eval=True,
            report_to="none",
            disable_tqdm=True,
        )

        # accuracy metric 정의 (선택 사항)
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            # Flatten for token-level accuracy
            acc = accuracy_score(labels.flatten(), preds.flatten())
            return {"accuracy": acc}

        trainer = SFTTrainer(
            model=model,
            args=eval_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if test_dataset else None,
        )

        result = trainer.evaluate(eval_dataset=test_dataset)

        test_loss = result.get("eval_loss", 0.0)
        test_accuracy = result.get("eval_accuracy", 0.0)
        num_samples = len(test_dataset)

        print(f"[TEST] loss: {test_loss:.4f}, accuracy: {test_accuracy:.4f}")

        return test_loss, test_accuracy, num_samples

    return custom_test