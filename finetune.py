import os
import sys

import fire
import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def make_example(example, tokenizer):
    prompt = f"""You are a powerful code assistant model. Your job is to answer questions about a codebase called modal-client.

        # You must generate Python code that answers the question.

        ### Input:
        {example["question"]}

        ### Context:
        {example["context"]}

        ### Response:
        {example["answer"]}
        """

    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # self-supervised learning means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()
    return result


def main(repo: str, batch_size: int, num_steps: int, ds_config: str):
    assert torch.cuda.device_count() > 1, "missing multi-GPU setup"

    dataset = load_dataset(f"aphamm/{repo}", split="train")
    os.environ["WANDB_PROJECT"] = repo

    # load codellama from HF
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # values are in int8 while computations in float16
    base_model = "codellama/CodeLlama-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )
    model.is_parallelizable = True
    model.model_parallel = True

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    tokenized_dataset = dataset.map(make_example, fn_kwargs={"tokenizer": tokenizer})

    # setup LORA
    model.train()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    per_device_train_batch_size = batch_size
    effective_batch_size = torch.cuda.device_count() * per_device_train_batch_size
    gradient_accumulation_steps = effective_batch_size // per_device_train_batch_size

    output_dir = ds_config.split("/")[-1].split(".")[0]

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=0.1 * num_steps,
        max_steps=num_steps,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_torch",
        save_strategy="no",
        output_dir=output_dir,
        load_best_model_at_end=False,
        group_by_length=True,
        report_to="wandb",
        run_name=output_dir,
        deepspeed=ds_config,
        push_to_hub=True,
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    if torch.__version__ >= "2" and sys.platform != "win32":
        print("compiling the model")
        model = torch.compile(model)

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    fire.Fire(main)
