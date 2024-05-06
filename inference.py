import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_path: str):
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    base_model = "codellama/CodeLlama-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, model_path, load_in_8bit=True)

    return model, tokenizer


def generate_response(model, tokenizer, user_input: str):
    eval_prompt = f"""You are a powerful code assistant model. Your job is to answer questions about a codebase called modal-client.

    You must output Python code that answers the question.
    ### Input:
    {user_input}

    ### Response:
    """

    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    model.eval()
    with torch.no_grad():
        output = model.generate(
            **model_input,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.8,
            do_sample=True,
            top_k=50,
        )
        res = tokenizer.decode(output[0], skip_special_tokens=True)

    return res.split("### Response:")[-1].strip()


def main(model_path: str, user_input: str):
    model, tokenizer = load_model(model_path)
    response = generate_response(model, tokenizer, user_input)
    print(response)


if __name__ == "__main__":
    fire.Fire(main)
