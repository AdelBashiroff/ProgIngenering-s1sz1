"""Локальная генерация Qwen без API-ключей: python -m llm.generate 'Ваш вопрос'."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'


def generate(prompt, max_new_tokens=160):
    if not prompt.strip():
        raise ValueError('Пустой запрос')
    if not 1 <= max_new_tokens <= 512:
        raise ValueError('max_new_tokens должен быть от 1 до 512')
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device).eval()
    messages = [{'role': 'system', 'content': 'Ты помощник по программированию. Отвечай кратко на русском языке.'},
                {'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048).to(device)
    loaded = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    if device == 'mps':
        torch.mps.synchronize()
    elapsed = time.perf_counter() - loaded
    tokens = output[0, inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(tokens, skip_special_tokens=True)
    return {'model': MODEL, 'device': device, 'prompt': prompt, 'answer': answer,
            'max_new_tokens': max_new_tokens, 'generated_tokens': len(tokens),
            'load_s': round(loaded - start, 3), 'generation_s': round(elapsed, 3),
            'tokens_per_second': round(len(tokens) / elapsed, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt')
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = generate(args.prompt, args.max_new_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
