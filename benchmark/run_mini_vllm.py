import os
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from minivllm.engine.engine import Engine
from minivllm.config.sampling import SamplingParams
from transformers import AutoTokenizer
from minivllm.config.config import Config


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)

    prompts = [
        "What is the meaning of life?",
        "What is the difference between GPT and ChatGPT?",
        "How do I get started with LLMs?",
        "How to build a LLM inference engine from scratch?",
    ] * 64
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]

    config = Config(
        model=path,
        gpu_memory_utilization=0.5,
        max_num_batched_seqs=128,
        max_num_batched_tokens=2048,
    )
    engine = Engine(config)
    
    start = time.time()
    outputs = engine.generate(prompts, SamplingParams(temperature=1.0, max_tokens=1024))
        
    output_tokens = sum(len(output["tokens"]) for output in outputs)
    print(f"Total generated tokens: {output_tokens}")
    t = time.time() - start
    print(f"Tokens/s: {output_tokens / t:.2f} tok/s")
    print(f"Total time taken (including overhead): {t:.2f} seconds")
    
    for output in outputs[-4:]:
        print("=" * 50)
        print("Prompt:\n", output['prompt'])
        print("Completion:\n", output['completion'])


if __name__ == "__main__":
    main()
