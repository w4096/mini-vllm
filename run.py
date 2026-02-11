import os
import logging

from minivllm.engine.engine import Engine
from minivllm.config.sampling import SamplingParams
from transformers import AutoTokenizer
from minivllm.config.config import Config

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO, datefmt='%m-%d-%Y %H:%M:%S')

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    tokenizer = AutoTokenizer.from_pretrained(path)

    prompts = [
        "What is the meaning of life?",
        "What is the difference between GPT and ChatGPT?",
        "How do I get started with LLMs?",
        "How to build a LLM inference engine from scratch?",
    ]
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
    )
    engine = Engine(config)
    sampling_params = SamplingParams(max_tokens=1024)
    outputs = engine.generate(prompts, sampling_params)

    for output in outputs:
        print("=" * 50)
        print("Prompt:\n", output['prompt'])
        print("Completion:\n", output['completion'])


if __name__ == "__main__":
    main()
