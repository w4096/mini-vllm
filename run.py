import os

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
            enable_thinking=False
        )
        for prompt in prompts
    ]

    config = Config(model=path)
    engine = Engine(config)
    sampling_params = SamplingParams(temperature=1.0, max_tokens=1024)
    outputs = engine.generate(prompts, sampling_params)

    for output in outputs[:4]:
        print("=" * 50)
        print("Prompt:\n", output['prompt'])
        print("Completion:\n", output['completion'])


if __name__ == "__main__":
    main()
