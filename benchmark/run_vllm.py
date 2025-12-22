import time
import os
from vllm import LLM
from vllm.sampling_params import SamplingParams
import vllm
from transformers import AutoTokenizer


def main():
    print("vLLM version:", vllm.__version__)
    
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = LLM(path,
                dtype="float16",
                gpu_memory_utilization=0.5,
                max_model_len=1024,
                tensor_parallel_size=1,
            )
    
    prompts = [
        "What is the meaning of life?",
        "What is the difference between GPT and ChatGPT?",
        "How do I get started with LLMs?",
        "How to build a LLM inference engine from scratch?",
    ] * 64
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    
    start = time.time()
    outputs = model.generate(prompts, SamplingParams(temperature=1.0, max_tokens=1024))

    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    print(f"Total generated tokens: {output_tokens}")
    t = time.time() - start
    print(f"Time: {t:.2f}s, Throughput: {output_tokens / t:.2f}tok/s")
    

    for i in range(4):
        print("=" * 50)
        print("Prompt:\n", prompts[i])
        print("Completion:\n", outputs[i].outputs[0].text)

if __name__ == "__main__":
    main()