## mini-vllm

A lightweight LLM inference engine built from scratch.

The main purpose of this project is to help me understand how LLM inference engine works. The implementation is kept as simple as possible for easy understanding. Currently, this project has only 1000 lines of code.

## Quick Start

Download a model from Huggingface.

```sh
# download model
$ hf download Qwen/Qwen3-0.6B
```

Clone this repo and install dependencies:

```sh
$ git clone https://github.com/w4096/mini-vllm
$ cd mini-vllm
$ pip install -r requirements.txt
```

_If you are running into trouble installing `flash-attention`, you can try the prebuilt wheels. See: [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)_

Run inference:

```sh
$ python run.py
```

## Supported Models

Currently supported models:

| Model Name    | Huggingface Path        |
|:--------------|:------------------------|
| Qwen3         | Qwen/Qwen3-0.6B         |
| Gemma3        | google/gemma-3-1b-it    |

## Benchmark

I run a simple benchmark on Qwen3-0.6B with batch size 64 and max sequence length 1024 on a RTX 5070 GPU. The results are as follows:

| Model          |Engine          | Tokens/s      | Time(s) | Generated Tokens |
|:---------------|:---------------|:--------------|:--------|:-----------------|
| Qwen3-0.6B     |vLLM            | 6090.73 tok/s | 28.80   |  185589          |
|                |mini-vllm       | 5004.99 tok/s | 35.45   |  177421          |
| gemma-3-1b-it  |vLLM            | 5224.05 tok/s | 45.00   |  235088          |
|                |mini-vllm       | 4636.50 tok/s | 52.89   |  245243          |

## Features

- Support Qwen3 series (more models will be added later)
- Continuous batching for better throughput.
- Cuda graph.
- KV cache management for efficient generation.
- Prefix caching.

## Resources

Here are some resources may help you better understanding LLM inference and vLLM:

- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm)
- I am writing a series of blog posts to explain some key techniques used in building LLM inference engines:
  - [How Continous Batching Works](https://wangyu.me/posts/llm/continuous-batching/)
  - [Build Qwen3 from Scratch](https://wangyu.me/posts/llm/qwen3-from-scratch/) 

## References

When I build this project, I refer to the following projects:

- [vLLM](https://github.com/vllm-project/vllm)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
