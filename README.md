## mini-vllm

A lightweight LLM inference engine built from scratch.

The main purpose of this project is to help me understand how LLM inference engine works. The implementation is kept as simple as possible for easy understanding. Currently, this project has only 1000 lines of code.

## Quick Start

Download a model from Huggingface. Now, only Qwen3 series are supported and only Qwen3-0.6B is tested. More models will be added later. 

```sh
# download model
$ huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
    --local-dir ~/huggingface/Qwen3-0.6B/ \
    --local-dir-use-symlinks False
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

## Benchmark

I run a simple benchmark on Qwen3-0.6B with batch size 64 and max sequence length 1024 on a RTX 5070 GPU. The results are as follows:

|Engine          | Tokens/s      | Time(s) | Generated Tokens |
|:---------------|:--------------|:--------|:-----------------|
|vLLM            | 6090.73 tok/s | 28.80   |  185589          |
|mini-vllm       | 5004.99 tok/s | 35.45   |  177421          |

## Features

- Support Qwen3 series (more models will be added later)
- Continuous batching for better throughput.
- Cuda graph.
- KV cache management for efficient generation.
- Prefix caching.

## TODOs

- [x] CUDA graph.
- [ ] Chunked prefilling.
- [ ] More models support (Llama2, Mistral, etc.)
- [ ] Implement FlashAttention from scratch and replace the current FlashAttention implementation.
- [ ] Performance tuning.

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
