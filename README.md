## mini-vllm

A simple vLLM like inference engine.

## Quick Start

Download a model from Huggingface. Now, only Qwen3-0.6B is supported for now.

```sh
# download model
$ huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
    --local-dir ~/huggingface/Qwen3-0.6B/ \
    --local-dir-use-symlinks False
```

Run the inference server.

```sh
$ git clone https://github.com/w4096/mini-vllm
$ cd mini-vllm
$ pip install -r requirements.txt
$ python run.py
```

You may need to install `transformers`, `torch`, and `flash-attention`.

If you have trouble installing `flash-attention`, you can try the prebuilt wheels. See: [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)


## References

- [vLLM](https://github.com/vllm-project/vllm)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
