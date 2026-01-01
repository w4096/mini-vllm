from time import perf_counter
import logging
from tqdm.auto import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer

from minivllm.engine.request import Request
from minivllm.config.config import Config
from minivllm.config.sampling import SamplingParams
from minivllm.scheduler.scheduler import Scheduler, Batch
from minivllm.executor.executor import Executor
from minivllm.engine.metrics import Metrics


class Engine:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        self.executor = Executor(config)
        self.scheduler = Scheduler(config)
        self.metrics = Metrics()


    def submit(self, tokens: list[int], sampling_params: SamplingParams) -> Request:
        req = Request(tokens, sampling_params)
        self.scheduler.submit(req)
        return req


    def step(self) -> Batch:
        batch = self.scheduler.schedule()
        tokens = self.executor.execute(batch)
        self.scheduler.update(batch, tokens)
        self.metrics.update(batch)
        return batch

    @property
    def finished(self):
        return self.scheduler.finished

    def generate(
            self,
            prompts: list[list[int]],
            sampling_params: SamplingParams | list[SamplingParams],
            use_tqdm: bool = True,
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.submit(prompt, sp)

        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) if use_tqdm else None
        stats = self.metrics.stats()

        finished_requests: list[Request] = []
        while not self.finished:
            batch = self.step()

            if pbar:
                s = self.metrics.stats()
                pbar.set_postfix({
                    "Prefill(token/s)": f"{s.prefill_throughput:4.0f}",
                    "Decode(token/s)": f"{s.decode_throughput:4.0f}",
                    "TTFT": f"{s.time_to_first_token:4.2f}",
                    "ITL": f"{s.inter_token_latency:4.2f}",
                    "TPS": f"{s.tokens_per_second:4.2f}",
                    "RPS": f"{s.requests_per_second:4.2f}",
                })
                pbar.update(s.finished_requests - stats.finished_requests)
                stats = s

            for req in batch.requests:
                if req.finished:
                    finished_requests.append(req)


        finished_requests.sort(key=lambda req: req.id)

        outputs = []
        for req in finished_requests:
            outputs.append({
                "prompt": self.tokenizer.decode(req.prompt_tokens),
                "completion": self.tokenizer.decode(req.completion_tokens),
                "tokens": req.completion_tokens,
            })

        if pbar:
            pbar.close()

        return outputs

