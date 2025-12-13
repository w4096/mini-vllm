from time import perf_counter
from tqdm.auto import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer

from minivllm.engine.request import Request
from minivllm.config.config import Config
from minivllm.config.sampling import SamplingParams
from minivllm.sched.scheduler import Scheduler, Task, SchedulerConfig
from minivllm.executor.executor import Executor
from minivllm.kvcache.block_manager import KVCacheBlockManager


class Engine:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        self.executor = Executor(config)

        block_manager = KVCacheBlockManager(config.kvcache_num_blocks, config.kvcache_block_size)
        self.scheduler = Scheduler(SchedulerConfig(
            max_batched_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            eos=config.eos,
        ))
        self.scheduler.set_block_manager(block_manager)

    def submit(self, tokens: list[int], sampling_params: SamplingParams):
        req = Request(tokens, sampling_params)
        self.scheduler.submit(req)

    def step(self) -> Task:
        task = self.scheduler.schedule()
        tokens = self.executor.execute(task)
        self.scheduler.update(task, tokens)
        return task

    @property
    def finished(self):
        return self.scheduler.finished

    def generate(
            self,
            prompts: list[list[int]],
            sampling_params: SamplingParams | list[SamplingParams],
            use_tqdm: bool = True,
    ) -> list[dict]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.submit(prompt, sp)

        finished_requests: list[Request] = []
        prefill_throughput = decode_throughput = 0.
        i = 0
        while not self.finished:
            t = perf_counter()
            task = self.step()

            if use_tqdm:
                if task.type == Task.PREFILL:
                    processed_token_count = sum(req.prompt_token_count for req in task.requests)
                    prefill_throughput = processed_token_count / (perf_counter() - t)
                else:
                    processed_token_count = len(task.requests)
                    decode_throughput = processed_token_count / (perf_counter() - t)

                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput):<5} tokens/s",
                    "Decode": f"{int(decode_throughput):<5} tokens/s",
                })

            for req in task.requests:
                if req.finished:
                    if use_tqdm:
                        pbar.update(1)
                    finished_requests.append(req)

        finished_requests.sort(key=lambda req: req.id)
        outputs = []
        for req in finished_requests:
            outputs.append({
                "prompt": self.tokenizer.decode(req.prompt_tokens),
                "completion": self.tokenizer.decode(req.completion_tokens),
                "tokens": req.completion_tokens}
            )
        if use_tqdm:
            pbar.close()

        return outputs
