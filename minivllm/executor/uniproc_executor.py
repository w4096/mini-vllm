import torch

from minivllm.config.config import Config
from minivllm.engine.request import Request
from minivllm.executor.context import Context, set_forward_context
from minivllm.executor.executor import Executor
from minivllm.models import load_model
from minivllm.sched.task import Task
from minivllm.models.layers.sampler import Sampler


class UniProcExecutor(Executor):
    def __init__(self, config: Config):
        super().__init__(config)

        torch.set_default_device("cuda")

        self.model = load_model(config.model)

        self.sampler = Sampler()


    def _build_prefill_input(self, task: Task) -> tuple[torch.Tensor, Context]:
        tokens = []
        positions = []
        slot_mapping = []
        accum_seq_lens_q = [0]
        accum_seq_lens_k = [0]
        max_seq_len_q = 0
        max_seq_len_k = 0

        for req in task.requests:
            seq_len = len(req.tokens)
            tokens.extend(req.tokens)
            positions.extend(range(seq_len))
            accum_seq_lens_q.append(seq_len)
            accum_seq_lens_k.append(seq_len)
            max_seq_len_q = max(max_seq_len_q, seq_len)
            max_seq_len_k = max(max_seq_len_k, seq_len)

            for i in range(len(req.blocks)):
                start = req.blocks[i] * self.config.cache.block_size
                if i != len(req.blocks) - 1:
                    end = start + self.config.cache.block_size
                else:
                    last_block_tokens = seq_len % self.config.cache.block_size
                    end = start + last_block_tokens
                slot_mapping.extend(list(range(start, end)))

        ctx = Context()
        ctx.prefill = True
        tokens = torch.tensor(tokens, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        ctx.positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        ctx.accum_seq_lens_q = torch.tensor(accum_seq_lens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        ctx.accum_seq_lens_k = torch.tensor(accum_seq_lens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        ctx.slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        ctx.max_seq_len_q = max_seq_len_q
        ctx.max_seq_len_k = max_seq_len_k
        return tokens, ctx


    def _build_decode_input(self, task: Task) -> tuple[torch.Tensor, Context]:
        tokens = []
        positions = []
        slot_mapping = []
        context_lens = []

        for req in task.requests:
            tokens.append(req.tokens[-1])
            positions.append(len(req.tokens) - 1)
            context_lens.append(len(req.tokens))

            slot_base_index = req.blocks[-1] * self.config.cache.block_size
            last_block_tokens = len(req.tokens) % self.config.cache.block_size
            slot_mapping.append(slot_base_index + last_block_tokens)

        tokens = torch.tensor(tokens, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        ctx = Context()
        ctx.prefill = False
        ctx.positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        ctx.slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        ctx.context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        ctx.block_table = self._build_block_table(task)
        return tokens, ctx


    @staticmethod
    def _build_block_table(task: Task) -> torch.Tensor:
        max_block_len = max(len(req.blocks) for req in task.requests)
        block_table = [
            req.blocks + [-1] * (max_block_len - len(req.blocks))
            for req in task.requests
        ]
        return torch.tensor(block_table, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)


    @torch.inference_mode()
    def forward(self, ctx: Context, tokens: torch.Tensor) -> torch.Tensor:
        set_forward_context(ctx)
        logits = self.model.compute_logits(self.model(tokens, ctx.positions))
        return logits


    def sample(self, logits: torch.Tensor, requests: list[Request]) -> list[int]:
        temperatures = []
        for req in requests:
            temperatures.append(req.sampling_params.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return self.sampler(logits, temperatures).tolist()


    def execute(self, task: Task) -> list[int]:
        # TODO: Add decode support
        if task.type == Task.PREFILL or 1 == 1:
            tokens, ctx = self._build_prefill_input(task)
        elif task.type == Task.DECODE:
            tokens, ctx = self._build_decode_input(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")

        logits = self.forward(ctx, tokens)
        return self.sample(logits, task.requests)
