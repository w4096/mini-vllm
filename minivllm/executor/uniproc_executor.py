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

        self.config = config

        torch.set_default_device("cuda")
        torch.set_default_dtype(config.hf_config.dtype)
        self.model = load_model(config)
        self.sampler = Sampler()

        self.warmup_model()

        self.allocate_kv_cache()

        
    def allocate_kv_cache(self):
        config = self.config
        hf_config = self.config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * config.kvcache_block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        kvcache_num_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert kvcache_num_blocks > 0
        kvcache_num_blocks = min(kvcache_num_blocks, config.kvcache_num_blocks)
        
        # update the config with the new value
        config.kvcache_num_blocks = kvcache_num_blocks
        
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, kvcache_num_blocks, config.kvcache_block_size, num_kv_heads, head_dim)
        
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        max_model_len = 64
        num_seqs = 10
        reqs = [Request([0] * max_model_len) for _ in range(num_seqs)]
        
        self.execute(Task(Task.PREFILL, reqs))
        
        torch.cuda.empty_cache()


    def _build_prefill_input(self, task: Task) -> tuple[torch.Tensor, Context]:
        tokens = []
        positions = []
        slot_mapping = []
        accum_seq_lens_q = [0]
        accum_seq_lens_k = [0]
        max_seq_len_q = 0
        max_seq_len_k = 0

        for req in task.requests:
            seqlen = len(req.tokens)
            tokens.extend(req.tokens[req.num_cached_tokens:])
            positions.extend(range(req.num_cached_tokens, seqlen))
            
            seqlen_q = seqlen - req.num_cached_tokens
            seqlen_k = seqlen
            accum_seq_lens_q.append(accum_seq_lens_q[-1] + seqlen_q)
            accum_seq_lens_k.append(accum_seq_lens_k[-1] + seqlen_k)
            max_seq_len_q = max(max_seq_len_q, seqlen_q)
            max_seq_len_k = max(max_seq_len_k, seqlen_k)

            num_cached_blocks = req.num_cached_tokens // self.config.kvcache_block_size
            for i in range(num_cached_blocks, len(req.blocks)):
                start = req.blocks[i] * self.config.kvcache_block_size
                if i != len(req.blocks) - 1:
                    end = start + self.config.kvcache_block_size
                else:
                    last_block_tokens = len(req.tokens) - (len(req.blocks) - 1) * self.config.kvcache_block_size
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

            slot_base_index = req.blocks[-1] * self.config.kvcache_block_size
            last_block_tokens = len(req.tokens) - (len(req.blocks) - 1) * self.config.kvcache_block_size
            slot_mapping.append(slot_base_index + last_block_tokens - 1)

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


    

    def sample(self, logits: torch.Tensor, requests: list[Request]) -> list[int]:
        temperatures = []
        for req in requests:
            temperatures.append(req.sampling_params.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return self.sampler(logits, temperatures).tolist()


    @torch.inference_mode()
    def execute(self, task: Task) -> list[int]:
        if task.type == Task.PREFILL:
            tokens, ctx = self._build_prefill_input(task)
        elif task.type == Task.DECODE:
            tokens, ctx = self._build_decode_input(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")

        set_forward_context(ctx)
        logits = self.model.compute_logits(self.model(tokens, ctx.positions))
        return self.sample(logits, task.requests)
