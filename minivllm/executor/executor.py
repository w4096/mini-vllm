import torch
import logging

from minivllm.config.config import Config
from minivllm.engine.request import Request
from minivllm.executor.context import Context, set_forward_context
from minivllm.models.loader import load_model
from minivllm.scheduler.batch import Batch
from minivllm.models.layers.sampler import Sampler
from minivllm.executor.graph import CUDAGraphExecutor

logger = logging.getLogger(__name__)

class Executor:
    def __init__(self, config: Config):
        self.config = config

        torch.set_default_device("cuda")
        torch.set_default_dtype(config.hf_config.dtype)
        self.model = load_model(config)
        self.sampler = Sampler()

        self._warmup_model()

        self.kv_cache = None
        self._init_kv_cache()
        
        self.graph_executor = CUDAGraphExecutor(self.model, self.config, self.config.max_num_batched_seqs)
        self.graph_executor.capture()

        
    def _init_kv_cache(self):
        config = self.config
        hf_config = self.config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * config.kv_cache_block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        kv_cache_num_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert kv_cache_num_blocks > 0

        kv_cache_num_blocks = min(kv_cache_num_blocks, config.kv_cache_num_blocks)

        # update the config with the new value
        config.kv_cache_num_blocks = kv_cache_num_blocks
        
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, kv_cache_num_blocks, config.kv_cache_block_size, num_kv_heads, head_dim)
        
        logger.info(f'Allocated {kv_cache_num_blocks} key-value cache blocks.')
        
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def _warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_batched_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_batched_seqs)

        # for fast start
        max_model_len = 64
        num_batched_seqs = 10
        reqs = [Request([0] * max_model_len) for _ in range(num_batched_seqs)]
        
        self.execute(Batch(Batch.PREFILL, reqs))
        
        torch.cuda.empty_cache()


    def _build_prefill_input(self, requests: list[Request]) -> tuple[torch.Tensor, Context]:
        input_ids = []
        positions = []
        slot_mapping = []
        cu_seq_lens_q = [0]
        cu_seq_lens_k = [0]
        max_seq_len_q = 0
        max_seq_len_k = 0
        block_table = None

        for req in requests:
            seqlen = len(req.tokens)
            input_ids.extend(req.tokens[req.num_cached_tokens:])
            positions.extend(range(req.num_cached_tokens, seqlen))
            
            seqlen_q = seqlen - req.num_cached_tokens
            seqlen_k = seqlen
            cu_seq_lens_q.append(cu_seq_lens_q[-1] + seqlen_q)
            cu_seq_lens_k.append(cu_seq_lens_k[-1] + seqlen_k)
            max_seq_len_q = max(max_seq_len_q, seqlen_q)
            max_seq_len_k = max(max_seq_len_k, seqlen_k)

            num_cached_blocks = req.num_cached_tokens // self.config.kv_cache_block_size
            for i in range(num_cached_blocks, len(req.blocks)):
                start = req.blocks[i] * self.config.kv_cache_block_size
                if i != len(req.blocks) - 1:
                    end = start + self.config.kv_cache_block_size
                else:
                    last_block_tokens = len(req.tokens) - (len(req.blocks) - 1) * self.config.kv_cache_block_size
                    end = start + last_block_tokens
                slot_mapping.extend(list(range(start, end)))
                
        if cu_seq_lens_k[-1] > cu_seq_lens_q[-1]:
            block_table = self._build_block_table(requests)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        ctx = Context(
            prefill=True,
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
            cu_seq_lens_q = torch.tensor(cu_seq_lens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seq_lens_k = torch.tensor(cu_seq_lens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seq_len_q = max_seq_len_q,
            max_seq_len_k = max_seq_len_k,
            block_table = block_table
        )
        return input_ids, ctx


    def _build_decode_input(self, requests: list[Request]) -> tuple[torch.Tensor, Context]:
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for req in requests:
            input_ids.append(req.tokens[-1])
            positions.append(len(req.tokens) - 1)
            context_lens.append(len(req.tokens))

            slot_base_index = req.blocks[-1] * self.config.kv_cache_block_size
            last_block_tokens = len(req.tokens) - (len(req.blocks) - 1) * self.config.kv_cache_block_size
            slot_mapping.append(slot_base_index + last_block_tokens - 1)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        ctx = Context(
            prefill=False,
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True),
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            block_table = self._build_block_table(requests),
        )
        return input_ids, ctx


    @staticmethod
    def _build_block_table(requests: list[Request]) -> torch.Tensor:
        max_block_len = max(len(req.blocks) for req in requests)
        block_table = [
            req.blocks + [-1] * (max_block_len - len(req.blocks))
            for req in requests
        ]
        return torch.tensor(block_table, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def _build_sample_input(self, requests: list[Request]) -> torch.Tensor:
        temperatures = []
        for req in requests:
            temperatures.append(req.sampling_params.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures


    def forward(self, ctx: Context, tokens: torch.Tensor) -> torch.Tensor:
        set_forward_context(ctx)
        
        if ctx.prefill or not self.config.use_cuda_graph or self.graph_executor.max_batch_size < tokens.size(0):
            logits = self.model(tokens, ctx.positions)
        else:
            logits = self.graph_executor.replay(ctx, tokens)
            
        return logits
    
    def sample(self, logits: torch.Tensor, temperatures: torch.Tensor|None):
        return self.sampler(logits, temperatures).tolist()

    @torch.inference_mode()
    def execute(self, batch: Batch) -> list[int]:
        if batch.type == Batch.PREFILL:
            input_ids, ctx = self._build_prefill_input(batch.requests)
        elif batch.type == Batch.DECODE:
            input_ids, ctx = self._build_decode_input(batch.requests)
        else:
            raise ValueError(f"Unknown batch type: {batch.type}")

        temperatures = self._build_sample_input(batch.requests)

        logits = self.forward(ctx, input_ids)
        output_tokens = self.sample(logits, temperatures)
        return output_tokens
    