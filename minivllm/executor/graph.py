import torch
from tqdm import tqdm
from minivllm.config.config import Config
from minivllm.executor.context import Context


class CudaGraphRunner:
    def __init__(self, model, config: Config, max_batch_size: int):
        self.model = model
        self.config = config
        self.max_batch_size = min(max_batch_size, 256)
        self.batch_size_list = [1, 2, 4, 8] + list(range(16, self.max_batch_size + 1, 16))
        self.graphs = {}
        self.pool = None

        self.input_ids = torch.zeros(self.max_batch_size, dtype=torch.int64, device='cuda')
        self.positions = torch.zeros(self.max_batch_size, dtype=torch.int64, device='cuda')
        self.slot_mapping = torch.zeros(self.max_batch_size, dtype=torch.int32, device='cuda')
        self.cache_seqlens = torch.zeros(self.max_batch_size, dtype=torch.int32, device='cuda')
        self.block_table = torch.zeros((self.max_batch_size, config.kv_cache_num_blocks), dtype=torch.int32, device='cuda')
        self.outputs = torch.zeros(self.max_batch_size, config.hf_config.vocab_size, device='cuda')

    @torch.inference_mode()
    def capture(self):
        pbar = tqdm(
            reversed(self.batch_size_list),
            desc="Capturing CUDA graphs...",
        )

        for batch_size in pbar:
            pbar.set_postfix({
                "Batch Size": batch_size,
            })
            self._capture_batch(batch_size)
        pbar.close()

    def _capture_batch(self, batch_size: int):
        ctx = Context(
            prefill=False,
            positions=self.positions[:batch_size],
            slot_mapping=self.slot_mapping[:batch_size],
            cache_seqlens=self.cache_seqlens[:batch_size],
            block_table=self.block_table[:batch_size],
        )

        # we must run the model once before capturing the graph, since some pytorch ops need compile.
        self.outputs[:batch_size] = self.model(
            ctx, self.input_ids[:batch_size], self.positions[:batch_size])

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            self.outputs[:batch_size] = self.model(ctx, self.input_ids[:batch_size], self.positions[:batch_size])

        # if this graph uses a new memory pool, we save it for next graphs.
        self.pool = g.pool()

        self.graphs[batch_size] = g
        torch.cuda.synchronize()

    @torch.inference_mode()
    def replay(self, ctx: Context, input_ids: torch.Tensor) -> torch.Tensor:
        bs = input_ids.size(0)
        graph = self.graphs[next(b for b in self.batch_size_list if b >= bs)]
        self.input_ids[:bs] = input_ids
        self.positions[:bs] = ctx.positions
        self.slot_mapping.fill_(-1)
        self.slot_mapping[:bs] = ctx.slot_mapping
        self.cache_seqlens.zero_()
        self.cache_seqlens[:bs] = ctx.cache_seqlens
        self.block_table[:bs, :ctx.block_table.size(1)] = ctx.block_table
        graph.replay()
        return self.outputs[:bs]
