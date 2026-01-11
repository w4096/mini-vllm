import logging
from collections import deque

from minivllm.config.config import Config
from minivllm.engine.request import Request
from minivllm.kvcache.block_manager import KVCacheBlockManager
from minivllm.scheduler.batch import Batch

logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, config: Config):
        self.max_num_batched_seqs = config.max_num_batched_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos_token_ids = config.eos_token_ids

        self.block_manager = KVCacheBlockManager(config.kv_cache_num_blocks, config.kv_cache_block_size)
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()


    @property
    def finished(self):
        return not self.waiting and not self.running


    def submit(self, req: Request):
        """
        Add a request to the waiting queue and wait for it to be scheduled.
        """
        self.waiting.append(req)


    def _schedule_prefill(self) -> Batch | None:
        """
        Schedule prefill requests.
        :return: None
        """
        reqs = []
        num_batched_tokens = 0
        while self.waiting and len(reqs) < self.max_num_batched_seqs:
            req = self.waiting[0]
            num_prefill_tokens = len(req.tokens) - req.num_cached_tokens
            if num_batched_tokens + num_prefill_tokens > self.max_num_batched_tokens:
                break
            if not self.block_manager.can_allocate_new_block(req):
                break
            num_batched_tokens += num_prefill_tokens
            self.block_manager.allocate_blocks_for_prefill(req)
            req.state = Request.RUNNING
            self.waiting.popleft()
            self.running.append(req)
            reqs.append(req)
        if reqs:
            return Batch(Batch.PREFILL, reqs)
        return None


    def _schedule_decode(self) -> Batch | None:
        """
        Schedule decode requests.
        :return: None
        """
        reqs = []
        while self.running and len(reqs) < self.max_num_batched_seqs:
            req = self.running.popleft()
            while not self.block_manager.can_allocate_new_block(req):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(req)
                    break

            if req.state == Request.RUNNING:
                self.block_manager.allocate_block_for_decode(req)
                reqs.append(req)
        if reqs:
            self.running.extendleft(reversed(reqs))
            return Batch(Batch.DECODE, reqs)
        else:
            return None

    def schedule(self) -> Batch | None:
        out = self._schedule_prefill()
        if out is not None:
            return out

        out = self._schedule_decode()
        return out

    def preempt(self, req: Request):
        """
        No more KVCacheBlocks available for the next request, we have to preempt this request
        and release its KVCacheBlocks.
        """
        req.state = Request.WAITING
        self.block_manager.deallocate(req)
        self.waiting.appendleft(req)


    def update(self, batch: Batch, tokens: list[int]):
        for req, token in zip(batch.requests, tokens):
            req.append_output_token(token)

            eos_reached = self.eos_token_ids and token in self.eos_token_ids
            max_len_reached = len(req.completion_tokens) >= req.sampling_params.max_tokens
            if  max_len_reached or (eos_reached and not req.sampling_params.ignore_eos):
                req.state = Request.FINISHED
                self.block_manager.deallocate(req)
                self.running.remove(req)
            else:
                self.block_manager.cache_block_if_needed(req)
