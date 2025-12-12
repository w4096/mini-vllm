import logging
from collections import deque
from dataclasses import dataclass

from minivllm.config.config import Config
from minivllm.engine.request import Request, RequestState

from minivllm.kvcache.block_manager import KVCacheBlockManager
from minivllm.sched.task import Task

logger = logging.getLogger(__name__)

@dataclass
class SchedulerConfig:
    max_batched_seqs: int
    max_num_batched_tokens: int
    eos: int

class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.max_batched_seqs = config.max_batched_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos

        self.block_manager: KVCacheBlockManager | None = None
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()


    def set_block_manager(self, block_manager: KVCacheBlockManager):
        self.block_manager = block_manager

    @property
    def finished(self):
        return not self.waiting and not self.running


    def submit(self, req: Request):
        """
        Add a request to the waiting queue and wait for it to be scheduled.
        """
        self.waiting.append(req)


    def _schedule_prefill(self) -> Task | None:
        """
        Schedule prefill requests.
        :return: None
        """
        reqs = []
        num_batched_tokens = 0
        while self.waiting and len(reqs) < self.max_batched_seqs:
            req = self.waiting[0]
            num_prefill_tokens = len(req.tokens) - req.num_cached_tokens
            if num_batched_tokens + num_prefill_tokens > self.max_num_batched_tokens:
                break
            if not self.block_manager.can_allocate_new_block(req):
                break
            num_batched_tokens += num_prefill_tokens
            self.block_manager.allocate_blocks_for_prefill(req)
            req.state = RequestState.RUNNING
            self.waiting.popleft()
            self.running.append(req)
            reqs.append(req)
        if reqs:
            return Task(Task.PREFILL, reqs)
        return None


    def _schedule_decode(self) -> Task | None:
        """
        Schedule decode requests.
        :return: None
        """
        reqs = []
        while self.running and len(reqs) < self.max_batched_seqs:
            req = self.running.popleft()
            while not self.block_manager.can_allocate_new_block(req):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(req)
                    break

            if req.state == RequestState.RUNNING:
                self.block_manager.append_block_if_needed(req)
                reqs.append(req)
        self.running.extendleft(reqs)
        return Task(Task.DECODE, reqs)

    def schedule(self) -> Task | None:
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
        req.state = RequestState.WAITING
        self.block_manager.deallocate(req)
        self.waiting.appendleft(req)


    def update(self, task: Task, tokens: list[int]):
        for req, token in zip(task.requests, tokens):
            req.append_output_token(token)

            eos_reached = token == self.eos
            max_len_reached = len(req.completion_tokens) == req.sampling_params.max_tokens
            if eos_reached or max_len_reached:
                req.state = RequestState.FINISHED
                self.block_manager.deallocate(req)
                self.running.remove(req)
            else:
                self.block_manager.cache_block_if_needed(req)
