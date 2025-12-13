import time
from minivllm.sched.task import Task

class Stats:
    def __init__(self):
        self.prefill_throughput = 0
        self.decode_throughput = 0
        self.time_to_first_token = 0
        self.inter_token_latency = 0
        self.tokens_per_second = 0
        self.requests_per_second = 0
        self.finished_requests = 0

class Metrics:
    def __init__(self):
        self.prefill_time = 0
        self.prefill_steps = 0
        self.prefill_tokens = 0

        self.decode_time = 0
        self.decode_tokens = 0
        self.decode_steps = 0

        self.finished_request_count = 0
        self.start_time = time.perf_counter()

    def update(self, task: Task):
        if task.type == Task.PREFILL:
            self._update_prefill_metrics(task)
        else:
            self._update_decode_metrics(task)

    def _update_prefill_metrics(self, task: Task):
        self.prefill_steps += 1
        self.prefill_time += time.perf_counter() - task.create_time

        for req in task.requests:
            self.prefill_tokens += len(req.tokens) - req.num_cached_tokens

    def _update_decode_metrics(self, task: Task):
        self.decode_steps += 1
        self.decode_tokens += len(task.requests)
        self.decode_time += time.perf_counter() - task.create_time

        for req in task.requests:
            if req.finished:
                self.finished_request_count += 1

    def reset(self):
        self.prefill_time = 0
        self.decode_time = 0
        self.prefill_steps = 0
        self.decode_steps = 0
        self.finished_request_count = 0
        self.start_time = time.perf_counter()


    def stats(self) -> Stats:
        tokens = self.prefill_tokens + self.decode_tokens
        elapsed = time.perf_counter() - self.start_time

        stats = Stats()
        stats.time_to_first_token = self.prefill_time / self.prefill_steps if self.prefill_steps > 0 else 0
        stats.inter_token_latency = self.decode_time / self.decode_steps if self.decode_steps > 0 else 0
        stats.tokens_per_second = tokens / elapsed if elapsed > 0 else 0
        stats.requests_per_second = self.finished_request_count / elapsed if elapsed > 0 else 0
        stats.prefill_throughput = self.prefill_tokens / self.prefill_time if self.prefill_time > 0 else 0
        stats.decode_throughput = self.decode_tokens / self.decode_time if self.decode_time > 0 else 0
        stats.finished_requests = self.finished_request_count
        return stats
