import time
from minivllm.engine.request import Request

class Batch:
    PREFILL = 0
    DECODE = 1

    def __init__(self, typ: int, reqs: list[Request]):
        self.type = typ
        self.requests = reqs

        self.create_time = time.perf_counter()
