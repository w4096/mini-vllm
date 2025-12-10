from minivllm.engine.request import Request

class Task:
    PREFILL = 0
    DECODE = 1

    def __init__(self, typ: int, reqs: list[Request]):
        self.type = typ
        self.requests = reqs
