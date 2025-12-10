import copy
from itertools import count
from typing import List
from enum import Enum, auto
from minivllm.config.sampling import SamplingParams

class RequestState(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Request:
    counter = count()
    def __init__(self, prompt_tokens: List[int], sampling_params: SamplingParams = SamplingParams()):
        self.id = next(Request.counter)
        self.state = RequestState.WAITING
        self.prompt_token_count = len(prompt_tokens)
        self.tokens = copy.copy(prompt_tokens)

        # The block ids that are used to store the KVCache for this request.
        self.blocks: list[int] = []
        self.sampling_params = sampling_params
        self.cached_token_count = 0

    def __getitem__(self, idx) -> int:
        return self.tokens[idx]

    def __len__(self):
        return len(self.tokens)

    @property
    def finished(self):
        return self.state == RequestState.FINISHED

    def append_output_token(self, token: int):
        self.tokens.append(token)

    @property
    def prompt_tokens(self):
        return self.tokens[:self.prompt_token_count]

    @property
    def completion_tokens(self):
        return self.tokens[self.prompt_token_count:]
