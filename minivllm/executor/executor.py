import logging
import time
from abc import ABC, abstractmethod
from minivllm.config.config import Config
from minivllm.sched.task import Task

logger = logging.getLogger(__name__)


class Executor(ABC):
    """Abstract base class for LLM executors.

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
    """
    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def execute(self, task: Task) ->  list[int]:
        pass

    def shutdown(self) -> None:
        """Shutdown the executor."""
        pass








