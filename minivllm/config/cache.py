from dataclasses import dataclass


@dataclass()
class CacheConfig:
    num_blocks: int = 1024
    block_size: int = 32