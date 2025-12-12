import pytest
import random
from minivllm.kvcache.block_manager import KVCacheBlockManager
from minivllm.engine.request import Request


def test_block_manager():
    BLOCK_SIZE = 256
    manager = KVCacheBlockManager(10240, BLOCK_SIZE)
    
    tokens = [random.randint(1, 25600) for _ in range(25600)]

    req = Request(tokens)
    manager.allocate_blocks_for_prefill(req)
    assert len(req.blocks) == len(tokens) // BLOCK_SIZE
    assert len(manager.hash_to_block_id) == len(req.blocks)
    assert len(manager.used_block_ids) == len(req.blocks)
    
    requests = []
    for i in range(1, 100):
        req = Request(tokens[:i*BLOCK_SIZE])
        if not manager.can_allocate_new_block(req):
            for r in requests:
                manager.deallocate(r)
            requests.clear()
        manager.allocate_blocks_for_prefill(req)
        assert len(req.blocks) == i
        assert req.num_cached_tokens == len(req.tokens)
        requests.append(req)