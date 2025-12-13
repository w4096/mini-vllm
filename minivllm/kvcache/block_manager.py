from minivllm.engine.request import Request, RequestState
from collections import deque
from minivllm.utils import utils
import xxhash
import numpy as np



class KVCacheBlock:
    def __init__(self, bid: int):
        self.id = bid
        self.refcount = 0
        self.hash = -1
        self.tokens: list[int] = []

    def update(self, h: int, tokens: list[int]):
        self.hash = h
        self.tokens = tokens

    def reset(self):
        self.refcount = 1
        self.hash = -1
        self.tokens = []

    @staticmethod
    def compute_hash(tokens: list[int], prefix: int = - 1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(tokens).tobytes())
        return h.intdigest()
    

class KVCacheBlockManager:
    def __init__(self, capacity: int, block_size: int):
        """
        :param capacity: The number of KVCacheBlocks
        :param block_size: The number of slots in each KVCacheBlock.

        A KVCacheBlock has `block_size` slots, each slot can store a token.
        +--------+--------+-----+--------+
        | slot 1 | slot 2 | ... | slot n |
        +--------+--------+-----+--------+
        """

        self.block_size: int = block_size
        self.blocks: list[KVCacheBlock] = [KVCacheBlock(i) for i in range(capacity)]
        self.free_block_ids: deque[int] = deque(range(capacity))
        self.used_block_ids: set[int] = set()
        self.hash_to_block_id: dict[int, int] = {}
        


    def allocate_blocks_for_prefill(self, req: Request):
        assert req.state == RequestState.WAITING
        assert len(req.blocks) == 0

        hash_ = -1
        prefix_cache_miss = False
        required_blocks = utils.cdiv(len(req.tokens), self.block_size)
        for i in range(required_blocks):
            tokens = req.tokens[i * self.block_size: (i + 1) * self.block_size]
            
            if len(tokens) != self.block_size:
                block = self._allocate()
            else:
                hash_ = KVCacheBlock.compute_hash(tokens, hash_)
                block_id = self.hash_to_block_id.get(hash_, -1)
                if block_id == -1 or self.blocks[block_id].tokens != tokens:
                    prefix_cache_miss = True
                
                if prefix_cache_miss:
                    block = self._allocate()
                else:
                    req.num_cached_tokens += self.block_size
                    if block_id in self.used_block_ids:
                        block = self.blocks[block_id]
                        assert block.refcount >= 1
                        block.refcount += 1
                    else:
                        block = self._allocate(block_id)
                block.update(hash_, tokens)
                self.hash_to_block_id[hash_] = block.id

            req.blocks.append(block.id)


    def append_block_if_needed(self, req: Request):
        """
        In decode stage, we only need to allocate a new block if all the allocated blocks are used.
        """

        assert req.state == RequestState.RUNNING
        assert len(req.blocks) > 0
        if self.request_required_blocks(req) > 0:
            block = self._allocate()
            assert block.refcount == 1
            req.blocks.append(block.id)


    def deallocate(self, req: Request):
        for bid in reversed(req.blocks):
            block = self.blocks[bid]
            assert block.refcount > 0
            block.refcount -= 1
            if block.refcount == 0:
                self.used_block_ids.remove(bid)
                self.free_block_ids.append(bid)
        req.blocks = []
        req.num_cached_tokens = 0

    
    def _allocate(self, bid=-1):
        """
        Allocate a new KVCacheBlock from free_block_ids.
        :return: the allocated KVCacheBlock.
        """
        assert len(self.free_block_ids) > 0

        if bid == -1:
            bid = self.free_block_ids.pop()
        else:
            # alloc certain block
            self.free_block_ids.remove(bid)
        block = self.blocks[bid]
        assert block.refcount == 0
        block.reset()
        self.used_block_ids.add(bid)
        if block.hash != -1 and block.hash in self.hash_to_block_id:
            del self.hash_to_block_id[block.id]
        return block
    

    def can_allocate_new_block(self, req: Request):
        """
        check if the kv cache manager can allocate enough blocks for the request.
        """
        return len(self.free_block_ids) >= self.request_required_blocks(req)

    def request_required_blocks(self, req: Request):
        """
        check if the request need append a new block for the request.
        """
        required_blocks = utils.cdiv(len(req.tokens), self.block_size)
        return required_blocks - len(req.blocks)

    def cache_block_if_needed(self, req: Request):
        """
        Cache the last block of the request if it is full
        """
        if len(req.tokens) % self.block_size == 0:
            # This means the last block is full. We don't need
            # alloc new block since the kv cache will be saved in
            # the last slot of last block
            assert self.blocks[req.blocks[-1]].hash == -1
            tokens = req.tokens[-self.block_size:]
            prefix = self.blocks[req.blocks[-2]].hash if len(req.blocks) > 1 else -1
            h = KVCacheBlock.compute_hash(tokens, prefix)
            self.blocks[req.blocks[-1]].update(h, tokens)
            self.hash_to_block_id[h] = req.blocks[-1]
