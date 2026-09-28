"""Byte-budgeted LRU cache (ARCHITECTURE §6.2). Thread-safe."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable, Hashable


class LRUCache:
    def __init__(self, budget_bytes: int, size_of: Callable[[Any], int] | None = None) -> None:
        self.budget = budget_bytes
        self._size_of = size_of or (lambda v: int(getattr(v, "nbytes", 0)))
        self._items: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return entry[0]

    def put(self, key: Hashable, value: Any) -> None:
        size = self._size_of(value)
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            if size > self.budget:
                return  # would evict everything and still not fit
            self._items[key] = (value, size)
            self._bytes += size
            while self._bytes > self.budget:
                _, (_, s) = self._items.popitem(last=False)
                self._bytes -= s

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._items

    @property
    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0
