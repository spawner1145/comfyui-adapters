import os
import threading
from collections import defaultdict
from functools import lru_cache, wraps

from typing import Literal

__all__ = ['ts_lru_cache']

LevelTyping = Literal['global', 'process', 'thread']


def _get_context_key(level: LevelTyping = 'global'):
    if level == 'global':
        return None
    elif level == 'process':
        return os.getpid()
    elif level == 'thread':
        return os.getpid(), threading.get_ident()
    else:
        raise ValueError(f'Invalid cache level, '
                         f'\'global\', \'process\' or \'thread\' expected but {level!r} found.')


def ts_lru_cache(level: LevelTyping = 'global', **options):
    _ = _get_context_key(level)

    def _decorator(func):

        @lru_cache(**options)
        @wraps(func)
        def _cached_func(*args, __context_key=None, **kwargs):
            return func(*args, **kwargs)

        lock_pool = defaultdict(threading.Lock)
        lock = threading.Lock()

        @wraps(_cached_func)
        def _new_func(*args, **kwargs):
            context_key = _get_context_key(level=level)
            with lock:
                _context_lock = lock_pool[context_key]
            with _context_lock:
                return _cached_func(*args, __context_key=context_key, **kwargs)

        if hasattr(_cached_func, 'cache_info'):
            _new_func.cache_info = _cached_func.cache_info
        if hasattr(_cached_func, 'cache_clear'):
            _new_func.cache_clear = _cached_func.cache_clear

        return _new_func

    return _decorator
