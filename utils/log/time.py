import functools
import time

def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        print(f"Function {func.__name__} took {duration:.4f} seconds")
        return result
    return wrapper