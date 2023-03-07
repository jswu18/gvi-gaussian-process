from functools import wraps


def static(decorator_function):
    def decorator(f):
        @wraps(f)
        def inner(*args, **kwargs):
            return f(**decorator_function(*args, **kwargs))

        return inner

    return decorator


def instance(decorator_function):
    def decorator(f):
        @wraps(f)
        def inner(self, *args, **kwargs):
            return f(self, **decorator_function(*args, **kwargs))

        return inner

    return decorator
