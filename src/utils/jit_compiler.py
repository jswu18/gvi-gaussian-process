from typing import Any, Callable

import jax


class JitCompiler:
    def __init__(self, f: Callable[[Any, Any], Any]):
        self.is_warmed_up = False
        self.f = f
        self.jit_compiled_f = jax.jit(lambda parameters, *data: f(parameters, *data))

    def __call__(self, parameters, *data):
        if not self.is_warmed_up:
            _ = self.jit_compiled_f(parameters, *(x[:1, ...] for x in data))
            self.is_warmed_up = True
        return self.jit_compiled_f(parameters, *data)
