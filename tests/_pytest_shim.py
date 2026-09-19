"""Minimal pytest stand-in (stdlib only) for environments without pytest installed.

Implements just the surface the test suite uses: raises, approx and trivial
fixtures. Not a substitute for the real pytest in development/CI.
"""
import math


class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other):
        if isinstance(self.expected, (tuple, list)):
            if not isinstance(other, (tuple, list)) or len(other) != len(self.expected):
                return False
            return all(math.isclose(a, b, rel_tol=self.rel, abs_tol=self.abs)
                       for a, b in zip(other, self.expected))
        return math.isclose(other, self.expected, rel_tol=self.rel, abs_tol=self.abs)

    def __repr__(self):
        return f'approx({self.expected!r})'


def approx(expected, rel=1e-6, abs=1e-12):
    return _Approx(expected, rel, abs)


class _Raises:
    def __init__(self, expected):
        self.expected = expected
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f'{self.expected.__name__} was not raised')
        if not issubclass(exc_type, self.expected):
            return False
        self.value = exc
        return True


def raises(expected):
    return _Raises(expected)


def fixture(*args, **kwargs):
    def wrap(fn):
        fn._is_fixture = True
        return fn
    if args and callable(args[0]):
        return wrap(args[0])
    return wrap
