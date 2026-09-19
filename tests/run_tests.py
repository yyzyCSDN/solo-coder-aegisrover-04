#!/usr/bin/env python3
"""Stdlib test runner that injects the pytest shim and runs *_test modules.

Usage: python3 tests/run_tests.py [test_module ...]
"""
import importlib.util
import inspect
import os
import sys
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
sys.path.insert(0, os.path.abspath(SRC))
sys.path.insert(0, HERE)

# Inject the shim as the `pytest` module before any test module imports it.
import _pytest_shim  # noqa: E402
sys.modules['pytest'] = _pytest_shim


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def collect(module):
    fixtures = {}
    tests = []
    for name, obj in vars(module).items():
        if callable(obj) and getattr(obj, '_is_fixture', False):
            fixtures[name] = obj
        if name.startswith('test_') and callable(obj):
            tests.append((name, obj))
    return fixtures, tests


def run_module(path):
    name = os.path.splitext(os.path.basename(path))[0]
    module = load_module(path, name)
    fixtures, tests = collect(module)
    passed = failed = 0
    failures = []
    for test_name, fn in tests:
        kwargs = {}
        for param in inspect.signature(fn).parameters:
            if param in fixtures:
                value = fixtures[param]()
                kwargs[param] = next(value) if inspect.isgenerator(value) else value
        try:
            fn(**kwargs)
            passed += 1
        except Exception:
            failed += 1
            failures.append((f'{name}::{test_name}', traceback.format_exc()))
    return passed, failed, failures


def main(argv):
    if argv:
        paths = argv
    else:
        paths = [os.path.join(HERE, f) for f in sorted(os.listdir(HERE))
                 if f.startswith('test_') and f.endswith('.py')]
    total_passed = total_failed = 0
    all_failures = []
    for path in paths:
        try:
            p, f, failures = run_module(path)
        except Exception:
            print(f'ERROR collecting {path}')
            traceback.print_exc()
            total_failed += 1
            continue
        total_passed += p
        total_failed += f
        all_failures.extend(failures)
        status = 'ok' if f == 0 else 'FAIL'
        print(f'[{status}] {os.path.basename(path)}: {p} passed, {f} failed')
    for label, tb in all_failures:
        print('\n' + '=' * 70 + f'\nFAIL {label}\n' + '-' * 70)
        print(tb)
    print(f'\n{total_passed} passed, {total_failed} failed')
    return 1 if total_failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
