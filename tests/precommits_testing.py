# Uncorrect version - please uncomment to test

# import os, sys  # Style error (flake8)
#
#
# def greet(name: str) -> None:
#     print(f"Hello, {name}")  # Formatting error (black)
#
#
# greet("World")
# greet(123)  # Type error (mypy)

# Correct version

import os
import sys


def greet(name: str) -> None:
    print(f"Hello, {name}")
    print(f"Python version: {sys.version}")
    print(f"File path: {os.getcwd()}")


greet("World")
greet("123")
