"""Import torch before xgboost gets imported anywhere in the test session.

On this platform, importing xgboost (via test_baselines.py -> src.baselines)
before torch has ever been imported causes a segfault the first time a torch
op runs -- a known conflict between XGBoost's and PyTorch's bundled OpenMP
runtimes on macOS. Loading torch first here (conftest.py is imported before
test modules are collected) avoids it; the actual test files don't need to
care about import order.
"""

import torch  # noqa: F401
