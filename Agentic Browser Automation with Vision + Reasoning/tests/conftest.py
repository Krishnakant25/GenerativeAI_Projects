"""pytest configuration — asyncio mode marker registration."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
