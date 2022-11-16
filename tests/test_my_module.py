"""Tests for the edg_acoustics.my_module module.
"""
import pytest

from edg_acoustics.my_module import hello


def test_hello():
    assert hello('nlesc') == 'Hello nlesc!'


def test_hello_with_error():
    with pytest.raises(ValueError) as excinfo:
        hello('nobody')
    assert 'Can not say hello to nobody' in str(excinfo.value)


@pytest.fixture
def some_name():
    return 'Jane Smith'


def test_hello_with_fixture(some_name):
    assert hello(some_name) == 'Hello Jane Smith!'
