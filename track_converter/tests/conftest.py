from pathlib import Path

import pytest


@pytest.fixture
def ctc_test_data_dir():
    return Path(__file__).parent.resolve() / "data" / "CTC"


@pytest.fixture
def btrack_test_data_dir():
    return Path(__file__).parent.resolve() / "data" / "btrack"


@pytest.fixture
def trackmate_test_data_dir():
    return Path(__file__).parent.resolve() / "data" / "trackmate"


@pytest.fixture
def mastodon_test_data_dir():
    return Path(__file__).parent.resolve() / "data" / "mastodon"
