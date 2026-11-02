"""Validate provenance failures without downloading data in the test suite."""

import pytest

from secom.data import CHECKSUMS, FILES, load_dataset


def test_pinned_source_hashes_cover_both_raw_inputs():
    assert set(CHECKSUMS) == set(FILES)
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef") for value in CHECKSUMS.values()
    )


def test_corrupted_cached_data_is_rejected_before_parsing_or_fitting(tmp_path):
    for name in FILES:
        (tmp_path / name).write_text("corrupted cache\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_dataset(tmp_path)
