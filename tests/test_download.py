import hashlib

import pytest

from ct_mae.download import fetch_asset


def test_download_checks_hash_and_reuses_verified_cache(tmp_path):
    source = tmp_path / "source.npz"
    source.write_bytes(b"official bytes fixture")
    asset = {
        "filename": "organamnist.npz",
        "url": source.as_uri(),
        "bytes": source.stat().st_size,
        "md5": hashlib.md5(source.read_bytes(), usedforsecurity=False).hexdigest(),
    }
    destination = fetch_asset(asset, tmp_path / "data")
    assert destination.read_bytes() == source.read_bytes()
    source.unlink()
    assert fetch_asset(asset, destination.parent) == destination


def test_wrong_checksum_never_promotes_partial_file(tmp_path):
    source = tmp_path / "source.npz"
    source.write_bytes(b"corrupt fixture")
    asset = {
        "filename": "organamnist.npz",
        "url": source.as_uri(),
        "bytes": source.stat().st_size,
        "md5": "0" * 32,
    }
    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch_asset(asset, tmp_path / "data")
    assert not (tmp_path / "data" / "organamnist.npz").exists()
