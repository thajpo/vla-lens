from __future__ import annotations

import logging

from vla_lens.table_io import read_optional_parquet


def test_read_optional_parquet_logs_warning_for_unreadable_table(tmp_path, caplog):
    path = tmp_path / "bad.parquet"
    path.write_text("not parquet", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="vla_lens.table_io"):
        frame = read_optional_parquet(path, context="test table")

    assert frame.empty
    assert "Failed to read test table parquet table" in caplog.text
    assert str(path) in caplog.text
