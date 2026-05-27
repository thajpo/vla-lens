"""Tables workbench primitives."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import pandas as pd

from vla_lens.traces import TraceBundle, TraceDataset
from vla_lens.workbench.schema import (
    CONTEXT_TABLE_IDS,
    TRACE_TABLE_ALIASES,
    TRACE_TABLE_PATHS,
)
from vla_lens.workbench.utils import (
    _jsonable_record,
)


def query_table(
    dataset: TraceDataset,
    *,
    table: str,
    filters: Mapping[str, Any] | None = None,
    columns: Sequence[str] | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Query metadata/index tables through DuckDB.

    Physical bundle indexes are read from Parquet. Dataset-level episode rows
    are registered as a small in-memory DuckDB relation because their source of
    truth is the JSON manifest per trace.
    """
    filtered = _query_table_duckdb(
        dataset,
        table=table,
        filters=filters or {},
        columns=columns,
        limit=limit,
    )
    if filtered is None:
        raise KeyError(f"Unknown metadata table '{table}'")
    total = int(filtered.attrs.get("total", len(filtered)))
    limited = filtered
    if columns:
        requested = [column for column in columns if column in limited]
        limited = limited.loc[:, requested]
    return {
        "table": table,
        "total": total,
        "returned": int(len(limited)),
        "columns": [str(column) for column in limited.columns],
        "rows": [_jsonable_record(row) for row in limited.to_dict("records")],
    }

def _table_frame(dataset: TraceDataset, table: str) -> pd.DataFrame:
    table_name = TRACE_TABLE_ALIASES.get(str(table), str(table))
    if table_name in {"episodes", "episode_index"}:
        return dataset.episode_index.copy()
    if table_name in {"timesteps", "timestep_index"}:
        return dataset.timestep_index.copy()
    if table_name == "artifact_index":
        return dataset.artifact_index.copy()
    if table_name == "context":
        frames: list[pd.DataFrame] = []
        for context_table in CONTEXT_TABLE_IDS:
            frame = _table_frame(dataset, context_table)
            if not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if table_name in TRACE_TABLE_PATHS and table_name != "timesteps":
        frames = []
        for bundle in dataset.bundles:
            frame = _bundle_trace_table(bundle, table_name).copy()
            if frame.empty:
                continue
            frame["trace_id"] = bundle.manifest.trace_id
            frame["episode_id"] = bundle.manifest.episode_id
            frame["bundle_path"] = str(bundle.path)
            if table_name in CONTEXT_TABLE_IDS:
                frame["context_table"] = table_name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raise KeyError(f"Unknown metadata table '{table}'")

def _query_table_duckdb(
    dataset: TraceDataset,
    *,
    table: str,
    filters: Mapping[str, Any],
    columns: Sequence[str] | None,
    limit: int,
) -> pd.DataFrame | None:
    con = duckdb.connect(database=":memory:")
    try:
        source = _duckdb_source_sql(dataset, table, con)
        if source is None:
            return None
        available_columns = set(con.execute(f"SELECT * FROM ({source}) AS source LIMIT 0").df())
        if any(column not in available_columns for column in filters):
            empty_columns = list(columns) if columns else sorted(available_columns)
            frame = pd.DataFrame(columns=empty_columns)
            frame.attrs["total"] = 0
            return frame
        where_sql, params = _duckdb_where(filters)
        total = con.execute(
            f"SELECT count(*) AS total FROM ({source}) AS source {where_sql}",
            params,
        ).fetchone()[0]
        selected_columns = "*"
        if columns:
            selected = [column for column in columns if column in available_columns]
            selected_columns = ", ".join(_quote_identifier(column) for column in selected) or "*"
        frame = con.execute(
            f"""
            SELECT {selected_columns}
            FROM ({source}) AS source
            {where_sql}
            LIMIT ?
            """,
            [*params, max(0, int(limit))],
        ).df()
        frame.attrs["total"] = int(total)
        return frame
    finally:
        con.close()

def _duckdb_source_sql(
    dataset: TraceDataset,
    table: str,
    con: duckdb.DuckDBPyConnection,
) -> str | None:
    selects: list[str] = []
    table_name = TRACE_TABLE_ALIASES.get(str(table), str(table))
    if table_name in {"episodes", "episode_index"}:
        con.register("episode_index_source", dataset.episode_index.copy())
        return "SELECT * FROM episode_index_source"
    if table_name == "context":
        for context_table in CONTEXT_TABLE_IDS:
            bundle_path = TRACE_TABLE_PATHS[context_table]
            for bundle in dataset.bundles:
                constants = {
                    "trace_id": bundle.manifest.trace_id,
                    "episode_id": bundle.manifest.episode_id,
                    "bundle_path": str(bundle.path),
                    "context_table": context_table,
                }
                select = _parquet_select(bundle.path / bundle_path, constants)
                if not select:
                    select = _dataframe_select(
                        con,
                        f"context_{context_table}_{len(selects)}",
                        _bundle_trace_table(bundle, context_table),
                        constants,
                    )
                selects.append(select)
    elif table_name in {"artifacts", "artifact_index"}:
        dataset_index = dataset.dataset_artifact_index.copy()
        if not dataset_index.empty and not (dataset.root / TraceBundle.MANIFEST).exists():
            selects.append(
                _dataframe_select(
                    con,
                    f"dataset_artifacts_{len(selects)}",
                    dataset_index,
                    {
                        "trace_id": None,
                        "episode_id": None,
                        "bundle_path": None,
                        "dataset_path": str(dataset._dataset_artifact_root()),
                        "artifact_scope": "dataset",
                    },
                )
            )
        for bundle in dataset.bundles:
            constants = {
                "trace_id": bundle.manifest.trace_id,
                "episode_id": bundle.manifest.episode_id,
                "bundle_path": str(bundle.path),
                "dataset_path": str(dataset.root),
                "artifact_scope": "bundle",
            }
            select = _parquet_select(bundle.path / TraceBundle.ARTIFACT_INDEX, constants)
            if not select:
                select = _dataframe_select(
                    con,
                    f"bundle_artifacts_{len(selects)}",
                    bundle.artifact_index,
                    constants,
                )
            selects.append(select)
    elif table_name in TRACE_TABLE_PATHS:
        bundle_path = TRACE_TABLE_PATHS[table_name]
        for bundle in dataset.bundles:
            constants = {
                "trace_id": bundle.manifest.trace_id,
                "episode_id": bundle.manifest.episode_id,
                "bundle_path": str(bundle.path),
            }
            select = _parquet_select(bundle.path / bundle_path, constants)
            if not select:
                select = _dataframe_select(
                    con,
                    f"{table_name}_{len(selects)}",
                    _bundle_trace_table(bundle, table_name),
                    constants,
                )
            selects.append(select)
    else:
        return None
    selects = [select for select in selects if select]
    if not selects:
        return None
    return "\nUNION ALL BY NAME\n".join(selects)

def _bundle_trace_table(bundle: TraceBundle, table_name: str) -> pd.DataFrame:
    readers = {
        "timesteps": bundle.timesteps,
        "policy_calls": bundle.policy_calls,
        "generation_steps": bundle.generation_steps,
        "streams": bundle.streams,
        "token_spaces": bundle.token_spaces,
        "tokens": bundle.tokens,
        "array_index": bundle.array_index,
        "model_sites": bundle.model_sites,
        "artifact_index": bundle.artifact_index,
        "robot_state": bundle.robot_state,
        "scene_state": bundle.scene_state,
        "camera_state": bundle.camera_state,
        "evaluation": bundle.evaluation,
        "image_preprocessing": bundle.image_preprocessing,
        "prompt_metadata": bundle.prompt_metadata,
        "action_normalization": bundle.action_normalization,
    }
    if table_name not in readers:
        raise KeyError(f"Unknown metadata table '{table_name}'")
    return readers[table_name]

def _parquet_select(path: Path, constants: Mapping[str, Any]) -> str:
    if not path.exists():
        return ""
    try:
        columns = pd.read_parquet(path).columns
        if columns.empty:
            return ""
    except Exception:
        return ""
    extra = []
    existing_columns = {str(column) for column in columns}
    for key, value in constants.items():
        if key in existing_columns:
            continue
        if value is None:
            extra.append(f"NULL AS {_quote_identifier(key)}")
        else:
            extra.append(f"{_quote_literal(str(value))} AS {_quote_identifier(key)}")
    suffix = ", " + ", ".join(extra) if extra else ""
    return f"SELECT *{suffix} FROM read_parquet({_quote_literal(str(path))})"

def _dataframe_select(
    con: duckdb.DuckDBPyConnection,
    name: str,
    frame: pd.DataFrame,
    constants: Mapping[str, Any],
) -> str:
    if frame.empty:
        return ""
    table = frame.copy()
    for key, value in constants.items():
        if key not in table:
            table[key] = value
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    con.register(safe_name, table)
    return f"SELECT * FROM {_quote_identifier(safe_name)}"

def _duckdb_where(filters: Mapping[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for column, expected in filters.items():
        ident = _quote_identifier(str(column))
        if isinstance(expected, Mapping):
            if "start" in expected:
                clauses.append(f"try_cast({ident} AS DOUBLE) >= ?")
                params.append(float(expected["start"]))
            if "end" in expected:
                clauses.append(f"try_cast({ident} AS DOUBLE) <= ?")
                params.append(float(expected["end"]))
        elif isinstance(expected, (list, tuple, set, frozenset)):
            values = [str(item) for item in expected]
            if values:
                clauses.append(f"cast({ident} AS VARCHAR) IN ({', '.join('?' for _ in values)})")
                params.extend(values)
            else:
                clauses.append("FALSE")
        else:
            clauses.append(f"cast({ident} AS VARCHAR) = ?")
            params.append(str(expected))
    if not clauses:
        return "", []
    return "WHERE " + " AND ".join(clauses), params

def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'

def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"

def _filter_table(frame: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    out = frame
    for column, expected in filters.items():
        if column not in out:
            return out.iloc[0:0].copy()
        if isinstance(expected, Mapping):
            out = _filter_table_range(out, column, expected)
        elif isinstance(expected, (list, tuple, set, frozenset)):
            allowed = {str(item) for item in expected}
            out = out.loc[out[column].astype(str).isin(allowed)]
        else:
            out = out.loc[out[column].astype(str) == str(expected)]
    return out

def _filter_table_range(
    frame: pd.DataFrame,
    column: str,
    expected: Mapping[str, Any],
) -> pd.DataFrame:
    values = pd.to_numeric(frame[column], errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if "start" in expected:
        mask = mask & (values >= float(expected["start"]))
    if "end" in expected:
        mask = mask & (values <= float(expected["end"]))
    return frame.loc[mask]
