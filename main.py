"""Conversational data analyst over CSV files stored in Google Drive.

The model has no code execution: it can only call the fixed tools below and
ask the Google Drive connector for file content. All numbers come from
pandas computation inside these tools.
"""

from __future__ import annotations

import io
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from mistralai import agents

_MAX_SAMPLE_ROWS = 100
_MAX_GROUP_ROWS = 200
_DESCRIBE_TOP_VALUES = 5

FilterOperator = Literal[
    "eq", "ne", "gt", "gte", "lt", "lte", "contains", "in", "is_null", "not_null",
]


class Filter(BaseModel):
    """One filter condition applied to the rows of a dataset."""

    column: str = Field(description="Column name, exactly as returned by describe_columns.")
    operator: FilterOperator = Field(
        description="Comparison to apply: eq, ne, gt, gte, lt, lte, contains, in, is_null, not_null."
    )
    value: str | float | int | None = Field(
        default=None,
        description="Value to compare against. For 'in', a comma-separated list, e.g. 'FR,DE,IT'.",
    )


_DATASETS: dict[str, dict[str, Any]] = {}

_PANDAS_AGGREGATIONS = {
    "count": "count", "nunique": "nunique", "sum": "sum", "mean": "mean",
    "median": "median", "min": "min", "max": "max", "std": "std",
}


def _session_store(session_id: str) -> dict[str, Any]:
    return _DATASETS.setdefault(session_id, {"datasets": {}, "active": None})


def _active_dataset_name(session_id: str) -> str | None:
    store = _DATASETS.get(session_id)
    return store["active"] if store else None


def _get_dataframe(session_id: str, dataset: str) -> pd.DataFrame:
    store = _DATASETS.get(session_id)
    if not store or not store["datasets"]:
        raise agents.ToolError(
            "No dataset is loaded in this session yet. "
            "Find the file with the Google Drive connector, then call load_csv with its content."
        )
    name = dataset or store["active"]
    if not name or name not in store["datasets"]:
        available = ", ".join(sorted(store["datasets"]))
        raise agents.ToolError(f"No dataset named {dataset!r}. Datasets loaded in this session: {available}.")
    return store["datasets"][name]


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise agents.ToolError(f"Unknown column {column!r}. Columns: {list(df.columns)}.")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _is_mostly_numeric(series: pd.Series) -> bool:
    numeric = _numeric(series)
    total = int(series.notna().sum())
    return total > 0 and int(numeric.notna().sum()) / total > 0.95


def _apply_filters(df: pd.DataFrame, filters: list[Filter] | None) -> pd.DataFrame:
    if not filters:
        return df
    filters = [f if isinstance(f, Filter) else Filter.model_validate(f) for f in filters]
    mask = pd.Series(True, index=df.index)
    for condition in filters:
        _require_column(df, condition.column)
        column = df[condition.column]
        if condition.operator == "is_null":
            mask &= column.isna()
            continue
        if condition.operator == "not_null":
            mask &= column.notna()
            continue
        if condition.value is None:
            raise agents.ToolError(f"Operator {condition.operator!r} requires a value.")
        if condition.operator == "in":
            values = [part.strip() for part in str(condition.value).split(",")]
            if _is_mostly_numeric(column):
                values = [v for v in (pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0] for v in values) if v == v]
                mask &= _numeric(column).isin(values)
            else:
                mask &= column.astype("string").isin(values)
            continue
        if condition.operator == "contains":
            mask &= column.fillna("").astype("string").str.contains(str(condition.value), case=False, regex=False)
            continue
        if _is_mostly_numeric(column):
            value = pd.to_numeric(pd.Series([condition.value]), errors="coerce").iloc[0]
            if pd.isna(value):
                raise agents.ToolError(f"Column {condition.column!r} is numeric; value {condition.value!r} is not.")
            left: pd.Series = _numeric(column)
        elif isinstance(condition.value, str):
            value = condition.value
            left = column.astype("string")
        else:
            value = condition.value
            left = column
        if condition.operator == "eq":
            mask &= left == value
        elif condition.operator == "ne":
            mask &= left != value
        elif condition.operator == "gt":
            mask &= left > value
        elif condition.operator == "gte":
            mask &= left >= value
        elif condition.operator == "lt":
            mask &= left < value
        elif condition.operator == "lte":
            mask &= left <= value
        else:
            raise agents.ToolError(f"Unknown operator {condition.operator!r}.")
    return df[mask]


def _compact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _compact(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_compact(value) for value in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return round(obj, 6)
    if pd.isna(obj):
        return None
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


@agents.tool(model_access="direct")
def load_csv(
    ctx: agents.ToolCallbackContext,
    /,
    csv_text: str,
    dataset_name: str = "data",
) -> dict[str, Any]:
    """Load a CSV file's text content as a dataset of this conversation. Call it again with another dataset_name to hold several datasets, or with the same name to replace one; the last loaded dataset becomes the active one that every other tool uses."""
    if not csv_text.strip():
        raise agents.ToolError("The CSV content is empty.")
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as error:
        raise agents.ToolError(f"Could not parse the content as CSV: {error}") from error
    store = _session_store(ctx.session.id)
    store["datasets"][dataset_name] = df
    store["active"] = dataset_name
    return _compact(
        {
            "dataset": dataset_name,
            "rows": int(len(df)),
            "columns": [
                {"name": column, "dtype": str(df[column].dtype), "nulls": int(df[column].isna().sum())}
                for column in df.columns
            ],
            "datasets_in_session": sorted(store["datasets"]),
            "active": store["active"],
        }
    )


@agents.tool(model_access="direct")
def list_datasets(ctx: agents.ToolCallbackContext, /) -> dict[str, Any]:
    """List the datasets loaded in this conversation, with the active one, their row counts and column names."""
    store = _DATASETS.get(ctx.session.id)
    if not store or not store["datasets"]:
        raise agents.ToolError("No dataset is loaded in this session yet. Call load_csv with the CSV content first.")
    return {
        "active": store["active"],
        "datasets": [
            {"name": name, "rows": int(len(df)), "columns": list(df.columns)}
            for name, df in sorted(store["datasets"].items())
        ],
    }


@agents.tool(model_access="direct")
def describe_columns(ctx: agents.ToolCallbackContext, /, dataset: str = "") -> dict[str, Any]:
    """Describe every column of a dataset: inferred kind, null count, unique count, min/max/mean for numeric columns, min/max dates for datetime columns, and the most frequent values otherwise."""
    df = _get_dataframe(ctx.session.id, dataset)
    profiles: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        non_null = int(series.notna().sum())
        profile: dict[str, Any] = {
            "name": column,
            "dtype": str(series.dtype),
            "nulls": int(series.isna().sum()),
            "unique": int(series.nunique()),
        }
        if non_null and _is_mostly_numeric(series):
            valid = _numeric(series).dropna()
            profile.update(
                {"kind": "numeric", "min": float(valid.min()), "max": float(valid.max()), "mean": float(valid.mean())}
            )
        elif non_null and _is_mostly_datetime(series):
            valid = pd.to_datetime(series, errors="coerce", format="mixed").dropna()
            profile.update({"kind": "datetime", "min": str(valid.min()), "max": str(valid.max())})
        else:
            valid = series.dropna()
            profile.update(
                {
                    "kind": "categorical",
                    "sample_value": str(valid.iloc[0]) if non_null else None,
                    "top_values": [
                        {"value": str(value), "count": int(count)}
                        for value, count in valid.value_counts().head(_DESCRIBE_TOP_VALUES).items()
                    ],
                }
            )
        profiles.append(profile)
    return _compact(
        {"dataset": dataset or _active_dataset_name(ctx.session.id), "rows": int(len(df)), "columns": profiles}
    )


def _is_mostly_datetime(series: pd.Series) -> bool:
    if series.dtype.kind in "M":
        return True
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    total = int(series.notna().sum())
    return total > 0 and int(parsed.notna().sum()) / total > 0.95


@agents.tool(model_access="direct")
def filter_rows(
    ctx: agents.ToolCallbackContext,
    /,
    column: str,
    operator: FilterOperator,
    value: str | float | int | None = None,
    limit: Annotated[int, Field(ge=1, le=_MAX_SAMPLE_ROWS)] = 20,
    columns: list[str] | None = None,
    dataset: str = "",
) -> dict[str, Any]:
    """Filter rows by one condition, e.g. column='country', operator='eq', value='FR', and return the matched row count plus a sample of matching rows."""
    df = _get_dataframe(ctx.session.id, dataset)
    _require_column(df, column)
    matched = _apply_filters(df, [Filter(column=column, operator=operator, value=value)])
    selected = matched if columns is None else matched[[c for c in columns if c in matched.columns]]
    return _compact(
        {
            "dataset": dataset or _active_dataset_name(ctx.session.id),
            "matched_rows": int(len(matched)),
            "total_rows": int(len(df)),
            "sample": selected.head(limit).to_dict(orient="records"),
        }
    )


@agents.tool(model_access="direct")
def group_by(
    ctx: agents.ToolCallbackContext,
    /,
    group_by: Annotated[list[str], Field(min_length=1, max_length=5, description="Columns to group by.")],
    aggregations: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=10,
            description="Measures, one per entry: 'row_count' or 'column:agg' with agg in count, nunique, sum, mean, median, min, max, std. Example: ['row_count', 'revenue:sum', 'price:mean'].",
        ),
    ],
    filters: list[Filter] | None = None,
    order_by: str | None = None,
    limit: Annotated[int, Field(ge=1, le=_MAX_GROUP_ROWS)] = 50,
    dataset: str = "",
) -> dict[str, Any]:
    """Compute group-by aggregates over the rows that match optional filters, e.g. group_by=['country'], aggregations=['row_count', 'revenue:sum'] or with filters=[{column:'year', operator:'eq', value:2024}], and return every group with its numbers."""
    df = _get_dataframe(ctx.session.id, dataset)
    for column in group_by:
        _require_column(df, column)
    filtered = _apply_filters(df, filters)
    if filtered.empty:
        raise agents.ToolError("No rows match the given filters.")
    measures: list[tuple[str, str, str]] = []
    for spec in aggregations:
        if spec == "row_count":
            measures.append((spec, "", ""))
            continue
        if ":" not in spec:
            raise agents.ToolError(f"Bad aggregation {spec!r}: use 'row_count' or 'column:agg'.")
        column, agg = spec.split(":", 1)
        if agg not in _PANDAS_AGGREGATIONS:
            raise agents.ToolError(
                f"Bad aggregation {agg!r} for {column!r}. Agg must be one of {sorted(_PANDAS_AGGREGATIONS)}, or 'row_count'."
            )
        _require_column(df, column)
        measures.append((spec, column, _PANDAS_AGGREGATIONS[agg]))
    work = filtered.copy()
    for _, column, agg in measures:
        if agg not in ("", "nunique") and _is_mostly_numeric(filtered[column]):
            work[column] = _numeric(filtered[column])
    grouped = work.groupby(group_by, dropna=False, observed=True)
    result = pd.DataFrame(index=grouped.size().index)
    result["row_count"] = grouped.size().astype("int64")
    for spec, column, agg in measures:
        if agg == "":
            continue
        if agg == "count":
            result[spec] = grouped[column].count()
        else:
            result[spec] = grouped[column].agg(agg)
    result = result.reset_index()
    if order_by:
        _require_column(result, order_by)
        result = result.sort_values(order_by, ascending=False, na_position="last")
    result = result.head(limit)
    return _compact(
        {
            "dataset": dataset or _active_dataset_name(ctx.session.id),
            "rows_before_filters": int(len(df)),
            "rows_used": int(len(filtered)),
            "groups_returned": int(len(result)),
            "rows": result.to_dict(orient="records"),
        }
    )


@agents.tool(model_access="direct")
def sample_rows(
    ctx: agents.ToolCallbackContext,
    /,
    limit: Annotated[int, Field(ge=1, le=_MAX_SAMPLE_ROWS)] = 10,
    columns: list[str] | None = None,
    dataset: str = "",
) -> dict[str, Any]:
    """Show the first rows of a dataset, optionally limited to given columns, to check its shape and contents."""
    df = _get_dataframe(ctx.session.id, dataset)
    selected = df if columns is None else df[[c for c in columns if c in df.columns]]
    return _compact(
        {
            "dataset": dataset or _active_dataset_name(ctx.session.id),
            "total_rows": int(len(df)),
            "rows": selected.head(limit).to_dict(orient="records"),
        }
    )


AGENT_NAME = "data-analyst"

agent = agents.Agent(
    model="mistral-large-4",
    name=AGENT_NAME,
    description="Conversational data analyst over CSV files in Google Drive.",
    instructions=(
        "You are a conversational data analyst. The user gives you a link to a CSV file in their "
        "Google Drive and asks questions about it in plain language.\n"
        "\n"
        "Getting the data:\n"
        "- From the user's Google Drive link, get the file (its id is in the URL) or find it by name "
        "with your Google Drive connector, then read that file's content with the connector.\n"
        "- Pass the file's CSV text to load_csv, which computes the schema and stores the dataset in "
        "this conversation. Never compute numbers from the raw CSV text yourself.\n"
        "\n"
        "Answering questions:\n"
        "- Every number you give must come from a tool result: load_csv, list_datasets, "
        "describe_columns, filter_rows, group_by or sample_rows. No invented numbers.\n"
        "- For 'what is in this file' questions, call describe_columns first, then sample_rows for "
        "concrete examples.\n"
        "- For questions over a subset of rows, use filter_rows; for per-group totals, means, counts "
        "or comparisons between categories, use group_by and report the aggregates with their group "
        "labels.\n"
        "- Remember the dataset name, the columns and everything discussed; answer follow-ups without "
        "asking the user to re-share the file. If a tool reports a dataset or column is missing, say "
        "so plainly and suggest the closest existing names.\n"
        "- Be compact: lead with the headline number, then the supporting aggregates. Only show raw "
        "rows when the user asks to see them."
    ),
    tools=[load_csv, list_datasets, describe_columns, filter_rows, group_by, sample_rows],
    connectors=[agents.connector("google_drive_mcp")],
    builtins=agents.Builtins(skills=False, subagents=False),
)


async def main() -> None:
    result = await agent.run("I will give you a CSV link from my Google Drive. What can you do with it?")
    print(result.text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
