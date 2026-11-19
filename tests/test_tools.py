"""Offline tests of the fixed analysis tools: no model, no API key needed."""

import pandas as pd
import pytest

from main import _DATASETS, _session_store, describe_columns, filter_rows, group_by, load_csv, list_datasets, sample_rows

CSV = """order_id,country,product,category,quantity,unit_price,order_date
1,FR,Widget A,widgets,10,9.99,2024-01-05
2,DE,Widget B,widgets,3,19.50,2024-01-08
3,FR,Gadget C,gadgets,7,4.25,2024-02-01
4,IT,Widget A,widgets,,9.99,2024-02-14
5,DE,Gadget D,gadgets,12,4.25,2024-03-03
6,FR,Widget A,widgets,10,9.99,2024-03-15
"""


@pytest.fixture(autouse=True)
def store():
    _DATASETS.clear()
    result = load_csv(csv_text=CSV, dataset_name="orders")
    assert result["rows"] == 6
    assert result["active"] == "orders"
    yield _session_store("local")
    _DATASETS.clear()


def test_load_csv_reports_schema():
    report = load_csv(csv_text=CSV, dataset_name="orders2")
    assert report["dataset"] == "orders2"
    assert report["active"] == "orders2"
    names = [c["name"] for c in report["columns"]]
    assert names == ["order_id", "country", "product", "category", "quantity", "unit_price", "order_date"]
    assert report["columns"][4]["nulls"] == 1


def test_list_datasets():
    result = list_datasets()
    names = [d["name"] for d in result["datasets"]]
    assert names == ["orders"]
    assert result["active"] == "orders"


def test_describe_columns_kinds():
    result = describe_columns()
    by_name = {c["name"]: c for c in result["columns"]}
    assert by_name["quantity"]["kind"] == "numeric"
    assert by_name["quantity"]["mean"] == pytest.approx((10 + 3 + 7 + 12 + 10) / 5)
    assert by_name["order_date"]["kind"] == "datetime"
    assert by_name["order_date"]["min"].startswith("2024-01-05")
    assert by_name["order_date"]["max"].startswith("2024-03-15")
    assert by_name["country"]["kind"] == "categorical"
    top = dict((v["value"], v["count"]) for v in by_name["country"]["top_values"])
    assert top["FR"] == 3


def test_filter_rows_eq_numeric_and_contains():
    result = filter_rows(column="country", operator="eq", value="FR")
    assert result["matched_rows"] == 3
    result = filter_rows(column="product", operator="contains", value="gadget")
    assert result["matched_rows"] == 2
    result = filter_rows(column="unit_price", operator="gt", value="5")
    assert result["matched_rows"] == 4


def test_group_by_sum_mean_count():
    result = group_by(
        group_by=["country"],
        aggregations=["row_count", "unit_price:sum", "unit_price:mean", "product:nunique"],
    )
    rows = {r["country"]: r for r in result["rows"]}
    assert rows["FR"]["row_count"] == 3
    assert rows["FR"]["unit_price:sum"] == pytest.approx(9.99 + 4.25 + 9.99)
    assert rows["DE"]["product:nunique"] == 2
    assert result["rows_used"] == 6


def test_group_by_with_filters():
    result = group_by(
        group_by=["category"],
        aggregations=["row_count", "quantity:sum"],
        filters=[{"column": "country", "operator": "ne", "value": "IT"}],
    )
    rows = {r["category"]: r for r in result["rows"]}
    assert rows["gadgets"]["row_count"] == 2
    assert rows["gadgets"]["quantity:sum"] == 19
    assert result["rows_before_filters"] == 6
    assert result["rows_used"] == 5


def test_group_by_order_and_limit():
    result = group_by(group_by=["product"], aggregations=["row_count"], order_by="row_count", limit=1)
    assert result["groups_returned"] == 1
    assert result["rows"][0]["row_count"] == 3


def test_unknown_column_and_missing_dataset_errors():
    with pytest.raises(Exception, match="Unknown column"):
        filter_rows(column="nope", operator="eq", value="x")
    with pytest.raises(Exception, match="No dataset named"):
        filter_rows(column="country", operator="eq", value="FR", dataset="ghost")


def test_no_dataset_helpful_error():
    _DATASETS.clear()
    with pytest.raises(Exception, match="No dataset is loaded"):
        describe_columns()
    _session_store("local")


def test_in_row_count_in_group_by_is_all_rows():
    result = group_by(group_by=["category"], aggregations=["row_count"])
    total = sum(r["row_count"] for r in result["rows"])
    assert total == 6
