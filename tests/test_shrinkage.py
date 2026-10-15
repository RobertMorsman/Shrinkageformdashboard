from datetime import date

import pandas as pd
import pytest

from shrinkage import (
    aggregate,
    assign_total_cost,
    delta_text,
    euro,
    filter_frame,
    fmt_int,
    make_period_col,
    normalize,
    parse_dates,
    parse_money,
    previous_period,
)


@pytest.mark.parametrize("raw, expected", [
    ("€ 1.234,56", 1234.56),
    ("1,234.56", 1234.56),
    ("123,45", 123.45),
    ("12.50", 12.5),
    ("1.234.567", 1234567.0),
    (" € 7,00 ", 7.0),
    ("-3,5", -3.5),
    (4, 4.0),
    (2.25, 2.25),
    ("", 0.0),
    (None, 0.0),
    ("abc", 0.0),
])
def test_parse_money(raw, expected):
    assert parse_money(pd.Series([raw], dtype=object)).iloc[0] == pytest.approx(expected)


def test_parse_dates_dutch_dayfirst():
    out = parse_dates(pd.Series(["01-11-2025", "15-10-2025", "03-02-2025"]))
    assert list(out.dt.date) == [date(2025, 11, 1), date(2025, 10, 15), date(2025, 2, 3)]


def test_parse_dates_ambiguous_defaults_to_dayfirst():
    out = parse_dates(pd.Series(["03-02-2025"]))
    assert out.iloc[0].date() == date(2025, 2, 3)


def test_parse_dates_us_monthfirst_when_unambiguous():
    out = parse_dates(pd.Series(["10/15/2025", "11/01/2025"]))
    assert list(out.dt.date) == [date(2025, 10, 15), date(2025, 11, 1)]


def test_parse_dates_iso_and_garbage():
    out = parse_dates(pd.Series(["2025-10-03", "2025-10-15 14:30:00", "geen datum", ""]))
    assert out.iloc[0].date() == date(2025, 10, 3)
    assert out.iloc[1].date() == date(2025, 10, 15)
    assert out.iloc[2:].isna().all()


def test_formatting():
    assert euro(1234.5) == "€ 1.234,50"
    assert euro(-12) == "€ -12,00"
    assert fmt_int(1234567) == "1.234.567"
    assert delta_text(110, 100, money=False) == "+10 (+10.0%)"
    assert delta_text(90, 100, money=True) == "€ -10,00 (-10.0%)"
    assert delta_text(5, 0, money=False) == "+5"


def test_previous_period():
    assert previous_period(date(2025, 10, 8), date(2025, 10, 14)) == (date(2025, 10, 1), date(2025, 10, 7))
    assert previous_period(date(2025, 10, 1), date(2025, 10, 1)) == (date(2025, 9, 30), date(2025, 9, 30))


def test_week_starts_on_monday():
    df = pd.DataFrame({"Date": pd.to_datetime(["2025-10-13", "2025-10-15", "2025-10-19", "2025-10-20"])})
    periods = make_period_col(df, "Week")["Period"].dt.date.tolist()
    monday = date(2025, 10, 13)
    assert periods == [monday, monday, monday, date(2025, 10, 20)]


@pytest.mark.parametrize("freq, expected", [
    ("Maand", date(2025, 10, 1)),
    ("Kwartaal", date(2025, 10, 1)),
    ("Jaar", date(2025, 1, 1)),
])
def test_other_periods(freq, expected):
    df = pd.DataFrame({"Date": pd.to_datetime(["2025-11-15"])})
    got = make_period_col(df, freq)["Period"].iloc[0].date()
    assert got == (date(2025, 11, 1) if freq == "Maand" else expected)


def _raw_sheet():
    # Zoals gspread get_all_records() het teruggeeft
    return pd.DataFrame([
        {"Date": "01-10-2025", "Employee": "An", "Department": "Store A", "Product": "Bonbon",
         "Quantity": 2, "Reason": "Expiration date", "Cost price": "€ 3,00"},
        {"Date": "15-10-2025", "Employee": "Bo", "Department": "Store B", "Product": "Milk",
         "Quantity": "1", "Reason": "Breakage", "Cost price": "1,50"},
        {"Date": "", "Employee": "Cy", "Department": "Store A", "Product": "X",
         "Quantity": 1, "Reason": "Breakage", "Cost price": "9"},
        {"Date": "20-10-2025", "Employee": "Di", "Department": "", "Product": "Lid",
         "Quantity": 3, "Reason": "Breakage", "Cost price": "0,50"},
    ])


def test_normalize():
    df = normalize(_raw_sheet())
    assert len(df) == 3  # rij zonder datum valt weg
    assert df["Kostprijs"].tolist() == [3.0, 1.5, 0.5]
    assert df["Totaal_calc"].tolist() == [6.0, 1.5, 1.5]
    assert df["Department"].isna().tolist() == [False, False, True]
    assert (df["TotaleKost"] == 0).all()


def test_assign_total_cost_sources():
    df = normalize(_raw_sheet())
    assert assign_total_cost(df, "kostprijs")["TotaalKost"].sum() == pytest.approx(5.0)
    assert assign_total_cost(df, "berekend")["TotaalKost"].sum() == pytest.approx(9.0)
    # Geen TotaleKost-kolom gevuld → val terug op berekend
    assert assign_total_cost(df, "totalekost")["TotaalKost"].sum() == pytest.approx(9.0)


def test_filter_frame_includes_end_day_with_time():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2025-10-01 09:00", "2025-10-14 16:45", "2025-10-15 00:00"]),
        "Department": ["A", "B", "A"],
        "Reason": ["r1", "r2", "r1"],
    })
    out = filter_frame(df, date(2025, 10, 1), date(2025, 10, 14), None, None)
    assert len(out) == 2
    out = filter_frame(df, date(2025, 10, 1), date(2025, 10, 31), ["A"], ["r1"])
    assert len(out) == 2
    out = filter_frame(df, date(2025, 10, 1), date(2025, 10, 31), [], [])
    assert len(out) == 3


def test_aggregate():
    df = assign_total_cost(normalize(_raw_sheet()), "berekend")
    by_reason = aggregate(df, "Reason").set_index("Reason")
    assert by_reason.loc["Breakage", "TotaalKost"] == pytest.approx(3.0)
    assert by_reason.loc["Breakage", "Aantal"] == 2
    assert by_reason.loc["Expiration date", "Aantal"] == 1
