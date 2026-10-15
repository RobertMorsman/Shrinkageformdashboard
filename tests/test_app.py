"""Rookttest: draai de volledige Streamlit-app met een nep Google Sheet."""
from pathlib import Path
from unittest import mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")

RECORDS = [
    {"Date": "01-09-2025", "Employee": "An", "Department": "Store A", "Product": "Bonbon",
     "Quantity": 2, "Reason": "Expiration date", "Cost price": "€ 3,00"},
    {"Date": "15-10-2025", "Employee": "Bo", "Department": "Store B", "Product": "Milk",
     "Quantity": 1, "Reason": "Breakage", "Cost price": "1,50"},
    {"Date": "20-10-2025", "Employee": "Di", "Department": "Store A", "Product": "Lid",
     "Quantity": 3, "Reason": "Breakage", "Cost price": "0,50"},
]


def _run(records):
    st.cache_data.clear()
    st.cache_resource.clear()
    client = mock.MagicMock()
    client.open_by_key.return_value.worksheet.return_value.get_all_records.return_value = records
    with mock.patch("gspread.authorize", return_value=client), \
         mock.patch("google.oauth2.service_account.Credentials.from_service_account_info"):
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
    return at


def test_app_renders_with_data():
    at = _run(RECORDS)
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    labels = [m.label for m in at.metric]
    assert labels == ["Totale derving (EUR)", "Aantal meldingen", "Unieke winkels", "Unieke producten"]
    assert at.metric[0].value == "€ 5,00"
    assert at.metric[1].value == "3"


@pytest.mark.parametrize("freq", ["Week", "Kwartaal", "Jaar"])
def test_app_switch_frequency_and_metric(freq):
    at = _run(RECORDS)
    at.selectbox[0].select(freq).run()
    at.radio[1].set_value("Aantal meldingen").run()
    at.radio[0].set_value("berekend").run()
    assert not at.exception, at.exception
    assert at.metric[0].value == "€ 9,00"


def test_app_partial_date_range_does_not_crash():
    at = _run(RECORDS)
    from datetime import date
    at.date_input[0].set_value((date(2025, 10, 1),)).run()
    assert not at.exception, at.exception


def test_app_empty_sheet():
    at = _run([])
    assert not at.exception, at.exception
    assert any("Nog geen data" in i.value for i in at.info)
