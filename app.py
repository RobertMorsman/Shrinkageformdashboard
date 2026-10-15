import json
from datetime import date
from pathlib import Path

import altair as alt
import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials

from shrinkage import (
    COST_SOURCES,
    FREQUENCIES,
    aggregate,
    assign_total_cost,
    delta_text,
    euro,
    filter_frame,
    fmt_int,
    make_period_col,
    normalize,
    previous_period,
)

# -----------------------------------------------------------------------------
# Pagina-config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Shrinkage Dashboard", layout="wide")
st.title("📊 Shrinkage Dashboard")

# -----------------------------------------------------------------------------
# 1) Google Sheets connectie
#    - Gebruik st.secrets op Streamlit Cloud
#    - Fallback: lokaal client_secrets.json
# -----------------------------------------------------------------------------
SHEET_KEY = "1DDw-ocdH9MDWTnf6OZG5tlfouURGQcoVaoVUd2pLdv8"
SHEET_TAB = "Blad1"  # exact de tabnaam uit je bestand
LOCAL_CREDENTIALS = Path(__file__).parent / "client_secrets.json"

# Dashboard leest alleen: read-only scope is voldoende
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def _has_secret(key: str) -> bool:
    # st.secrets gooit een fout als er helemaal geen secrets.toml bestaat
    try:
        return key in st.secrets
    except Exception:
        return False


@st.cache_resource
def _gs_client():
    # 1) Streamlit Cloud secrets (aanbevolen)
    if _has_secret("gcp_service_account"):
        info = dict(st.secrets["gcp_service_account"])
    # 2) Lokale fallback
    elif LOCAL_CREDENTIALS.exists():
        info = json.loads(LOCAL_CREDENTIALS.read_text())
    else:
        st.error(
            "Geen Google‑credentials gevonden. "
            "Zet een service account in **Settings → Secrets** als `[gcp_service_account]`, "
            "of plaats lokaal `client_secrets.json`."
        )
        st.stop()

    if "private_key" in info:
        info["private_key"] = info["private_key"].replace("\\n", "\n")
    credentials = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(credentials)


# -----------------------------------------------------------------------------
# 2) Data laden uit Google Sheet
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Data laden uit Google Sheets…")
def load_data() -> pd.DataFrame:
    """
    Lees records uit Google Sheets en maak ze bruikbaar voor analyses.
    Kolommen in tab 'Blad1':
    Date, Employee, Department, Product, Quantity, Reason, Cost price
    → gemapt naar vaste interne namen (zie shrinkage.RENAME_MAP).
    """
    client = _gs_client()
    sh = client.open_by_key(SHEET_KEY)
    ws = sh.worksheet(SHEET_TAB) if SHEET_TAB else sh.sheet1
    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        return df
    return normalize(df)


def kpi_with_delta(container, label: str, current_val: float, prev_val: float, money: bool):
    value_txt = euro(current_val) if money else fmt_int(current_val)
    # 🔴⬆️ rood bij stijging, 🟢⬇️ groen bij daling
    container.metric(label, value_txt, delta=delta_text(current_val, prev_val, money),
                     delta_color="inverse")


try:
    df = load_data()
except gspread.exceptions.SpreadsheetNotFound:
    st.error("Google Sheet niet gevonden. Is de sheet gedeeld met het service-account e-mailadres?")
    st.stop()
except gspread.exceptions.WorksheetNotFound:
    st.error(f"Tabblad '{SHEET_TAB}' niet gevonden in de Google Sheet.")
    st.stop()
except Exception as e:  # netwerk / API / credentials
    st.error(f"Kon data niet laden uit Google Sheets: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3) Filters (sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 Filters")

    if df.empty:
        st.info("Nog geen data in de sheet.")
        st.stop()

    min_d = df["Date"].min().date()
    max_data_d = df["Date"].max().date()
    max_d = max(date.today(), max_data_d)   # selecteren mag t/m vandaag

    period = st.date_input(
        "Periode",
        value=(min_d, max_data_d),           # standaard eind = laatste datum in data
        min_value=min_d,
        max_value=max_d,
        format="DD-MM-YYYY",
    )
    # Tijdens het kiezen van een bereik geeft Streamlit eerst maar één datum terug
    if isinstance(period, (tuple, list)):
        start_d, end_d = (period[0], period[-1]) if period else (min_d, max_data_d)
    else:
        start_d = end_d = period

    freq_key = st.selectbox("Tijdsgroepering", FREQUENCIES, index=1)

    depts = sorted(df["Department"].dropna().astype(str).unique())
    reasons = sorted(df["Reason"].dropna().astype(str).unique())

    sel_depts = st.multiselect("Winkels/Afdelingen", depts, default=depts,
                               help="Leeg = alle winkels/afdelingen.")
    sel_reasons = st.multiselect("Dervingsredenen", reasons, default=reasons,
                                 help="Leeg = alle redenen.")

    # Standaard: 'Cost price'/'Kostprijs' als totaal per rij
    kost_bron = st.radio(
        "Bron voor Totale derving:",
        list(COST_SOURCES),
        format_func=COST_SOURCES.get,
        index=0,
    )

    metric = st.radio(
        "Te tonen metric",
        ["Totale kost (EUR)", "Aantal meldingen"],
        index=0,
        help="Toon totale dervingskosten of aantal meldingen.",
    )

    if st.button("↻ Data verversen"):
        load_data.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# 4) Filteren: huidige & vorige periode (voor KPI-delta's)
# -----------------------------------------------------------------------------
now_df = filter_frame(df, start_d, end_d, sel_depts, sel_reasons)
if now_df.empty:
    st.warning("Geen resultaten voor de huidige filters.")
    st.stop()

prev_start, prev_end = previous_period(start_d, end_d)
prev_df = filter_frame(df, prev_start, prev_end, sel_depts, sel_reasons)

now_df = assign_total_cost(now_df, kost_bron)
prev_df = assign_total_cost(prev_df, kost_bron)

# -----------------------------------------------------------------------------
# 5) KPI's + benchmark (delta)
# -----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
kpi_with_delta(c1, "Totale derving (EUR)", float(now_df["TotaalKost"].sum()),
               float(prev_df["TotaalKost"].sum()), money=True)
kpi_with_delta(c2, "Aantal meldingen", len(now_df), len(prev_df), money=False)
c3.metric("Unieke winkels", f"{now_df['Department'].nunique()}")
c4.metric("Unieke producten", f"{now_df['Product'].nunique()}")

st.caption(
    f"Benchmark vs. vorige periode: {prev_start.strftime('%d-%m-%Y')} t/m {prev_end.strftime('%d-%m-%Y')} "
    f"(zelfde lengte als huidige selectie)."
)
st.divider()

# -----------------------------------------------------------------------------
# 6) Tabs met grafieken en CSV-downloads
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📅 Per periode", "🏬 Per winkel", "🪪 Per reason", "📋 Details"])

y_col = "TotaalKost" if metric == "Totale kost (EUR)" else "Aantal"
y_title = "Totale derving (EUR)" if y_col == "TotaalKost" else "Aantal meldingen"


def _tooltip(dim: str, dim_type: str, title: str) -> list:
    return [
        alt.Tooltip(f"{dim}:{dim_type}", title=title),
        alt.Tooltip("TotaalKost:Q", title="Derving (EUR)", format=",.2f"),
        alt.Tooltip("Aantal:Q", title="Meldingen"),
    ]


def _csv(frame: pd.DataFrame) -> bytes:
    # utf-8-sig zodat Excel € en accenten goed toont
    return frame.to_csv(index=False).encode("utf-8-sig")


def bar_tab(dim: str, title: str, color: str, file_name: str):
    data = aggregate(now_df, dim).sort_values(y_col, ascending=False)
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            y=alt.Y(f"{dim}:N", sort="-x", title=title),
            x=alt.X(f"{y_col}:Q", title=y_title),
            tooltip=_tooltip(dim, "N", title),
            color=alt.value(color),
        )
        .properties(height=max(300, 28 * len(data)))
    )
    st.altair_chart(chart, width="stretch")
    st.download_button(
        f"⬇️ Download CSV: per {title.lower()}",
        data=_csv(data),
        file_name=file_name,
        mime="text/csv",
    )


with tab1:
    agg = aggregate(make_period_col(now_df, freq_key), "Period").sort_values("Period")
    chart = (
        alt.Chart(agg)
        .mark_line(point=True)
        .encode(
            x=alt.X("Period:T", title=freq_key),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            tooltip=_tooltip("Period", "T", freq_key),
            color=alt.value("#6A5ACD"),
        )
        .properties(height=350)
    )
    st.altair_chart(chart, width="stretch")
    st.download_button(
        "⬇️ Download CSV: aggregatie per periode",
        data=_csv(agg),
        file_name=f"derving_per_{freq_key.lower()}.csv",
        mime="text/csv",
    )

with tab2:
    bar_tab("Department", "Winkel/Afdeling", "#00A99D", "derving_per_winkel.csv")

with tab3:
    bar_tab("Reason", "Dervingsreden", "#FF7F50", "derving_per_reason.csv")

with tab4:
    show_cols = ["Date", "Employee", "Department", "Product", "Quantity", "Reason", "Kostprijs", "TotaalKost"]
    details = now_df[show_cols].sort_values("Date", ascending=False)
    st.dataframe(
        details,
        width="stretch",
        height=520,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Datum", format="DD-MM-YYYY"),
            "Kostprijs": st.column_config.NumberColumn("Kostprijs", format="€ %.2f"),
            "TotaalKost": st.column_config.NumberColumn("Totale kost", format="€ %.2f"),
        },
    )
    st.download_button(
        "⬇️ Download CSV: gefilterde details",
        data=_csv(details),
        file_name="derving_details_gefilterd.csv",
        mime="text/csv",
    )
