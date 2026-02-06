# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import altair as alt
import json
import gspread
import re
from pathlib import Path
from datetime import date, timedelta
from google.oauth2.service_account import Credentials

# -----------------------------------------------------------------------------
# Pagina-config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Shrinkage Dashboard", layout="wide")
st.title("📊 Shrinkage Dashboard")

# -----------------------------------------------------------------------------
# 1) Google Sheets connectie (zelfde sheet als je formulier-app)
# -----------------------------------------------------------------------------
SHEET_KEY = "1DDw-ocdH9MDWTnf6OZG5tlfouURGQcoVaoVUd2pLdv8"
SHEET_TAB = None  # None = eerste tabblad; of bv. "Sheet1"

@st.cache_resource
def _gs_client():
    """Maak 1x een geautoriseerde gspread client. Probeert eerst st.secrets, daarna local file."""
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = None

    # 1) Streamlit Cloud secrets (aanbevolen)
    if "gcp_service_account" in st.secrets:
        info = dict(st.secrets["gcp_service_account"])
        if "private_key" in info:
            info["private_key"] = info["private_key"].replace("\\n", "\n")
        credentials = Credentials.from_service_account_info(info, scopes=scope)

    # 2) Lokale fallback voor ontwikkeling
    elif Path("client_secrets.json").exists():
        with open("client_secrets.json", "r") as f:
            info = json.load(f)
            info["private_key"] = info["private_key"].replace("\\n", "\n")
        credentials = Credentials.from_service_account_info(info, scopes=scope)

    else:
        st.error(
            "Geen Google‑credentials gevonden. "
            "Voeg een service account toe in **Settings → Secrets** als `[gcp_service_account]`, "
            "of plaats lokaal `client_secrets.json`."
        )
        st.stop()

    client = gspread.authorize(credentials)
    return client

# -----------------------------------------------------------------------------
# 2) Helpers
# -----------------------------------------------------------------------------
def euro(x: float) -> str:
    """Formatteer getal als Europese euro-notatie."""
    return f"€ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def parse_money(series: pd.Series) -> pd.Series:
    """Zet bedragen uit Sheets om naar floats (€, spaties, NBSP, 1.234,56, 1,234.56, etc.)."""
    def _clean(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        s = (s.replace("€", "")
               .replace("EUR", "")
               .replace("\u00a0", "")  # NBSP
               .replace(" ", ""))
        if "," in s and "." in s:
            # NL-stijl met duizendtallen en komma als decimaal: 1.234,56
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            # Alleen komma → decimaal
            s = s.replace(",", ".")
        else:
            # Engels met duizendtallen: 1,234.56
            s = s.replace(",", "")
        s = re.sub(r"[^0-9\.\-]", "", s)  # laat alleen cijfers, punt, minus over
        try:
            return float(s) if s not in ("", "-", ".") else 0.0
        except Exception:
            return 0.0
    return series.apply(_clean)

def previous_period(start_d: date, end_d: date) -> tuple[date, date]:
    """Vorige periode met exact dezelfde lengte direct vóór de huidige."""
    length = (end_d - start_d).days + 1  # inclusief
    prev_end = start_d - timedelta(days=1)
    prev_start = prev_end - timedelta(days=length - 1)
    return prev_start, prev_end

def make_period_col(frame: pd.DataFrame, freq_key: str) -> pd.DataFrame:
    """Voeg 'Period' toe obv gekozen frequentie (Week/Maand/Kwartaal/Jaar)."""
    frame = frame.copy()
    if freq_key == "Week":
        p = frame["Date"].dt.to_period("W-MON")
        frame["Period"] = p.apply(lambda r: r.start_time)  # maandag als start
    elif freq_key == "Maand":
        frame["Period"] = frame["Date"].dt.to_period("M").dt.to_timestamp()
    elif freq_key == "Kwartaal":
        frame["Period"] = frame["Date"].dt.to_period("Q").dt.to_timestamp()
    elif freq_key == "Jaar":
        frame["Period"] = frame["Date"].dt.to_period("Y").dt.to_timestamp()
    else:
        frame["Period"] = frame["Date"]
    return frame

def kpi_with_delta(container, label: str, current_val: float, prev_val: float, money: bool):
    delta_val = current_val - prev_val
    if prev_val:
        pct = (delta_val / prev_val) * 100
        delta_txt = f"{'+' if delta_val >= 0 else ''}{euro(delta_val) if money else f'{int(delta_val):,}'.replace(',', '.')} ({pct:+.1f}%)"
    else:
        delta_txt = f"{'+' if delta_val >= 0 else ''}{euro(delta_val) if money else f'{int(delta_val):,}'.replace(',', '.')}"
    value_txt = euro(current_val) if money else f"{int(current_val):,}".replace(",", ".")
    container.metric(label, value_txt, delta=delta_txt)

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    """Lees records uit Google Sheets en maak ze bruikbaar voor analyses."""
    client = _gs_client()
    sh = client.open_by_key(SHEET_KEY)
    ws = sh.sheet1 if SHEET_TAB is None else sh.worksheet(SHEET_TAB)
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Kolommen normaliseren (verwachte structuur uit je formulier)
    rename_map = {
        "Datum": "Date",
        "Medewerker": "Employee",
        "Afdeling": "Department",
        "Dervingsreden": "Reason",
        "Kostprijs per stuk": "Kostprijs",
        "Kostprijs (per stuk)": "Kostprijs",
        "Totale kost": "TotaleKost",    # voor het geval je ooit een aparte kolom gebruikt
        "Total Cost": "TotaleKost",
        "Totaal": "TotaleKost",
    }
    df = df.rename(columns=rename_map)

    # Zorg dat kolommen bestaan
    for col in ["Date", "Employee", "Department", "Product", "Quantity", "Reason"]:
        if col not in df.columns:
            df[col] = None

    # Types & berekeningen
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    df["Quantity"] = pd.to_numeric(df.get("Quantity", 0), errors="coerce").fillna(0).astype(float)
    df["Kostprijs"] = parse_money(df.get("Kostprijs", pd.Series(dtype=object))).astype(float)
    if "TotaleKost" in df.columns:
        df["TotaleKost"] = parse_money(df["TotaleKost"]).astype(float)
    else:
        df["TotaleKost"] = 0.0

    # Altijd ook een berekende variant beschikbaar (qty × per-stuk prijs)
    df["Totaal_calc"] = df["Quantity"] * df["Kostprijs"]

    return df

def assign_total_cost(frame: pd.DataFrame, kost_bron: str) -> pd.DataFrame:
    """Bepaal de kolom 'TotaalKost' in een kopie van frame op basis van de gekozen bron."""
    df2 = frame.copy()
    if kost_bron.startswith("Gebruik kolom 'Kostprijs'"):
        # Interpreteer Kostprijs-kolom als het totale bedrag per rij
        df2["TotaalKost"] = df2["Kostprijs"]
    elif kost_bron.startswith("Quantity × Kostprijs"):
        df2["TotaalKost"] = df2["Totaal_calc"]
    else:
        # Gebruik aparte TotaleKost-kolom als die er is met niet-nul som, otherwise fallback
        if "TotaleKost" in df2.columns and df2["TotaleKost"].abs().sum() > 0:
            df2["TotaalKost"] = df2["TotaleKost"]
        else:
            df2["TotaalKost"] = df2["Totaal_calc"]
    return df2

# -----------------------------------------------------------------------------
# 3) Data laden
# -----------------------------------------------------------------------------
df = load_data()

# -----------------------------------------------------------------------------
# 4) Filters (sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 Filters")

    if df.empty:
        st.info("Nog geen data in de sheet.")
        st.stop()

    min_d = df["Date"].min().date()
    max_d = df["Date"].max().date()

    start_d, end_d = st.date_input(
        "Periode",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        format="DD-MM-YYYY",
    )

    freq_key = st.selectbox("Tijdsgroepering", ["Week", "Maand", "Kwartaal", "Jaar"], index=1)

    depts = sorted([d for d in df["Department"].dropna().unique()])
    reasons = sorted([r for r in df["Reason"].dropna().unique()])

    sel_depts = st.multiselect("Winkels/Afdelingen", depts, default=depts)
    sel_reasons = st.multiselect("Dervingsredenen", reasons, default=reasons)

    # ▼ Heel belangrijk: standaard jouw wens (Kostprijs = totaal per rij)
    kost_bron = st.radio(
        "Bron voor Totale derving:",
        [
            "Gebruik kolom 'Kostprijs' (interpreteer als totaalbedrag per rij)",
            "Quantity × Kostprijs (per stuk)",
            "Gebruik kolom 'TotaleKost' (als aanwezig in sheet)"
        ],
        index=0
    )

    # Kies wat je in grafieken wilt zien
    metric = st.radio(
        "Te tonen metric",
        ["Totale kost (EUR)", "Aantal meldingen"],
        index=0,
        help="Toon totale dervingskosten of aantal meldingen."
    )

    if st.button("↻ Data verversen"):
        load_data.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# 5) Filteren: huidige & vorige periode (voor KPI-delta's)
# -----------------------------------------------------------------------------
mask_now = (
    (df["Date"] >= pd.to_datetime(start_d)) &
    (df["Date"] <= pd.to_datetime(end_d)) &
    (df["Department"].isin(sel_depts) if sel_depts else True) &
    (df["Reason"].isin(sel_reasons) if sel_reasons else True)
)
now_df_raw = df.loc[mask_now].copy()
if now_df_raw.empty:
    st.warning("Geen resultaten voor de huidige filters.")
    st.stop()

prev_start, prev_end = previous_period(start_d, end_d)
mask_prev = (
    (df["Date"] >= pd.to_datetime(prev_start)) &
    (df["Date"] <= pd.to_datetime(prev_end)) &
    (df["Department"].isin(sel_depts) if sel_depts else True) &
    (df["Reason"].isin(sel_reasons) if sel_reasons else True)
)
prev_df_raw = df.loc[mask_prev].copy()

# Totale derving bepalen volgens gekozen bron (zowel nu als vorige periode)
now_df = assign_total_cost(now_df_raw, kost_bron)
prev_df = assign_total_cost(prev_df_raw, kost_bron) if not prev_df_raw.empty else prev_df_raw

# -----------------------------------------------------------------------------
# 6) KPI's + benchmark (delta)
# -----------------------------------------------------------------------------
current_cost = float(now_df["TotaalKost"].sum())
current_cnt  = int(now_df.shape[0])  # aantal meldingen (rijen)
current_stores = int(now_df["Department"].nunique())
current_products = int(now_df["Product"].nunique())

prev_cost = float(prev_df["TotaalKost"].sum()) if not prev_df.empty else 0.0
prev_cnt  = int(prev_df.shape[0]) if not prev_df.empty else 0

c1, c2, c3, c4 = st.columns(4)
kpi_with_delta(c1, "Totale derving (EUR)", current_cost, prev_cost, money=True)
kpi_with_delta(c2, "Aantal meldingen",     current_cnt,  prev_cnt,  money=False)
c3.metric("Unieke winkels", f"{current_stores}")
c4.metric("Unieke producten", f"{current_products}")

st.caption(
    f"Benchmark vs. vorige periode: {prev_start.strftime('%d-%m-%Y')} t/m {prev_end.strftime('%d-%m-%Y')} "
    f"(zelfde lengte als huidige selectie)."
)
st.divider()

# -----------------------------------------------------------------------------
# 7) Tabs met grafieken en CSV-downloads
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📅 Per periode", "🏬 Per winkel", "🪪 Per reason", "📋 Details"])

y_col = "TotaalKost" if metric == "Totale kost (EUR)" else "Aantal"
y_title = "Totale derving (EUR)" if y_col == "TotaalKost" else "Aantal meldingen"

with tab1:
    # Per periode (volgt freq_key)
    tdf = make_period_col(now_df, freq_key)
    agg = (
        tdf.groupby("Period")
        .agg(TotaalKost=("TotaalKost", "sum"), Aantal=("Product", "size"))
        .reset_index()
        .sort_values("Period")
    )

    chart = (
        alt.Chart(agg)
        .mark_line(point=True)
        .encode(
            x=alt.X("Period:T", title=freq_key),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            tooltip=[
                alt.Tooltip("Period:T", title=freq_key),
                alt.Tooltip("TotaalKost:Q", title="Derving (EUR)", format=",.2f"),
                alt.Tooltip("Aantal:Q", title="Meldingen"),
            ],
            color=alt.value("#6A5ACD"),
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

    st.download_button(
        "⬇️ Download CSV: aggregatie per periode",
        data=agg.to_csv(index=False).encode("utf-8"),
        file_name=f"derving_per_{freq_key.lower()}.csv",
        mime="text/csv",
    )

with tab2:
    # Per winkel
    by_dept = (
        now_df.groupby("Department", dropna=True)
        .agg(TotaalKost=("TotaalKost", "sum"), Aantal=("Product", "size"))
        .reset_index()
    )
    by_dept = by_dept.sort_values("TotaalKost" if y_col == "TotaalKost" else "Aantal", ascending=False)

    chart = (
        alt.Chart(by_dept)
        .mark_bar()
        .encode(
            y=alt.Y("Department:N", sort="-x", title="Winkel/Afdeling"),
            x=alt.X(f"{y_col}:Q", title=y_title),
            tooltip=[
                alt.Tooltip("Department:N", title="Winkel"),
                alt.Tooltip("TotaalKost:Q", title="Derving (EUR)", format=",.2f"),
                alt.Tooltip("Aantal:Q", title="Meldingen"),
            ],
            color=alt.value("#00A99D"),
        )
        .properties(height=500)
    )
    st.altair_chart(chart, use_container_width=True)

    st.download_button(
        "⬇️ Download CSV: per winkel",
        data=by_dept.to_csv(index=False).encode("utf-8"),
        file_name="derving_per_winkel.csv",
        mime="text/csv",
    )

with tab3:
    # Per reason
    by_reason = (
        now_df.groupby("Reason", dropna=True)
        .agg(TotaalKost=("TotaalKost", "sum"), Aantal=("Product", "size"))
        .reset_index()
    )
    by_reason = by_reason.sort_values("TotaalKost" if y_col == "TotaalKost" else "Aantal", ascending=False)

    chart = (
        alt.Chart(by_reason)
        .mark_bar()
        .encode(
            y=alt.Y("Reason:N", sort="-x", title="Dervingsreden"),
            x=alt.X(f"{y_col}:Q", title=y_title),
            tooltip=[
                alt.Tooltip("Reason:N", title="Reason"),
                alt.Tooltip("TotaalKost:Q", title="Derving (EUR)", format=",.2f"),
                alt.Tooltip("Aantal:Q", title="Meldingen"),
            ],
            color=alt.value("#FF7F50"),
        )
        .properties(height=500)
    )
    st.altair_chart(chart, use_container_width=True)

    st.download_button(
        "⬇️ Download CSV: per reason",
        data=by_reason.to_csv(index=False).encode("utf-8"),
        file_name="derving_per_reason.csv",
        mime="text/csv",
    )

with tab4:
    # Detailtabel (gefilterde data) + CSV
    show_cols = ["Date", "Employee", "Department", "Product", "Quantity", "Reason", "Kostprijs", "TotaalKost"]
    show_cols = [c for c in show_cols if c in now_df.columns]
    st.dataframe(now_df[show_cols].sort_values("Date", ascending=False), use_container_width=True, height=520)

    st.download_button(
        "⬇️ Download CSV: gefilterde details",
        data=now_df[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="derving_details_gefilterd.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# (Optioneel) Debug totals
# -----------------------------------------------------------------------------
# with st.expander("🔧 Debug totals (tijdelijk)"):
#     st.write("Som Kostprijs:", now_df["Kostprijs"].sum())
#     st.write("Som Totaal_calc (qty × prijs):", now_df["Totaal_calc"].sum())
#     st.write("Som TotaalKost (gebruikt):", now_df["TotaalKost"].sum())
``
