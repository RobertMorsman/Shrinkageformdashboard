"""
Pure data-helpers voor het Shrinkage Dashboard.

Geen Streamlit-afhankelijkheden, zodat alles los te testen is (zie tests/).
"""
from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd

# -----------------------------------------------------------------------------
# Kolommen
# -----------------------------------------------------------------------------
# Sheet-headers (NL/EN) → vaste interne namen
RENAME_MAP = {
    "Date": "Date",
    "Datum": "Date",

    "Employee": "Employee",
    "Medewerker": "Employee",

    "Department": "Department",
    "Afdeling": "Department",

    "Product": "Product",

    "Quantity": "Quantity",
    "Aantal": "Quantity",

    "Reason": "Reason",
    "Dervingsreden": "Reason",

    "Cost price": "Kostprijs",
    "Cost Price": "Kostprijs",
    "Kostprijs": "Kostprijs",

    "Totale kost": "TotaleKost",
    "Total Cost": "TotaleKost",
    "Totaal": "TotaleKost",
}

REQUIRED_COLS = ["Date", "Employee", "Department", "Product", "Quantity", "Reason", "Kostprijs"]

# Keuzes voor de bron van "Totale derving" (key → label in de UI)
COST_SOURCES = {
    "kostprijs": "Gebruik kolom 'Cost price' / 'Kostprijs' (interpreteer als totaalbedrag per rij)",
    "berekend": "Quantity × Kostprijs (per stuk)",
    "totalekost": "Gebruik kolom 'TotaleKost' (als aanwezig in sheet)",
}

FREQUENCIES = ["Week", "Maand", "Kwartaal", "Jaar"]


# -----------------------------------------------------------------------------
# Formatteren
# -----------------------------------------------------------------------------
def euro(x: float) -> str:
    """Formatteer getal als Europese euro-notatie: € 1.234,56."""
    return f"€ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_int(x: float) -> str:
    """Formatteer geheel getal met punt als duizendtal-scheiding: 1.234."""
    return f"{int(round(x)):,}".replace(",", ".")


def delta_text(current_val: float, prev_val: float, money: bool) -> str:
    """Tekst voor een KPI-delta, bv. '+€ 12,50 (+4.2%)'."""
    delta_val = current_val - prev_val
    sign = "+" if delta_val >= 0 else ""
    txt = f"{sign}{euro(delta_val) if money else fmt_int(delta_val)}"
    if prev_val:
        txt += f" ({delta_val / prev_val * 100:+.1f}%)"
    return txt


# -----------------------------------------------------------------------------
# Parsen
# -----------------------------------------------------------------------------
def _parse_money_value(x) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    s = (s.replace("€", "").replace("EUR", "")
           .replace(" ", "").replace(" ", ""))
    if "," in s and "." in s:
        # Het laatst voorkomende scheidingsteken is de decimaal
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")  # 1.234,56 → 1234.56
        else:
            s = s.replace(",", "")                    # 1,234.56 → 1234.56
    elif "," in s:
        s = s.replace(",", ".")                       # 123,45 → 123.45
    elif s.count(".") > 1:
        s = s.replace(".", "")                        # 1.234.567 → 1234567
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s) if s not in ("", "-", ".") else 0.0
    except ValueError:
        return 0.0


def parse_money(series: pd.Series) -> pd.Series:
    """Zet bedragen uit Sheets om naar floats (€, spaties, NBSP, 1.234,56 of 1,234.56)."""
    return series.apply(_parse_money_value).astype(float)


_DMY = re.compile(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})")


def parse_dates(series: pd.Series) -> pd.Series:
    """
    Zet datums uit Sheets om naar datetime.

    Zonder hint leest pandas '01-11-2025' als 11 januari (maand eerst) en laat
    '15-10-2025' daarna stilletjes vallen. Daarom: bij dd-mm-jjjj-achtige
    waarden bepalen we de volgorde uit de data zelf (dag > 12 of maand > 12)
    en vallen we bij twijfel terug op dag-eerst (Nederlandse notatie).
    ISO (2025-10-15) blijft gewoon werken.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    s = series.astype("string").str.strip()
    parts = s.str.extract(_DMY).apply(pd.to_numeric, errors="coerce")
    # Alleen maand-eerst als de data dat ondubbelzinnig aangeeft
    dayfirst = not (parts[1].gt(12).any() and not parts[0].gt(12).any())
    return pd.to_datetime(s, format="mixed", dayfirst=dayfirst, errors="coerce")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Hernoem kolommen, vul ontbrekende kolommen aan en zet types goed."""
    df = df.rename(columns=RENAME_MAP)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = None
    if "TotaleKost" not in df.columns:
        df["TotaleKost"] = 0.0

    df["Date"] = parse_dates(df["Date"])
    df = df.dropna(subset=["Date"]).copy()

    # Lege cellen komen uit Sheets als "" — behandel ze als ontbrekend
    # en zet alles om naar tekst (Sheets kan getallen als int teruggeven)
    for col in ["Employee", "Department", "Product", "Reason"]:
        df[col] = df[col].replace("", None)
        df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())

    df["Quantity"] = parse_money(df["Quantity"])
    df["Kostprijs"] = parse_money(df["Kostprijs"])
    df["TotaleKost"] = parse_money(df["TotaleKost"])
    df["Totaal_calc"] = df["Quantity"] * df["Kostprijs"]
    return df


# -----------------------------------------------------------------------------
# Rekenen
# -----------------------------------------------------------------------------
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
        # W-SUN = week eindigt op zondag → start op maandag
        frame["Period"] = frame["Date"].dt.to_period("W-SUN").dt.start_time
    elif freq_key == "Maand":
        frame["Period"] = frame["Date"].dt.to_period("M").dt.to_timestamp()
    elif freq_key == "Kwartaal":
        frame["Period"] = frame["Date"].dt.to_period("Q").dt.to_timestamp()
    elif freq_key == "Jaar":
        frame["Period"] = frame["Date"].dt.to_period("Y").dt.to_timestamp()
    else:
        frame["Period"] = frame["Date"]
    return frame


def assign_total_cost(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Bepaal 'TotaalKost' in een kopie van frame op basis van de gekozen bron-key."""
    df2 = frame.copy()
    if source == "kostprijs":
        df2["TotaalKost"] = df2["Kostprijs"]          # 'Cost price' is totaal per regel
    elif source == "berekend":
        df2["TotaalKost"] = df2["Totaal_calc"]        # aantal × (per-stuk)
    elif "TotaleKost" in df2.columns and df2["TotaleKost"].abs().sum() > 0:
        df2["TotaalKost"] = df2["TotaleKost"]         # aparte totalen-kolom
    else:
        df2["TotaalKost"] = df2["Totaal_calc"]
    return df2


def filter_frame(df: pd.DataFrame, start_d: date, end_d: date,
                 depts: list | None, reasons: list | None) -> pd.DataFrame:
    """Filter op datumbereik (inclusief einddag) en optioneel afdelingen/redenen."""
    day = df["Date"].dt.normalize()
    mask = (day >= pd.Timestamp(start_d)) & (day <= pd.Timestamp(end_d))
    if depts:
        mask &= df["Department"].isin(depts)
    if reasons:
        mask &= df["Reason"].isin(reasons)
    return df.loc[mask].copy()


def aggregate(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Som van TotaalKost en aantal meldingen per groep."""
    return (
        frame.groupby(by, dropna=True)
        .agg(TotaalKost=("TotaalKost", "sum"), Aantal=("TotaalKost", "size"))
        .reset_index()
    )
