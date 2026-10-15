# Shrinkage Dashboard

Streamlit-dashboard voor dervingsmeldingen (shrinkage) uit een Google Sheet:
totale kosten en aantal meldingen per periode, winkel/afdeling en reden, met
vergelijking t.o.v. de vorige periode van dezelfde lengte en CSV-exports.

## Bestanden

| Bestand | Wat |
|---|---|
| `app.py` | Streamlit-app (UI, Google Sheets-connectie) |
| `shrinkage.py` | Pure data-helpers: kolommen mappen, bedragen/datums parsen, filteren, aggregeren |
| `tests/` | Unit-tests voor `shrinkage.py` + rooktest van de hele app met een nep-sheet |
| `options-dash.xlsx` | Productlijst/kostprijzen en dervingsredenen (referentie) |

## Databron

Sheet-key en tabblad staan bovenaan `app.py` (`SHEET_KEY`, `SHEET_TAB = "Blad1"`).
Verwachte kolommen (NL of EN):
`Date/Datum`, `Employee/Medewerker`, `Department/Afdeling`, `Product`,
`Quantity/Aantal`, `Reason/Dervingsreden`, `Cost price/Kostprijs`,
optioneel `Totale kost/Total Cost/Totaal`.

Datums mogen `dd-mm-jjjj`, `jjjj-mm-dd` of (ondubbelzinnig) `mm/dd/jjjj` zijn.
Bedragen mogen `€ 1.234,56` of `1,234.56` zijn.

## Credentials

De sheet moet gedeeld zijn met het e-mailadres van het service-account.
De app zoekt credentials in deze volgorde:

1. `st.secrets["gcp_service_account"]` — op Streamlit Cloud via **Settings → Secrets**,
   lokaal via `.streamlit/secrets.toml` (zie `.streamlit/secrets.toml.example`).
2. `client_secrets.json` naast `app.py`.

De app vraagt alleen **read-only** toegang tot Sheets.

## Lokaal draaien

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

De tests praten niet met Google; de Sheets-client wordt gemockt.
Ze draaien automatisch via GitHub Actions bij elke push.
