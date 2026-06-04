# Raw data — provenance

| | |
|---|---|
| **File** | `BMW_sales_data_(2010-2024).csv` |
| **Source** | [BMW Sales Dataset — Kaggle](https://www.kaggle.com/datasets/eshummalik/bmw-sales-dataset) |
| **Author** | *eshummalik* (Kaggle) |
| **Shape** | 50,000 rows × 11 columns, no missing values |
| **Columns** | `Model`, `Year`, `Region`, `Color`, `Fuel_Type`, `Transmission`, `Engine_Size_L`, `Mileage_KM`, `Price_USD`, `Sales_Volume`, `Sales_Classification` |

## Important caveat

A formal audit (`make eda` / `make audit`) shows this dataset is **structurally
pristine but statistically signal-free** (features are mutually independent), and
that `Sales_Classification` is a **leaked** deterministic threshold on
`Sales_Volume`. This is documented and handled transparently — see
[ADR-0002](../../docs/adr/0002-data-integrity.md),
[ADR-0006](../../docs/adr/0006-signal-audit.md) and the generated
`reports/data_integrity_report.md` / `reports/signal_audit.md`.

All macro-economic, fuel-price, CO₂-regulation and FX data used in the project is
**not** part of this file — it is fetched/synthesised by the augmentation layer
(see [ADR-0003](../../docs/adr/0003-api-augmentation.md)).

> Please refer to the Kaggle page for the dataset's own licence and terms of use.
