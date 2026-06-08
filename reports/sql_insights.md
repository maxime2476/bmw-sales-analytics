# SQL Business Insights - BMW Sales

*Generated: 2026-06-08 · Author: Maxime GOURGUECHON*

> Decision-oriented analytics run with **DuckDB** directly over the raw CSV (no ETL). Queries live in `sql/queries/`. Reproduce with `make sql`.

## Electrification share by region

`electrification_by_region.sql`

| region        |   total_volume |   electrified_volume |   electrified_pct |
|:--------------|---------------:|---------------------:|------------------:|
| Asia          |    4.29743e+07 |          2.20192e+07 |             51.24 |
| North America |    4.24026e+07 |          2.16699e+07 |             51.11 |
| Europe        |    4.25551e+07 |          2.14157e+07 |             50.32 |
| Africa        |    4.15653e+07 |          2.08151e+07 |             50.08 |
| South America |    4.15518e+07 |          2.07473e+07 |             49.93 |
| Middle East   |    4.23266e+07 |          2.10225e+07 |             49.67 |

## 'High' classification rate by region

`high_rate_by_region.sql`

| region        |    n |   n_high |   high_pct |
|:--------------|-----:|---------:|-----------:|
| Europe        | 8334 |     2609 |      31.31 |
| Asia          | 8454 |     2618 |      30.97 |
| North America | 8335 |     2548 |      30.57 |
| South America | 8251 |     2495 |      30.24 |
| Middle East   | 8373 |     2507 |      29.94 |
| Africa        | 8253 |     2469 |      29.92 |

## Price distribution by model (USD)

`price_stats_by_model.sql`

| model    |    n |   avg_price_usd |   p25_price_usd |   median_price_usd |   p75_price_usd |
|:---------|-----:|----------------:|----------------:|-------------------:|----------------:|
| 7 Series | 4666 |           75570 |           53753 |              76093 |           97626 |
| 3 Series | 4595 |           75566 |           53101 |              75461 |           98454 |
| i8       | 4606 |           75366 |           53052 |              75886 |           97052 |
| 5 Series | 4592 |           75288 |           52198 |              75039 |           98679 |
| X1       | 4570 |           75262 |           52385 |              75610 |           98161 |
| X3       | 4497 |           75017 |           51835 |              75373 |           97704 |
| M3       | 4413 |           74842 |           52086 |              74313 |           97193 |
| i3       | 4618 |           74800 |           52593 |              74511 |           97357 |
| X5       | 4487 |           74708 |           52320 |              74599 |           96917 |
| M5       | 4478 |           74475 |           51805 |              74062 |           97735 |
| X6       | 4478 |           74435 |           51766 |              74296 |           97046 |

## Sales volume by region

`top_regions_by_volume.sql`

| region        |   n_records |   total_volume |   avg_volume |   pct_of_total |
|:--------------|------------:|---------------:|-------------:|---------------:|
| Asia          |        8454 |    4.29743e+07 |       5083.3 |          16.96 |
| Europe        |        8334 |    4.25551e+07 |       5106.2 |          16.8  |
| North America |        8335 |    4.24026e+07 |       5087.3 |          16.74 |
| Middle East   |        8373 |    4.23266e+07 |       5055.1 |          16.71 |
| Africa        |        8253 |    4.15653e+07 |       5036.4 |          16.4  |
| South America |        8251 |    4.15518e+07 |       5036   |          16.4  |

## Year-over-year total volume

`yoy_volume.sql`

|   year |   total_volume |         yoy_delta |   yoy_pct |
|-------:|---------------:|------------------:|----------:|
|   2010 |    1.69334e+07 |     nan           |    nan    |
|   2011 |    1.67589e+07 | -174504           |     -1.03 |
|   2012 |    1.67519e+07 |   -7046           |     -0.04 |
|   2013 |    1.68667e+07 |  114838           |      0.69 |
|   2014 |    1.6959e+07  |   92227           |      0.55 |
|   2015 |    1.70102e+07 |   51247           |      0.3  |
|   2016 |    1.69576e+07 |  -52657           |     -0.31 |
|   2017 |    1.66208e+07 | -336739           |     -1.99 |
|   2018 |    1.64123e+07 | -208538           |     -1.25 |
|   2019 |    1.7192e+07  |  779683           |      4.75 |
|   2020 |    1.63108e+07 | -881113           |     -5.13 |
|   2021 |    1.68847e+07 |  573823           |      3.52 |
|   2022 |    1.79209e+07 |       1.03628e+06 |      6.14 |
|   2023 |    1.62687e+07 |      -1.65229e+06 |     -9.22 |
|   2024 |    1.75279e+07 |       1.2592e+06  |      7.74 |

## Reading the results

The flat distributions across regions (~16.7% each) and models (~$75k each) are the SQL view of the same finding the audit proves statistically: the data is uniform noise (see ADR-0002 and `signal_audit.md`).
