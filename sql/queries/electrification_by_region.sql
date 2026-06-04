-- Share of electrified (Hybrid + Electric) sales volume by region.
SELECT
    Region                                              AS region,
    SUM(Sales_Volume)                                   AS total_volume,
    SUM(CASE WHEN Fuel_Type IN ('Hybrid', 'Electric')
             THEN Sales_Volume ELSE 0 END)              AS electrified_volume,
    ROUND(100.0 * SUM(CASE WHEN Fuel_Type IN ('Hybrid', 'Electric')
             THEN Sales_Volume ELSE 0 END)
          / SUM(Sales_Volume), 2)                       AS electrified_pct
FROM bmw
GROUP BY region
ORDER BY electrified_pct DESC;
