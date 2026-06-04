-- Total and average sales volume by region, with each region's share of the total.
SELECT
    Region                                              AS region,
    COUNT(*)                                            AS n_records,
    SUM(Sales_Volume)                                   AS total_volume,
    ROUND(AVG(Sales_Volume), 1)                         AS avg_volume,
    ROUND(100.0 * SUM(Sales_Volume)
          / SUM(SUM(Sales_Volume)) OVER (), 2)          AS pct_of_total
FROM bmw
GROUP BY region
ORDER BY total_volume DESC;
