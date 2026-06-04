-- Year-over-year total sales volume and percentage change (window LAG).
WITH yearly AS (
    SELECT Year AS year, SUM(Sales_Volume) AS total_volume
    FROM bmw
    GROUP BY year
)
SELECT
    year,
    total_volume,
    total_volume - LAG(total_volume) OVER (ORDER BY year)           AS yoy_delta,
    ROUND(100.0 * (total_volume - LAG(total_volume) OVER (ORDER BY year))
          / NULLIF(LAG(total_volume) OVER (ORDER BY year), 0), 2)   AS yoy_pct
FROM yearly
ORDER BY year;
