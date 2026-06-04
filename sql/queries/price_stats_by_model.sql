-- List-price distribution per model: mean and quartiles (P25 / median / P75).
SELECT
    Model                                               AS model,
    COUNT(*)                                            AS n,
    ROUND(AVG(Price_USD), 0)                            AS avg_price_usd,
    ROUND(quantile_cont(Price_USD, 0.25), 0)            AS p25_price_usd,
    ROUND(quantile_cont(Price_USD, 0.50), 0)            AS median_price_usd,
    ROUND(quantile_cont(Price_USD, 0.75), 0)            AS p75_price_usd
FROM bmw
GROUP BY model
ORDER BY avg_price_usd DESC;
