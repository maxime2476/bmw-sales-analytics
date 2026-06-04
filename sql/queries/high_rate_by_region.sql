-- Share of transactions classified 'High' by region (the leaked label, audited).
SELECT
    Region                                              AS region,
    COUNT(*)                                            AS n,
    SUM(CASE WHEN Sales_Classification = 'High' THEN 1 ELSE 0 END)  AS n_high,
    ROUND(100.0 * AVG(CASE WHEN Sales_Classification = 'High'
             THEN 1.0 ELSE 0.0 END), 2)                 AS high_pct
FROM bmw
GROUP BY region
ORDER BY high_pct DESC;
