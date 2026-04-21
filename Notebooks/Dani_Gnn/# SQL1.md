# SQL
```sql
SELECT 
    AVG(avg_degree_in_cusip) AS avg_degree_in_cusip,
    AVG(avg_degree_out_cik) AS avg_degree_out_cik,
    AVG(num_cusip) AS avg_num_cusip,
    AVG(num_cik) AS avg_num_cik,
    AVG(avg_change_in_shares) AS avg_change_in_shares,
    AVG(avg_change_in_weight) AS avg_change_in_weight,
    AVG(avg_change_in_adjusted_weight) AS avg_change_in_adjusted_weight
FROM changed_stas LIMIT 100
```
# Data
|avg_degree_in_cusip|avg_degree_out_cik|avg_num_cusip|avg_num_cik|avg_change_in_shares|avg_change_in_weight|avg_change_in_adjusted_weight|
|-------------------|------------------|-------------|-----------|--------------------|--------------------|-----------------------------|
|282.3809573991832  |189.96414219937927|2942.1458333333333333|4596.9375000000000000|8626.736802739057   |1.6224808329274386e-19|0.1516199674728033           