# Raw data

Place the three challenge datasets in this folder (they are git-ignored):

| File                          | Description                                                                 |
| ----------------------------- | --------------------------------------------------------------------------- |
| `Fraud_Data.csv`              | E-commerce transactions with the `class` target (1 = fraud, 0 = legitimate) |
| `IpAddress_to_Country.csv`    | Maps integer IP-address ranges to countries                                 |
| `creditcard.csv`              | Bank credit-card transactions (PCA features `V1..V28`) with `Class` target  |

Expected columns:

**Fraud_Data.csv** — `user_id, signup_time, purchase_time, purchase_value, device_id,
source, browser, sex, age, ip_address, class`

**IpAddress_to_Country.csv** — `lower_bound_ip_address, upper_bound_ip_address, country`

**creditcard.csv** — `Time, V1..V28, Amount, Class`
