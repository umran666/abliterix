# Damage metrics retain estimator identity

Abliterix records full-distribution KL divergence and fixed-continuation NLL drift as different Damage Metrics, even when both feed the same optimization objective. They estimate different quantities and must not share a leaderboard field or display label, because doing so makes backend comparisons scientifically invalid.
