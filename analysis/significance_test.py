from scipy import stats
# McNemar exact (binomial) on discordant pairs, fired subset
b, c = 16, 8   # wrong->right, right->wrong
p = stats.binomtest(max(b,c), b+c, 0.5, alternative='two-sided').pvalue
print(f"McNemar exact on fired subset: b={b} c={c} n_disc={b+c}  p={p:.4f}")
# Same test on the not-fired subset (pure generation noise, sanity floor)
