import numpy as np
import pandas as pd
from scipy.stats import (
    ttest_ind,
    mannwhitneyu,
    levene,
    ks_2samp,
    norm,
    f as f_dist,
    anderson_ksamp
)


class ClusterRegimeTester:


    def __init__(self, df, cluster_col="cluster", value_col="returns"):
        self.cluster_col = cluster_col
        self.value_col = value_col
        self.df = df.copy().dropna(subset=[cluster_col, value_col])
        self.clusters = sorted(self.df[cluster_col].unique())

    # ------------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------------

    @staticmethod
    def sharpe_ratio(x: np.ndarray):
        x = np.asarray(x, dtype=float)
        if x.size < 2:
            return np.nan
        s = x.std(ddof=1)
        return x.mean() / s if s > 0 else np.nan

    @staticmethod
    def memmel_test(x: np.ndarray, y: np.ndarray):
        """Memmel (2003) test for difference in Sharpe ratios."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        nx, ny = len(x), len(y)
        if nx < 3 or ny < 3:
            return np.nan, np.nan

        sx = ClusterRegimeTester.sharpe_ratio(x)
        sy = ClusterRegimeTester.sharpe_ratio(y)

        mx, my = x.mean(), y.mean()
        vx, vy = x.var(ddof=1), y.var(ddof=1)

        cov_xy = 0.0  # assume independence

        denom = np.sqrt(
            (vx / ((nx - 1) * mx**2)) +
            (vy / ((ny - 1) * my**2)) -
            (2 * cov_xy / (np.sqrt(nx * ny) * mx * my))
        )

        if denom == 0 or np.isnan(denom):
            return np.nan, np.nan

        z_stat = (sx - sy) / denom
        p_val = 2 * (1 - norm.cdf(abs(z_stat)))
        return z_stat, p_val

    @staticmethod
    def grs_test(x: np.ndarray, y: np.ndarray):
        """Simplified 2-portfolio GRS (mean equality)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        nx, ny = len(x), len(y)
        if nx < 3 or ny < 3:
            return np.nan, np.nan

        mx, my = x.mean(), y.mean()
        vx, vy = x.var(ddof=1), y.var(ddof=1)

        sp = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
        if sp == 0 or np.isnan(sp):
            return np.nan, np.nan

        n = min(nx, ny)
        grs = ((mx - my) ** 2) / (2 * sp) * (n - 2)
        p_val = 1 - f_dist.cdf(grs, 1, n - 2)
        return grs, p_val

    # ------------------------------------------------------------------------
    # Core Test Runner
    # ------------------------------------------------------------------------

    @staticmethod
    def _run_basic_tests(a: np.ndarray, b: np.ndarray):

        a = np.asarray(a, float)
        b = np.asarray(b, float)

        out = {}

        # If not enough samples → return NaN
        if len(a) < 2 or len(b) < 2:
            for k in [
                "t_stat", "t_p",
                "u_stat", "u_p",
                "lev_stat", "lev_p",
                "cohen_d",
                "ks_stat", "ks_p",
                "ad_stat", "ad_p",
                "memmel_z", "memmel_p",
                "grs_stat", "grs_p",
                "sharpe_a", "sharpe_b",
            ]:
                out[k] = np.nan
            return out

        # Welch t-test
        t_stat, t_p = ttest_ind(a, b, equal_var=False)

        # Mann–Whitney U
        u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")

        # Levene
        lev_stat, lev_p = levene(a, b)

        # Cohen's d
        pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        cohen_d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan

        # KS test
        ks_stat, ks_p = ks_2samp(a, b)

        # Anderson-Darling k-sample test
        ad_result = anderson_ksamp([a, b])
        ad_stat = ad_result.statistic
        ad_p = ad_result.significance_level

        # Memmel (Sharpe)
        memmel_z, memmel_p = ClusterRegimeTester.memmel_test(a, b)

        # GRS
        grs_stat, grs_p = ClusterRegimeTester.grs_test(a, b)

        out.update(
            t_stat=t_stat, t_p=t_p,
            u_stat=u_stat, u_p=u_p,
            lev_stat=lev_stat, lev_p=lev_p,
            cohen_d=cohen_d,
            ks_stat=ks_stat, ks_p=ks_p,
            ad_stat=ad_stat, ad_p=ad_p,
            memmel_z=memmel_z, memmel_p=memmel_p,
            grs_stat=grs_stat, grs_p=grs_p,
            sharpe_a=ClusterRegimeTester.sharpe_ratio(a),
            sharpe_b=ClusterRegimeTester.sharpe_ratio(b),
        )
        return out

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def bull_vs_rest(self):

        means = self.df.groupby(self.cluster_col)[self.value_col].mean()
        bull_cluster = means.idxmax()

        bull_rets = self.df.loc[self.df[self.cluster_col] == bull_cluster, self.value_col].values
        bear_rets = self.df.loc[self.df[self.cluster_col] != bull_cluster, self.value_col].values

        tests = self._run_basic_tests(bull_rets, bear_rets)

        result = {
            "bull_cluster": bull_cluster,
            "bear_clusters": [c for c in self.clusters if c != bull_cluster],
            "bull_n": len(bull_rets),
            "bear_n": len(bear_rets),
            "bull_mean": bull_rets.mean(),
            "bear_mean": bear_rets.mean(),
            **tests
        }
        return result

    def pairwise_cluster_tests(self):

        results = {}
        for i, c1 in enumerate(self.clusters):
            a = self.df.loc[self.df[self.cluster_col] == c1, self.value_col].values
            for c2 in self.clusters[i + 1:]:
                b = self.df.loc[self.df[self.cluster_col] == c2, self.value_col].values
                results[(c1, c2)] = {
                    "cluster_1": c1,
                    "cluster_2": c2,
                    **self._run_basic_tests(a, b)
                }
        return results

    # ------------------------------------------------------------------------
    # Pretty printing helpers
    # ------------------------------------------------------------------------

    def pretty_print_bull_vs_rest(self, res=None):
        if res is None:
            res = self.bull_vs_rest()

        print("==================================================")
        print("        BULL vs REST – Cluster Regime Tests       ")
        print("==================================================")
        print(f"Bull cluster      : {int(res['bull_cluster'])}")
        print(f"Other clusters    : {[int(c) for c in res['bear_clusters']]}")
        print(f"N (bull / rest)   : {res['bull_n']} / {res['bear_n']}")
        print(f"Mean (bull / rest): {res['bull_mean']:.2f}% / {res['bear_mean']:.2f}%")
        print(f"Sharpe (bull/rest): {res['sharpe_a']:.3f} / {res['sharpe_b']:.3f}")
        print()

        print("---- Mean & Distribution tests ----")
        print(f"Welch t-test      : t = {res['t_stat']:.3f},  p = {res['t_p']:.3f}")
        print(f"Mann–Whitney U    : U = {res['u_stat']:.3f},  p = {res['u_p']:.3f}")
        print(f"Levene (variance) : stat = {res['lev_stat']:.3f}, p = {res['lev_p']:.3f}")
        print(f"Cohen's d         : d = {res['cohen_d']:.3f}")
        print(f"KS test           : KS = {res['ks_stat']:.3f}, p = {res['ks_p']:.3f}")
        print(f"Anderson–Darling  : AD = {res['ad_stat']:.3f}, p ≈ {res['ad_p']:.3f}")
        print()

        print("---- Sharpe / Portfolio-level tests ----")
        print(f"Memmel (Sharpe Δ) : z = {res['memmel_z']:.3f}, p = {res['memmel_p']:.3f}")
        print(f"GRS (mean eq.)    : F = {res['grs_stat']:.3f}, p = {res['grs_p']:.3f}")
        print("==================================================")

    def pairwise_results_df(self, pair_results=None):
   
        if pair_results is None:
            pair_results = self.pairwise_cluster_tests()

        rows = []
        for (c1, c2), r in pair_results.items():
            row = {
                "cluster_1": int(c1),
                "cluster_2": int(c2),
                "t_stat": r["t_stat"],
                "t_p": r["t_p"],
                "u_stat": r["u_stat"],
                "u_p": r["u_p"],
                "lev_stat": r["lev_stat"],
                "lev_p": r["lev_p"],
                "cohen_d": r["cohen_d"],
                "ks_stat": r["ks_stat"],
                "ks_p": r["ks_p"],
                "ad_stat": r["ad_stat"],
                "ad_p": r["ad_p"],
                "memmel_z": r["memmel_z"],
                "memmel_p": r["memmel_p"],
                "grs_stat": r["grs_stat"],
                "grs_p": r["grs_p"],
                "sharpe_1": r["sharpe_a"],
                "sharpe_2": r["sharpe_b"],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.set_index(["cluster_1", "cluster_2"], inplace=True)
        return df