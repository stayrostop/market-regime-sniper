import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RegimeMagnitudePlotter:
    """
    Helper class to:
      1) Detect bull/bear regimes from clustered annual returns
      2) Plot signed regime magnitude per year using UMAP distances
         and overlay log(S&P 500 Close).

    Typical usage:
        plotter = RegimeMagnitudePlotter()
        plotter.plot(
            df_clusters=df_clusters_all,
            umap_2d=umap_2d_all,
            spx=spx,
            regime_source_df=df_clusters_train,   # or None
        )
    """

    def __init__(
        self,
        cluster_col: str = "cluster",
        ret_col: str = "returns",
        title_prefix: str = "Cluster Regime Magnitude",
    ):
        self.cluster_col = cluster_col
        self.ret_col = ret_col
        self.title_prefix = title_prefix

    # ------------------------------------------------------------------
    # 1. Bull / Bear detection
    # ------------------------------------------------------------------
    def get_regime_ids(self, df_clusters: pd.DataFrame):
        """
        Find bull/bear regimes based on mean annual return per cluster.

        Returns dict:
          - bull : id of cluster with maximum mean return
          - bear : id of cluster with minimum mean return
          - other: list of remaining cluster ids
          - means: Series with mean returns per cluster (sorted by index)
        """
        grouped = (
            df_clusters.groupby(self.cluster_col)[self.ret_col]
            .mean()
            .sort_index()
        )
        bull_id = grouped.idxmax()
        bear_id = grouped.idxmin()
        other_ids = [c for c in grouped.index if c not in [bull_id, bear_id]]

        return {
            "bull": bull_id,
            "bear": bear_id,
            "other": other_ids,
            "means": grouped,
        }

    # ------------------------------------------------------------------
    # 2. Main plotting method
    # ------------------------------------------------------------------
    def plot(
        self,
        df_clusters: pd.DataFrame,
        umap_2d: np.ndarray,
        spx: pd.DataFrame,
        regime_source_df: pd.DataFrame | None = None,
        title_prefix: str | None = None,
    ):
        """
        Plot signed regime magnitude per year, using UMAP distance from cluster centroids.

        Parameters
        ----------
        df_clusters : DataFrame
            Index = years (int), columns include:
                - self.cluster_col (e.g. 'cluster')
                - self.ret_col     (e.g. 'returns')
            This is the dataset that will be plotted (x-axis = years).

        umap_2d : np.ndarray, shape (n_years, 2)
            UMAP 2D embedding corresponding to df_clusters.index order.

        spx : DataFrame
            Daily SPX data with DateTimeIndex and column 'Close'
            (used only to overlay log(SPX Close) line).

        regime_source_df : DataFrame or None
            If None  → bull/bear detection is done on df_clusters.
            If not None → bull/bear are computed on regime_source_df
            (e.g. train-only), but the sign/colors are then applied to df_clusters.

        title_prefix : str or None
            Title prefix for the plot. If None, uses self.title_prefix.
        """
        if title_prefix is None:
            title_prefix = self.title_prefix

        # --------------------------------------------------------------
        # 1) Choose source df for regime detection (train or full)
        # --------------------------------------------------------------
        if regime_source_df is None:
            regime_source_df = df_clusters

        reg_info = self.get_regime_ids(regime_source_df)
        bull_id = reg_info["bull"]
        bear_id = reg_info["bear"]
        cluster_means_src = reg_info["means"]

        print("Regime means (source df):")
        print(cluster_means_src)
        print(f"\nBull cluster: {bull_id} | Bear cluster: {bear_id}")

        # --------------------------------------------------------------
        # 2) Data to plot
        # --------------------------------------------------------------
        years = df_clusters.index.values
        labels = df_clusters[self.cluster_col].values.astype(int)

        # Means on the plotted df (may differ slightly from train-only)
        cluster_means_plot = df_clusters.groupby(self.cluster_col)[self.ret_col].mean()

        # --------------------------------------------------------------
        # 3) UMAP centroids & magnitudes (in plotted df space)
        # --------------------------------------------------------------
        centroids = {
            c: umap_2d[labels == c].mean(axis=0) for c in np.unique(labels)
        }

        magnitudes = np.array(
            [np.linalg.norm(umap_2d[i] - centroids[labels[i]]) for i in range(len(labels))]
        )

        if magnitudes.max() == 0:
            magnitudes_scaled = np.ones_like(magnitudes)
        else:
            magnitudes_scaled = 10 * magnitudes / magnitudes.max()

        x_pos = np.arange(len(years))

        # --------------------------------------------------------------
        # 4) Signed magnitude per cluster (based on regime_source_df)
        # --------------------------------------------------------------
        global_mean = regime_source_df[self.ret_col].mean()

        sign_per_cluster = {}
        for c in cluster_means_src.index:
            if c == bull_id:
                sign_per_cluster[c] = 1.0
            elif c == bear_id:
                sign_per_cluster[c] = -1.0
            else:
                # intermediate: above global_mean → +0.5, below → -0.5
                sign_per_cluster[c] = (
                    0.5 if cluster_means_src[c] >= global_mean else -0.5
                )

        signed_mag = np.array(
            [magnitudes_scaled[i] * sign_per_cluster.get(labels[i], 0.0)
             for i in range(len(labels))]
        )

        # --------------------------------------------------------------
        # 5) Colors per cluster (unique, with bull/bear override)
        # --------------------------------------------------------------
        unique_clusters = sorted(np.unique(labels))
        base_cmap = plt.colormaps.get_cmap("tab20")

        color_map = {}
        for idx, c in enumerate(unique_clusters):
            base_color = base_cmap(idx % 20)

            if c == bull_id:      # strong green for bull
                color_map[c] = "tab:green"
            elif c == bear_id:    # strong red for bear
                color_map[c] = "tab:red"
            else:
                color_map[c] = base_color

        colors = [color_map[c] for c in labels]

        # --------------------------------------------------------------
        # 6) S&P 500 annual close (log)
        # --------------------------------------------------------------
        sp_annual_close = spx["Close"].resample("YE").last()
        sp_annual_close.index = sp_annual_close.index.year
        sp_annual_close = sp_annual_close.loc[years]
        sp_log = np.log1p(sp_annual_close.values)

        # --------------------------------------------------------------
        # 7) Plot
        # --------------------------------------------------------------
        plt.figure(figsize=(26, 10))

        # Bars (signed magnitude)
        plt.bar(x_pos, signed_mag, color=colors, alpha=0.9)

        # Legend by hand
        handles = []
        labels_legend = []
        for c in unique_clusters:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color_map[c]))
            mean_for_label = cluster_means_plot.get(c, np.nan)
            labels_legend.append(f"Cluster {c} (mean={mean_for_label:.2f}%)")

        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(x_pos[::5], years[::5], rotation=90)
        plt.ylabel("Signed UMAP distance (cluster magnitude)")

        plt.title(
            f"{title_prefix}\n"
            f"Bull (green, up) / Bear (red, down) based on regime_source_df"
        )

        # S&P line (secondary axis)
        ax2 = plt.twinx()
        ax2.plot(
            x_pos,
            sp_log,
            color="steelblue",
            linewidth=3.0,
            label="S&P 500 (log Close)",
        )
        ax2.set_ylabel("S&P 500 Close (log)")

        plt.legend(
            handles + [ax2.lines[0]],
            labels_legend + ["S&P 500 (log Close)"],
            loc="upper left",
        )

        plt.tight_layout()
        plt.grid(True, axis="y", alpha=0.3)
        plt.show()

        # Επιστρέφω reg_info αν θες να το ξαναχρησιμοποιήσεις
        return reg_info