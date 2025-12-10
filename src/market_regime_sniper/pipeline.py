from __future__ import annotations

import numpy as np
import pandas as pd
import hdbscan
import umap.umap_ as umap

from .data import build_cosine_norm_features, load_spx_from_csv
from .my_stats import ClusterRegimeTester
from .plot import RegimeMagnitudePlotter
from .sniper_strategy import run_hybrid_sniper
from .ssa_utils import ssa_decompose


def run_ssa_umap_hdbscan(
    features: pd.DataFrame,
    L_cos: int = 55,
    L_norm: int = 63,
    umap_neighbors: int = 6,
    umap_min_dist: float = 0.0,
    hdb_min_cluster: int = 6,
    hdb_min_samples: int = 6,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    End‑to‑end SSA → UMAP → HDBSCAN pipeline on cosine/norm features.

    Returns:
      - df_ssa     : SSA‑smoothed feature matrix
      - umap_2d    : UMAP 2D embedding
      - labels_hdb : HDBSCAN cluster labels
      - df_clusters: DataFrame[index=year, cols=['cluster','returns']]
    """
    cos_cols = [c for c in features.columns if c.startswith("p_cos_")]
    norm_cols = [c for c in features.columns if c.startswith("p_norm_")]

    df_ssa = pd.DataFrame(index=features.index, columns=features.columns, dtype=float)

    for yr in features.index:
        row = features.loc[yr]

        x_cos = row[cos_cols].values.astype(float)
        comps_cos = ssa_decompose(x_cos, L=L_cos)
        df_ssa.loc[yr, cos_cols] = comps_cos[0]

        x_norm = row[norm_cols].values.astype(float)
        comps_norm = ssa_decompose(x_norm, L=L_norm)
        df_ssa.loc[yr, norm_cols] = comps_norm[0]

    um = umap.UMAP(
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    umap_2d = um.fit_transform(df_ssa.values)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdb_min_cluster,
        min_samples=hdb_min_samples,
        cluster_selection_epsilon=0.0,
    )
    labels_hdb = clusterer.fit_predict(umap_2d)

    return df_ssa, umap_2d, labels_hdb, df_ssa.index.to_series().to_frame(name="year")


def build_clusters_df(
    years_index: pd.Index,
    labels: np.ndarray,
    annual_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble a cluster summary DataFrame with annual returns.
    """
    years = years_index.values
    df_clusters = pd.DataFrame(index=years)
    df_clusters["cluster"] = labels
    df_clusters["returns"] = annual_returns.loc[years, "returns"].values
    return df_clusters


def run_full_pipeline() -> pd.DataFrame:
    """
    Convenience function to run the full portfolio pipeline:
      1) Load SPX from local CSV
      2) Build cosine/norm features and annual returns
      3) Run SSA + UMAP + HDBSCAN clustering
      4) Validate regimes statistically
      5) Plot regime magnitude
      6) Run Hybrid Sniper backtest
    """
    spx = load_spx_from_csv()

    features, annual = build_cosine_norm_features(spx)
    df_ssa, umap_2d, labels, years_df = run_ssa_umap_hdbscan(features)
    years = years_df["year"].astype(int)

    df_clusters = build_clusters_df(years.index, labels, annual)
    tester = ClusterRegimeTester(df_clusters, cluster_col="cluster", value_col="returns")
    tester.pretty_print_bull_vs_rest()

    plotter = RegimeMagnitudePlotter()
    plotter.plot(df_clusters=df_clusters, umap_2d=umap_2d, spx=spx)

    print("\n=== Hybrid Sniper Backtest ===")
    results = run_hybrid_sniper(spx, df_clusters)
    print(results)

    return results


if __name__ == "__main__":
    run_full_pipeline()
