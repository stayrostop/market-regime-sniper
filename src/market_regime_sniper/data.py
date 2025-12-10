from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_spx_from_csv(path: str | Path = "spx.csv") -> pd.DataFrame:
    """
    Load daily S&P 500 prices from a local CSV (offline-friendly).

    CSV columns: Date,Open,High,Low,Close,Volume
    """
    data_path = Path(path)
    fallback = Path(__file__).resolve().parent / "spx.csv"

    if not data_path.exists() and fallback.exists():
        data_path = fallback

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find spx.csv at '{path}' or '{fallback}'.")

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.loc["1950-01-01":"2025-12-31"].copy()
    return df


def create_yearly_vectors(df: pd.DataFrame, col_name: str = "Close_norm") -> dict[int, np.ndarray]:
    """
    For each year build 2D vectors [x_i, x_{i+1}] from a normalized column.
    Used to derive cosine similarity and norm curves per year.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    yearly_vectors: dict[int, np.ndarray] = {}
    for year, group in df.groupby(df.index.year):
        series = group[col_name].dropna().values
        if len(series) < 3:
            continue
        vectors = np.column_stack((series[:-1], series[1:]))
        yearly_vectors[year] = vectors
    return yearly_vectors


def calculate_cosine_and_norm_df(yearly_vectors: dict[int, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    From yearly [x_i, x_{i+1}] vectors build:
      - cos_df  : cosine similarities per year (columns cos_1, cos_2, ...)
      - norm_df : vector norms per year (columns norm_1, norm_2, ...)
    """
    cos_results: dict[int, np.ndarray] = {}
    norm_results: dict[int, np.ndarray] = {}

    for year, vectors in yearly_vectors.items():
        if len(vectors) < 2:
            continue

        A = vectors[:-1]
        B = vectors[1:]

        dot_product = np.sum(A * B, axis=1)
        norm_a = np.linalg.norm(A, axis=1)
        norm_b = np.linalg.norm(B, axis=1)

        cos = dot_product / (norm_a * norm_b + 1e-10)
        norm_vec = norm_a

        cos_results[year] = cos
        norm_results[year] = norm_vec

    cos_df = pd.DataFrame.from_dict(cos_results, orient="index")
    norm_df = pd.DataFrame.from_dict(norm_results, orient="index")

    cos_df.columns = [f"cos_{i+1}" for i in range(cos_df.shape[1])]
    norm_df.columns = [f"norm_{i+1}" for i in range(norm_df.shape[1])]

    cos_df.sort_index(inplace=True)
    norm_df = norm_df.reindex(cos_df.index)
    return cos_df, norm_df


def build_cosine_norm_features(spx: pd.DataFrame, max_len: int = 231) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    High-level feature construction:
      1) Per-year normalization of Close
      2) Yearly cosine & norm vectors
      3) Concatenate into [p_cos_* | p_norm_*] feature matrix
      4) Annual returns for each year
    """
    df_norm = spx.copy()
    df_norm["year"] = df_norm.index.year
    df_norm["Close_norm"] = df_norm["Close"] / df_norm.groupby("year")["Close"].transform("first")

    vectors_price = create_yearly_vectors(df_norm, col_name="Close_norm")
    cos_p, norm_p = calculate_cosine_and_norm_df(vectors_price)

    common_years = cos_p.index
    cos_p = cos_p.loc[common_years]
    norm_p = norm_p.loc[common_years]

    max_len_eff = min(cos_p.shape[1], norm_p.shape[1], max_len)
    cos_p_fixed = cos_p.iloc[:, :max_len_eff]
    norm_p_fixed = norm_p.iloc[:, :max_len_eff]

    features = pd.concat(
        [
            cos_p_fixed.add_prefix("p_cos_"),
            norm_p_fixed.add_prefix("p_norm_"),
        ],
        axis=1,
    ).dropna()

    annual = spx["Close"].resample("YE").agg(["first", "last"])
    annual.index = annual.index.year
    annual["returns"] = (annual["last"] / annual["first"] - 1.0) * 100.0
    annual = annual.loc[features.index]

    return features, annual
