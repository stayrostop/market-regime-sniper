import numpy as np
import pandas as pd
import umap.umap_ as umap
import hdbscan
from sklearn.metrics import silhouette_score


class SSAUMAPRegimeClusterer:
    """
    Pipeline:
      1) Παίρνει features ανά έτος (rows=years, cols=p_cos_* / p_norm_*)
      2) Κάνει SSA ξεχωριστά σε cos και norm με διαφορετικό L
      3) Τρέχει UMAP -> HDBSCAN
      4) Κάνει ελαφρύ hyperparameter tuning πάνω σε UMAP + HDBSCAN
    """
    def __init__(
        self,
        L_cos=60,
        L_norm=40,
        max_len=None,
        random_state=42,
    ):
        """
        L_cos   : window length SSA για cos features
        L_norm  : window length SSA για norm features
        max_len : αν θες να κόβεις τις σειρές σε fixed length (π.χ. 231).
                  Αν None, κρατάει όλα τα columns όπως είναι.
        """
        self.L_cos = L_cos
        self.L_norm = L_norm
        self.max_len = max_len
        self.random_state = random_state

        # Θα γεμίζουν μετά το fit
        self.cos_cols_ = None
        self.norm_cols_ = None
        self.features_ = None
        self.df_ssa_ = None
        self.umap_2d_ = None
        self.labels_ = None
        self.best_params_ = None
        self.tuning_results_ = None

    # =========================
    # 1. SSA βασική συνάρτηση
    # =========================
    @staticmethod
    def ssa_decompose(series, L=10):
        """
        Singular Spectrum Analysis decomposition για 1D χρονοσειρά.
        Επιστρέφει: list με components (κάθε component είναι 1D array μήκους N).
        """
        x = np.asarray(series, dtype=float)
        N = len(x)
        if L < 2 or L > N:
            raise ValueError(f"L must be in [2, N], got L={L}, N={N}")

        K = N - L + 1
        # Trajectory matrix (L x K)
        X = np.column_stack([x[i:i+L] for i in range(K)])  # shape (L, K)

        # SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        d = len(s)

        components = []

        # Reconstruct + diagonal averaging
        for i in range(d):
            Xi = s[i] * np.outer(U[:, i], Vt[i, :])  # (L, K)
            recon = np.zeros(N)
            counts = np.zeros(N)

            for row in range(L):
                for col in range(K):
                    t = row + col          # 0 .. N-1
                    recon[t] += Xi[row, col]
                    counts[t] += 1

            components.append(recon / counts)

        return components

    # 2. Προετοιμασία feature matrix (cos/norm)
    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        - Εντοπίζει p_cos_*, p_norm_*
        - Κόβει σε κοινό max_len αν χρειάζεται
        - Επιστρέφει "καθαρό" features DF
        """
        df = features.copy()

        cos_cols = [c for c in df.columns if c.startswith("p_cos_")]
        norm_cols = [c for c in df.columns if c.startswith("p_norm_")]

        if len(cos_cols) == 0 or len(norm_cols) == 0:
            raise ValueError("Δεν βρέθηκαν p_cos_* και p_norm_* columns στο features DataFrame.")

        # ✅ Sort by numeric suffix στο τέλος του ονόματος (π.χ. "..._123")
        def sort_by_index(col_list):
            def extract_idx(c):
                # Παίρνουμε το τελευταίο κομμάτι μετά το '_'
                # π.χ. "p_cos_cos_12" -> "12"
                last = c.rsplit("_", 1)[-1]
                try:
                    return int(last)
                except ValueError:
                    # fallback για περίεργα ονόματα
                    return 10**9
            return sorted(col_list, key=extract_idx)

        cos_cols = sort_by_index(cos_cols)
        norm_cols = sort_by_index(norm_cols)

        if self.max_len is not None:
            max_len = min(len(cos_cols), len(norm_cols), self.max_len)
            cos_cols = cos_cols[:max_len]
            norm_cols = norm_cols[:max_len]

        cols_keep = cos_cols + norm_cols
        df = df[cols_keep].dropna()

        self.cos_cols_ = cos_cols
        self.norm_cols_ = norm_cols
        self.features_ = df

        return df

    # ===========================================
    # 3. SSA ανά έτος, με L_cos / L_norm
    # ===========================================
    def build_ssa_matrix(self) -> pd.DataFrame:
        """
        Τρέχει SSA για όλα τα χρόνια (rows) και για cos/norm ξεχωριστά.
        Χρησιμοποιεί self.features_, self.cos_cols_, self.norm_cols_.
        Επιστρέφει df_ssa (ίδιο σχήμα με features).
        """
        if self.features_ is None:
            raise RuntimeError("Πρέπει να καλέσεις πρώτα _prepare_features ή fit().")

        df = self.features_
        cos_cols = self.cos_cols_
        norm_cols = self.norm_cols_

        df_ssa = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

        for yr in df.index:
            row = df.loc[yr]

            # COS
            x_cos = row[cos_cols].values.astype(float)
            comps_cos = self.ssa_decompose(x_cos, L=self.L_cos)
            df_ssa.loc[yr, cos_cols] = comps_cos[0]

            # NORM
            x_norm = row[norm_cols].values.astype(float)
            comps_norm = self.ssa_decompose(x_norm, L=self.L_norm)
            df_ssa.loc[yr, norm_cols] = comps_norm[0]

        self.df_ssa_ = df_ssa
        return df_ssa

    # 4. Lightweight Hyperparameter Tuning
    def tune_umap_hdbscan(
        self,
        umap_neighbors_list=None,
        umap_mindist_list=None,
        hdb_min_cluster_sizes=None,
        hdb_min_samples_mode="paired",
    ):
        """

        - umap_neighbors_list: λίστα με n_neighbors (default: [3, 6, 10])
        - umap_mindist_list  : λίστα με min_dist (default: [0.0, 0.0001])
        - hdb_min_cluster_sizes: λίστα (default: [5, 10, 15])
        - hdb_min_samples_mode:
              "paired" -> [min_cluster, max(2, min_cluster//2)]
        Αποθηκεύει:
            - self.umap_2d_
            - self.labels_
            - self.best_params_
            - self.tuning_results_
        """
        if self.df_ssa_ is None:
            raise RuntimeError("Πρέπει να έχεις χτίσει df_ssa_ (κάλεσε build_ssa_matrix ή fit).")

        X = self.df_ssa_.values

        if umap_neighbors_list is None:
            umap_neighbors_list = [3, 6, 10]

        if umap_mindist_list is None:
            umap_mindist_list = [0.0, 0.0001]

        if hdb_min_cluster_sizes is None:
            hdb_min_cluster_sizes = [5, 10, 15]

        results = []
        best_score = -np.inf
        best_conf = None
        best_labels = None
        best_umap_emb = None

        for n_neighbors in umap_neighbors_list:
            for min_dist in umap_mindist_list:
                um_model = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric="euclidean",
                    n_components=2,
                    random_state=self.random_state,
                )
                umap_emb = um_model.fit_transform(X)

                for min_cluster in hdb_min_cluster_sizes:
                    if hdb_min_samples_mode == "paired":
                        candidates_min_samples = [min_cluster, max(2, min_cluster // 2)]
                    else:
                        candidates_min_samples = [min_cluster]

                    for min_samples in candidates_min_samples:
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=min_cluster,
                            min_samples=min_samples,
                            cluster_selection_epsilon=0.0,
                        )
                        labels = clusterer.fit_predict(umap_emb)

                        unique_lbls = set(labels)
                        n_clusters = len(unique_lbls) - (1 if -1 in unique_lbls else 0)
                        noise_ratio = (labels == -1).sum() / len(labels)

                        if n_clusters < 2:
                            score = -1.0
                        else:
                            try:
                                score = silhouette_score(umap_emb, labels)
                            except Exception:
                                score = -1.0

                        results.append({
                            "umap_n_neighbors": n_neighbors,
                            "umap_min_dist": min_dist,
                            "hdb_min_cluster": min_cluster,
                            "hdb_min_samples": min_samples,
                            "n_clusters": n_clusters,
                            "noise_ratio": round(noise_ratio, 3),
                            "silhouette": round(score, 4),
                        })

                        # Βάζουμε προτεραιότητα στα:
                        # - καλό silhouette
                        # - όχι τεράστιο noise
                        if score > best_score and noise_ratio < 0.5 and n_clusters >= 2:
                            best_score = score
                            best_conf = {
                                "n_neighbors": n_neighbors,
                                "min_dist": min_dist,
                                "min_cluster_size": min_cluster,
                                "min_samples": min_samples,
                            }
                            best_labels = labels
                            best_umap_emb = umap_emb

        self.tuning_results_ = pd.DataFrame(results).sort_values(
            by="silhouette", ascending=False
        )

        self.best_params_ = best_conf
        self.labels_ = best_labels
        self.umap_2d_ = best_umap_emb

        return self.tuning_results_

    # 5. High-level fit
    def fit(
        self,
        features: pd.DataFrame,
        do_tuning=True,
        **tuning_kwargs,
    ):
        """
        High-level:
          1) προετοιμασία features (cos/norm, max_len)
          2) SSA
          3) optional: hyperparam tuning
        """
        self._prepare_features(features)
        self.build_ssa_matrix()

        if do_tuning:
            self.tune_umap_hdbscan(**tuning_kwargs)
        else:
            # Αν δεν κάνουμε tuning, ένα default run:
            um_model = umap.UMAP(
                n_neighbors=6,
                min_dist=0.0001,
                metric="euclidean",
                n_components=2,
                random_state=self.random_state,
            )
            self.umap_2d_ = um_model.fit_transform(self.df_ssa_.values)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=10,
                min_samples=10,
                cluster_selection_epsilon=0.0,
            )
            self.labels_ = clusterer.fit_predict(self.umap_2d_)
            self.best_params_ = {
                "n_neighbors": 6,
                "min_dist": 0.0001,
                "min_cluster_size": 10,
                "min_samples": 10,
            }

        return self

    # 6. Helper: clusters per year
    def get_year_cluster_df(self) -> pd.DataFrame:
        """
        Επιστρέφει DataFrame:
            index = years (όπως features.index)
            columns = ['cluster']
        """
        if self.features_ is None or self.labels_ is None:
            raise RuntimeError("Πρέπει πρώτα να καλέσεις fit().")

        years = self.features_.index.values
        return pd.DataFrame({"cluster": self.labels_}, index=years)


def ssa_decompose(series, L=10):
    """
    Convenience wrapper to reuse the SSA implementation outside the class.
    """
    return SSAUMAPRegimeClusterer.ssa_decompose(series, L=L)
