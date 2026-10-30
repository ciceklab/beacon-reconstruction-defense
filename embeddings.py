import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

def fit_beacon_embeddings(beacon_matrix: np.ndarray, n_components: int=16):
    matrix = np.asarray(beacon_matrix, dtype=np.float64)
    n_individuals, n_snps = matrix.shape
    max_rank = min(n_individuals, n_snps) - 1
    if max_rank < 1:
        k = 1
    else:
        k = min(n_components, max_rank)
    svd = TruncatedSVD(n_components=k, random_state=0)
    svd.fit(matrix)
    human_embeddings = svd.transform(matrix)
    snp_embeddings = svd.components_.T
    snp_embeddings = StandardScaler().fit_transform(snp_embeddings)
    centroid = human_embeddings.mean(axis=0)
    return (snp_embeddings, human_embeddings, centroid)

def mia_vulnerability_score(snp_idx: int, beacon_matrix: np.ndarray, human_embeddings: np.ndarray) -> float:
    carriers = beacon_matrix[:, snp_idx] > 0
    if carriers.sum() == 0:
        return 0.0
    centroid = human_embeddings.mean(axis=0)
    dists = np.linalg.norm(human_embeddings[carriers] - centroid, axis=1)
    return float(dists.mean())
