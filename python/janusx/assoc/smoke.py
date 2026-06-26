from __future__ import annotations

import numpy as np

from .api import ASSOC


def _assert_common_fit_state(model: ASSOC, *, route: str, kinship_kind: str) -> None:
    assert model.fitted_ is True
    assert model.route_ == route
    assert model.null_fit_ is not None
    assert model.fit_result_ is not None
    assert model.fit_result_["route"] == route
    assert model.fit_result_["kinship_kind"] == kinship_kind
    assert model.fit_result_["n_samples"] == 8
    assert model.fit_result_["n_covariates"] == 2
    assert model.fit_result_["threads"] >= 1


def run_assoc_api_smoke() -> None:
    toy = ASSOC.toy_data(seed=42)
    y = toy["y"]
    X = toy["X"]
    G = toy["G"]
    K = toy["K"]

    lm = ASSOC(model="glm", model_args={"threads": 1})
    lm.fit(y, X)
    lm_res = lm.assoc(G)
    _assert_common_fit_state(lm, route="lm", kinship_kind="none")
    assert lm_res.shape == (5, 3)
    assert list(lm_res.columns) == ["beta", "se", "pwald"]
    assert lm.fit_result_["backend"] == "LM"
    assert lm.null_fit_["lambda"] is None
    lm_chunk = lm.assoc(G, threads=2, chunk_size=2)
    assert np.allclose(lm_res.to_numpy(dtype=float), lm_chunk.to_numpy(dtype=float), equal_nan=True)

    lmm = ASSOC(model="lmm", model_args={"threads": 1})
    lmm.fit(y, X, K)
    lmm_res = lmm.assoc(G)
    _assert_common_fit_state(lmm, route="lmm", kinship_kind="dense")
    assert lmm_res.shape == (5, 3)
    assert lmm.fit_result_["backend"] == "LMM"
    assert lmm.null_fit_["lambda"] is not None
    lmm_chunk = lmm.assoc(G, threads=2, chunk_size=2)
    assert np.allclose(lmm_res.to_numpy(dtype=float), lmm_chunk.to_numpy(dtype=float), equal_nan=True)

    splmm = ASSOC(model="splmm", model_args={"threads": 1, "sparse_cutoff": 0.0})
    splmm.fit(y, X, K)
    splmm_res = splmm.assoc(G)
    _assert_common_fit_state(splmm, route="splmm", kinship_kind="sparse")
    assert splmm_res.shape == (5, 3)
    assert splmm.fit_result_["backend"] is not None
    assert splmm.fit_result_["sparse_grm_path"] is not None
    assert splmm.null_fit_["lambda"] is not None
    assert splmm.null_fit_["va"] is not None
    assert splmm.null_fit_["ve"] is not None
    splmm_chunk = splmm.assoc(G, threads=2, chunk_size=2)
    assert np.allclose(splmm_res.to_numpy(dtype=float), splmm_chunk.to_numpy(dtype=float), equal_nan=True)

    if toy["K_sparse"] is not None:
        auto_sparse = ASSOC(model="lmm", model_args={"threads": 1, "sparse_cutoff": 0.0})
        auto_sparse.fit(y, X, toy["K_sparse"])
        auto_res = auto_sparse.assoc(G.iloc[::-1])
        _assert_common_fit_state(auto_sparse, route="splmm", kinship_kind="sparse")
        assert auto_sparse.effective_model_ == "splmm"
        assert auto_res.shape == (5, 3)
        assert np.all(np.isfinite(auto_res.to_numpy(dtype=float)))


def main() -> None:
    run_assoc_api_smoke()


if __name__ == "__main__":
    main()
