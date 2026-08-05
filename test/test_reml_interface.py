import argparse
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from janusx.script import reml
from janusx.script._common.cli_args import add_common_trait_selector_args


class RemlCliContractTests(unittest.TestCase):
    def test_new_pheno_and_effect_flags_parse(self):
        args = reml.build_parser().parse_args([
            "-p", "pheno.tsv", "-n", "PH", "-c", "loc,year",
            "-rc", "block", "-gxe", "loc:year", "-gxc", "temperature",
            "-k", "kinship.npy",
        ])
        self.assertEqual(args.pheno, "pheno.tsv")
        self.assertEqual(args.ncol, ["PH"])
        self.assertEqual(args.cov, ["loc,year"])
        self.assertEqual(args.rcov, ["block"])
        self.assertEqual(args.gxe, ["loc:year"])
        self.assertEqual(args.gxc, ["temperature"])

    def test_legacy_reml_file_and_effect_flags_are_rejected(self):
        for flag in ("-file", "--file", "-f", "-r", "-e", "-l", "-grm", "--n"):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    reml.build_parser().parse_args([flag, "x"])

    def test_grm_inputs_are_optional_but_mutually_exclusive(self):
        args = reml.build_parser().parse_args(["-p", "p.tsv", "-n", "PH"])
        self.assertIsNone(args.grm)
        self.assertIsNone(args.grm_sparse)
        with self.assertRaises(SystemExit):
            reml.build_parser().parse_args([
                "-p", "p.tsv", "-n", "PH", "-k", "a.npy", "-spk", "a.spgrm"
            ])

    def test_shared_trait_selector_exposes_ncol_only(self):
        parser = argparse.ArgumentParser()
        add_common_trait_selector_args(parser)
        self.assertEqual(parser.parse_args(["-n", "0"]).ncol, ["0"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--n", "0"])


class RemlEffectCompilerTests(unittest.TestCase):
    def test_numeric_type_inference_logs_rule_metadata(self):
        low_card = pd.Series([1, 1, 2, 1] * 10)
        continuous = pd.Series(np.linspace(0.1, 4.0, 40))
        categorical = pd.Series(["A", "B", "A", "C"] * 10)
        self.assertEqual(reml._infer_column_type(low_card), "categorical")
        self.assertEqual(reml._infer_column_type(continuous), "continuous")
        self.assertEqual(reml._infer_column_type(categorical), "categorical")
        details = reml._infer_column_type_details(low_card)
        self.assertEqual(details["valid_count"], 40)
        self.assertEqual(details["unique_count"], 2)
        self.assertIn("low_cardinality", details["reason"])

    def test_categorical_pair_compiles_to_combined_factor(self):
        df = pd.DataFrame({"loc": ["A", "A", "B"], "year": ["Y1", "Y2", "Y1"]})
        spec = reml._parse_effect_specs(
            ["loc:year"], "fixed", list(df.columns), df
        )[0]
        matrix, names = reml._compile_effect_matrix(df, spec, for_random=False)
        self.assertEqual(spec.source_types, ("categorical", "categorical"))
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(len(names), 2)

    def test_numeric_product_and_categorical_slopes(self):
        df = pd.DataFrame({
            "dose": [1.0, 2.0, 3.0],
            "temp": [2.0, 4.0, 5.0],
            "treatment": ["A", "B", "A"],
        })
        product = reml._parse_effect_specs(
            ["dose:temp"], "fixed", list(df.columns), df
        )[0]
        product_matrix, _ = reml._compile_effect_matrix(
            df, product, for_random=False
        )
        np.testing.assert_allclose(product_matrix.ravel(), [2.0, 8.0, 15.0])

        slope = reml._parse_effect_specs(
            ["treatment:temp"], "random", list(df.columns), df
        )[0]
        slope_matrix, slope_names = reml._compile_effect_matrix(
            df, slope, for_random=True
        )
        self.assertEqual(slope_matrix.shape, (3, 2))
        self.assertEqual(len(slope_names), 2)

    def test_gxe_requires_categorical_and_gxc_requires_continuous(self):
        with self.assertRaisesRegex(ValueError, "categorical"):
            reml._parse_effect_specs(
                ["temp"],
                "gxe",
                ["temp"],
                pd.DataFrame({"temp": [1.0, 2.0]}),
            )

    def test_gxe_comma_list_is_independent_and_gxe_colon_is_one_term(self):
        df = pd.DataFrame({
            "loc": ["A", "B", "A", "B"],
            "year": ["Y1", "Y1", "Y2", "Y2"],
        })
        independent = reml._parse_effect_specs(
            ["loc,year"], "gxe", list(df.columns), df
        )
        combined = reml._parse_effect_specs(
            ["loc:year"], "gxe", list(df.columns), df
        )
        self.assertEqual([spec.label for spec in independent], ["loc", "year"])
        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0].result_type, "categorical")

    def test_gxc_centers_without_rescaling(self):
        df = pd.DataFrame({
            "line": ["L1", "L1", "L2", "L2"],
            "temperature": [10.0, 12.0, 14.0, 16.0],
        })
        block, _names = reml._compile_line_slope_matrix(
            df, line_col="line", column="temperature"
        )
        values = np.asarray(block.toarray(), dtype=float)
        self.assertAlmostEqual(float(values.sum()), 0.0)
        np.testing.assert_allclose(values[:, 0], [-3.0, -1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(values.max() - values.min()), 6.0)
        with self.assertRaisesRegex(ValueError, "continuous"):
            reml._parse_effect_specs(
                ["loc"],
                "gxc",
                ["loc"],
                pd.DataFrame({"loc": ["A", "B"]}),
            )


class RemlDesignPropagationTests(unittest.TestCase):
    def test_legacy_stage1_term_helper_keeps_within_line_fixed_terms(self):
        sub = pd.DataFrame({
            "line": ["L1", "L1", "L2", "L2"],
            "loc": ["A", "B", "A", "B"],
        })
        fixed = [reml._TermSpec(name="loc", force_onehot=True)]
        random = [reml._TermSpec(name="block", force_onehot=True)]
        _random_out, fixed_out = reml._build_stage1_blue_terms(
            sub,
            line_col="line",
            trait="yield",
            fixed_terms_all=fixed,
            random_terms_all=random,
            logger=logging.getLogger("test"),
        )
        self.assertEqual([term.name for term in fixed_out], ["loc"])

    def test_varying_fixed_covariate_reaches_blue_stage(self):
        sub = pd.DataFrame({
            "line": ["L1", "L1", "L2", "L2"],
            "loc": ["A", "B", "A", "B"],
            "yield": [1.0, 4.0, 2.0, 5.0],
        })
        fixed = reml._parse_effect_specs(
            ["loc"], "fixed", list(sub.columns), sub
        )
        compiled = reml._compile_model_terms(
            sub,
            line_col="line",
            fixed_specs=fixed,
            random_specs=[],
            gxe_specs=[],
            gxc_specs=[],
        )
        self.assertIn("loc", compiled.fixed_labels)
        captured = {}

        class _FakeBlueModel:
            def __init__(self, x):
                captured["X"] = np.asarray(x, dtype=float)
                n_beta = int(self.X.shape[1] + 1)
                self.beta = np.zeros(n_beta, dtype=float)
                self._cov_beta = np.eye(n_beta, dtype=float)

            @property
            def X(self):
                return captured["X"]

        with patch.object(reml, "BLUP", side_effect=lambda y, X, Z, maxiter, progress: _FakeBlueModel(X)):
            reml._fit_stage1_blue(
                y_obs=sub["yield"].to_numpy(),
                sub=sub,
                line_col="line",
                trait="yield",
                compiled=compiled,
                maxiter=20,
                logger=logging.getLogger("test"),
            )
        # One compiled loc column plus one treatment-coded line column.  If
        # the old within-line filter is accidentally restored, only the line
        # column would reach this BLUE design.
        self.assertEqual(captured["X"].shape[1], 2)


class RemlOutputContractTests(unittest.TestCase):
    def test_no_grm_writes_blue_blup_and_raw_variances_to_log(self):
        table = "\n".join([
            "line\tloc\tblock\tPH",
            "L1\tA\tB1\t1.0",
            "L1\tB\tB2\t2.0",
            "L2\tA\tB1\t3.0",
            "L2\tB\tB2\t4.0",
            "L3\tA\tB1\t5.0",
            "L3\tB\tB2\t6.0",
        ])
        with tempfile.TemporaryDirectory(prefix="jx_reml_test_") as tmp:
            root = Path(tmp)
            pheno = root / "pheno.tsv"
            pheno.write_text(table + "\n", encoding="utf-8")
            out = root / "result"
            reml.main([
                "-p", str(pheno), "-n", "PH", "-c", "loc", "-rc", "block",
                "-o", str(out), "-t", "1", "-maxiter", "5",
            ])
            self.assertTrue(Path(f"{out}.blue.txt").exists())
            self.assertTrue(Path(f"{out}.blup.txt").exists())
            self.assertFalse(Path(f"{out}.gblup.txt").exists())
            log_text = Path(f"{out}.reml.log").read_text(encoding="utf-8")
            self.assertIn("variance", log_text.lower())
            self.assertIn("residual", log_text.lower())
            self.assertNotIn("narrow(h2)", log_text)


class RemlNarrowCorrectionTests(unittest.TestCase):
    def test_dense_route_passes_stage1_uncertainty_to_joint_fit(self):
        captured = {}

        def fake_joint(y_line, *, kinship, noise_diag, x_fixed, maxiter):
            captured["noise_diag"] = np.asarray(noise_diag, dtype=float).copy()
            return reml._JointKernelResult(
                va=1.0,
                vline=1.0,
                h2_raw=0.25,
                beta=np.asarray([0.0]),
                add_blup=np.zeros(3),
                line_blup=np.zeros(3),
                noise_mean=float(np.mean(noise_diag)),
            )

        with patch.object(reml, "_fit_joint_line_kernel_exact", side_effect=fake_joint):
            state = reml._fit_dense_narrow_corrected(
                np.asarray([1.0, 2.0, 3.0]),
                kinship=np.eye(3),
                noise_diag=np.asarray([0.5, 1.0, 2.0]),
                x_fixed=None,
                maxiter=10,
            )
        np.testing.assert_allclose(captured["noise_diag"], [0.5, 1.0, 2.0])
        self.assertEqual(state.noise_mean, 1.1666666666666667)

    def test_more_stage1_uncertainty_lowers_phenotype_scale_h2(self):
        kinship = np.asarray([
            [1.0, 0.4, 0.2, 0.1],
            [0.4, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.5],
            [0.1, 0.2, 0.5, 1.0],
        ])
        y = np.asarray([0.0, 1.0, 2.0, 3.0])
        low = reml._fit_joint_line_kernel_exact(
            y, kinship=kinship, noise_diag=np.zeros(4), x_fixed=None, maxiter=20
        )
        high = reml._fit_joint_line_kernel_exact(
            y, kinship=kinship, noise_diag=np.ones(4) * 10.0, x_fixed=None, maxiter=20
        )
        self.assertLess(high.h2_raw, low.h2_raw)

    def test_sparse_route_adds_stage1_uncertainty_to_reported_pve(self):
        backend_result = {
            "pve": 0.8,
            "pve_pheno_scale": 0.8,
            "pve_vc_ratio_raw": 0.9,
            "sigma_g2": 4.0,
            "sigma_e2": 1.0,
            "mean_diag_k": 1.0,
        }
        with patch.object(
            reml,
            "_splmm_sparse_null_fit",
            return_value=dict(backend_result),
        ):
            corrected = reml._fit_sparse_narrow_corrected(
                jxgrm_path="fixture.spgrm",
                sample_idx=np.asarray([0, 1, 2]),
                y_vec=np.asarray([1.0, 2.0, 3.0]),
                x_cov=None,
                noise_diag=np.asarray([1.0, 2.0, 3.0]),
                objective_mode="raw",
                threads=1,
            )
        self.assertAlmostEqual(corrected["stage1_noise_mean"], 2.0)
        self.assertAlmostEqual(corrected["pve"], 4.0 / 7.0)
        self.assertEqual(corrected["pve_vc_ratio_raw"], 0.9)


class RemlDeveloperHelpTests(unittest.TestCase):
    def test_spk_mode_is_hidden_unless_dev_help_is_requested(self):
        normal_help = reml.build_parser([]).format_help()
        dev_help = reml.build_parser(["-dev"]).format_help()
        self.assertNotIn("--spk-mode", normal_help)
        self.assertIn("--spk-mode", dev_help)


if __name__ == "__main__":
    unittest.main()
