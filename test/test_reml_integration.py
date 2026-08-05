from pathlib import Path
import tempfile
import unittest

import numpy as np

from janusx.script import reml


class RemlIntegrationTests(unittest.TestCase):
    def test_no_grm_and_dense_grm_complete_with_first_column_ids(self):
        rows = [
            ("L1", "A", "B1", 10.0, 1.0),
            ("L1", "B", "B2", 11.0, 2.0),
            ("L2", "A", "B1", 12.0, 3.0),
            ("L2", "B", "B2", 13.0, 3.5),
            ("L3", "A", "B1", 14.0, 5.0),
            ("L3", "B", "B2", 15.0, 5.5),
            ("L4", "A", "B1", 16.0, 7.0),
            ("L4", "B", "B2", 17.0, 8.0),
        ]
        text = "\n".join(
            [
                "line\tloc\tblock\ttemperature\tPH",
                *(
                    f"{line}\t{loc}\t{block}\t{temperature}\t{trait}"
                    for line, loc, block, temperature, trait in rows
                ),
            ]
        ) + "\n"
        with tempfile.TemporaryDirectory(prefix="jx_reml_integration_") as tmp:
            root = Path(tmp)
            pheno = root / "pheno.tsv"
            pheno.write_text(text, encoding="utf-8")

            no_grm = root / "no_grm"
            reml.main([
                "-p", str(pheno), "-n", "PH", "-c", "loc", "-rc", "block",
                "-o", str(no_grm), "-t", "1", "-maxiter", "8",
            ])
            self.assertTrue(Path(f"{no_grm}.blue.txt").exists())
            self.assertTrue(Path(f"{no_grm}.blup.txt").exists())
            self.assertFalse(Path(f"{no_grm}.gblup.txt").exists())

            grm = root / "kinship.npy"
            grm_matrix = np.asarray([
                [1.0, 0.35, 0.10, 0.05],
                [0.35, 1.0, 0.25, 0.15],
                [0.10, 0.25, 1.0, 0.30],
                [0.05, 0.15, 0.30, 1.0],
            ])
            np.save(grm, grm_matrix)
            dense = root / "dense"
            reml.main([
                "-p", str(pheno), "-n", "PH", "-c", "loc", "-rc", "block",
                "-k", str(grm), "-o", str(dense), "-t", "1", "-maxiter", "12",
            ])
            self.assertTrue(Path(f"{dense}.gblup.txt").exists())
            log_text = Path(f"{dense}.reml.log").read_text(encoding="utf-8")
            self.assertIn("joint additive", log_text)
            self.assertIn("narrow(h2)", log_text)
            self.assertIn("stage1 BLUE uncertainty diag", log_text)
            summary = Path(f"{dense}.reml.summary.tsv").read_text(encoding="utf-8")
            self.assertIn("h2_narrow", summary)

    def test_gxe_and_gxc_terms_reach_observation_model(self):
        rows = []
        envs = [("A", 10.0), ("B", 15.0), ("C", 20.0)]
        for line_idx in range(1, 7):
            for env_idx, (loc, base_temperature) in enumerate(envs):
                for rep in range(2):
                    temperature = base_temperature + 0.25 * line_idx + 0.9 * rep + 0.13 * env_idx * rep
                    value = (
                        2.0 * line_idx
                        + 0.8 * env_idx
                        + 0.05 * temperature * (1.0 + 0.1 * line_idx)
                        + 0.15 * rep
                    )
                    rows.append((f"L{line_idx}", loc, temperature, value))
        text = "\n".join(
            [
                "line\tloc\ttemperature\tPH",
                *(f"{line}\t{loc}\t{temperature}\t{trait}" for line, loc, temperature, trait in rows),
            ]
        ) + "\n"
        with tempfile.TemporaryDirectory(prefix="jx_reml_gxe_gxc_") as tmp:
            root = Path(tmp)
            pheno = root / "pheno.tsv"
            pheno.write_text(text, encoding="utf-8")
            out = root / "result"
            reml.main([
                "-p", str(pheno), "-n", "PH", "-gxe", "loc", "-gxc", "temperature",
                "-o", str(out), "-t", "1", "-maxiter", "5",
            ])
            self.assertTrue(Path(f"{out}.blue.txt").exists())
            self.assertTrue(Path(f"{out}.blup.txt").exists())
            log_text = Path(f"{out}.reml.log").read_text(encoding="utf-8")
            self.assertIn("random variance [line×loc]", log_text)
            self.assertIn("random variance [line×temperature]", log_text)
            self.assertIn("residual variance", log_text)


if __name__ == "__main__":
    unittest.main()
