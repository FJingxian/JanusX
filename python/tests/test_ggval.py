from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from janusx import ggval


def _touch(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_args_accepts_common_cli_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ggval",
            "--mode",
            "smoke",
            "--outdir",
            "validation",
            "--logdir",
            "validation/logs",
            "--only",
            "gwas,gs-file",
            "--skip",
            "gwas",
            "--threads",
            "4",
            "--cv",
            "3",
            "--no-postgs",
            "--no-backend-thread-checks",
        ],
    )

    args = ggval.parse_args()

    assert args.mode == "smoke"
    assert args.outdir == "validation"
    assert args.logdir == "validation/logs"
    assert args.only == "gwas,gs-file"
    assert args.skip == "gwas"
    assert args.threads == 4
    assert args.cv == 3
    assert args.no_postgs is True
    assert args.no_backend_thread_checks is True


@pytest.mark.parametrize(
    ("mode", "only", "skip", "expected"),
    [
        ("smoke", None, None, ["gwas", "gs-file"]),
        (
            "full",
            None,
            "bench",
            ["gwas", "gs-file", "gs-bfile", "gs-vcf", "gs-hmp", "gs-ml", "reml"],
        ),
        ("smoke", "bench,gwas,gs-file", None, ["gwas", "gs-file", "bench"]),
        ("smoke", "gwas gs-file gwas", None, ["gwas", "gs-file"]),
        ("full", "gs-bfile", "gs-file", ["gs-bfile"]),
    ],
)
def test_resolve_requested_suites(
    mode: str,
    only: str | None,
    skip: str | None,
    expected: list[str],
) -> None:
    assert ggval.resolve_requested_suites(mode, only, skip) == expected


def test_resolve_requested_suites_rejects_unknown_or_empty_selection() -> None:
    with pytest.raises(SystemExit):
        ggval.resolve_requested_suites("smoke", "gwas,unknown", None)
    with pytest.raises(SystemExit):
        ggval.resolve_requested_suites("smoke", "gwas", "gwas")


def test_environment_defaults_feed_smoke_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JX_GGVAL_SMOKE_NSNP_K", "7")
    monkeypatch.setenv("JX_GGVAL_SMOKE_NIND", "123")
    monkeypatch.setenv("JX_GGVAL_SIM_SEED", "99")

    assert ggval._resolve_sim_config("smoke") == (7, 123, 99)


def test_run_uses_single_thread_plotting_defaults_and_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen_env: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        seen_env.update(kwargs["env"])
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(ggval.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        ggval.run(
            ["jx", "postgwas", "-gwasfile", "result.tsv"],
            stdout_path=tmp_path / "postgwas.log",
        )

    assert seen_env["OMP_NUM_THREADS"] == "1"
    assert seen_env["OPENBLAS_NUM_THREADS"] == "1"
    assert seen_env["RAYON_NUM_THREADS"] == "1"


def test_build_validation_dataset_builds_common_cli_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(list(cmd))
        if cmd[1] == "sim":
            prefix = Path(cmd[4])
            for suffix in (
                ".bed",
                ".bim",
                ".fam",
                ".pheno",
                ".pheno.txt",
                ".pheno.NA.txt",
            ):
                _touch(Path(f"{prefix}{suffix}"))
            return subprocess.CompletedProcess(cmd, 0)

        if cmd[1] != "gformat":
            raise AssertionError(f"Unexpected command: {cmd}")
        fmt = cmd[cmd.index("-fmt") + 1]
        outdir = Path(cmd[cmd.index("-o") + 1])
        prefix_name = cmd[cmd.index("-prefix") + 1]
        prefix = outdir / prefix_name
        if fmt == "txt":
            for suffix in (".txt", ".id", ".site"):
                _touch(Path(f"{prefix}{suffix}"))
        elif fmt == "plink":
            for suffix in (".bed", ".bim", ".fam"):
                _touch(Path(f"{prefix}{suffix}"))
        elif fmt == "hmp":
            _touch(prefix.with_suffix(".hmp"))
        elif fmt == "vcf":
            _touch(Path(f"{prefix}.vcf.gz"))
        else:
            raise AssertionError(f"Unexpected gformat target: {fmt}")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ggval, "run", fake_run)

    paths = ggval.build_validation_dataset(
        tmp_path,
        nsnp_k=2,
        n_individuals=80,
        seed=20260707,
        export_debug_formats=True,
    )

    assert [cmd[1] for cmd in commands] == [
        "sim",
        "gformat",
        "gformat",
        "gformat",
        "gformat",
    ]
    assert commands[1][commands[1].index("-fmt") + 1] == "txt"
    assert commands[2][commands[2].index("-fmt") + 1] == "plink"
    assert commands[3][commands[3].index("-fmt") + 1] == "hmp"
    assert commands[4][commands[4].index("-fmt") + 1] == "vcf"
    assert paths["txt_matrix"].is_file()
    assert Path(f"{paths['roundtrip_prefix']}.bed").is_file()
    assert paths["hmp_prefix"].with_suffix(".hmp").is_file()
    assert Path(f"{paths['vcf_prefix']}.vcf.gz").is_file()


def test_gs_file_input_grm_cache_uses_current_plink_bed_name(tmp_path: Path) -> None:
    cache = _touch(
        tmp_path / "~ggval_sim_txt.bed.maf0.02.geno0.05.snp0.cGRM.npy"
    )

    assert ggval._gs_file_input_grm_cache_candidates(
        tmp_path,
        prefix_name=ggval.SIM_TXT_PREFIX_NAME,
    ) == [cache]


def test_validate_gs_file_input_accepts_current_cache_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefix = ggval.SIM_TXT_PREFIX_NAME
    summary = _touch(tmp_path / "summary.json", "{}")
    _touch(tmp_path / f"~{prefix}.snp0.bed")
    _touch(tmp_path / f"~{prefix}.bed.maf0.02.geno0.05.snp0.cGRM.npy")

    monkeypatch.setattr(
        ggval,
        "validate_gs_output_basics",
        lambda *args, **kwargs: summary,
    )
    monkeypatch.setattr(
        ggval,
        "validate_text_effect_artifact",
        lambda *args, **kwargs: None,
    )

    assert (
        ggval.validate_gs_file_input_debug_outputs(
            tmp_path,
            prefix_name=prefix,
            trait_name=ggval.TRAIT_NAME,
            expected_models=["BLUP"],
        )
        == summary
    )


def test_validate_gs_file_input_rejects_legacy_cache_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefix = ggval.SIM_TXT_PREFIX_NAME
    _touch(tmp_path / f"~{prefix}.snp0.bed")
    _touch(tmp_path / f"~{prefix}.maf0.02.geno0.05.snp0.cGRM.npy")

    monkeypatch.setattr(
        ggval,
        "validate_gs_output_basics",
        lambda *args, **kwargs: tmp_path / "summary.json",
    )
    monkeypatch.setattr(
        ggval,
        "validate_text_effect_artifact",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(SystemExit):
        ggval.validate_gs_file_input_debug_outputs(
            tmp_path,
            prefix_name=prefix,
            trait_name=ggval.TRAIT_NAME,
            expected_models=["BLUP"],
        )


def test_run_gwas_suite_emits_common_pca_gwas_and_postgwas_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pheno = _touch(tmp_path / "ggval_sim.pheno.txt")
    sim_prefix = tmp_path / "ggval_sim"
    _touch(Path(f"{sim_prefix}.cGRM.npy"))
    _touch(Path(f"{sim_prefix}.eigenvec"))
    gwas_file = _touch(tmp_path / "ggval_sim.test.lm.tsv")
    commands: list[list[str]] = []
    pca_commands: list[list[str]] = []

    def record_run(cmd, **kwargs):
        commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    def record_pca(cmd, **kwargs):
        pca_commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ggval, "run", record_run)
    monkeypatch.setattr(ggval, "run_pca_with_backend_report", record_pca)
    monkeypatch.setattr(
        ggval,
        "find_gwas_tsv_files",
        lambda outdir, prefix_name: [gwas_file],
    )
    monkeypatch.setattr(
        ggval,
        "require_any_file",
        lambda message, candidates: next(iter(candidates), tmp_path / "postgwas.pdf"),
    )

    ggval.run_gwas_suite(
        {"sim_prefix": sim_prefix, "pheno_txt": pheno},
        tmp_path,
        tmp_path / "logs",
        threads=3,
    )

    assert [cmd[1] for cmd in pca_commands] == ["pca"]
    assert [cmd[1] for cmd in commands] == [
        "grm",
        "grm",
        "pca",
        "gwas",
        "gwas",
        "postgwas",
    ]
    gwas_commands = [cmd for cmd in commands if cmd[1] == "gwas"]
    assert all("-farmcpu" in cmd for cmd in gwas_commands)
    assert all("-splmm-exact" in cmd for cmd in gwas_commands)
    assert "-force-model" not in gwas_commands[0]
    assert "-force-model" in gwas_commands[1]
    postgwas = commands[-1]
    assert postgwas[:3] == ["jx", "postgwas", "-gwasfile"]
    assert str(gwas_file) in postgwas


@pytest.mark.parametrize("postgs_enabled", [False, True])
def test_run_gs_file_suite_emits_file_input_and_optional_postgs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    postgs_enabled: bool,
) -> None:
    txt_matrix = _touch(tmp_path / "ggval_sim_txt.txt")
    pheno = _touch(tmp_path / "ggval_sim.pheno.txt")
    summary = _touch(tmp_path / "summary.json", "{}")
    commands: list[list[str]] = []

    def record_run(cmd, **kwargs):
        commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ggval, "run", record_run)
    monkeypatch.setattr(
        ggval,
        "validate_gs_file_input_debug_outputs",
        lambda *args, **kwargs: summary,
    )
    monkeypatch.setattr(
        ggval,
        "validate_postgs_outputs",
        lambda *args, **kwargs: None,
    )

    ggval.run_gs_file_suite(
        {"txt_matrix": txt_matrix, "pheno_txt": pheno},
        tmp_path,
        threads=2,
        cv_folds=3,
        postgs_enabled=postgs_enabled,
    )

    assert commands[0][:3] == ["jx", "gs", "-file"]
    assert str(txt_matrix) in commands[0]
    assert "-BayesC" in commands[0]
    assert "-save-model" in commands[0]
    if postgs_enabled:
        assert commands[1] == [
            "jx",
            "postgs",
            "-json",
            str(summary),
            "-o",
            str(tmp_path / ggval.SIM_TXT_PREFIX_NAME),
        ]
    else:
        assert len(commands) == 1


def test_run_gs_bfile_and_reml_suites_emit_common_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pheno = _touch(tmp_path / "ggval_sim.pheno.txt")
    sim_prefix = tmp_path / "ggval_sim"
    _touch(Path(f"{sim_prefix}.bim"), "1 rs1 0 10 A G\n")
    _touch(Path(f"{sim_prefix}.cGRM.npy"))
    summary = _touch(tmp_path / "summary.json", "{}")
    commands: list[list[str]] = []

    def record_run(cmd, **kwargs):
        commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ggval, "run", record_run)
    monkeypatch.setattr(
        ggval,
        "validate_gs_output_basics",
        lambda *args, **kwargs: summary,
    )
    monkeypatch.setattr(
        ggval,
        "validate_text_effect_artifact",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        ggval,
        "validate_reml_outputs",
        lambda *args, **kwargs: None,
    )

    paths = {"sim_prefix": sim_prefix, "pheno_txt": pheno}
    ggval.run_gs_bfile_suite(paths, tmp_path, threads=2, cv_folds=2)
    ggval.run_reml_suite(paths, tmp_path)

    assert commands[0][:3] == ["jx", "gs", "-bfile"]
    assert commands[0][commands[0].index("-prefix") + 1] == ggval.GS_BFILE_PREFIX_NAME
    assert commands[1][:3] == ["jx", "reml", "-file"]
    assert commands[1][commands[1].index("-k") + 1] == str(
        Path(f"{sim_prefix}.cGRM.npy")
    )


def test_run_validation_flow_dispatches_common_suites(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    paths = {"sim_prefix": tmp_path / "ggval_sim"}

    monkeypatch.setattr(ggval, "cleanup_validation_artifacts", lambda outdir: None)
    monkeypatch.setattr(ggval, "_resolve_sim_config", lambda mode: (2, 80, 1))
    monkeypatch.setattr(
        ggval,
        "build_validation_dataset",
        lambda *args, **kwargs: paths,
    )
    for name in (
        "run_gwas_suite",
        "run_gs_file_suite",
        "run_gs_bfile_suite",
        "run_gs_vcf_suite",
        "run_gs_hmp_suite",
        "run_reml_suite",
    ):
        monkeypatch.setattr(
            ggval,
            name,
            lambda *args, _name=name, **kwargs: calls.append(_name),
        )

    statuses = ggval.run_validation_flow(
        tmp_path,
        tmp_path / "logs",
        mode="smoke",
        suites=["gwas", "gs-file", "gs-bfile", "gs-vcf", "gs-hmp", "reml"],
        threads=2,
        cv_folds=2,
        postgs_enabled=False,
    )

    assert calls == [
        "run_gwas_suite",
        "run_gs_file_suite",
        "run_gs_bfile_suite",
        "run_gs_vcf_suite",
        "run_gs_hmp_suite",
        "run_reml_suite",
    ]
    assert statuses == {
        "gwas": "ok",
        "gs-file": "ok",
        "gs-bfile": "ok",
        "gs-vcf": "ok",
        "gs-hmp": "ok",
        "reml": "ok",
    }


def test_main_runs_selected_suite_and_reports_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_flow(outdir, logdir, **kwargs):
        captured.update({"outdir": outdir, "logdir": logdir, **kwargs})
        return {"gs-file": "ok"}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ggval",
            "--mode",
            "smoke",
            "--outdir",
            str(tmp_path / "out"),
            "--only",
            "gwas,gs-file",
            "--skip",
            "gwas",
        ],
    )
    monkeypatch.setattr(ggval, "check_jx_available", lambda: None)
    monkeypatch.setattr(ggval, "run_validation_flow", fake_flow)

    assert ggval.main() == 0
    assert captured["mode"] == "smoke"
    assert captured["suites"] == ["gs-file"]
    assert captured["postgs_enabled"] is True
    output = capsys.readouterr().out
    assert "Suites    : gs-file" in output
    assert "Status    : gs-file -> ok" in output


def test_main_rejects_mutually_exclusive_cli_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ggval",
            "--multicore",
            "--only",
            "gwas",
            "--outdir",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(ggval, "check_jx_available", lambda: None)

    with pytest.raises(SystemExit):
        ggval.main()
