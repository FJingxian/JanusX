import csv
import re

import pandas as pd


_PSEUDO_COMPONENT_SPLIT_RE = re.compile(r"[|&]")


def _normalize_assoc_key_value(value: object) -> str:
    txt = str(value).strip()
    if txt == "":
        return txt
    try:
        return str(int(float(txt)))
    except Exception:
        return txt


def _split_pseudo_components(snp_name: str) -> list[tuple[str, str, str]]:
    components: list[tuple[str, str, str]] = []
    for raw_part in _PSEUDO_COMPONENT_SPLIT_RE.split(str(snp_name)):
        part = raw_part.strip()
        if not part:
            continue
        if part.startswith("!"):
            part = part[1:].strip()
        if not part:
            continue
        if "_" not in part:
            raise ValueError(f"Invalid pseudo SNP component without chrom_pos form: {snp_name}")
        chrom, pos = part.rsplit("_", 1)
        chrom = chrom.strip()
        pos_norm = _normalize_assoc_key_value(pos)
        if chrom == "" or pos_norm == "":
            raise ValueError(f"Invalid pseudo SNP component coordinates: {snp_name}")
        components.append((part, chrom, pos_norm))
    return components


def _is_composite_pseudo_snp(snp_name: str) -> bool:
    try:
        return len(_split_pseudo_components(snp_name)) > 1
    except ValueError:
        return False


def _component_alleles(raw: object, n_components: int) -> list[str]:
    parts = [part.strip() for part in str(raw).split(",")]
    if len(parts) < n_components:
        parts.extend([""] * (n_components - len(parts)))
    return parts[:n_components]


def _assoc_row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("chrom", "")).strip(),
        _normalize_assoc_key_value(row.get("pos", "")),
        str(row.get("snp", "")).strip(),
    )


def _placeholder_single_row(
    header: list[str],
    chrom: str,
    pos: str,
    snp_name: str,
    allele0: str,
    allele1: str,
) -> dict[str, str]:
    row = {col: "NA" for col in header}
    row["chrom"] = chrom
    row["pos"] = pos
    row["snp"] = snp_name
    row["allele0"] = allele0
    row["allele1"] = allele1
    if "miss" in row:
        row["miss"] = "0"
    return row


def expand_pseudo_gwas_tsv(tsv_path: str) -> dict[str, int]:
    """
    Expand GARFIELD pseudo-GWAS TSV rows for downstream inspection.

    For each composite pseudo SNP, the output TSV is rewritten so that:
    - each constituent SNP has a singleton row nearby; real singleton rows are
      moved to the first referencing composite site, otherwise a placeholder row
      is synthesized with singleton coordinates and alleles.
    - the composite pseudo SNP itself is duplicated once per constituent
      coordinate, keeping the original composite SNP name/statistics.

    The tuple (chrom, pos, snp) remains unique after expansion.
    """
    with open(tsv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = list(reader.fieldnames or [])
        rows = [{col: str(row.get(col, "")) for col in header} for row in reader]

    if not header or len(rows) == 0:
        return {"rows_in": len(rows), "rows_out": len(rows), "composite_rows": 0}

    singleton_rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        snp_name = str(row.get("snp", "")).strip()
        if _is_composite_pseudo_snp(snp_name):
            continue
        singleton_rows.setdefault(_assoc_row_key(row), row)

    emitted_singletons: set[tuple[str, str, str]] = set()
    out_rows: list[dict[str, str]] = []
    composite_rows = 0

    for row in rows:
        snp_name = str(row.get("snp", "")).strip()
        components = _split_pseudo_components(snp_name)
        if len(components) <= 1:
            key = _assoc_row_key(row)
            if key in emitted_singletons:
                continue
            out_rows.append(dict(row))
            emitted_singletons.add(key)
            continue

        composite_rows += 1
        allele0_parts = _component_alleles(row.get("allele0", ""), len(components))
        allele1_parts = _component_alleles(row.get("allele1", ""), len(components))

        for idx, (component_snp, chrom, pos) in enumerate(components):
            singleton_key = (chrom, pos, component_snp)
            if singleton_key in emitted_singletons:
                continue
            actual_single = singleton_rows.get(singleton_key)
            if actual_single is not None:
                out_rows.append(dict(actual_single))
            else:
                out_rows.append(
                    _placeholder_single_row(
                        header,
                        chrom,
                        pos,
                        component_snp,
                        allele0_parts[idx],
                        allele1_parts[idx],
                    )
                )
            emitted_singletons.add(singleton_key)

        for _idx, (_component_snp, chrom, pos) in enumerate(components):
            expanded = dict(row)
            expanded["chrom"] = chrom
            expanded["pos"] = pos
            out_rows.append(expanded)

    with open(tsv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=header,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(out_rows)

    return {
        "rows_in": len(rows),
        "rows_out": len(out_rows),
        "composite_rows": composite_rows,
    }

def decode(gwasresult: pd.DataFrame, pseudodf: pd.DataFrame, pseudochrom: str = 'pseudo'):
    """
    Decode GARFIELD pseudo SNP GWAS hits back to the original SNP coordinates.

    Parameters
    ----------
    gwasresult : pd.DataFrame
        GWAS results on pseudo SNPs with required columns (chrom, pos, pwald).
    pseudodf : pd.DataFrame
        Mapping table from pseudo SNPs to expression strings with required columns (chrom, pos, expression).
    pseudochrom : str, default "pseudo"
        Pseudo chromosome label used to select rows in gwasresult.

    Returns
    -------
    pd.DataFrame
        Expanded table with columns:
        - batch : pseudochrom label
        - pseudo : original expression string
        - chrom : original SNP chromosome
        - pos : original SNP position
        - pwald : p-value from gwasresult

    Notes
    -----
    - Each expression is expanded into multiple rows (one per SNP).
    - Duplicate expressions are removed before expansion.
    """
    df = gwasresult
    df = df.set_index(df.columns[[0,1]].tolist())

    pseudoIdx = pseudodf
    pseudoIdx = pseudoIdx.set_index(pseudoIdx.columns[[0,1]].tolist())
    df[pseudochrom] = pseudoIdx.loc[df.index,'expression']
    df = df.loc[~df[pseudochrom].duplicated()]
    df_pseudo = df.xs(pseudochrom, level=0)
    df_pseudo = df_pseudo.reset_index().rename(columns={'index': 'idx'})
    df_pseudo = df_pseudo.rename(columns={df_pseudo.columns[0]: 'idx'})

    pattern = r'(?P<flag>!)?(?P<chrom>\d+)_(?P<pos>\d+)'
    matches = df_pseudo[pseudochrom].str.extractall(pattern)

    matches = matches.reset_index().rename(
        columns={'level_0': 'row_id', 'level_1': 'match_id'}
    )
    matches = matches.merge(
        df_pseudo[['pwald', pseudochrom]],
        left_on='row_id',
        right_index=True,
        how='left'
    )

    # Type conversion
    matches['chrom'] = matches['chrom'].astype(str).str.strip()
    matches['pos'] = matches['pos'].astype(int)
    matches['batch'] = pseudochrom
    matches['pseudo'] = matches[pseudochrom]

    # 最终结果
    final = matches[['batch','pseudo', 'chrom', 'pos', 'pwald']]
    return final
