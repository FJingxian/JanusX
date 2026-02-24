import pandas as pd
import re
import gzip
from urllib.parse import unquote


def _extract_attr_value(attr_series: pd.Series, key: str) -> pd.Series:
    """
    Extract one GFF attribute value by key.
    Supports both "...;key=value;..." and "...;key=value" at line end.
    """
    key_pat = re.escape(str(key))
    # capture value until next ';' or end-of-string
    pattern = rf"(?:^|;){key_pat}=([^;]*)(?:;|$)"
    out = attr_series.astype(str).str.extract(pattern, expand=False)
    out = out.fillna("NA")
    def _normalize_text(x: object) -> object:
        if not isinstance(x, str):
            return x
        text = unquote(x).strip()
        # Normalize tabs/newlines inside malformed attributes to single spaces.
        return re.sub(r"\s+", " ", text)
    return out.map(_normalize_text)


def _read_gff_basic(annofile: str) -> pd.DataFrame:
    """
    Read GFF/GFF3 robustly and keep full attribute column (field 9).
    Uses split('\\t', 8) to avoid truncation when extra tabs appear inside attributes.
    """
    opener = gzip.open if annofile.endswith(".gz") else open
    rows: list[list[object]] = []
    with opener(annofile, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            line = line.rstrip("\n")
            parts = line.split("\t", 8)
            if len(parts) < 9:
                continue
            rows.append([parts[0], parts[2], parts[3], parts[4], parts[8]])
    if not rows:
        return pd.DataFrame(columns=[0, 2, 3, 4, 8])
    return pd.DataFrame(rows, columns=[0, 2, 3, 4, 8])


def readanno(annofile:str,descItem:str='description'):
    """After processing: anno columns are 0=chr, 1=start, 2=end, 3=geneID, 4=desc1, 5=desc2."""
    suffix = annofile.replace('.gz','').split('.')[-1]
    if suffix == 'bed':
        anno = pd.read_csv(annofile,sep='\t',header=None,).fillna('NA')
        if anno.shape[1]<=4:
            anno[4] = ['NA' for _ in anno.index]
        if anno.shape[1]<=5:
            anno[5] = ['NA' for _ in anno.index]
    elif suffix == 'gff' or suffix == 'gff3':
        anno = _read_gff_basic(annofile)
        anno = anno[anno[2] == 'gene']
        del anno[2]
        anno[0] = anno[0].astype(str).str.strip()
        anno[3] = pd.to_numeric(anno[3], errors='coerce')
        anno[4] = pd.to_numeric(anno[4], errors='coerce')
        anno = anno.dropna(subset=[3, 4])
        anno[3] = anno[3].astype(int)
        anno[4] = anno[4].astype(int)
        anno = anno.sort_values([0,3])
        anno.columns = range(anno.shape[1])
        anno[4] = _extract_attr_value(anno[3], descItem)
        anno[3] = _extract_attr_value(anno[3], 'ID')
        # Strip prefix like "gene:" -> "Zm00001d027230"
        anno[3] = anno[3].str.replace(r'^[^:]*:', '', regex=True)
        anno[5] = ['NA' for _ in anno.index]
    return anno
