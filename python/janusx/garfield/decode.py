import pandas as pd

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
