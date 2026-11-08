use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GarfieldInputKind {
    Auto,
    Bfile,
    Vcf,
    Hmp,
    File,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum GarfieldBinMode {
    Bin,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum GarfieldMlKeepPolicy {
    TopK,
    TopKPlusRandom { top_frac: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GarfieldResponse {
    Binary,
    Continuous,
}

#[inline]
pub(super) fn parse_response(response: &str) -> Result<GarfieldResponse, String> {
    let t = response.trim().to_ascii_lowercase();
    match t.as_str() {
        "binary" | "bin" | "b" => Ok(GarfieldResponse::Binary),
        "continuous" | "cont" | "c" => Ok(GarfieldResponse::Continuous),
        _ => Err("response must be 'binary' or 'continuous'".to_string()),
    }
}

#[inline]
pub(super) fn parse_input_kind(input_kind: &str) -> Result<GarfieldInputKind, String> {
    let t = input_kind.trim().to_ascii_lowercase();
    match t.as_str() {
        "" | "auto" => Ok(GarfieldInputKind::Auto),
        "bfile" | "plink" | "bed" => Ok(GarfieldInputKind::Bfile),
        "vcf" => Ok(GarfieldInputKind::Vcf),
        "hmp" | "hapmap" => Ok(GarfieldInputKind::Hmp),
        "file" | "txt" | "npy" | "bin" => Ok(GarfieldInputKind::File),
        _ => Err("input_kind must be one of: auto, bfile, vcf, hmp, file".to_string()),
    }
}

#[inline]
pub(super) fn parse_bin_mode(mode: &str) -> Result<GarfieldBinMode, String> {
    let t = mode.trim().to_ascii_lowercase();
    match t.as_str() {
        "bin" | "bin02" => Ok(GarfieldBinMode::Bin),
        _ => Err("mode must be one of: bin, bin02".to_string()),
    }
}

#[inline]
pub(super) fn parse_beam_rank_mode(mode: &str) -> Result<BeamRankMode, String> {
    let t = mode.trim().to_ascii_lowercase();
    if let Some(rest) = t.strip_prefix("gain_from_layer:") {
        let start = rest
            .parse::<usize>()
            .map_err(|_| "rank_score gain_from_layer:<N> requires integer N >= 1".to_string())?;
        if start < 1 {
            return Err("rank_score gain_from_layer:<N> requires integer N >= 1".to_string());
        }
        return Ok(BeamRankMode::GainFromLayer(start));
    }
    match t.as_str() {
        "" | "interaction_gain" | "gain" | "interaction-gain" | "interactiongain" => {
            Ok(BeamRankMode::InteractionGain)
        }
        "exhaustive_then_gain"
        | "exhaustive-then-gain"
        | "exhaustivethengain"
        | "staged_gain"
        | "staged-gain"
        | "stagedgain"
        | "beam_gain"
        | "beam-gain"
        | "beamgain" => Ok(BeamRankMode::ExhaustiveThenGain),
        "raw" | "score" => Ok(BeamRankMode::Raw),
        _ => Err(
            "rank_score must be one of: raw, interaction_gain, exhaustive_then_gain, gain_from_layer:<N>"
                .to_string(),
        ),
    }
}
