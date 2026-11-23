use super::*;

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
