#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ProposalFamily {
    Singleton,
    TwoPlus,
    ThreePlus,
}

impl ProposalFamily {
    fn index(self) -> usize {
        match self {
            Self::Singleton => 0,
            Self::TwoPlus => 1,
            Self::ThreePlus => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MaxOnlyReducer {
    maxima: [Option<f64>; 3],
}

impl MaxOnlyReducer {
    pub(crate) fn new(_families: [ProposalFamily; 3]) -> Self {
        Self { maxima: [None; 3] }
    }

    pub(crate) fn observe(&mut self, family: ProposalFamily, value: f64) {
        if !value.is_finite() {
            return;
        }
        let slot = &mut self.maxima[family.index()];
        if slot.map(|current| value > current).unwrap_or(true) {
            *slot = Some(value);
        }
    }

    pub(crate) fn max(&self, family: ProposalFamily) -> Option<f64> {
        self.maxima[family.index()]
    }

    #[cfg(test)]
    pub(crate) fn storage_len(&self) -> usize {
        self.maxima.len()
    }
}

impl SharedNullMaxima {
    pub(crate) fn observe_rule(&mut self, rule_len: usize, absolute: f64, delta: f64) {
        let family = match rule_len {
            0 | 1 => ProposalFamily::Singleton,
            2 => ProposalFamily::TwoPlus,
            _ => ProposalFamily::ThreePlus,
        };
        self.absolute.observe(family, absolute);
        // Best-parent delta is defined only for composite rules.  Keeping
        // singleton deltas out of the reducer avoids presenting a redundant
        // family threshold and matches the formal 2+/3+ calibration contract.
        if rule_len >= 2 {
            self.delta.observe(family, delta);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SharedNullMaxima {
    pub(crate) absolute: MaxOnlyReducer,
    pub(crate) delta: MaxOnlyReducer,
}

impl Default for SharedNullMaxima {
    fn default() -> Self {
        let families = [
            ProposalFamily::Singleton,
            ProposalFamily::TwoPlus,
            ProposalFamily::ThreePlus,
        ];
        Self {
            absolute: MaxOnlyReducer::new(families),
            delta: MaxOnlyReducer::new(families),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SharedMaxTAccumulator {
    absolute: [Vec<f64>; 3],
    delta: [Vec<f64>; 3],
    current: SharedNullMaxima,
}

impl SharedMaxTAccumulator {
    pub(crate) fn begin_replicate(&mut self) {
        self.current = SharedNullMaxima::default();
    }

    pub(crate) fn observe_rule(&mut self, rule_len: usize, absolute: f64, delta: f64) {
        self.current.observe_rule(rule_len, absolute, delta);
    }

    pub(crate) fn finish_replicate(&mut self) {
        for family in [
            ProposalFamily::Singleton,
            ProposalFamily::TwoPlus,
            ProposalFamily::ThreePlus,
        ] {
            if let Some(value) = self.current.absolute.max(family) {
                self.absolute[family.index()].push(value);
            }
            if let Some(value) = self.current.delta.max(family) {
                self.delta[family.index()].push(value);
            }
        }
    }

    pub(crate) fn values(&self, family: ProposalFamily, delta: bool) -> &[f64] {
        if delta {
            &self.delta[family.index()]
        } else {
            &self.absolute[family.index()]
        }
    }

    pub(crate) fn q99(&self, family: ProposalFamily, delta: bool) -> Option<f64> {
        quantile(self.values(family, delta), 0.99)
    }
}

fn quantile(values: &[f64], probability: f64) -> Option<f64> {
    if values.is_empty() || !probability.is_finite() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let index =
        ((sorted.len().saturating_sub(1) as f64) * probability.clamp(0.0, 1.0)).ceil() as usize;
    sorted
        .get(index.min(sorted.len().saturating_sub(1)))
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_only_reducer_keeps_one_value_per_family() {
        let families = [
            ProposalFamily::Singleton,
            ProposalFamily::TwoPlus,
            ProposalFamily::ThreePlus,
        ];
        let mut reducer = MaxOnlyReducer::new(families);
        for score in [1.0, 4.0, 2.0] {
            reducer.observe(ProposalFamily::TwoPlus, score);
        }
        assert_eq!(reducer.max(ProposalFamily::TwoPlus), Some(4.0));
        assert_eq!(reducer.storage_len(), 3);
    }

    #[test]
    fn shared_null_maxima_tracks_absolute_and_delta_separately() {
        let mut maxima = SharedNullMaxima::default();
        maxima.absolute.observe(ProposalFamily::ThreePlus, 2.0);
        maxima.delta.observe(ProposalFamily::ThreePlus, 1.5);
        assert_eq!(maxima.absolute.max(ProposalFamily::ThreePlus), Some(2.0));
        assert_eq!(maxima.delta.max(ProposalFamily::ThreePlus), Some(1.5));
    }

    #[test]
    fn shared_max_t_stores_one_maximum_per_family_and_replicate() {
        let mut accumulator = SharedMaxTAccumulator::default();
        accumulator.begin_replicate();
        accumulator.observe_rule(1, 1.0, 99.0);
        accumulator.observe_rule(3, 2.0, 1.0);
        accumulator.observe_rule(3, 4.0, 0.5);
        accumulator.finish_replicate();
        accumulator.begin_replicate();
        accumulator.observe_rule(3, 3.0, 2.0);
        accumulator.finish_replicate();
        assert_eq!(
            accumulator.values(ProposalFamily::ThreePlus, false),
            &[4.0, 3.0]
        );
        assert!(accumulator
            .values(ProposalFamily::Singleton, true)
            .is_empty());
        assert_eq!(accumulator.q99(ProposalFamily::ThreePlus, true), Some(2.0));
    }
}
