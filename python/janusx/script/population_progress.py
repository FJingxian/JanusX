"""Low-overhead terminal progress for population scans."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ._common.progress import ProgressAdapter, stdout_is_tty


def _stage_bounds(total: int, weights: Sequence[float]) -> tuple[int, ...]:
    """Return monotonic integer bounds for weighted progress stages."""

    total_i = max(1, int(total))
    values = tuple(float(weight) for weight in weights)
    if not values or any((not math.isfinite(weight) or weight <= 0.0) for weight in values):
        values = tuple(1.0 for _ in values) or (1.0,)
    weight_sum = sum(values)
    bounds = [0]
    cumulative = 0.0
    for weight in values[:-1]:
        cumulative += weight
        boundary = int(round(total_i * cumulative / weight_sum))
        bounds.append(min(total_i, max(bounds[-1], boundary)))
    bounds.append(total_i)
    return tuple(bounds)


class PopulationScanProgress:
    """Render coarse Rust scan progress without entering the SNP hot loop.

    The Rust backend reports ``(stage, done, total)`` only at BED block
    boundaries.  Progress is disabled entirely for non-interactive output so
    batch jobs and redirected logs do not pay callback or terminal costs.
    """

    def __init__(
        self,
        *,
        description: str,
        stages: int = 1,
        stage_labels: Sequence[str] | None = None,
        log_unit: str = "site",
        stage_weights: Sequence[float] | None = None,
    ) -> None:
        self.enabled = bool(stdout_is_tty())
        self._stages = max(1, int(stages))
        supplied_weights = tuple(float(weight) for weight in (stage_weights or ()))
        if len(supplied_weights) != self._stages:
            supplied_weights = tuple(1.0 for _ in range(self._stages))
        self._stage_weights = supplied_weights
        labels = tuple(str(label) for label in (stage_labels or (description,)))
        self._labels = labels if labels else (str(description),)
        self._bar = None
        if self.enabled:
            try:
                self._bar = ProgressAdapter(
                    total=1,
                    desc=str(description),
                    show_spinner=True,
                    show_postfix=False,
                    keep_display=False,
                    show_remaining=True,
                    emit_done=False,
                    force_animate=True,
                    log_unit=str(log_unit),
                )
            except Exception:
                self.enabled = False
        self._last_stage = -1
        self._last_done = 0

    def _label(self, stage: int) -> str:
        index = min(max(0, int(stage)), len(self._labels) - 1)
        return self._labels[index]

    def callback(self, stage: int, done: int, total: int) -> None:
        """Consume a Rust callback; terminal failures never abort analysis."""

        if not self.enabled or self._bar is None:
            return
        try:
            stage_i = min(max(0, int(stage)), self._stages - 1)
            total_i = max(1, int(total))
            bounds = _stage_bounds(total_i, self._stage_weights)
            global_total = total_i
            if self._last_stage != stage_i:
                self._bar.set_desc(self._label(stage_i))
                self._bar.set_total(global_total)
                self._last_stage = stage_i
            elif self._bar.total != global_total:
                self._bar.set_total(global_total)

            stage_start = bounds[stage_i]
            stage_end = bounds[stage_i + 1]
            stage_fraction = min(1.0, max(0.0, float(done) / total_i))
            global_done = min(
                global_total,
                stage_start + int(round((stage_end - stage_start) * stage_fraction)),
            )
            delta = global_done - self._last_done
            if delta > 0:
                self._bar.update(delta)
                self._last_done = global_done
        except Exception:
            # A broken/closed terminal must not change the analysis result.
            self.enabled = False

    def finish(self) -> None:
        if self._bar is None:
            return
        try:
            self._bar.finish()
            self._bar.close(show_done=False)
        except Exception:
            self.enabled = False

    def close(self) -> None:
        if self._bar is None:
            return
        try:
            self._bar.close(show_done=False)
        except Exception:
            self.enabled = False
