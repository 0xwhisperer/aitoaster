"""
Micro time-warp: an experimental, UNVERIFIED-against-real-systems feature.

Hypothesis (stated explicitly by the user, not established fact): audio
fingerprint matchers (Shazam/ACRCloud/Pex-style constellation/landmark
matching) are commonly built to be robust to a GLOBAL, uniform time-stretch
or pitch-shift, but may implicitly assume a consistent internal tempo map
when aligning a query's landmark constellation against a reference. A very
slight, NON-UNIFORM time warp - the effective playback speed drifting by a
few milliseconds, unevenly, across the track (not a single flat stretch
factor) - might desynchronize relative landmark timing without producing
an audible pitch or tempo artifact, since human timing perception is far
coarser than a fingerprint matcher's alignment tolerance.

THIS IS NOT VERIFIED AGAINST ANY REAL FINGERPRINT SYSTEM. There is no local
access to ACRCloud/Pex to test against. What IS verified locally (see
app/fingerprint_proxy.py and the accompanying test) is a much narrower,
honest claim: whether this warp measurably shifts the relative timing of
a local landmark/constellation extraction built as a simplified proxy for
the same core technique those real systems use. A positive result there is
evidence the MECHANISM has some effect; it is NOT proof this defeats any
specific real commercial system, whose actual alignment tolerances,
windowing, and matching logic are unknown and could easily be more (or
less) tolerant of this kind of drift than the proxy.

DEFAULT DRIFT (4ms). Measured against the local landmark proxy on a real
track: fingerprint landmarks cluster in the low end - 94% below 500Hz, none
above 4kHz - because constellation matching anchors on the strongest, most
stable spectral peaks. Landmark timing is quantized by the analysis hop
(512 samples, ~11.6ms), so once the drift displaces a landmark past one hop
it cannot move further. Displacement therefore SATURATES: averaged over five
seeds, 4ms, 8ms and 15ms all produced the same 11.61ms p90 landmark shift,
while 2ms produced none.

Higher values are not free. Resampling through a drifting time axis smears
fast high-frequency content, and sibilants ("s"/"t") are exactly that. On a
measured vocal the sibilant retained 98.2% of its energy at 4ms but only
91.3% at 15ms, with the loss concentrated in 1-4kHz (-8.9%) while the bass
was untouched (+0.3%). Since no landmarks live in that band, drift above
4ms costs audible high-frequency detail without adding any disruption.
"""
import numpy as np
from scipy.interpolate import interp1d


def generate_warp_curve(n_samples, sr, seed=None, max_drift_ms=4.0, n_components=5):
    """Builds a smooth, slowly-varying warp curve: a sum of a few
    low-frequency sinusoids (random periods/phases/amplitudes, seeded),
    summing to a curve with no sharp transitions (which WOULD be audible
    as a click/glitch) and a bounded total drift (max_drift_ms caps how far
    any single moment can be shifted from its true position - this bounds
    both audibility risk and how much this can plausibly do against a real
    matcher's tolerance).

    Returns an array of per-sample time offsets, in seconds, the same
    length as n_samples - offset[i] is how much sample i's EFFECTIVE
    position drifts from its nominal position i/sr."""
    rng = np.random.default_rng(seed)
    duration_sec = n_samples / sr
    t = np.linspace(0, duration_sec, n_samples)

    curve = np.zeros(n_samples)
    for _ in range(n_components):
        # periods spread across several seconds to tens of seconds - slow
        # enough that the LOCAL rate-of-change stays small (a fast-varying
        # warp would start to sound like audible flutter/wow)
        period_sec = rng.uniform(3.0, 25.0)
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = rng.uniform(0.3, 1.0)
        curve += amplitude * np.sin(2 * np.pi * t / period_sec + phase)

    # normalize to exactly [-1, 1], then scale to the requested max drift
    curve = curve / (np.abs(curve).max() + 1e-9)
    max_drift_sec = max_drift_ms / 1000.0
    return curve * max_drift_sec


def apply_time_warp(mono_audio, sr, seed=None, max_drift_ms=4.0):
    """mono_audio: 1D float32 numpy array. Returns a new 1D float32 array,
    same length, with a smooth non-uniform time-warp applied.

    Mechanism: build a per-sample time-offset curve (see
    generate_warp_curve), add it to each sample's nominal timestamp to get
    an "effective" timestamp for where that sample's content should
    actually be read from, then interpolate the ORIGINAL signal at those
    warped-back positions. This resamples through a smoothly-varying time
    axis rather than a single fixed rate, which is what makes it a WARP
    (local speed varies) rather than a STRETCH (one global rate change,
    trivially undone by a matcher that already normalizes for tempo)."""
    n = len(mono_audio)
    offsets = generate_warp_curve(n, sr, seed=seed, max_drift_ms=max_drift_ms)

    original_t = np.arange(n) / sr
    # sample i's warped effective position is (i/sr + offset[i]) - to
    # produce output sample i, read the ORIGINAL signal at that warped
    # position via interpolation (this is a time-domain resample through a
    # non-constant map, not a spectral operation).
    warped_t = original_t + offsets
    warped_t = np.clip(warped_t, original_t[0], original_t[-1])

    interpolator = interp1d(original_t, mono_audio, kind="cubic",
                             bounds_error=False, fill_value=0.0)
    return interpolator(warped_t).astype(np.float32)
