"""
Local landmark/constellation extraction proxy - built ONLY to test the
time-warp hypothesis in app/timewarp.py, NOT a real fingerprinting system
and NOT a reproduction of any specific commercial product (Shazam/
ACRCloud/Pex). Real systems have proprietary peak-selection, hashing, and
matching logic this does not attempt to replicate.

What this DOES capture, faithfully enough to be a meaningful proxy: the
core idea common to constellation-based audio fingerprinting - identify
prominent, well-separated peaks in a time-frequency representation (a
spectrogram), and represent the audio as a set of (time, frequency)
landmark points. Real systems then hash PAIRS of nearby landmarks (their
relative time/frequency offset) for fast database lookup; this proxy stops
at extracting the landmark set itself and comparing landmark TIMING
directly, since that's the specific property the time-warp hypothesis is
about (does the warp measurably shift landmark timing).

This cannot tell you whether time-warping defeats any real matcher's
actual tolerance - that tolerance is unknown/proprietary. It CAN tell you
whether the warp has ANY measurable effect on the kind of representation
these systems are built on, which is the honest, narrower question this
experiment can actually answer locally.
"""
import numpy as np
from scipy.signal import stft as scipy_stft
from scipy.ndimage import maximum_filter


def extract_landmarks(mono_audio, sr, n_fft=2048, hop=512,
                        neighborhood_time=15, neighborhood_freq=15,
                        min_peak_db=-40.0, max_landmarks=400):
    """Returns a list of (time_sec, freq_hz, magnitude_db) tuples - the
    "constellation" of prominent, locally-maximal spectral peaks.

    A peak is kept only if it's the loudest point in a (neighborhood_time x
    neighborhood_freq) local window AND above min_peak_db - this is the
    same core "local maximum in a 2D neighborhood" peak-picking principle
    real constellation fingerprinting uses to find robust, well-separated
    landmarks rather than every noisy bin."""
    freqs, times, Zxx = scipy_stft(mono_audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag_db = 20 * np.log10(np.abs(Zxx) + 1e-8)

    local_max = maximum_filter(mag_db, size=(neighborhood_freq, neighborhood_time))
    is_peak = (mag_db == local_max) & (mag_db > min_peak_db)

    peak_freq_idx, peak_time_idx = np.where(is_peak)
    peak_mags = mag_db[peak_freq_idx, peak_time_idx]

    # keep only the strongest landmarks up to max_landmarks, sorted by time -
    # real systems similarly cap landmark density (too many landmarks per
    # second is itself computationally/storage-wasteful and doesn't improve
    # matching, since the point is a SPARSE, distinctive set of anchors)
    if len(peak_mags) > max_landmarks:
        top_idx = np.argsort(peak_mags)[-max_landmarks:]
        peak_freq_idx = peak_freq_idx[top_idx]
        peak_time_idx = peak_time_idx[top_idx]
        peak_mags = peak_mags[top_idx]

    landmarks = [
        (float(times[t]), float(freqs[f]), float(m))
        for f, t, m in zip(peak_freq_idx, peak_time_idx, peak_mags)
    ]
    landmarks.sort(key=lambda x: x[0])
    return landmarks


def match_landmarks(landmarks_a, landmarks_b, freq_tolerance_hz=50.0,
                      time_search_window_sec=0.5):
    """For each landmark in A, find the landmark in B that's the same
    FREQUENCY (within freq_tolerance_hz) and CLOSEST IN TIME. Returns a
    list of (time_shift_sec) values - one per matched pair - representing
    how much that landmark's timing appears to have moved between A and B.

    BUG FIX (found via direct testing): an earlier version of this
    function filtered candidates by time-proximity first, then picked
    whichever had the closest FREQUENCY - but many landmarks recur at the
    same frequency multiple times across a clip (bass notes, harmonics),
    so when several same-frequency candidates tied on frequency distance
    (often exactly 0, since it's genuinely the same recurring tone),
    argmin's first-wins tie-break could pick a candidate at the WRONG
    occurrence of that frequency, even when matching a clip against an
    EXACT COPY OF ITSELF. Verified directly: a self-match control (which
    should show ~0ms shift for every single landmark, since the input is
    identical) showed the same large spurious "shifts" as the real
    warped-vs-original comparison, revealing the matcher itself was the
    noise source, not anything about the audio being compared. Fixed by
    filtering to same-frequency-band candidates FIRST, then picking
    whichever of those is closest in TIME - the natural fix once a
    landmark's frequency is treated as its stable identity and time as
    the thing that can legitimately drift (which is the actual property
    this whole experiment is trying to measure)."""
    shifts = []
    b_by_freq = sorted(landmarks_b, key=lambda x: x[1])
    b_freqs = np.array([b[1] for b in b_by_freq])
    b_times = np.array([b[0] for b in b_by_freq])

    for t_a, f_a, _ in landmarks_a:
        candidates_idx = np.where(np.abs(b_freqs - f_a) <= freq_tolerance_hz)[0]
        if len(candidates_idx) == 0:
            continue
        candidate_times = b_times[candidates_idx]
        time_diffs = np.abs(candidate_times - t_a)
        best_local = np.argmin(time_diffs)
        if time_diffs[best_local] > time_search_window_sec:
            continue
        best_idx = candidates_idx[best_local]
        t_b = b_times[best_idx]
        shifts.append(t_b - t_a)

    return shifts
