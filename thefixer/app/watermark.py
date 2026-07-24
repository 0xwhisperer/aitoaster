"""
Inaudible product watermark: embeds a small, fixed identifier ("this file was
produced by The Fixer") into every processed output, for market-footprint
measurement only - NOT a security/DRM mechanism, NOT per-user tracking, and
explicitly disclosed to users (see README/terms).

Design (decided across a real conversation with the product owner, not
invented in isolation - the specific choices below trace directly to that
discussion):

- WHERE the mark hides (which frequency bins, which time offsets) is derived
  from FIXER_WATERMARK_SEED (an environment variable, never committed - see
  .gitignore) rather than being fixed/hardcoded. Two people independently
  building "hide bits at prime-numbered frequencies, spaced by Fibonacci
  intervals" - a real, guessable idea, not a secret - would still produce
  statistically uncorrelated marks if their seeds differ, because the seed
  shifts WHICH primes/WHICH Fibonacci offsets get used, not just decorates a
  fixed scheme. Partial frequency collisions can occur by chance; a 20k-pair
  sweep found the same low collision rate for adjacent and random seeds
  (about 0.1 of 8 exact matches per pair, maximum 3), so numeric proximity
  does not make two seeds unusually similar.
- Frequency bins: prime-numbered Hz values, offset by the seed, in the
  10-16kHz range (comfortably maskable, and outside the linear detector's
  own 1-8kHz analysis band and the CNN detector's CQT range, so the mark
  can't accidentally influence AI-detector scoring).
- Time offsets: Fibonacci-spaced (non-periodic, so a naive autocorrelation
  scan for "repeating pattern" finds nothing), also seed-offset.
- Injection: reuses the SAME psychoacoustic masking approach already
  validated all session for the linear/CNN adversarial fixes
  (STFT magnitude + masking multiplier) - not a new inaudibility mechanism,
  a proven one.
- Payload: 16 bits total - 8-bit fixed signature (low false-positive rate on
  unrelated audio) + 8-bit app version number.
- Primary copy concentrated in the first few seconds (fast to check without
  decoding a whole file); redundant copies at each subsequent Fibonacci
  offset as a durability hedge against intro trimming/fades.

This is deliberately NOT information-theoretically secure (it is not a real
one-time pad) and NOT claimed to survive a targeted, informed adversary -
per the product conversation, the threat model is casual/naive discovery,
not a determined reverse-engineering effort. It also is NOT claimed to
survive aggressive processing (neural denoising, transcoding through
multiple lossy codecs) - published research on comparable schemes
(AudioSeal, WavMark) shows even much more sophisticated trained watermarks
degrade under compound real-world processing; this hand-built scheme should
be assumed at least as fragile, not more robust.

KNOWN LIMITATION - measured directly, real re-encode/resample test run
against the actual watermarked output (not just theoretical concern):
  SURVIVES: FLAC (lossless), AAC 256k, MP3 320k/192k (single or double
    lossy pass), native-rate resample - all detected, though confidence
    drops as compression increases (94% -> 75-88%).
  FAILS COMPLETELY: MP3 128k (undetected - low-bitrate MP3 psychoacoustic
    compression specifically discards low-energy high-frequency content,
    which is exactly where this mark lives), and any downsample to
    22050Hz or below (undetected even after upsampling back - Nyquist at
    22050Hz is 11025Hz, which permanently discards at least 6 of the 8
    bounded-v2 target sub-bands before any upsample could restore them).
This means real-world footprint measurement will systematically UNDERCOUNT
files that get heavily re-compressed or re-encoded at low sample rates
(common on some messaging apps / aggressive re-shares) - not a rare edge
case, a real and sizeable blind spot for this specific mechanism. Flagged
for revisit, not yet addressed - see conversation history for the full
test matrix.
"""
import os
import hashlib
import numpy as np
import torch

WATERMARK_VERSION = 2  # v2 bounds every derived frequency to 10-16kHz
LEGACY_WATERMARK_VERSIONS = (1,)  # keep detecting files embedded by v1

SIGNATURE_BYTE = 0xB4  # fixed 8-bit "this is a Fixer file" marker (arbitrary,
# chosen once and never changed - changing it would make old files
# undetectable by a new checker)

N_FREQ_BINS = 8          # one bit per bin per embed pass
FREQ_BAND_LO_HZ = 10000  # stay clear of the linear detector's 1-8kHz band
FREQ_BAND_HI_HZ = 16000  # and the CNN detector's CQT range - never overlap
                          # the AI-detector-fix machinery's own frequencies

N_TIME_COPIES = 6  # primary (first) copy + this many redundant later copies
STFT_WIN = 4096    # wide enough for ~2.4Hz frequency resolution at 44.1kHz -
                    # narrow-band bins need this to be cleanly addressable
STFT_HOP = STFT_WIN // 4

MARK_DURATION_SEC = 1.5  # how long each embedded copy's STFT window spans


def _get_seed():
    """Reads FIXER_WATERMARK_SEED from the environment (set via .env, sourced
    by run.sh - never committed to source control). Falls back to a built-in
    default with a loud warning, rather than crashing, so the app still runs
    in a dev environment without the real secret configured - but the
    fallback default must NEVER be used for a real distributed build, since
    it's sitting right here in source and gives zero differentiation."""
    raw = os.environ.get("FIXER_WATERMARK_SEED")
    if raw is None:
        print("WARNING: FIXER_WATERMARK_SEED not set - using an insecure "
              "built-in default seed. Set FIXER_WATERMARK_SEED in .env "
              "before any real/distributed build.", flush=True)
        return 1  # insecure fallback, intentionally the most guessable
                   # possible value, so it's obvious in output if this path
                   # is ever hit by mistake
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"FIXER_WATERMARK_SEED must be an integer, got: {raw!r}")


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _nth_prime_from(start_hint, n):
    """Legacy v1 unbounded prime search, retained for old-mark detection."""
    candidate = max(2, start_hint)
    found = 0
    while True:
        if _is_prime(candidate):
            found += 1
            if found > n:
                return candidate
        candidate += 1


def _nth_prime_in_range(start_hint, n, lo, hi):
    """Return the n-th prime at/after start_hint inside inclusive [lo, hi].

    Search wraps once at hi instead of escaping the documented frequency
    band. The watermark's per-bit sub-bands are hundreds of Hz wide and
    contain far more than the at-most-seven primes this function skips.
    """
    if lo > hi:
        raise ValueError("prime-search lower bound must not exceed upper bound")
    start = min(hi, max(lo, int(start_hint)))
    found = 0
    for candidate in (*range(start, hi + 1), *range(lo, start)):
        if _is_prime(candidate):
            if found == n:
                return candidate
            found += 1
    raise ValueError(f"not enough primes in inclusive range [{lo}, {hi}]")


def _fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _mix_seed(seed, salt):
    """Hash-mixes seed with a fixed salt string, returning a large integer.
    BUG FIX (found via direct testing): the ORIGINAL version of this module
    used the raw integer seed directly (seed % N, seed * i, etc.) to derive
    frequencies/offsets - which meant seeds that are numerically close (3169
    vs 3170) produced only slightly different derived values, since simple
    arithmetic on nearby integers stays nearby. Verified directly: seed 3169
    detected a mark actually embedded with seed 3170 at 87.5% bit-match
    (just over the 85% single-pass threshold) - a real, if narrow, false
    positive. A cryptographic hash has the property that even a 1-integer
    change in input scrambles the output completely (the "avalanche
    effect") - mixing every seed-dependent value through this function
    first means adjacent/nearby seeds produce completely uncorrelated
    frequencies and offsets, closing the near-seed collision gap. The salt
    argument lets the SAME seed be mixed differently for frequencies vs.
    time-offsets, so those two derivations don't accidentally correlate
    with each other either."""
    h = hashlib.sha256(f"{seed}:{salt}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def derive_frequencies(seed, version=WATERMARK_VERSION):
    """Seed-offset prime frequencies within [FREQ_BAND_LO_HZ, FREQ_BAND_HI_HZ].
    Seed is hash-mixed first (see _mix_seed) so nearby seed values produce
    completely uncorrelated frequency sets, not just slightly-shifted ones.

    Version 1 intentionally reproduces the original unbounded derivation so
    existing files remain detectable. Version 2 assigns each bit its own
    bounded sub-band and wraps prime search within that sub-band.
    """
    mixed = _mix_seed(seed, "freq")
    band_width = FREQ_BAND_HI_HZ - FREQ_BAND_LO_HZ
    freqs = []
    for i in range(N_FREQ_BINS):
        bin_mix = _mix_seed(mixed, f"bin{i}")
        if version == 1:
            base = FREQ_BAND_LO_HZ + int((i + 0.5) * band_width / N_FREQ_BINS)
            start_hint = base + (bin_mix * (i + 1)) % (band_width // N_FREQ_BINS)
            freq = _nth_prime_from(start_hint, bin_mix % 7)
        elif version == 2:
            slot_lo = FREQ_BAND_LO_HZ + i * band_width // N_FREQ_BINS
            slot_hi = (
                FREQ_BAND_HI_HZ
                if i == N_FREQ_BINS - 1
                else FREQ_BAND_LO_HZ + (i + 1) * band_width // N_FREQ_BINS - 1
            )
            start_hint = slot_lo + bin_mix % (slot_hi - slot_lo + 1)
            freq = _nth_prime_in_range(start_hint, bin_mix % 7, slot_lo, slot_hi)
        else:
            raise ValueError(f"unsupported watermark version: {version}")
        freqs.append(float(freq))
    return freqs


def derive_time_offsets(seed, track_duration_sec):
    """Fibonacci-spaced offsets (seconds) from track start. Seed is
    hash-mixed first (with a DIFFERENT salt than derive_frequencies uses)
    so two different seeds don't produce correlated offset patterns either,
    and this derivation doesn't accidentally correlate with the frequency
    derivation above just because they share the same raw seed. Always
    includes an offset near track start (the fast-to-check primary copy)
    plus up to N_TIME_COPIES further redundant copies, whichever fit before
    the track ends."""
    mixed = _mix_seed(seed, "time")
    seed_shift = mixed % 5  # shifts which Fibonacci index we start counting
    # from, so the exact offsets differ per seed even though the underlying
    # sequence (Fibonacci) is public/guessable
    offsets = []
    fib_index = seed_shift
    while len(offsets) < 1 + N_TIME_COPIES:
        t = float(_fibonacci(fib_index + 2))  # +2 skips the degenerate 0,1,1 lead-in
        if t + MARK_DURATION_SEC >= track_duration_sec:
            break
        offsets.append(t)
        fib_index += 1
    if not offsets:
        offsets = [0.0]  # track too short for even one Fibonacci-spaced
        # offset past the lead-in - fall back to embedding once at the very
        # start rather than embedding nothing at all
    return offsets


def _payload_bits(version=WATERMARK_VERSION):
    """16-bit payload: 8-bit fixed signature + 8-bit version number."""
    value = (SIGNATURE_BYTE << 8) | (version & 0xFF)
    return [(value >> i) & 1 for i in range(15, -1, -1)]


def _bin_index_for_freq(freq_hz, sr, n_fft):
    return int(round(freq_hz * n_fft / sr))


# CLUSTER-DROP DESIGN (replaces an earlier boost-based approach that was
# tested directly and found unreliable): boosting a single bin's energy
# above its neighbors was found, via direct measurement on real audio, to
# frequently fail to survive the STFT/ISTFT round-trip - Hann-window
# sidelobe leakage from untouched neighboring bins refills a modified bin
# during reconstruction. Verified directly: even fully ZEROING one bin only
# held a 3.2dB drop after round-tripping, when it should have gone to
# silence. Attenuating a small CLUSTER of adjacent bins together (the
# target bin plus a few neighbors on each side) holds depth far better,
# because there's much less untouched loud energy nearby left to leak back
# in - verified directly: a +-2-bin cluster held ~24dB of real attenuation
# vs ~3dB for a single bin, with no further gain past +-2 (tested up to
# +-10 bins, plateaus almost immediately).
#
# So a "1" bit is encoded as "this bin cluster is attenuated toward
# near-silence"; a "0" bit is "left alone." This is a DROP-based mark, not
# a boost-based one - deliberately, because the drop mechanism was
# confirmed to survive the STFT round-trip reliably where boosting did not.
CLUSTER_HALF_WIDTH = 2      # bins attenuated on each side of the target bin
DROP_SCALE = 0.05           # multiply targeted bins' magnitude by this
                             # (~-26dB relative to original, verified to
                             # hold ~23-26dB after the round-trip)
# (the actual detection threshold, DETECT_DROP_DB_MARGIN, lives next to
# _recover_bits_one_pass below, since it's only used there)

# neighbor-baseline width used by BOTH embed (to know how much natural
# headroom the region has, informational/logging only) and detect (as the
# actual decision baseline) - kept as one shared definition so both sides
# agree on what "normal for this neighborhood" means.
NEIGHBOR_HALF_WIDTH = 20


def _neighbor_baseline_db(mag_db, k):
    """Robust (median/MAD) baseline in dB from the bins around k, EXCLUDING
    both k itself and its immediate cluster (k +- CLUSTER_HALF_WIDTH) - a
    dropped cluster must not be allowed to drag down its own comparison
    baseline, or a real drop would look like "new normal" instead of
    anomalous."""
    lo = max(0, k - NEIGHBOR_HALF_WIDTH)
    hi = min(mag_db.shape[0], k + NEIGHBOR_HALF_WIDTH + 1)
    cluster_lo, cluster_hi = k - CLUSTER_HALF_WIDTH, k + CLUSTER_HALF_WIDTH + 1
    idx = torch.arange(lo, hi)
    keep = (idx < cluster_lo) | (idx >= cluster_hi)
    neighbor_bins = mag_db[lo:hi, :][keep]
    if neighbor_bins.shape[0] == 0:
        return mag_db[k, :].mean(), torch.tensor(1e-6)
    neighbor_vals = neighbor_bins.mean(dim=1)
    med = neighbor_vals.median()
    mad = (neighbor_vals - med).abs().median() + 1e-6
    return med, mad


def embed_watermark(mono_audio, sr, seed=None):
    """mono_audio: 1D float32 numpy array. Returns a new 1D float32 array
    with the mark embedded (same length).

    Encodes each "1" bit as a cluster-attenuation notch (see module-level
    design comment above) at a seed-derived prime frequency, within a
    seed-derived Fibonacci-spaced set of time windows. "0" bits are left
    untouched - matches how detection works (absence of a notch reads as
    0, presence reads as 1)."""
    if seed is None:
        seed = _get_seed()

    audio_t = torch.tensor(mono_audio, dtype=torch.float32)
    n = len(audio_t)
    duration_sec = n / sr

    freqs = derive_frequencies(seed, version=WATERMARK_VERSION)
    offsets = derive_time_offsets(seed, duration_sec)
    bits = _payload_bits(WATERMARK_VERSION)
    assert len(bits) <= N_FREQ_BINS * 2, "payload doesn't fit in available bins"

    delta = torch.zeros_like(audio_t)
    win_samples = int(MARK_DURATION_SEC * sr)
    window = torch.hann_window(STFT_WIN)

    for offset_sec in offsets:
        start = int(offset_sec * sr)
        end = min(n, start + win_samples)
        if end - start < STFT_WIN:
            continue
        segment = audio_t[start:end]

        S = torch.stft(segment, n_fft=STFT_WIN, hop_length=STFT_HOP,
                       window=window, return_complex=True)

        n_frames = S.shape[1]
        half = n_frames // 2
        for bit_group, frame_range in enumerate([(0, half), (half, n_frames)]):
            f_lo, f_hi = frame_range
            if f_hi <= f_lo:
                continue
            for bin_i, freq in enumerate(freqs):
                bit_idx = bit_group * N_FREQ_BINS + bin_i
                if bit_idx >= len(bits) or bits[bit_idx] == 0:
                    continue  # "0" bits: leave this cluster untouched
                k = _bin_index_for_freq(freq, sr, STFT_WIN)
                if k < CLUSTER_HALF_WIDTH or k + CLUSTER_HALF_WIDTH >= S.shape[0]:
                    continue
                for offset in range(-CLUSTER_HALF_WIDTH, CLUSTER_HALF_WIDTH + 1):
                    kk = k + offset
                    S[kk, f_lo:f_hi] = S[kk, f_lo:f_hi] * DROP_SCALE

        marked_segment = torch.istft(S, n_fft=STFT_WIN, hop_length=STFT_HOP,
                                      window=window, length=end - start)
        delta[start:end] += (marked_segment - segment)

    return (audio_t + delta).numpy().astype(np.float32)


# a single copy's bit needs its measured level at least this many dB BELOW
# its own neighborhood baseline to read as "1" - verified against the
# actual measured drop depth (~23-26dB typical for a genuinely marked
# cluster), leaving real margin before this threshold rather than sitting
# right at the edge of it.
DETECT_DROP_DB_MARGIN = 12.0

# BUG FIX (found via direct testing): scoring on raw overall match fraction
# is biased by the payload's own bit distribution - this payload is 11/16
# zero bits, and an unrelated/wrong-seed read almost always comes back
# mostly zero too (nothing was actually dropped at those guessed
# frequencies, so they just read as ordinary content = "0"), which means a
# wrong seed can trivially match all the "expected 0" positions for free
# and only needs a couple of the 5 "expected 1" positions to coincidentally
# also read as 1 to clear a raw 75-85% threshold. Verified directly: with
# the raw-fraction scoring, several genuinely WRONG seeds falsely read as
# "found" at 0.75 raw match, purely from this base-rate bias, not from any
# real seed-derivation collision. Fixed by requiring the recall on the "1"
# bits specifically (the actual injected signal, not the default-state "0"
# bits) to clear its own threshold, in addition to overall accuracy - a
# wrong seed's coincidental agreement on defaulted-zero bits can no longer
# substitute for actually finding the real injected ones.
ONE_BIT_RECALL_THRESHOLD = 0.8   # fraction of the payload's actual "1" bits
                                   # that must be correctly recovered as 1
ZERO_BIT_PRECISION_THRESHOLD = 0.7  # fraction of the payload's actual "0"
                                   # bits that must be correctly recovered
                                   # as 0 (guards against a read that's
                                   # just "everything looks like a 1")

# lower recall bar for the majority-vote fallback, since it's meant to
# recover a mark that no single noisy copy confirmed alone - still well
# above chance, just not as strict as a single clean pass.
MAJORITY_VOTE_ONE_BIT_RECALL_THRESHOLD = 0.6
MAJORITY_VOTE_ZERO_BIT_PRECISION_THRESHOLD = 0.6


def _score_match(recovered, expected):
    """Returns (passes_one_bit_recall, passes_zero_bit_precision, overall_frac)
    - see the constants above for why raw overall match fraction alone is
    not a safe scoring rule for this payload's bit distribution."""
    one_positions = [i for i, b in enumerate(expected) if b == 1]
    zero_positions = [i for i, b in enumerate(expected) if b == 0]
    one_recall = (sum(1 for i in one_positions if recovered[i] == 1) / len(one_positions)
                  if one_positions else 1.0)
    zero_precision = (sum(1 for i in zero_positions if recovered[i] == 0) / len(zero_positions)
                       if zero_positions else 1.0)
    overall = sum(1 for a, b in zip(recovered, expected) if a == b) / len(expected)
    return one_recall, zero_precision, overall


def _recover_bits_one_pass(audio_t, sr, start, end, freqs):
    """Returns a list of ints (0/1) for this one time-offset's read, one
    entry per payload bit position - or None if this window is unusable.

    A "1" reads as: this bin cluster's measured level sits notably BELOW
    its own neighborhood's robust baseline (a genuine attenuation notch,
    matching what embed_watermark's cluster-drop actually produces).
    Ordinary unmarked content doesn't naturally show deep, sustained drops
    at one exact narrow cluster relative to its own neighborhood - that's
    what makes this detectable at all."""
    segment = audio_t[start:end]
    if len(segment) < STFT_WIN:
        return None
    window = torch.hann_window(STFT_WIN)
    with torch.no_grad():
        S = torch.stft(segment, n_fft=STFT_WIN, hop_length=STFT_HOP,
                       window=window, return_complex=True)
        mag_db = 20 * torch.log10(S.abs() + 1e-8)

    n_frames = S.shape[1]
    half = n_frames // 2
    recovered = []
    for frame_range in [(0, half), (half, n_frames)]:
        f_lo, f_hi = frame_range
        if f_hi <= f_lo:
            recovered.extend([0] * N_FREQ_BINS)
            continue
        sub_mag_db = mag_db[:, f_lo:f_hi]
        for freq in freqs:
            k = _bin_index_for_freq(freq, sr, STFT_WIN)
            if k < NEIGHBOR_HALF_WIDTH or k + NEIGHBOR_HALF_WIDTH >= mag_db.shape[0]:
                recovered.append(0)
                continue
            med_db, mad_db = _neighbor_baseline_db(sub_mag_db, k)
            target_val = sub_mag_db[k, :].mean()
            drop_db = med_db - target_val  # positive when target is QUIETER
            # than its neighborhood baseline - the signature of a real notch
            recovered.append(1 if drop_db.item() > DETECT_DROP_DB_MARGIN else 0)
    return recovered


def detect_watermark(mono_audio, sr, seed=None):
    """Returns (found: bool, version: int|None, detail: dict).

    Checks the fast front-matter copy first (cheap common case: most files
    either clearly have it there or clearly don't). If no single pass clears
    the separate one-bit-recall and zero-bit-precision thresholds, it falls
    back to reading every Fibonacci-spaced copy and MAJORITY-VOTES each bit
    position across all of them - this is real error
    correction, not just "try again": a bit that reads noisy/wrong on one
    copy (a genuinely marked cluster happened to have very little natural
    energy there to begin with, so the drop measurement is noisier) is far
    less likely to read wrong the same way on multiple independent copies
    at different track positions, so combining copies recovers a mark that
    no single noisy copy would have confirmed alone."""
    if seed is None:
        seed = _get_seed()

    audio_t = torch.tensor(mono_audio, dtype=torch.float32)
    n = len(audio_t)
    duration_sec = n / sr

    offsets = derive_time_offsets(seed, duration_sec)
    win_samples = int(MARK_DURATION_SEC * sr)
    versions_to_check = (WATERMARK_VERSION, *LEGACY_WATERMARK_VERSIONS)
    passes_read_by_version = {}

    for candidate_version in versions_to_check:
        freqs = derive_frequencies(seed, version=candidate_version)
        expected_bits = _payload_bits(candidate_version)
        all_reads = []  # per-pass recovered-bit lists for majority vote

        for pass_idx, offset_sec in enumerate(offsets):
            start = int(offset_sec * sr)
            end = min(n, start + win_samples)
            recovered = _recover_bits_one_pass(audio_t, sr, start, end, freqs)
            if recovered is None:
                continue
            all_reads.append(recovered)

            one_recall, zero_precision, match_frac = _score_match(recovered, expected_bits)
            if (one_recall >= ONE_BIT_RECALL_THRESHOLD
                    and zero_precision >= ZERO_BIT_PRECISION_THRESHOLD):
                return True, candidate_version, {
                    "method": "single_pass", "pass_index": pass_idx,
                    "offset_sec": offset_sec, "match_fraction": match_frac,
                }

        passes_read_by_version[candidate_version] = len(all_reads)
        if len(all_reads) >= 2:
            voted = []
            for bit_i in range(len(expected_bits)):
                votes = [read[bit_i] for read in all_reads if bit_i < len(read)]
                voted.append(1 if sum(votes) > len(votes) / 2 else 0)
            one_recall, zero_precision, match_frac = _score_match(voted, expected_bits)
            if (one_recall >= MAJORITY_VOTE_ONE_BIT_RECALL_THRESHOLD
                    and zero_precision >= MAJORITY_VOTE_ZERO_BIT_PRECISION_THRESHOLD):
                return True, candidate_version, {
                    "method": "majority_vote", "n_passes_combined": len(all_reads),
                    "match_fraction": match_frac,
                }

    return False, None, {
        "checked_offsets": offsets,
        "passes_read_by_version": passes_read_by_version,
    }
