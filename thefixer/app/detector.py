"""
Torch-free reimplementation of the lofcz/ai-music-detector inference pipeline.

Both models (linear fakeprint-logistic-regression, CQT-cepstrum CNN) are
evaluated purely via onnxruntime + numpy/scipy/librosa. No gradient/adversarial
code lives here - this module only SCORES audio, matching the original
inference.py / inference_cnn.py logic exactly.
"""
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from scipy.ndimage import minimum_filter1d
from scipy.fft import dct

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _load_config():
    with open(MODELS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def load_audio_mono(path, sr):
    """Decode any audio file to mono float32 PCM at the given sample rate via ffmpeg."""
    cmd = ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le",
           "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


class LinearDetector:
    """Fakeprint logistic-regression model - exact port of inference.py."""

    def __init__(self, config=None):
        config = config or _load_config()
        audio_cfg = config["audio"]
        fp_cfg = config["fakeprint"]

        self.sample_rate = audio_cfg["sample_rate"]
        self.n_fft = audio_cfg["n_fft"]
        self.max_duration = audio_cfg["max_duration"]

        self.freq_min = fp_cfg["freq_min"]
        self.freq_max = fp_cfg["freq_max"]
        self.hull_area = fp_cfg["hull_area"]
        self.max_db = fp_cfg["max_db"]
        self.min_db = fp_cfg["min_db"]

        freq_bins = np.linspace(0, self.sample_rate / 2, num=(self.n_fft // 2) + 1)
        self.freq_mask = (freq_bins >= self.freq_min) & (freq_bins <= self.freq_max)

        self.session = ort.InferenceSession(str(MODELS_DIR / "linear_detector.onnx"))
        self.input_name = self.session.get_inputs()[0].name
        self.expected_features = self.session.get_inputs()[0].shape[1]

    def _power_spectrogram(self, audio):
        """Matches torchaudio.transforms.Spectrogram(n_fft, power=2, normalized=False)
        EXACTLY as it's actually configured in the real model pipeline and in the
        differentiable reference (linear_differentiable.py's
        torchaudio_style_spectrogram): hop = n_fft // 4, a PERIODIC Hann window
        (not numpy's default symmetric one), and center=True with reflect-mode
        padding (torch.stft's default) - the first frame is centered ON sample 0,
        not starting at it. Getting any of these three wrong silently produces a
        different (and previously verified-wrong: 84.57% vs 0.042% on the same
        test tone) fakeprint, since the whole detector is built on this exact
        spectrogram shape."""
        hop = self.n_fft // 4
        # periodic Hann: numpy's hanning() is symmetric (includes both endpoints);
        # torch.hann_window(..., periodic=True) is the first N samples of an
        # N+1-point symmetric window, which is NOT the same function.
        window = np.hanning(self.n_fft + 1)[:-1]

        # center=True, pad_mode="reflect": reflect-pad n_fft//2 samples on each
        # side before framing, matching torch.stft's default centering behavior.
        pad = self.n_fft // 2
        audio = np.pad(audio, (pad, pad), mode="reflect")

        n = len(audio)
        n_frames = 1 + (n - self.n_fft) // hop if n >= self.n_fft else 0
        if n_frames <= 0:
            padded = np.zeros(self.n_fft, dtype=np.float64)
            padded[:n] = audio
            audio = padded
            n_frames = 1
        power = np.empty((self.n_fft // 2 + 1, n_frames), dtype=np.float64)
        for i in range(n_frames):
            frame = audio[i * hop: i * hop + self.n_fft] * window
            spec = np.fft.rfft(frame)
            power[:, i] = (spec.real ** 2 + spec.imag ** 2)
        return power

    def compute_fakeprint(self, audio_mono, sr):
        if sr != self.sample_rate:
            raise ValueError("audio must already be resampled to model sample_rate")
        max_samples = self.max_duration * self.sample_rate
        if len(audio_mono) > max_samples:
            audio_mono = audio_mono[:max_samples]

        spec = self._power_spectrogram(audio_mono.astype(np.float64))
        spec_db = 10 * np.log10(np.clip(spec, 1e-10, 1e6))
        mean_spectrum = spec_db.mean(axis=1)

        freq_spectrum = mean_spectrum[self.freq_mask]
        hull = minimum_filter1d(freq_spectrum, size=self.hull_area, mode="nearest")
        hull = np.clip(hull, self.min_db, None)
        residue = np.clip(freq_spectrum - hull, 0, None)
        residue = np.clip(residue, 0, self.max_db)
        max_val = np.max(residue) + 1e-6
        return (residue / max_val).astype(np.float32)

    def predict(self, path):
        audio = load_audio_mono(path, self.sample_rate)
        fakeprint = self.compute_fakeprint(audio, self.sample_rate)
        if len(fakeprint) != self.expected_features:
            old_x = np.linspace(0, 1, len(fakeprint))
            new_x = np.linspace(0, 1, self.expected_features)
            fakeprint = np.interp(new_x, old_x, fakeprint).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: fakeprint.reshape(1, -1)})
        probability = float(outputs[0][0, 0])
        return {"probability": probability, "is_ai": probability >= 0.5}

    def predict_array(self, audio_44k, sr_in):
        """Score an in-memory mono array (resampled to model rate via ffmpeg roundtrip)."""
        import tempfile, os, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        try:
            sf.write(tmp_path, audio_44k, sr_in, subtype="PCM_16")
            return self.predict(tmp_path)
        finally:
            os.unlink(tmp_path)


class CNNDetector:
    """CQT-cepstrum CNN model - exact port of inference_cnn.py (ONNX path only)."""

    def __init__(self, config=None):
        config = config or _load_config()
        audio_cfg = config["audio"]
        cqt_cfg = config["cqt"]
        cepstrum_cfg = config["cepstrum"]

        self.sample_rate = audio_cfg["sample_rate"]
        self.segment_seconds = cepstrum_cfg["segment_seconds"]
        self.segment_samples = int(self.segment_seconds * self.sample_rate)

        self.fmin = cqt_cfg["fmin"]
        self.n_bins = cqt_cfg["n_bins"]
        self.bins_per_octave = cqt_cfg["bins_per_octave"]
        self.hop_length = cqt_cfg["hop_length"]
        self.n_coeffs = cepstrum_cfg["n_coeffs"]

        self.session = ort.InferenceSession(str(MODELS_DIR / "cnn_detector.onnx"))
        self.input_name = self.session.get_inputs()[0].name

    def extract_cqt_cepstrum(self, audio):
        import librosa
        cqt = librosa.cqt(
            audio, sr=self.sample_rate, fmin=self.fmin,
            n_bins=self.n_bins, bins_per_octave=self.bins_per_octave,
            hop_length=self.hop_length,
        )
        cqt_mag = np.abs(cqt)
        log_cqt = np.log(cqt_mag + 1e-6)
        cepstrum = dct(log_cqt, type=2, axis=0, norm="ortho")
        return cepstrum[: self.n_coeffs, :].astype(np.float32)

    def extract_segments(self, audio, n_segments):
        segments, positions = [], []
        skip = 5 * self.sample_rate

        if len(audio) > self.segment_samples + 2 * skip:
            start_offset, end_offset = skip, len(audio) - skip
        else:
            start_offset, end_offset = 0, len(audio)

        usable = end_offset - start_offset
        if usable <= self.segment_samples:
            if len(audio) <= self.segment_samples:
                padded = np.zeros(self.segment_samples, dtype=np.float32)
                padded[: len(audio)] = audio
                segments.append(padded)
                positions.append(0)
            else:
                center = len(audio) // 2
                start = max(0, center - self.segment_samples // 2)
                segments.append(audio[start: start + self.segment_samples])
                positions.append(start)
        else:
            available = usable - self.segment_samples
            if n_segments == 1:
                pos_list = [start_offset + available // 2]
            else:
                step = available / (n_segments - 1)
                pos_list = [start_offset + int(i * step) for i in range(n_segments)]
            for start in pos_list:
                segments.append(audio[start: start + self.segment_samples])
                positions.append(start)
        return segments, positions

    def predict_cepstrum_batch(self, cepstra):
        batch = np.stack([c[np.newaxis, :, :] for c in cepstra], axis=0).astype(np.float32)
        output = self.session.run(None, {self.input_name: batch})[0]
        probs = 1 / (1 + np.exp(-output[:, 0]))
        return probs.tolist()

    def predict(self, path, n_segments=5):
        audio = load_audio_mono(path, self.sample_rate)
        segments, positions = self.extract_segments(audio, n_segments)
        cepstra = [self.extract_cqt_cepstrum(seg) for seg in segments]
        probabilities = self.predict_cepstrum_batch(cepstra)
        final_prob = float(np.median(probabilities))
        return {
            "probability": final_prob,
            "is_ai": final_prob > 0.5,
            # The WORST window, exposed alongside the median. The median
            # answers "is this track AI-generated?", which is what this
            # detector is for. It is the wrong question for "is this
            # delivered file certified?": a file can show a passing median
            # while individual windows sit near 100%. Measured on a real
            # delivered file, the median understated the worst window by
            # roughly 260x. Delivery gating uses this; the detector's own
            # verdict is unchanged.
            "worst_segment_prob": float(max(probabilities)) if probabilities else final_prob,
            "segment_probs": probabilities,
            "segment_positions_sec": [p / self.sample_rate for p in positions],
        }


class Scorer:
    """Combined linear + CNN scoring, cached model load."""

    def __init__(self):
        config = _load_config()
        self.linear = LinearDetector(config)
        self.cnn = CNNDetector(config)

    def score(self, path):
        lin = self.linear.predict(path)
        cnn = self.cnn.predict(path)
        return {
            "linear": lin,
            "cnn": cnn,
            "linear_pct": lin["probability"] * 100,
            "cnn_pct": cnn["probability"] * 100,
            "cnn_worst_pct": cnn.get("worst_segment_prob", cnn["probability"]) * 100,
            "passes_linear": lin["probability"] < 0.01,
            "passes_cnn": cnn["probability"] < 0.5,
            # A file is only certified when its WORST window passes, not its
            # median. See worst_segment_prob above.
            "passes_cnn_worst": cnn.get("worst_segment_prob", cnn["probability"]) < 0.5,
            "passes_both": lin["probability"] < 0.01 and cnn["probability"] < 0.5,
        }
