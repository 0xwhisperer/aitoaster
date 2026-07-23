import numpy as np
import torch
import torch.nn.functional as F
import subprocess
import yaml
import onnxruntime as ort
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

with open(MODELS_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)

audio_cfg = config["audio"]
fp_cfg = config["fakeprint"]
SAMPLE_RATE = audio_cfg["sample_rate"]
N_FFT = audio_cfg["n_fft"]
MAX_DURATION = audio_cfg["max_duration"]
FREQ_MIN = fp_cfg["freq_min"]
FREQ_MAX = fp_cfg["freq_max"]
HULL_AREA = fp_cfg["hull_area"]
MAX_DB = fp_cfg["max_db"]
MIN_DB = fp_cfg["min_db"]

freq_bins = np.linspace(0, SAMPLE_RATE / 2, num=(N_FFT // 2) + 1)
FREQ_MASK = (freq_bins >= FREQ_MIN) & (freq_bins <= FREQ_MAX)
FREQ_MASK_IDX = np.where(FREQ_MASK)[0]

w = np.load(MODELS_DIR / "linear_weights.npz")
WEIGHTS = torch.tensor(w["weights"][0], dtype=torch.float32)
BIAS = torch.tensor(w["bias"][0], dtype=torch.float32)

_session = ort.InferenceSession(str(MODELS_DIR / "linear_detector.onnx"))
_input_name = _session.get_inputs()[0].name


def torchaudio_style_spectrogram(audio_1d, n_fft):
    """Reproduce torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2) exactly:
    center-padded STFT with a Hann window, hop_length = n_fft//4 (torchaudio default),
    power spectrogram (magnitude squared)."""
    hop_length = n_fft // 4
    window = torch.hann_window(n_fft, periodic=True)
    spec = torch.stft(
        audio_1d, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, center=True, pad_mode="reflect", return_complex=True,
    )
    power_spec = spec.abs() ** 2
    return power_spec  # [freq_bins, time]


def differentiable_min_filter_1d(x, size):
    """Sliding-window minimum, matching scipy.ndimage.minimum_filter1d(mode='nearest').
    Implemented via -maxpool1d(-x) which has well-defined gradients in PyTorch."""
    pad_left = size // 2
    pad_right = size - 1 - pad_left
    x_padded = F.pad(x.unsqueeze(0).unsqueeze(0), (pad_left, pad_right), mode="replicate")
    neg_min = F.max_pool1d(-x_padded, kernel_size=size, stride=1)
    return (-neg_min).squeeze(0).squeeze(0)


def compute_fakeprint_differentiable(audio_1d):
    """audio_1d: 1D torch tensor, mono, at SAMPLE_RATE. Returns the 3585-dim fakeprint,
    fully differentiable end-to-end."""
    power_spec = torchaudio_style_spectrogram(audio_1d, N_FFT)  # [freq, time]
    spec_db = 10 * torch.log10(torch.clamp(power_spec, min=1e-10, max=1e6))
    mean_spectrum = spec_db.mean(dim=1)  # [freq]

    freq_spectrum = mean_spectrum[FREQ_MASK_IDX]  # [3585]
    hull = differentiable_min_filter_1d(freq_spectrum, HULL_AREA)
    hull = torch.clamp(hull, min=MIN_DB)
    residue = torch.clamp(freq_spectrum - hull, min=0)
    residue = torch.clamp(residue, max=MAX_DB)
    max_val = residue.max() + 1e-6
    fakeprint = residue / max_val
    return fakeprint


def forward_logit_differentiable(audio_1d):
    fakeprint = compute_fakeprint_differentiable(audio_1d)
    logit = torch.dot(WEIGHTS, fakeprint) + BIAS
    return logit


def forward_score_differentiable(audio_1d):
    return torch.sigmoid(forward_logit_differentiable(audio_1d))


def get_real_score(path):
    """Ground truth: run the actual ONNX model via the exact same pipeline as inference.py"""
    import torchaudio
    from scipy.ndimage import minimum_filter1d
    audio, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
    max_samples = MAX_DURATION * SAMPLE_RATE
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    stft = torchaudio.transforms.Spectrogram(n_fft=N_FFT, power=2, normalized=False)
    with torch.no_grad():
        spec = stft(audio)
    spec_db = 10 * torch.log10(torch.clamp(spec, min=1e-10, max=1e6))
    mean_spectrum = spec_db.mean(dim=(0, 2)).cpu().numpy()
    freq_spectrum = mean_spectrum[FREQ_MASK]
    hull = minimum_filter1d(freq_spectrum, size=HULL_AREA, mode="nearest")
    hull = np.clip(hull, MIN_DB, None)
    residue = np.clip(freq_spectrum - hull, 0, None)
    residue = np.clip(residue, 0, MAX_DB)
    max_val = np.max(residue) + 1e-6
    fakeprint = (residue / max_val).astype(np.float32)
    output = _session.run(None, {_input_name: fakeprint.reshape(1, -1)})
    return float(output[0][0, 0])


def load_audio_torch(path, max_seconds=None):
    cmd = ["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    if max_seconds:
        audio = audio[: int(max_seconds * SAMPLE_RATE)]
    return torch.tensor(audio, dtype=torch.float32)


if __name__ == "__main__":
    # verify our differentiable pipeline matches the real model closely
    path = "/Users/daniel/Desktop/audio/northstar.wav"
    audio = load_audio_torch(path, max_seconds=MAX_DURATION)
    with torch.no_grad():
        our_score = forward_score_differentiable(audio).item()
    real_score = get_real_score(path)
    print(f"Our differentiable pipeline score: {our_score:.6f}")
    print(f"Real ONNX model score:             {real_score:.6f}")
    print(f"Difference: {abs(our_score - real_score):.6f}")
