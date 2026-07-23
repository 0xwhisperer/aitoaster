import numpy as np
import torch
import subprocess
import yaml
import onnxruntime as ort
import librosa
from scipy.fft import dct
from onnx2torch import convert
from nnAudio.features import CQT
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

with open(MODELS_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)

SR = config["audio"]["sample_rate"]  # 16000
CQT_CFG = config["cqt"]
N_COEFFS = config["cepstrum"]["n_coeffs"]
SEGMENT_SECONDS = config["cepstrum"]["segment_seconds"]
SEGMENT_SAMPLES = int(SEGMENT_SECONDS * SR)
SKIP_SAMPLES = 5 * SR

_model = convert(str(MODELS_DIR / "cnn_detector.onnx"))
_model.eval()
for p in _model.parameters():
    p.requires_grad = False

_cqt_transform = CQT(sr=SR, fmin=CQT_CFG["fmin"], n_bins=CQT_CFG["n_bins"],
                      bins_per_octave=CQT_CFG["bins_per_octave"], hop_length=CQT_CFG["hop_length"],
                      output_format="Magnitude", verbose=False)
for p in _cqt_transform.parameters():
    p.requires_grad = False


def _dct_matrix(N):
    n = torch.arange(N).float()
    k = torch.arange(N).float().unsqueeze(1)
    basis = torch.cos(np.pi / N * (n + 0.5) * k)
    basis[0, :] *= 1 / np.sqrt(N)
    basis[1:, :] *= np.sqrt(2 / N)
    return basis


_dctmat = _dct_matrix(CQT_CFG["n_bins"])


def forward_logit_differentiable(audio_1d):
    """audio_1d: [1, T] -> raw logit (pre-sigmoid), differentiable end-to-end
    via nnAudio's CQT (a verified-close surrogate for librosa's CQT)."""
    cqt_mag = _cqt_transform(audio_1d)
    log_cqt = torch.log(cqt_mag + 1e-6)
    cepstrum = torch.einsum('kb,cbt->ckt', _dctmat, log_cqt)
    cepstrum = cepstrum[:, :N_COEFFS, :]
    model_input = cepstrum.unsqueeze(1)
    logit = _model(model_input)
    return logit[0, 0]


def forward_score_differentiable(audio_1d):
    return torch.sigmoid(forward_logit_differentiable(audio_1d))


_session = ort.InferenceSession(
    str(MODELS_DIR / "cnn_detector.onnx"),
    providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name


def get_real_score_segment(audio_np):
    """Ground truth: exact librosa-based pipeline the real model uses."""
    cqt = librosa.cqt(audio_np, sr=SR, fmin=CQT_CFG["fmin"], n_bins=CQT_CFG["n_bins"],
                       bins_per_octave=CQT_CFG["bins_per_octave"], hop_length=CQT_CFG["hop_length"])
    cqt_mag = np.abs(cqt)
    log_cqt = np.log(cqt_mag + 1e-6)
    cepstrum = dct(log_cqt, type=2, axis=0, norm='ortho')[:N_COEFFS, :]
    batch = cepstrum[np.newaxis, np.newaxis, :, :].astype(np.float32)
    output = _session.run(None, {_input_name: batch})[0]
    return 1 / (1 + np.exp(-output[0, 0]))


def load_audio_mono(path, sr=SR):
    cmd = ["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def get_real_evaluator_segments(audio_np, n_segments=5):
    """Reproduce the real detector's own segment-selection logic exactly."""
    n = len(audio_np)
    if n > SEGMENT_SAMPLES + 2 * SKIP_SAMPLES:
        start_offset = SKIP_SAMPLES
        end_offset = n - SKIP_SAMPLES
    else:
        start_offset = 0
        end_offset = n
    usable_length = end_offset - start_offset
    available = usable_length - SEGMENT_SAMPLES
    if n_segments == 1:
        positions = [start_offset + available // 2]
    else:
        step = available / (n_segments - 1)
        positions = [start_offset + int(i * step) for i in range(n_segments)]
    return positions


if __name__ == "__main__":
    path = "/Users/daniel/Desktop/audio/northstar_cleaned.wav"
    audio = load_audio_mono(path)
    positions = get_real_evaluator_segments(audio)
    print(f"real evaluator segment positions: {[f'{p/SR:.1f}s' for p in positions]}")

    for pos in positions:
        seg = audio[pos:pos + SEGMENT_SAMPLES]
        seg_t = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            ours = forward_score_differentiable(seg_t).item()
        real = get_real_score_segment(seg)
        print(f"  segment @ {pos/SR:.1f}s: ours={ours:.5f}  real={real:.5f}  diff={abs(ours-real):.5f}")
