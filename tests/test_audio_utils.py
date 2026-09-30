import numpy as np

from audio_utils import load_audio, resample, save_wav, slice_audio, trim_silence


def test_resample_length_and_identity():
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    assert len(resample(x, 16000, 8000)) == 8000
    assert resample(x, 16000, 16000) is not None
    assert len(resample(x, 8000, 16000)) == 32000


def test_save_and_load_roundtrip_with_resample(tmp_path):
    x = (0.1 * np.sin(np.linspace(0, 200, 8000))).astype(np.float32)
    save_wav(tmp_path / "a.wav", x, 8000)
    y = load_audio(tmp_path / "a.wav", sr=16000)
    assert y.dtype == np.float32 and len(y) == 16000


def test_stereo_is_downmixed(tmp_path):
    import soundfile as sf

    sf.write(str(tmp_path / "st.wav"), np.zeros((1600, 2), dtype=np.float32), 16000)
    assert load_audio(tmp_path / "st.wav").ndim == 1


def test_slice_audio_clamps_and_pads():
    x = np.arange(16000, dtype=np.float32)
    assert len(slice_audio(x, 16000, 0.25, 0.75)) == 8000
    assert len(slice_audio(x, 16000, 0.0, 1.0, pad=0.5)) == 16000  # clamped
    assert len(slice_audio(x, 16000, 0.5, 0.75, pad=0.25)) == 12000  # 0.25s..1.0s


def test_trim_silence_reports_offset():
    sr = 16000
    x = np.zeros(sr * 3, dtype=np.float32)
    x[sr : 2 * sr] = 0.5 * np.sin(np.linspace(0, 3000, sr))
    trimmed, offset = trim_silence(x, sr)
    assert abs(offset - 1.0) < 0.05
    assert abs(len(trimmed) / sr - 1.0) < 0.05


def test_normalize_rms_hits_target_and_caps_gain():
    from audio_utils import normalize_rms, rms

    quiet = (0.003 * np.sin(np.linspace(0, 400, 16000))).astype(np.float32)
    assert abs(rms(normalize_rms(quiet, 0.05)) - 0.05) < 0.002

    hush = (1e-5 * np.sin(np.linspace(0, 400, 16000))).astype(np.float32)
    assert rms(normalize_rms(hush, 0.05, max_gain=50.0)) <= 50 * rms(hush) + 1e-9   # noise not blown up

    assert np.array_equal(normalize_rms(np.zeros(100, np.float32)), np.zeros(100, np.float32))   # silence untouched
    loud = np.full(1000, 0.9, np.float32)
    assert np.abs(normalize_rms(loud, 0.5)).max() <= 0.99                                        # never clips past range
