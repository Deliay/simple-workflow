"""Local audio tool tests (mono / mix / reverb)."""

from __future__ import annotations

import asyncio
import io
import wave

import numpy as np

from simple_workflow.tools import AudioTool


def make_wav(samples: np.ndarray, rate: int = 8000) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(pcm.shape[1])
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(pcm.tobytes())
        return buffer.getvalue()


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as reader:
        channels = reader.getnchannels()
        rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    array = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return array.reshape(-1, channels), rate


def run(tool: AudioTool, params: list[object]) -> bytes:
    return asyncio.run(tool.run(params, ctx=None))  # type: ignore[arg-type]


def test_mono_downsamples_channels() -> None:
    stereo = np.zeros((100, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = -0.5
    out = run(AudioTool(), ["mono", make_wav(stereo)])
    samples, _ = read_wav(out)
    assert samples.shape[1] == 1


def test_mix_sums_inputs() -> None:
    left = np.full((100, 1), 0.2, dtype=np.float32)
    right = np.full((50, 1), 0.1, dtype=np.float32)
    out = run(AudioTool(), ["mix", make_wav(left), make_wav(right)])
    samples, _ = read_wav(out)
    assert samples.shape[0] == 100
    assert abs(float(samples[0, 0]) - 0.3) < 1e-3


def test_reverb_preserves_length() -> None:
    signal = np.zeros((200, 1), dtype=np.float32)
    signal[0, 0] = 1.0
    out = run(AudioTool(), ["reverb", make_wav(signal)])
    samples, _ = read_wav(out)
    assert samples.shape[0] == 200
    # Reverb should spread energy past the initial impulse.
    assert float(np.sum(np.abs(samples[20:]))) > 0


def test_mix_resamples_differing_rates() -> None:
    slow = np.full((100, 1), 0.2, dtype=np.float32)
    fast = np.full((200, 1), 0.1, dtype=np.float32)
    out = run(AudioTool(), ["mix", make_wav(slow, 8000), make_wav(fast, 16000)])
    samples, rate = read_wav(out)
    assert rate == 16000
    # The 8 kHz stream is resampled up to 16 kHz -> 200 frames, matching.
    assert samples.shape[0] == 200
    assert abs(float(samples[0, 0]) - 0.3) < 1e-3


def test_mix_upmixes_mono_to_stereo() -> None:
    mono = np.full((100, 1), 0.2, dtype=np.float32)
    stereo = np.full((100, 2), 0.1, dtype=np.float32)
    out = run(AudioTool(), ["mix", make_wav(mono, 8000), make_wav(stereo, 8000)])
    samples, _ = read_wav(out)
    assert samples.shape[1] == 2


def test_unknown_operation() -> None:
    import pytest

    from simple_workflow.errors import ToolExecutionError

    with pytest.raises(ToolExecutionError, match="unknown operation"):
        run(AudioTool(), ["bogus", make_wav(np.zeros((10, 1), dtype=np.float32))])
