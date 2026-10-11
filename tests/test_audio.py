"""Local audio tool tests (mono / mix / reverb / amp / mp3)."""

from __future__ import annotations

import asyncio
import io
import wave
import zipfile

import numpy as np
import pytest

from simple_workflow.media import ffmpeg_available
from simple_workflow.tools import AudioTool, ZipTool
from simple_workflow.types import Tagged


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


def test_mix_balances_track_loudness() -> None:
    left = np.full((100, 1), 0.2, dtype=np.float32)
    right = np.full((50, 1), 0.1, dtype=np.float32)
    out = run(AudioTool(), ["mix", make_wav(left), make_wav(right)])
    samples, _ = read_wav(out)
    assert samples.shape[0] == 100
    # Each track is brought to TARGET_RMS before summing: the loud 0.2 track is
    # attenuated and the quiet 0.1 track is left alone, so neither dominates.
    assert abs(float(samples[0, 0]) - 2 * AudioTool.TARGET_RMS) < 1e-3
    assert abs(float(samples[60, 0]) - AudioTool.TARGET_RMS) < 1e-3


def test_mix_keeps_peak_under_ceiling() -> None:
    tracks = [make_wav(np.full((100, 1), 0.5, dtype=np.float32)) for _ in range(20)]
    out = run(AudioTool(), ["mix", *tracks])
    samples, _ = read_wav(out)
    assert float(np.max(np.abs(samples))) <= AudioTool.PEAK_CEILING + 1e-2
    assert float(np.max(np.abs(samples))) > 0.0


def test_reverb_preserves_length() -> None:
    signal = np.zeros((200, 1), dtype=np.float32)
    signal[0, 0] = 1.0
    out = run(AudioTool(), ["reverb", make_wav(signal)])
    samples, _ = read_wav(out)
    assert samples.shape[0] == 200
    # Reverb should spread energy past the initial impulse.
    assert float(np.sum(np.abs(samples[20:]))) > 0


def test_reverb_is_subtle_and_has_predelay() -> None:
    rate = 8000
    signal = np.zeros((4000, 1), dtype=np.float32)
    signal[0, 0] = 1.0
    out = run(AudioTool(), ["reverb", make_wav(signal, rate)])
    samples, _ = read_wav(out)
    pre = int(AudioTool.REVERB_PRE_DELAY * rate)
    # The dry impulse dominates and nothing precedes the reverb tail.
    assert abs(float(samples[0, 0]) - AudioTool.REVERB_DRY) < 1e-2
    assert float(np.max(np.abs(samples[1:pre, 0]))) < 1e-4
    # The tail is present but well below the dry signal (not swamped).
    wet_peak = float(np.max(np.abs(samples[pre:, 0])))
    assert 0.0 < wet_peak < 0.1
    assert float(np.max(np.abs(samples))) <= 1.0


def test_mix_resamples_differing_rates() -> None:
    slow = np.full((100, 1), 0.2, dtype=np.float32)
    fast = np.full((200, 1), 0.1, dtype=np.float32)
    out = run(AudioTool(), ["mix", make_wav(slow, 8000), make_wav(fast, 16000)])
    samples, rate = read_wav(out)
    assert rate == 16000
    # The 8 kHz stream is resampled up to 16 kHz -> 200 frames, matching, and
    # both tracks are loudness-matched to TARGET_RMS before summing.
    assert samples.shape[0] == 200
    assert abs(float(samples[0, 0]) - 2 * AudioTool.TARGET_RMS) < 1e-3


def test_mix_upmixes_mono_to_stereo() -> None:
    mono = np.full((100, 1), 0.2, dtype=np.float32)
    stereo = np.full((100, 2), 0.1, dtype=np.float32)
    out = run(AudioTool(), ["mix", make_wav(mono, 8000), make_wav(stereo, 8000)])
    samples, _ = read_wav(out)
    assert samples.shape[1] == 2


def test_mix_lifts_vocal_role() -> None:
    signal = np.full((400, 1), 0.2, dtype=np.float32)
    plain = read_wav(run(AudioTool(), ["mix", make_wav(signal)]))[0]
    lifted = read_wav(
        run(AudioTool(), ["mix", Tagged(make_wav(signal), role="vocal")])
    )[0]
    ratio = float(np.max(np.abs(lifted))) / float(np.max(np.abs(plain)))
    assert abs(ratio - AudioTool.MIX_VOCAL_GAIN) < 0.02


def test_mix_role_gains() -> None:
    signal = np.full((400, 1), 0.2, dtype=np.float32)
    wav = make_wav(signal)
    base = float(np.max(np.abs(read_wav(run(AudioTool(), ["mix", wav]))[0])))
    for role, gain in (
        ("vocal", AudioTool.MIX_VOCAL_GAIN),
        ("harmony", AudioTool.MIX_HARMONY_GAIN),
        ("inst", AudioTool.MIX_INST_GAIN),
    ):
        out = read_wav(run(AudioTool(), ["mix", Tagged(wav, role=role)]))[0]
        assert abs(float(np.max(np.abs(out))) / base - gain) < 0.02


def test_mix_harmony_ducks_less_than_inst() -> None:
    tool = AudioTool()
    vocal = np.zeros((2000, 1), dtype=np.float32)
    vocal[:1000, 0] = tool.TARGET_RMS * tool.MIX_VOCAL_GAIN
    inst_duck = tool._duck_gain(vocal, rate=8000)
    harmony_duck = tool._duck_gain(
        vocal, rate=8000, depth=tool.MIX_HARMONY_DUCK_DEPTH
    )
    assert float(harmony_duck[:900].mean()) > float(inst_duck[:900].mean())
    assert float(harmony_duck[:900].mean()) < 1.0


def test_mix_duck_gain_dips_under_vocal() -> None:
    tool = AudioTool()
    vocal = np.zeros((2000, 1), dtype=np.float32)
    vocal[:1000, 0] = tool.TARGET_RMS * tool.MIX_VOCAL_GAIN
    duck = tool._duck_gain(vocal, rate=8000)
    # The backing is lowered while the vocal sounds and recovers when it stops.
    assert float(duck[:900].mean()) < 1.0 - AudioTool.MIX_DUCK_DEPTH / 2
    assert float(duck[1100:].mean()) > 0.99
    assert float(duck[:900].mean()) < float(duck[1100:].mean())


def test_mix_emphasises_vocal_over_backing() -> None:
    rate = 8000
    vocal = np.zeros((4000, 1), dtype=np.float32)
    vocal[:2000, 0] = 0.3
    backing = np.full((4000, 1), 0.2, dtype=np.float32)
    out = read_wav(
        run(
            AudioTool(),
            ["mix", Tagged(make_wav(vocal, rate), role="vocal"), make_wav(backing, rate)],
        )
    )[0]
    # While the vocal is present it dominates; afterwards only backing remains.
    assert abs(float(out[500, 0])) > abs(float(out[3500, 0])) * 2


def test_mix_without_a_vocal_tag_is_unchanged() -> None:
    left = np.full((100, 1), 0.2, dtype=np.float32)
    right = np.full((100, 1), 0.1, dtype=np.float32)
    out = read_wav(run(AudioTool(), ["mix", make_wav(left), make_wav(right)]))[0]
    assert abs(float(out[0, 0]) - 2 * AudioTool.TARGET_RMS) < 1e-3


def test_amp_normalises_rms() -> None:
    quiet = np.full((100, 1), 0.25, dtype=np.float32)
    out = run(AudioTool(), ["amp", make_wav(quiet)])
    samples, _ = read_wav(out)
    rms = float(np.sqrt(np.mean(np.square(samples))))
    assert abs(rms - AudioTool.TARGET_RMS) < 1e-2


def test_amp_levels_differing_signals() -> None:
    # Two constant signals at different levels should land at the same RMS.
    quiet = run(AudioTool(), ["amp", make_wav(np.full((100, 1), 0.2, np.float32))])
    loud = run(AudioTool(), ["amp", make_wav(np.full((100, 1), 0.8, np.float32))])
    q, _ = read_wav(quiet)
    loud_samples, _ = read_wav(loud)
    q_rms = float(np.sqrt(np.mean(np.square(q))))
    loud_rms = float(np.sqrt(np.mean(np.square(loud_samples))))
    assert abs(q_rms - loud_rms) < 1e-3


def test_amp_respects_peak_ceiling() -> None:
    # A sparse transient: RMS gain would overshoot, so the peak ceiling kicks in.
    transient = np.zeros((100, 1), dtype=np.float32)
    transient[0, 0] = 0.9
    out = run(AudioTool(), ["amp", make_wav(transient)])
    samples, _ = read_wav(out)
    assert float(np.max(np.abs(samples))) <= AudioTool.PEAK_CEILING + 1e-2


def test_amp_leaves_silence_untouched() -> None:
    silence = np.zeros((100, 1), dtype=np.float32)
    out = run(AudioTool(), ["amp", make_wav(silence)])
    samples, _ = read_wav(out)
    assert float(np.max(np.abs(samples))) == 0.0


def test_unknown_operation() -> None:
    from simple_workflow.errors import ToolExecutionError

    with pytest.raises(ToolExecutionError, match="unknown operation"):
        run(AudioTool(), ["bogus", make_wav(np.zeros((10, 1), dtype=np.float32))])


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not installed")
def test_mp3_operation_encodes() -> None:
    wav = make_wav(np.zeros((8000, 1), dtype=np.float32), 8000)
    out = run(AudioTool(), ["mp3", wav])
    assert out
    assert not out.startswith(b"RIFF")


def test_zip_packs_tagged_inputs() -> None:
    out = run(
        ZipTool(),
        [Tagged(b"one", "a.txt"), Tagged(b"two", "b.txt")],
    )
    with zipfile.ZipFile(io.BytesIO(out)) as archive:
        assert sorted(archive.namelist()) == ["a.txt", "b.txt"]
        assert archive.read("a.txt") == b"one"
        assert archive.read("b.txt") == b"two"


def test_zip_requires_a_file_name() -> None:
    from simple_workflow.errors import ToolExecutionError

    with pytest.raises(ToolExecutionError, match=r"file=\.\.\."):
        run(ZipTool(), [b"untagged"])
