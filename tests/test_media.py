"""Media/transcoding tests."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from simple_workflow.media import ensure_wav, ffmpeg_available, is_wav, to_wav, wav_info

from .test_audio import make_wav


def test_is_wav() -> None:
    assert is_wav(make_wav(np.zeros((10, 1), dtype=np.float32)))
    assert not is_wav(b"\x00\x00\x00\x20ftypiso5")


def test_ensure_wav_passthrough() -> None:
    wav = make_wav(np.zeros((10, 1), dtype=np.float32))
    assert ensure_wav(wav) is wav


def test_ensure_wav_raises_for_undecodable_without_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import simple_workflow.media as media
    from simple_workflow.errors import ToolExecutionError

    monkeypatch.setattr(media, "ffmpeg_available", lambda: False)
    # An M4A-style payload with no ffmpeg must not be forwarded unchanged.
    with pytest.raises(ToolExecutionError, match="ffmpeg"):
        ensure_wav(b"\x00\x00\x00\x20ftypiso5", channels=1)


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not installed")
def test_ensure_wav_downmixes_when_channels_requested() -> None:
    stereo = make_wav(np.zeros((100, 2), dtype=np.float32))
    converted = ensure_wav(stereo, channels=1)
    info = wav_info(converted)
    assert info is not None
    assert info[1] == 1


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not installed")
def test_to_wav_decodes_aac() -> None:
    wav = make_wav(np.zeros((8000, 2), dtype=np.float32), 8000)
    m4a = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "mp4", "-c:a", "aac", "-movflags", "+frag_keyframe+empty_moov", "pipe:1"],
        input=wav,
        capture_output=True,
        check=True,
    ).stdout
    assert not is_wav(m4a)
    decoded = to_wav(m4a)
    assert is_wav(decoded)
