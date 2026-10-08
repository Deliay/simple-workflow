"""Audio container helpers (ffmpeg-backed).

The MSST service only accepts formats it can decode (wav/flac/...).  Bilibili's
``bv`` tool returns an MP4/M4A (AAC) stream, so the audio API tools transcode
non-WAV inputs to 16-bit PCM WAV first.  Decoding is done through ffmpeg pipes;
no temporary files are written.
"""

from __future__ import annotations

import shutil
import subprocess

from .errors import ToolExecutionError

_FFMPEG = "ffmpeg"


def is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def ffmpeg_available() -> bool:
    return shutil.which(_FFMPEG) is not None


def to_wav(
    data: bytes,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> bytes:
    """Transcode arbitrary audio bytes into PCM WAV using ffmpeg."""
    if is_wav(data) and sample_rate is None and channels is None:
        return data
    if not ffmpeg_available():
        raise ToolExecutionError(
            "ffmpeg is required to decode this audio format but was not found on PATH"
        )
    command = [
        _FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "wav",
        "-acodec",
        "pcm_s16le",
    ]
    if sample_rate is not None:
        command += ["-ar", str(sample_rate)]
    if channels is not None:
        command += ["-ac", str(channels)]
    command += ["pipe:1"]

    process = subprocess.run(command, input=data, capture_output=True)
    if process.returncode != 0 or not process.stdout:
        detail = process.stderr.decode("utf-8", errors="replace")[:300]
        raise ToolExecutionError(f"ffmpeg failed to decode audio: {detail}")
    return process.stdout


def wav_info(data: bytes) -> tuple[int, int] | None:
    """Return ``(sample_rate, channels)`` for a WAV stream, else ``None``."""
    if not is_wav(data):
        return None
    import io
    import wave

    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            return reader.getframerate(), reader.getnchannels()
    except Exception:  # noqa: BLE001 - malformed header, fall back to ffmpeg
        return None


def ensure_wav(
    data: bytes,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> bytes:
    """Convert ``data`` to PCM WAV, transcoding with ffmpeg when necessary.

    A stream that is already a WAV with the requested rate/channel layout is
    returned unchanged.  Anything else must be decoded by ffmpeg; if that is
    impossible the error is propagated instead of silently forwarding bytes the
    downstream API cannot decode (e.g. the M4A returned by ``bv``).
    """
    info = wav_info(data)
    if info is not None:
        rate_ok = sample_rate is None or info[0] == sample_rate
        channels_ok = channels is None or info[1] == channels
        if rate_ok and channels_ok:
            return data
    try:
        return to_wav(data, sample_rate=sample_rate, channels=channels)
    except ToolExecutionError:
        # A WAV whose header ``wave`` cannot parse is still a WAV; keep it.
        if is_wav(data):
            return data
        raise
