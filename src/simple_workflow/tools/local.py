"""Local (in-process) tools: ``unzip``, ``zip`` and ``audio``.

``audio`` implements several operations on 16-bit PCM WAV data using only the
standard library ``wave`` module and ``numpy``:

* ``mono``   - down-mix every input to a single channel,
* ``reverb`` - apply a short, damped, dry-dominant convolution reverb tuned
  for pop material (pre-delay, high-frequency damping, rumble high-pass),
* ``mix``    - match every stream's loudness, then sum them. The ``mix`` tag
  picks a role: ``vocal`` leads, ``harmony`` supports and ``inst`` (or no tag)
  is the bed; the lead side-chain ducks the other roles under it,
* ``amp``    - automatic gain: RMS-normalise to a target loudness, with a
  peak ceiling as a fallback so the result never clips,
* ``mp3``    - encode the input to MP3 (ABR/VBR V0) through ffmpeg.

``zip`` packs every input into a single archive.  Each input must have been
tagged with a ``{file=name}`` key, which supplies the member's filename.
"""

from __future__ import annotations

import io
import os
import zipfile
from typing import Any

import numpy as np

from ..errors import ToolExecutionError
from ..media import to_mp3, to_wav
from ..signature import Param, Signature
from ..types import ValueType
from .base import RunContext, Tool

_TEXT = ValueType.TEXT
_BINARY = ValueType.BINARY


class UnzipTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            "unzip",
            Signature(
                params=(Param("path", _TEXT), Param("archive", _BINARY)),
                output=_BINARY,
            ),
        )

    async def run(self, params: list[Any], ctx: RunContext) -> bytes:  # noqa: ANN401
        path, archive = params[0], params[1]
        try:
            zf = zipfile.ZipFile(io.BytesIO(bytes(archive)))
        except zipfile.BadZipFile as exc:
            raise ToolExecutionError(f"unzip: not a valid zip archive: {exc}") from exc
        with zf:
            member = _find_member(zf.namelist(), path)
            if member is None:
                raise ToolExecutionError(
                    f"unzip: {path!r} not found in archive "
                    f"(members: {', '.join(zf.namelist()[:20])})"
                )
            return zf.read(member)


def _find_member(names: list[str], wanted: str) -> str | None:
    if wanted in names:
        return wanted
    lower = wanted.lower()
    for name in names:
        if name.lower() == lower:
            return name
    base = os.path.basename(lower)
    for name in names:
        if os.path.basename(name).lower() == base:
            return name
    # Fall back to matching on the stem, ignoring extensions, so that a request
    # for "noreverb" matches the archive member "noreverb.wav".
    want_stem = os.path.splitext(base)[0]
    for name in names:
        if os.path.splitext(os.path.basename(name).lower())[0] == want_stem:
            return name
    return None


class ZipTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            "zip",
            Signature(variadic=Param("files", _BINARY), output=_BINARY),
        )

    async def run(self, params: list[Any], ctx: RunContext) -> bytes:  # noqa: ANN401
        if not params:
            raise ToolExecutionError("zip: requires at least one input")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for index, value in enumerate(params):
                filename = getattr(value, "filename", None)
                if not filename:
                    raise ToolExecutionError(
                        f"zip: input {index + 1} has no {{file=...}} tag"
                    )
                archive.writestr(str(filename), bytes(value))
        return buffer.getvalue()


class AudioTool(Tool):
    TARGET_RMS = 0.1  # target RMS amplitude for ``amp`` (~ -20 dBFS)
    PEAK_CEILING = 0.95  # ``amp`` falls back to this peak when RMS would clip
    SILENCE_FLOOR = 1e-4  # do not boost streams quieter than this RMS
    # Pop-friendly reverb: a short, damped tail with a touch of pre-delay so the
    # dry performance stays clear and the effect supports rather than swamps it.
    REVERB_PRE_DELAY = 0.02  # seconds of silence before the tail starts
    REVERB_RT60 = 0.8  # seconds for the tail to decay by 60 dB
    REVERB_DAMP_HZ = 4000.0  # high-frequency damping (smoother, less sibilant)
    REVERB_HIGHPASS_HZ = 150.0  # remove rumble that would muddy the mix
    REVERB_DRY = 0.9  # dry signal level
    REVERB_WET = 0.25  # reverb level, well below the dry signal
    # ``mix`` roles: vocals lead, harmony supports, everything else is the bed.
    MIX_VOCAL_GAIN = 1.5  # lead vocal (~ +3.5 dB)
    MIX_HARMONY_GAIN = 1.0  # harmony vocal, below the lead
    MIX_INST_GAIN = 0.85  # instrumental bed (~ -1.4 dB)
    MIX_DUCK_DEPTH = 0.4  # bed dips up to ~ -4.4 dB under the lead
    MIX_HARMONY_DUCK_DEPTH = 0.2  # harmony dips less than the bed
    MIX_DUCK_WINDOW = 0.03  # envelope window in seconds for the duck

    def __init__(self) -> None:
        super().__init__(
            "audio",
            Signature(
                params=(Param("op", _TEXT),),
                variadic=Param("inputs", _BINARY),
                output=_BINARY,
            ),
        )

    async def run(self, params: list[Any], ctx: RunContext) -> bytes:  # noqa: ANN401
        op = str(params[0])
        raw = list(params[1:])
        streams = [bytes(p) for p in raw]
        roles = [getattr(p, "role", None) for p in raw]
        if not streams:
            raise ToolExecutionError(f"audio:{op} requires at least one input stream")
        try:
            if op == "mono":
                return self._mono(streams)
            if op == "reverb":
                return self._reverb(streams)
            if op == "mix":
                return self._mix(streams, roles)
            if op == "amp":
                return self._amp(streams)
            if op == "mp3":
                return self._mp3(streams)
            if op in ("decode", "to-wav", "wav"):
                return self._decode(streams)
        except ToolExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolExecutionError(f"audio:{op} failed: {exc}") from exc
        raise ToolExecutionError(
            f"audio: unknown operation {op!r} "
            "(expected one of: mono, reverb, mix, amp, mp3, decode)"
        )

    def _decode(self, streams: list[bytes]) -> bytes:
        return to_wav(streams[0])

    def _mp3(self, streams: list[bytes]) -> bytes:
        return to_mp3(streams[0])

    def _mono(self, streams: list[bytes]) -> bytes:
        samples, rate = _decode_wav(streams[0])
        mono = samples.mean(axis=1, keepdims=True)
        return _encode_wav(mono, rate)

    def _mix(
        self, streams: list[bytes], roles: list[str | None] | None = None
    ) -> bytes:
        roles = list(roles) if roles is not None else [None] * len(streams)
        decoded = [_decode_wav(s) for s in streams]
        rate = max(stream_rate for _, stream_rate in decoded)
        channels = max(samples.shape[1] for samples, _ in decoded)
        prepared: list[np.ndarray] = []
        for samples, stream_rate in decoded:
            aligned = _resample(samples, stream_rate, rate)
            if aligned.shape[1] < channels:
                if aligned.shape[1] == 1:
                    aligned = np.repeat(aligned, channels, axis=1)
                else:
                    aligned = np.pad(aligned, ((0, 0), (0, channels - aligned.shape[1])))
            prepared.append(aligned)

        # Roles: ``{mix=vocal}`` leads, ``{mix=harmony}`` supports and anything
        # else (``{mix=inst}`` or untagged) is the bed.  Without any role the mix
        # stays a plain equal-loudness sum, so existing pipelines are unaffected.
        has_role = any(role is not None for role in roles)
        has_vocal = any(role == "vocal" for role in roles)
        length = max(samples.shape[0] for samples in prepared)
        vocal = np.zeros((length, channels), dtype=np.float32)
        harmony = np.zeros((length, channels), dtype=np.float32)
        inst = np.zeros((length, channels), dtype=np.float32)
        for samples, role in zip(prepared, roles, strict=True):
            track = samples * self._loudness_gain(samples)
            if has_role and role == "vocal":
                track = track * self.MIX_VOCAL_GAIN
                target = vocal
            elif has_role and role == "harmony":
                track = track * self.MIX_HARMONY_GAIN
                target = harmony
            else:
                if has_role:
                    track = track * self.MIX_INST_GAIN
                target = inst
            target[: samples.shape[0]] += track

        if has_vocal:
            if np.any(inst):
                inst = inst * self._duck_gain(vocal, rate)[:, None]
            if np.any(harmony):
                harmony = harmony * self._duck_gain(
                    vocal, rate, self.MIX_HARMONY_DUCK_DEPTH
                )[:, None]
        mixed = vocal + harmony + inst
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > self.PEAK_CEILING:
            mixed *= self.PEAK_CEILING / peak
        return _encode_wav(mixed, rate)

    def _duck_gain(
        self, vocal: np.ndarray, rate: int, depth: float | None = None
    ) -> np.ndarray:
        """Time-varying gain that dips while the vocal is sounding."""
        depth = self.MIX_DUCK_DEPTH if depth is None else depth
        mono = vocal.mean(axis=1)
        window = max(1, int(self.MIX_DUCK_WINDOW * rate))
        envelope = np.sqrt(np.maximum(_smooth(np.square(mono), window), 0.0))
        reference = self.TARGET_RMS * self.MIX_VOCAL_GAIN
        level = np.clip(envelope / (reference + 1e-9), 0.0, 1.0)
        duck = 1.0 - depth * level
        return _smooth(duck, window)

    def _loudness_gain(self, samples: np.ndarray) -> float:
        """Gain bringing ``samples`` to ``TARGET_RMS`` without clipping."""
        if samples.size == 0:
            return 1.0
        rms = float(np.sqrt(np.mean(np.square(samples))))
        if rms < self.SILENCE_FLOOR:
            return 1.0
        gain = self.TARGET_RMS / rms
        # Peak ceiling as a fallback: if the RMS gain would push a transient
        # past the ceiling, trade loudness for a clean, unclipped result.
        peak = float(np.max(np.abs(samples)))
        if peak > 0.0 and peak * gain > self.PEAK_CEILING:
            gain = self.PEAK_CEILING / peak
        return gain

    def _amp(self, streams: list[bytes]) -> bytes:
        samples, rate = _decode_wav(streams[0])
        return _encode_wav(samples * self._loudness_gain(samples), rate)

    def _reverb(self, streams: list[bytes]) -> bytes:
        samples, rate = _decode_wav(streams[0])
        impulse = _make_impulse(rate)
        wet = np.empty_like(samples)
        for channel in range(samples.shape[1]):
            wet[:, channel] = _fft_convolve(samples[:, channel], impulse)
        output = self.REVERB_DRY * samples + self.REVERB_WET * wet
        peak = float(np.max(np.abs(output))) if output.size else 0.0
        if peak > self.PEAK_CEILING:
            output *= self.PEAK_CEILING / peak
        return _encode_wav(output, rate)


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------


def _decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(io.BytesIO(data), "rb") as reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())

    if width == 1:
        array = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        array = (array - 128.0) / 128.0
    elif width == 2:
        array = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        values = (values ^ 0x800000) - 0x800000
        array = values.astype(np.float32) / 8388608.0
    elif width == 4:
        array = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ToolExecutionError(f"unsupported WAV sample width: {width} bytes")

    if channels <= 0:
        raise ToolExecutionError("WAV file reports zero channels")
    return array.reshape(-1, channels), rate


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler used to align mixed streams."""
    if src_rate == dst_rate or samples.shape[0] == 0:
        return samples
    n_out = max(1, int(round(samples.shape[0] * dst_rate / src_rate)))
    src_x = np.linspace(0.0, 1.0, samples.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, 1.0, n_out, endpoint=False)
    out = np.empty((n_out, samples.shape[1]), dtype=np.float32)
    for channel in range(samples.shape[1]):
        out[:, channel] = np.interp(dst_x, src_x, samples[:, channel])
    return out


def _encode_wav(samples: np.ndarray, rate: int) -> bytes:
    import wave

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(pcm.shape[1])
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(pcm.tobytes())
        return buffer.getvalue()


def _make_impulse(rate: int) -> np.ndarray:
    """Build a short, damped reverb impulse tuned for pop material.

    The tail is exponentially decaying (``REVERB_RT60``), smoothed to damp the
    highs instead of a bright noise wash, and high-passed to keep low rumble out
    of the mix.  It starts after a short pre-delay and is energy-normalised, so
    the wet signal sits at a predictable level under the dry one.
    """
    pre = max(1, int(AudioTool.REVERB_PRE_DELAY * rate))
    tail = max(1, int(2.0 * AudioTool.REVERB_RT60 * rate))
    t = np.arange(tail, dtype=np.float32) / rate
    envelope: np.ndarray = np.exp(-6.907755 * t / AudioTool.REVERB_RT60)
    rng = np.random.default_rng(0)
    noise: np.ndarray = rng.standard_normal(tail).astype(np.float32)
    noise = _high_pass(_low_pass(noise, rate, AudioTool.REVERB_DAMP_HZ), rate,
                       AudioTool.REVERB_HIGHPASS_HZ)
    impulse = np.zeros(pre + tail, dtype=np.float32)
    impulse[pre:] = envelope * noise
    energy = float(np.sqrt(np.sum(np.square(impulse))))
    if energy > 0.0:
        impulse /= energy
    return impulse


def _low_pass(signal: np.ndarray, rate: int, cutoff_hz: float) -> np.ndarray:
    """Gentle FIR low-pass: a Hann-windowed moving average."""
    if signal.size == 0 or cutoff_hz <= 0.0:
        return signal
    width = max(1, int(rate / cutoff_hz))
    kernel = np.hanning(width + 1).astype(np.float32)
    total = float(kernel.sum())
    if total == 0.0:
        return signal
    filtered: np.ndarray = np.convolve(signal, kernel / total, mode="same")
    return filtered


def _high_pass(signal: np.ndarray, rate: int, cutoff_hz: float) -> np.ndarray:
    """One-pole-style high-pass: remove the low-frequency moving average."""
    if signal.size == 0 or cutoff_hz <= 0.0:
        return signal
    return np.asarray(signal - _low_pass(signal, rate, cutoff_hz), dtype=np.float32)


def _smooth(signal: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average via prefix sums (O(n) regardless of window)."""
    if window <= 1 or signal.size == 0:
        return np.asarray(signal, dtype=np.float32)
    pad = window // 2
    padded = np.pad(signal, (pad, window - 1 - pad), mode="edge")
    cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    smoothed = (cumulative[window:] - cumulative[:-window]) / window
    return smoothed.astype(np.float32)


def _fft_convolve(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    size = len(signal) + len(impulse) - 1
    n = 1 << (size - 1).bit_length()
    spectrum = np.fft.rfft(signal, n) * np.fft.rfft(impulse, n)
    result: np.ndarray = np.asarray(np.fft.irfft(spectrum, n)[: len(signal)], dtype=np.float32)
    return result


def local_tools() -> list[Tool]:
    return [UnzipTool(), ZipTool(), AudioTool()]
