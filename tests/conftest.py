"""Shared test fixtures: a registry of deterministic stub tools."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from typing import Any

import pytest

from simple_workflow import Param, Signature, Tool, ToolRegistry, ValueType
from simple_workflow.tools import UnzipTool

TEXT = ValueType.TEXT
BINARY = ValueType.BINARY

LONG_SCRIPT = """\
[bv:BV1TqaR67E7f] -> band-roformer[msst:Kim_MelBandRoformer.ckpt]
[var:band-roformer] -> vocals[unzip:vocals.wav]
[var:vocals] -> karaoke-roformer[msst:bs_roformer_karaoke_frazer_becruily.ckpt]
[var:karaoke-roformer] -> vocal[unzip:Vocals.wav]
[var:vocal] -> mono-vocal[audio:mono]
[var:mono-vocal] -> dereverb-vocal[msst:dereverb_room_anvuew_sdr_13.7432.ckpt]
[var:dereverb-vocal] -> noreverb-vocal[unzip:noreverb]
[var:noreverb-vocal] -> viridis-vocal[rvc:viridis-v2_200e_5000s.pth]
[var:viridis-vocal] -> reverb-viridis-vocal[audio:reverb]
[var:band-roformer] -> bands[unzip:other.wav]
[var:karaoke-roformer] -> harmony[unzip:Instrumental.wav]
[var:viridis-vocal, bands, harmony] -> result[audio:mix]
[var:result] -> [final]
"""

SHORT_SCRIPT = """\
[bv:BV1TqaR67E7f]
-> band-roformer[msst:Kim_MelBandRoformer.ckpt]
-> [unzip:vocals.wav]
-> karaoke-roformer[msst:bs_roformer_karaoke_frazer_becruily.ckpt]
-> [unzip:Vocals.wav]
-> [audio:mono]
-> [msst:dereverb_room_anvuew_sdr_13.7432.ckpt]
-> [unzip:noreverb]
-> viridis-vocal[rvc:viridis-v2_200e_5000s.pth]
-> reverb-viridis-vocal[audio:reverb]
[var:band-roformer] -> bands[unzip:other.wav]
[var:karaoke-roformer] -> harmony[unzip:Instrumental.wav]
[var:viridis-vocal, var:bands, var:harmony]
-> result[audio:mix]
-> [final]
"""

# The literal shorthand from the specification.  Note that it calls the RVC
# step anonymously and then references `viridis-vocal`, so that name is never
# bound.  Kept here to document the (ambiguous) source text.
SHORT_SCRIPT_SPEC = SHORT_SCRIPT.replace(
    "-> viridis-vocal[rvc:viridis-v2_200e_5000s.pth]",
    "-> [rvc:viridis-v2_200e_5000s.pth]",
)


def make_zip(**members: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


MSST_ZIP = make_zip(
    **{
        "vocals.wav": b"V",
        "other.wav": b"O",
        "Vocals.wav": b"K",
        "Instrumental.wav": b"I",
        "noreverb": b"N",
    }
)


class StubTool(Tool):
    def __init__(
        self,
        name: str,
        signature: Signature,
        func: Callable[..., Any],
    ) -> None:
        super().__init__(name, signature)
        self._func = func
        self.calls: list[list[Any]] = []

    async def run(self, params: list[Any], ctx: Any) -> Any:  # noqa: ANN401
        self.calls.append(list(params))
        return self._func(*params)


def build_stub_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        StubTool(
            "bv",
            Signature(params=(Param("id", TEXT),), output=BINARY),
            lambda video_id: f"raw:{video_id}".encode(),
        )
    )
    registry.register(
        StubTool(
            "msst",
            Signature(params=(Param("model", TEXT), Param("audio", BINARY)), output=BINARY),
            lambda model, audio: MSST_ZIP,
        )
    )
    registry.register(
        StubTool(
            "rvc",
            Signature(params=(Param("model", TEXT), Param("audio", BINARY)), output=BINARY),
            lambda model, audio: b"VIRIDIS",
        )
    )
    registry.register(
        StubTool(
            "audio",
            Signature(
                params=(Param("op", TEXT),),
                variadic=Param("inputs", BINARY),
                output=BINARY,
            ),
            lambda op, *streams: f"{op}:".encode() + b"|".join(streams),
        )
    )
    registry.register(UnzipTool())
    return registry


@pytest.fixture
def registry() -> ToolRegistry:
    return build_stub_registry()
