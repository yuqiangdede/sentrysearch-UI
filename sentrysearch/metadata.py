"""Tesla dashcam SEI metadata extraction.

Parses Supplemental Enhancement Information (SEI) NAL units embedded in
Tesla dashcam MP4 files.  Requires firmware 2025.44.25+ and HW3+.
"""

import struct
from typing import Generator, Optional

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import DecodeError

from . import dashcam_pb2


def extract_metadata(video_path: str) -> list[dict]:
    """Extract per-frame Tesla SEI metadata from an MP4 file.

    Returns a list of dicts whose keys match the ``SeiMetadata`` proto fields
    (vehicle_speed_mps, latitude_deg, longitude_deg, blinker_on_left, etc.).

    Returns an empty list when no SEI metadata is found.
    """
    try:
        with open(video_path, "rb") as fp:
            offset, size = _find_mdat(fp)
            return [
                MessageToDict(meta, preserving_proto_field_name=True)
                for meta in _iter_sei_messages(fp, offset, size)
            ]
    except (RuntimeError, OSError):
        return []


# -- internal helpers (adapted from dashcam/sei_extractor.py) ---------------


def _find_mdat(fp) -> tuple[int, int]:
    """Return (offset, size) for the first mdat atom."""
    fp.seek(0)
    while True:
        header = fp.read(8)
        if len(header) < 8:
            raise RuntimeError("mdat atom not found")
        size32, atom_type = struct.unpack(">I4s", header)
        if size32 == 1:
            large = fp.read(8)
            if len(large) != 8:
                raise RuntimeError("truncated extended atom size")
            atom_size = struct.unpack(">Q", large)[0]
            header_size = 16
        else:
            atom_size = size32 if size32 else 0
            header_size = 8
        if atom_type == b"mdat":
            payload_size = atom_size - header_size if atom_size else 0
            return fp.tell(), payload_size
        if atom_size < header_size:
            raise RuntimeError("invalid MP4 atom size")
        fp.seek(atom_size - header_size, 1)


def _iter_nals(fp, offset: int, size: int) -> Generator[bytes, None, None]:
    """Yield SEI user-data-unregistered NAL units from the mdat atom."""
    NAL_ID_SEI = 6
    NAL_SEI_ID_USER_DATA_UNREGISTERED = 5

    fp.seek(offset)
    consumed = 0
    while size == 0 or consumed < size:
        header = fp.read(4)
        if len(header) < 4:
            break
        nal_size = struct.unpack(">I", header)[0]
        if nal_size < 2:
            fp.seek(nal_size, 1)
            consumed += 4 + nal_size
            continue

        first_two = fp.read(2)
        if len(first_two) != 2:
            break

        if (first_two[0] & 0x1F) != NAL_ID_SEI or first_two[1] != NAL_SEI_ID_USER_DATA_UNREGISTERED:
            fp.seek(nal_size - 2, 1)
            consumed += 4 + nal_size
            continue

        rest = fp.read(nal_size - 2)
        if len(rest) != nal_size - 2:
            break
        consumed += 4 + nal_size
        yield first_two + rest


def _extract_proto_payload(nal: bytes) -> Optional[bytes]:
    """Extract protobuf payload from a SEI NAL unit."""
    if not isinstance(nal, bytes) or len(nal) < 2:
        return None
    for i in range(3, len(nal) - 1):
        byte = nal[i]
        if byte == 0x42:
            continue
        if byte == 0x69:
            if i > 2:
                return _strip_emulation_prevention_bytes(nal[i + 1 : -1])
            break
        break
    return None


def _strip_emulation_prevention_bytes(data: bytes) -> bytes:
    """Remove H.264 emulation prevention bytes (0x03 after 0x00 0x00)."""
    stripped = bytearray()
    zero_count = 0
    for byte in data:
        if zero_count >= 2 and byte == 0x03:
            zero_count = 0
            continue
        stripped.append(byte)
        zero_count = 0 if byte != 0 else zero_count + 1
    return bytes(stripped)


def _iter_sei_messages(fp, offset: int, size: int):
    """Yield parsed ``SeiMetadata`` proto messages."""
    for nal in _iter_nals(fp, offset, size):
        payload = _extract_proto_payload(nal)
        if not payload:
            continue
        meta = dashcam_pb2.SeiMetadata()
        try:
            meta.ParseFromString(payload)
        except DecodeError:
            continue
        yield meta
