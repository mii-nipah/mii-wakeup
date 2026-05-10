from __future__ import annotations

import sys
import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_BLOCK_SIZE = 1_280


class AudioDeviceError(RuntimeError):
    """Raised when microphone input cannot be opened."""


@dataclass(frozen=True)
class MicrophoneConfig:
    device: int | str | None = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    block_size: int = DEFAULT_BLOCK_SIZE


def parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return value


def _load_sounddevice():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise AudioDeviceError(
            "Microphone input needs PortAudio. Install the system PortAudio "
            "library, or use --input with a WAV file or raw stdin."
        ) from exc

    return sd


def format_input_devices() -> str:
    sd = _load_sounddevice()
    devices = sd.query_devices()
    default_input = sd.default.device[0] if sd.default.device else None

    lines: list[str] = []
    for index, device in enumerate(devices):
        input_channels = int(device.get("max_input_channels", 0))
        if input_channels <= 0:
            continue

        marker = "*" if index == default_input else " "
        default_rate = int(device.get("default_samplerate", 0))
        lines.append(
            f"{marker} {index}: {device['name']} "
            f"({input_channels} input channels, {default_rate} Hz default)"
        )

    return "\n".join(lines) if lines else "No input devices found."


def microphone_frames(
    config: MicrophoneConfig,
    *,
    on_overflow: Callable[[], None] | None = None,
) -> Iterator[np.ndarray]:
    sd = _load_sounddevice()

    try:
        stream = sd.RawInputStream(
            samplerate=config.sample_rate,
            blocksize=config.block_size,
            channels=1,
            dtype="int16",
            device=config.device,
        )
    except Exception as exc:  # sounddevice raises several backend-specific errors.
        raise AudioDeviceError(f"Could not open microphone input: {exc}") from exc

    def frames() -> Iterator[np.ndarray]:
        with stream:
            while True:
                data, overflowed = stream.read(config.block_size)
                if overflowed and on_overflow is not None:
                    on_overflow()
                yield np.frombuffer(data, dtype=np.int16).copy()

    return frames()


def wav_frames(path: Path, *, block_size: int = DEFAULT_BLOCK_SIZE) -> Iterator[np.ndarray]:
    try:
        wav_file = wave.open(str(path), "rb")
    except (OSError, wave.Error) as exc:
        raise ValueError(f"Could not open WAV input {path}: {exc}") from exc

    try:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()

        if channels != 1 or sample_width != 2 or sample_rate != DEFAULT_SAMPLE_RATE:
            raise ValueError(
                "WAV input must be mono 16-bit PCM at 16000 Hz "
                f"(got channels={channels}, sample_width={sample_width}, "
                f"sample_rate={sample_rate})."
            )
    except Exception:
        wav_file.close()
        raise

    def frames() -> Iterator[np.ndarray]:
        with wav_file:
            while True:
                data = wav_file.readframes(block_size)
                if not data:
                    return
                if len(data) < block_size * sample_width:
                    data += b"\0" * (block_size * sample_width - len(data))
                yield np.frombuffer(data, dtype=np.int16).copy()

    return frames()


def raw_stdin_frames(*, block_size: int = DEFAULT_BLOCK_SIZE) -> Iterator[np.ndarray]:
    frame_bytes = block_size * np.dtype(np.int16).itemsize
    stream = sys.stdin.buffer

    while True:
        data = stream.read(frame_bytes)
        if not data:
            return
        if len(data) < frame_bytes:
            data += b"\0" * (frame_bytes - len(data))
        yield np.frombuffer(data, dtype=np.int16).copy()
