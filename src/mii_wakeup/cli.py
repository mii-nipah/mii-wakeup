from __future__ import annotations

import argparse
import signal
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from . import __version__
from .audio import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_SAMPLE_RATE,
    AudioDeviceError,
    MicrophoneConfig,
    format_input_devices,
    microphone_frames,
    parse_device,
    raw_stdin_frames,
    wav_frames,
)
from .detection import DetectionConfig, iter_wake_events
from .openwakeword_adapter import ModelLoadError, load_openwakeword_model
from .output import render_event


def main(argv: list[str] | None = None) -> int:
    _prefer_default_sigpipe()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        try:
            print(format_input_devices())
        except AudioDeviceError as exc:
            parser.exit(2, f"mii-wakeup: {exc}\n")
        return 0

    try:
        frames = _audio_frames(args)
        model = load_openwakeword_model(
            args.model,
            vad_threshold=args.vad_threshold,
            speex_noise_suppression=args.speex_noise_suppression,
        )
    except (AudioDeviceError, ModelLoadError, ValueError) as exc:
        parser.exit(2, f"mii-wakeup: {exc}\n")

    if not args.quiet:
        _status(args, model)

    config = DetectionConfig(
        threshold=args.threshold,
        cooldown_seconds=args.cooldown,
        max_events=None if args.continuous else 1,
    )

    try:
        events = iter_wake_events(
            model,
            frames,
            config,
            on_scores=_score_printer() if args.print_scores else None,
        )
        emitted_events = 0
        for event in events:
            emitted_events += 1
            print(render_event(event, args.output), flush=True)
    except (AudioDeviceError, ValueError) as exc:
        parser.exit(2, f"mii-wakeup: {exc}\n")
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 130

    if emitted_events == 0 and not args.continuous:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mii-wakeup",
        description="Listen for openWakeWord wake-word activations.",
    )
    parser.add_argument(
        "--model",
        action="append",
        type=Path,
        default=[],
        help="Path to a custom .onnx wake-word model. Can be provided more than once.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Read audio from a mono 16-bit 16 kHz WAV file, or '-' for raw "
            "mono signed 16-bit 16 kHz PCM on stdin. Defaults to the microphone."
        ),
    )
    parser.add_argument(
        "--device",
        help="Input device index or name for microphone mode.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available microphone input devices and exit.",
    )
    parser.add_argument(
        "--threshold",
        type=_score,
        default=0.5,
        help="Wake activation threshold between 0 and 1. Defaults to 0.5.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=_score,
        default=0.0,
        help="Enable openWakeWord's VAD filter with a threshold between 0 and 1.",
    )
    parser.add_argument(
        "--speex-noise-suppression",
        action="store_true",
        help="Enable openWakeWord's optional Speex noise suppression.",
    )
    parser.add_argument(
        "--block-size",
        type=_positive_int,
        default=DEFAULT_BLOCK_SIZE,
        help="Audio frame size in samples. 1280 is 80 ms at 16 kHz.",
    )
    parser.add_argument(
        "--cooldown",
        type=_non_negative_float,
        default=1.0,
        help="Seconds to suppress repeated events for the same label in continuous mode.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Keep listening and print one line per wake event instead of exiting after the first.",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Event output format. Text is '<label>\\t<score>'.",
    )
    parser.add_argument(
        "--print-scores",
        action="store_true",
        help="Print live model scores to stderr for tuning.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only write wake events to stdout.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _audio_frames(args: argparse.Namespace) -> Iterable[np.ndarray]:
    if args.input is None:
        return microphone_frames(
            MicrophoneConfig(
                device=parse_device(args.device),
                sample_rate=DEFAULT_SAMPLE_RATE,
                block_size=args.block_size,
            ),
            on_overflow=lambda: print("mii-wakeup: input overflow", file=sys.stderr),
        )

    if args.input == Path("-"):
        return raw_stdin_frames(block_size=args.block_size)

    return wav_frames(args.input, block_size=args.block_size)


def _status(args: argparse.Namespace, model: object) -> None:
    names = ", ".join(getattr(model, "models", {}).keys()) or "openWakeWord"
    source = "microphone" if args.input is None else str(args.input)
    print(
        f"mii-wakeup: listening on {source} for {names} "
        f"(threshold={args.threshold:.2f})",
        file=sys.stderr,
    )


def _score_printer():
    def print_scores(scores: Mapping[str, float]) -> None:
        rendered = " ".join(
            f"{label}={float(score):.3f}" for label, score in sorted(scores.items())
        )
        print(f"\r{rendered}", end="", file=sys.stderr, flush=True)

    return print_scores


def _score(value: str) -> float:
    try:
        score = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc

    if not 0 <= score <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return score


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative number") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _prefer_default_sigpipe() -> None:
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
