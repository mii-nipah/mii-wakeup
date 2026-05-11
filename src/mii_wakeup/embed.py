from __future__ import annotations

import base64
import os
import shutil
import stat
from pathlib import Path


class EmbedError(RuntimeError):
    """Raised when an embedded clone cannot be created."""


def create_embedded_clone(
    *,
    executable_path: Path,
    model_paths: list[Path],
    output_path: Path,
    force: bool = False,
) -> Path:
    executable_path = _resolve_executable_path(executable_path)
    output_path = output_path.resolve()
    model_paths = [model_path.resolve() for model_path in model_paths]

    if not model_paths:
        raise EmbedError("--embed requires at least one --model path.")
    if not executable_path.is_file():
        raise EmbedError(f"Could not find executable to embed: {executable_path}")
    if not _is_elf_executable(executable_path):
        raise EmbedError(
            "--embed must be run from a standalone mii-wakeup executable, "
            "not from the uv/python entrypoint."
        )
    for model_path in model_paths:
        if not model_path.is_file():
            raise EmbedError(f"Wake-word model does not exist: {model_path}")
    if output_path.exists() and not force:
        raise EmbedError(f"Output already exists: {output_path} (use --force to overwrite)")
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temp_path.open("wb") as output:
            _write_script_header(output, len(model_paths))
            _write_embedded_file(output, "MII_WAKEUP_BINARY", executable_path)
            for index, model_path in enumerate(model_paths):
                _write_embedded_file(output, f"MII_WAKEUP_MODEL_{index}", model_path)
            _write_script_footer(output, len(model_paths))

        mode = temp_path.stat().st_mode
        temp_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return output_path


def default_embedded_output_path(executable_path: Path) -> Path:
    executable_name = executable_path.resolve().name
    return Path.cwd() / f"{executable_name}-embedded"


def _is_elf_executable(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(4) == b"\x7fELF"
    except OSError:
        return False


def _resolve_executable_path(path: Path) -> Path:
    if path.is_file():
        return path.resolve()

    found_path = shutil.which(str(path))
    if found_path is not None:
        return Path(found_path).resolve()

    return path.resolve()


def _write_script_header(output, model_count: int) -> None:
    _line(output, "#!/bin/sh")
    _line(output, "set -eu")
    _line(output, "")
    _line(output, "if printf '' | base64 -d >/dev/null 2>&1; then")
    _line(output, "  decoder=base64_d")
    _line(output, "elif printf '' | base64 -D >/dev/null 2>&1; then")
    _line(output, "  decoder=base64_D")
    _line(output, "elif command -v openssl >/dev/null 2>&1; then")
    _line(output, "  decoder=openssl")
    _line(output, "else")
    _line(output, "  echo 'mii-wakeup: no base64 decoder found' >&2")
    _line(output, "  exit 127")
    _line(output, "fi")
    _line(output, "")
    _line(output, "decode_to() {")
    _line(output, "  case \"$decoder\" in")
    _line(output, "    base64_d) base64 -d > \"$1\" ;;")
    _line(output, "    base64_D) base64 -D > \"$1\" ;;")
    _line(output, "    openssl) openssl base64 -d > \"$1\" ;;")
    _line(output, "  esac")
    _line(output, "}")
    _line(output, "")
    _line(output, "tmp_root=${TMPDIR:-/tmp}")
    _line(output, "work=$(mktemp -d \"${tmp_root%/}/mii-wakeup-embedded.XXXXXX\")")
    _line(output, "cleanup() { rm -rf \"$work\"; }")
    _line(output, "trap cleanup EXIT INT TERM")
    _line(output, "")
    _line(output, "binary=\"$work/mii-wakeup\"")
    for index in range(model_count):
        _line(output, f"model_{index}=\"$work/model-{index}.onnx\"")
    _line(output, "")


def _write_script_footer(output, model_count: int) -> None:
    _line(output, "chmod +x \"$binary\"")
    model_args = " ".join(f"--model \"$model_{index}\"" for index in range(model_count))
    _line(output, f"\"$binary\" {model_args} \"$@\"")
    _line(output, "status=$?")
    _line(output, "exit \"$status\"")


def _write_embedded_file(output, marker: str, path: Path) -> None:
    target = '"$binary"' if marker == "MII_WAKEUP_BINARY" else _model_target(marker)
    _line(output, f"decode_to {target} <<'{marker}'")
    with path.open("rb") as source:
        base64.encode(source, output)
    _line(output, marker)
    _line(output, "")


def _model_target(marker: str) -> str:
    index = marker.rsplit("_", 1)[-1]
    return f"\"$model_{index}\""


def _line(output, text: str) -> None:
    output.write(text.encode("ascii"))
    output.write(b"\n")
