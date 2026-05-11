#!/usr/bin/env bash
set -euo pipefail

output_dir="dist"
output_filename="mii-wakeup"
extra_args=()

usage() {
  printf 'Usage: %s [--output-dir DIR] [--output-filename NAME] [-- NUITKA_ARGS...]\n' "$0"
}

while (($#)); do
  case "$1" in
    --output-dir)
      output_dir="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --output-filename)
      output_filename="${2:?missing value for --output-filename}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

site_packages="$(
  python - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)"

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'Required build input missing: %s\n' "$1" >&2
    exit 1
  fi
}

oww_models="$site_packages/openwakeword/resources/models"
ort_capi="$site_packages/onnxruntime/capi"

require_file "$oww_models/melspectrogram.onnx"
require_file "$oww_models/embedding_model.onnx"
require_file "$oww_models/silero_vad.onnx"
require_file "$ort_capi/libonnxruntime_providers_shared.so"

python -m nuitka \
  --mode=onefile \
  --output-dir="$output_dir" \
  --output-filename="$output_filename" \
  --include-data-files="$oww_models/melspectrogram.onnx=openwakeword/resources/models/melspectrogram.onnx" \
  --include-data-files="$oww_models/embedding_model.onnx=openwakeword/resources/models/embedding_model.onnx" \
  --include-data-files="$oww_models/silero_vad.onnx=openwakeword/resources/models/silero_vad.onnx" \
  --include-data-files="$ort_capi/libonnxruntime_providers_shared.so=onnxruntime/capi/libonnxruntime_providers_shared.so" \
  --include-distribution-metadata=mii-wakeup \
  --include-module=openwakeword.model \
  --include-module=openwakeword.utils \
  --include-module=openwakeword.vad \
  --include-package=sounddevice \
  --include-package=mii_wakeup \
  --nofollow-import-to=openwakeword.custom_verifier_model \
  --nofollow-import-to=scipy \
  --nofollow-import-to=sklearn \
  --nofollow-import-to=joblib \
  --nofollow-import-to=tqdm \
  --nofollow-import-to=onnxruntime.backend \
  --nofollow-import-to=onnxruntime.tools \
  --nofollow-import-to=onnxruntime.quantization \
  --nofollow-import-to=onnxruntime.transformers \
  --noinclude-unittest-mode=nofollow \
  --python-flag=no_docstrings \
  --python-flag=no_asserts \
  --assume-yes-for-downloads \
  "${extra_args[@]}" \
  main.py
