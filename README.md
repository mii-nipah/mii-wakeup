# mii-wakeup

Tiny Unix-style wake-word listener built on
[openWakeWord](https://github.com/dscripka/openWakeWord).

The default behavior is intentionally pipe-friendly: listen until a wake word is
detected, print the confidence score on stdout, then exit.

```sh
uv run mii-wakeup --model /home/nipah/Downloads/ai/weights/hey_meee.onnx \
  && echo "now I'm awake"
```

For shell pipelines, consume the event line:

```sh
uv run mii-wakeup --model /home/nipah/Downloads/ai/weights/hey_meee.onnx |
  while read -r _event; do
    echo "now I'm awake"
  done
```

## Usage

```sh
uv run mii-wakeup --model ./hey_meee.onnx
uv run mii-wakeup --model ./hey_meee.onnx --stream
uv run mii-wakeup --model ./hey_meee.onnx --stream --output json
uv run mii-wakeup --list-devices
```

When running from the uv environment, omitting `--model` lets openWakeWord load
its bundled pre-trained wake-word models. The slim Nuitka build below excludes
those demo wake-word models, so packaged binaries should be run with `--model`.
Custom `.onnx` models can be provided more than once with repeated `--model`
flags.

The microphone path uses mono 16 kHz signed 16-bit audio in 1280-sample frames,
matching openWakeWord's 80 ms frame recommendation. On Linux, `sounddevice`
needs PortAudio available on the host:

```sh
sudo apt install libportaudio2
```

For more Unix composition, mii-wakeup can also read raw PCM from stdin or a WAV
file. Stdin must be mono signed 16-bit 16 kHz PCM:

```sh
arecord -q -r 16000 -f S16_LE -c 1 |
  uv run mii-wakeup --model ./hey_meee.onnx --input -

uv run mii-wakeup --model ./hey_meee.onnx --input ./sample.wav
```

## Embedded Clones

A built binary can create a single-file clone that carries both mii-wakeup and
your wake-word model:

```sh
./dist/mii-wakeup --model ./hey_meee.onnx --embed ./mii-wakeup-hey-meee
./mii-wakeup-hey-meee --stream
```

The embedded clone is a POSIX shell executable. It extracts the bundled binary
and model into a temporary directory, runs mii-wakeup with the embedded
`--model`, then cleans up when the process exits. Use `--force` with `--embed`
to overwrite an existing output file.

## Nuitka

The package is structured so it can be compiled as a one-file executable. The
build command explicitly includes only openWakeWord's runtime support models.
It excludes openWakeWord's demo wake-word models and training/custom-verifier
path because packaged mii-wakeup expects users to pass their wake-word `.onnx`
with `--model`.

```sh
uv run python -m nuitka \
  --mode=onefile \
  --output-dir=dist \
  --output-filename=mii-wakeup \
  --include-data-files=.venv/lib/python3.13/site-packages/openwakeword/resources/models/melspectrogram.onnx=openwakeword/resources/models/melspectrogram.onnx \
  --include-data-files=.venv/lib/python3.13/site-packages/openwakeword/resources/models/embedding_model.onnx=openwakeword/resources/models/embedding_model.onnx \
  --include-data-files=.venv/lib/python3.13/site-packages/openwakeword/resources/models/silero_vad.onnx=openwakeword/resources/models/silero_vad.onnx \
  --include-data-files=.venv/lib/python3.13/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so=onnxruntime/capi/libonnxruntime_providers_shared.so \
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
  main.py
```

The user's custom wake-word `.onnx` file remains external and should be passed
with `--model`.
