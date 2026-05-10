# mii-wakeup

Tiny Unix-style wake-word listener built on
[openWakeWord](https://github.com/dscripka/openWakeWord).

The default behavior is intentionally pipe-friendly: listen until a wake word is
detected, print one event line on stdout, then exit.

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
uv run mii-wakeup --model ./hey_meee.onnx --continuous --output json
uv run mii-wakeup --list-devices
```

If `--model` is omitted, openWakeWord loads its bundled pre-trained models.
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

## Nuitka

The package is structured so it can be compiled as a one-file executable. The
build command explicitly includes `openwakeword` package data, which is where
openWakeWord keeps its bundled ONNX support models. It also excludes
openWakeWord's training/custom-verifier path because mii-wakeup only performs
runtime wake detection.

```sh
uv run python -m nuitka \
  --mode=onefile \
  --output-dir=dist \
  --output-filename=mii-wakeup \
  --include-package-data=openwakeword \
  --include-module=openwakeword.model \
  --include-module=openwakeword.utils \
  --include-module=openwakeword.vad \
  --include-package=onnxruntime \
  --include-package=sounddevice \
  --include-package=mii_wakeup \
  --nofollow-import-to=openwakeword.custom_verifier_model \
  --nofollow-import-to=scipy \
  --nofollow-import-to=sklearn \
  --nofollow-import-to=joblib \
  --nofollow-import-to=tqdm \
  --nofollow-import-to=onnxruntime.backend \
  --noinclude-unittest-mode=nofollow \
  main.py
```

The user's custom wake-word `.onnx` file remains external and should be passed
with `--model`.
