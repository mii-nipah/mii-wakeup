# mii-wakeup

Tiny Unix-style wake-word listener built on
[openWakeWord](https://github.com/dscripka/openWakeWord).

The default behavior is intentionally pipe-friendly: listen until a wake word is
detected, print the confidence score on stdout, then exit.

```sh
mii-wakeup --model /home/nipah/Downloads/ai/weights/hey_meee.onnx \
  && echo "now I'm awake"
```

For shell pipelines, consume the event line:

```sh
mii-wakeup --model /home/nipah/Downloads/ai/weights/hey_meee.onnx |
  while read -r _event; do
    echo "now I'm awake"
  done
```

## Usage

```sh
mii-wakeup --model ./hey_meee.onnx
mii-wakeup --model ./hey_meee.onnx --stream
mii-wakeup --model ./hey_meee.onnx --stream --output json
mii-wakeup --list-devices
```

Packaged binaries expect a wake-word `.onnx` model with `--model`. Custom
models can be provided more than once with repeated `--model` flags.

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
  mii-wakeup --model ./hey_meee.onnx --input -

mii-wakeup --model ./hey_meee.onnx --input ./sample.wav
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
uv run ./scripts/build-onefile.sh
```

The user's custom wake-word `.onnx` file remains external and should be passed
with `--model`. The script keeps Nuitka's build directories under `dist/`, so
repeated local builds can reuse as much work as Nuitka and the C compiler allow.

When developing from a checkout, run the command through uv:

```sh
uv run mii-wakeup --model ./hey_meee.onnx
```

## Releases

Releases are automated from the version in `pyproject.toml`. To ship a new
downloadable Linux x86_64 binary, edit the version, commit, and push to `main`.
The GitHub Actions release workflow skips pushes whose `v<version>` tag already
exists; new versions run the tests, build the Nuitka one-file binary, create the
tag, and upload a tarball plus `SHA256SUMS` to GitHub Releases.

## License

BSD-3-Clause.
