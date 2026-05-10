from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from mii_wakeup.audio import DEFAULT_SAMPLE_RATE, parse_device, wav_frames


class WavFrameTests(unittest.TestCase):
    def test_reads_mono_16khz_pcm_in_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "silence.wav"
            samples = np.zeros(1600, dtype=np.int16)

            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(DEFAULT_SAMPLE_RATE)
                wav_file.writeframes(samples.tobytes())

            frames = list(wav_frames(path, block_size=1280))

        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].shape, (1280,))
        self.assertEqual(frames[1].shape, (1280,))
        self.assertTrue(np.all(frames[1][320:] == 0))


class DeviceParsingTests(unittest.TestCase):
    def test_numeric_device_names_become_indexes(self) -> None:
        self.assertEqual(parse_device("3"), 3)
        self.assertEqual(parse_device("-1"), -1)
        self.assertEqual(parse_device("USB Mic"), "USB Mic")


if __name__ == "__main__":
    unittest.main()
