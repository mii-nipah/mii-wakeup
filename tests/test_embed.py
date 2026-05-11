from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mii_wakeup.embed import EmbedError, create_embedded_clone


class EmbedTests(unittest.TestCase):
    def test_creates_executable_shell_clone(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "mii-wakeup"
            model = root / "hey_meee.onnx"
            output = root / "mii-wakeup-hey-meee"
            executable.write_bytes(b"\x7fELFfake binary")
            model.write_bytes(b"fake model")

            create_embedded_clone(
                executable_path=executable,
                model_paths=[model],
                output_path=output,
            )

            clone = output.read_text(encoding="ascii")
            exists = output.exists()
            executable_bit = bool(output.stat().st_mode & stat.S_IXUSR)

        self.assertTrue(exists)
        self.assertTrue(executable_bit)
        self.assertIn("MII_WAKEUP_BINARY", clone)
        self.assertIn("MII_WAKEUP_MODEL_0", clone)
        self.assertIn('--model "$model_0"', clone)

    def test_rejects_non_standalone_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "mii-wakeup"
            model = root / "hey_meee.onnx"
            output = root / "embedded"
            executable.write_text("#!/usr/bin/env python\n", encoding="ascii")
            model.write_bytes(b"fake model")

            with self.assertRaises(EmbedError):
                create_embedded_clone(
                    executable_path=executable,
                    model_paths=[model],
                    output_path=output,
                )

    def test_refuses_to_overwrite_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "mii-wakeup"
            model = root / "hey_meee.onnx"
            output = root / "embedded"
            executable.write_bytes(b"\x7fELFfake binary")
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            model.write_bytes(b"fake model")
            output.write_text("existing", encoding="ascii")

            with self.assertRaises(EmbedError):
                create_embedded_clone(
                    executable_path=executable,
                    model_paths=[model],
                    output_path=output,
                )

    def test_resolves_executable_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "mii-wakeup-test"
            model = root / "hey_meee.onnx"
            output = root / "embedded"
            executable.write_bytes(b"\x7fELFfake binary")
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            model.write_bytes(b"fake model")

            with patch.dict("os.environ", {"PATH": str(root)}):
                create_embedded_clone(
                    executable_path=Path("mii-wakeup-test"),
                    model_paths=[model],
                    output_path=output,
                )

            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
