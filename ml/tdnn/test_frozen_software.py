"""Regression checks for the frozen software entry; no recording fixtures needed."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import identify_frozen as frozen


CONDITIONS = "desktop_members_plus_voxpopuli_public_fp32_v1"


class FrozenReleaseTests(unittest.TestCase):
    def test_matching_release(self):
        release = frozen.load_release(CONDITIONS)
        self.assertEqual(release["threshold"], 0.5757178068161011)
        self.assertEqual(release["members"], ["member02", "member03", "member04", "member05"])

    def test_changed_conditions_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conditions differ"):
            frozen.load_release("different_microphone_conditions")

    def test_changed_threshold_rejected(self):
        release = copy.deepcopy(frozen.load_release(CONDITIONS))
        release["threshold"] = 0.576
        with patch.object(frozen.json, "loads", return_value=release):
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                frozen.load_release(CONDITIONS)

    def test_changed_inference_source_rejected(self):
        with tempfile.TemporaryDirectory(dir=frozen.ROOT) as folder:
            root = Path(folder)
            for name in ["frozen_libri251_fp32.json", "identify.py", "small_model.py"]:
                (root / name).write_bytes((frozen.ROOT / name).read_bytes())
            with (root / "identify.py").open("ab") as stream:
                stream.write(b"\n# changed inference\n")
            with patch.object(frozen, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "dependency changed"):
                    frozen.load_release(CONDITIONS)

    def test_metadata_cli_does_not_claim_assets_verified(self):
        output = io.StringIO()
        with patch("sys.argv", ["identify_frozen.py", "--check-release", "--conditions-id", CONDITIONS]):
            with redirect_stdout(output):
                self.assertEqual(frozen.main(), 0)
        self.assertFalse(json.loads(output.getvalue())["runtime_assets_checked"])

    def test_session03_guard_runs_before_inference(self):
        identifier = frozen.FrozenTDNN.__new__(frozen.FrozenTDNN)
        with self.assertRaisesRegex(ValueError, "session03 remains reserved"):
            identifier.identify(Path("session03") / "held_out.wav")


@unittest.skipUnless(
    (frozen.ROOT / "checkpoints/libri251_m03_20h/best.pt").is_file()
    and (frozen.ROOT / "templates/libri251_fp32_multiwindow.json").is_file(),
    "Install the separately supplied runtime assets for model/input checks",
)
class InstalledRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identifier = frozen.FrozenTDNN(CONDITIONS)

    def test_wrong_template_fingerprint_rejected(self):
        with patch.object(frozen, "file_sha256", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "template database changed"):
                frozen.FrozenTDNN(CONDITIONS)

    def test_wrong_checkpoint_fingerprint_rejected(self):
        with patch("identify.checksum", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "Model/template SHA-256 mismatch"):
                frozen.FrozenTDNN(CONDITIONS)

    def test_invalid_audio_cli_exit_code(self):
        with tempfile.TemporaryDirectory(dir=frozen.ROOT) as folder:
            silent = Path(folder) / "silent.wav"
            with wave.open(str(silent), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\0\0" * 48000)
            output = io.StringIO()
            with patch("sys.argv", ["identify_frozen.py", "--wav", str(silent), "--conditions-id", CONDITIONS]):
                with redirect_stdout(output):
                    self.assertEqual(frozen.main(), 2)
            self.assertEqual(json.loads(output.getvalue())["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
