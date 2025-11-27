import json
import os
import subprocess
import sys
import tempfile
import unittest


class TestStartApiServerModeCLI(unittest.TestCase):
    """Tests that start_api_server CLI respects --mode and --config arguments.
    Uses --dry-run to avoid launching uvicorn.
    """

    def test_mode_and_config_env_vars(self):
        script = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "start_api_server.py")
        )

        # Create a temporary config file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            tmp.write(json.dumps({"model_path": "./tmp"}))
            tmp.flush()
            tmp_path = tmp.name

        try:
            cmd = [sys.executable, script, "--mode", "laptop", "--config", tmp_path, "--dry-run"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            stdout = result.stdout

            # Check for the expected env variables printed by dry-run
            self.assertIn("CUBO_LAPTOP_MODE= 1", stdout)
            self.assertIn("CUBO_CONFIG_PATH=", stdout)
            self.assertIn(tmp_path, stdout)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
