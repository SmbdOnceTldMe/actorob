from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def run_probe(script: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)
