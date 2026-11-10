"""Every Python example in README.md runs and prints exactly the output
shown in the block right after it (since 0.6.0; before, only the
tutorials were checked this way)."""
import pathlib
import re
import subprocess
import sys

import pytest

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def _examples():
    if not README.exists():             # e.g. tests run from a wheel
        return []
    blocks = re.findall(r"```(\w*)\n(.*?)```", README.read_text(), re.S)
    pairs = []
    for i, (lang, body) in enumerate(blocks):
        if lang == "python":
            assert i + 1 < len(blocks) and blocks[i + 1][0] == "", \
                f"README python block {len(pairs) + 1} has no output block"
            pairs.append((body, blocks[i + 1][1]))
    return pairs


EXAMPLES = _examples()


@pytest.mark.skipif(not EXAMPLES, reason="README.md is not here")
@pytest.mark.parametrize("number", range(1, len(EXAMPLES) + 1))
def test_readme_example_prints_what_is_shown(number):
    code, shown = EXAMPLES[number - 1]
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout == shown
