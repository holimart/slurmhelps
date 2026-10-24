import pytest

from slurmhelps.gpu_batch import build_sbatch_script


def _script(**kw):
    return build_sbatch_script(remote_name="cmp-x", remote_dir="/tmp/x", account="ffa", run_cmd="true", **kw)


def test_nice_line_is_written_as_a_directive():
    s = _script(nice=2147483645)
    lines = s.splitlines()
    assert "#SBATCH --nice=2147483645" in lines
    # Directives must come before the first command, or sbatch ignores them.
    assert lines.index("#SBATCH --nice=2147483645") < lines.index("set -euo pipefail")


def test_no_nice_by_default():
    assert "--nice" not in _script()


def test_negative_nice_is_refused():
    with pytest.raises(ValueError):
        _script(nice=-1)
