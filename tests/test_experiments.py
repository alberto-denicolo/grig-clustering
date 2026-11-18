from pathlib import Path

import pytest
import yaml

from grig.experiments import REGISTRY

QUICK = yaml.safe_load((Path(__file__).parents[1] / "configs" / "quick.yaml").read_text())


@pytest.mark.parametrize("name", list(REGISTRY))
def test_experiment_runs_on_quick_config(name):
    out = REGISTRY[name](QUICK[name], QUICK["seed"], n_jobs=1, log=lambda *_: None)
    frames = out if isinstance(out, tuple) else (out,)
    assert all(len(f) > 0 for f in frames)
