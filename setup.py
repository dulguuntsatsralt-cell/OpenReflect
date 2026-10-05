"""Compatibility shim for environments with an older setuptools build backend."""
from setuptools import setup

setup(
    name="arex-evaluation-suite",
    version="0.1.0",
    package_dir={"arex_eval": "evaluation/arex_eval", "openreflect": "src/openreflect"},
    packages=[
        "arex_eval", "arex_eval.backends",
        "openreflect", "openreflect.envs", "openreflect.scaffold", "openreflect.palm",
        "openreflect.selection", "openreflect.train", "openreflect.metrics",
    ],
    entry_points={"console_scripts": ["arex=arex_eval.cli:main", "openreflect=openreflect.cli:main"]},
)
