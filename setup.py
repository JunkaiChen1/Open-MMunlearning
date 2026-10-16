from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
requirements = [
    line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="open-mmunlearning",
    version="0.1.0",
    description="Multimodal model finetuning, unlearning, and evaluation",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    package_dir={"": "src"},
    py_modules=["eval_config"],
    packages=find_packages(where="src"),
    install_requires=requirements,
    extras_require={"dev": ["ruff>=0.6,<1"]},
    python_requires=">=3.11",
)
