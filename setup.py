from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="open-mmunlearning",
    version="0.1.0",
    author="open-mmunlearning contributors",
    description="An open-source framework for multimodal LLM unlearning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"attacks": ["assets/DnCNN/*.pth.tar"]},
    install_requires=requirements,  # Uses requirements.txt
    extras_require={
        "lm-eval": [
            "lm-eval==0.4.11",
        ],  # Install using `pip install ".[lm-eval]"`
        "dev": [
            "pre-commit==4.0.1",
            "ruff==0.6.9",
        ],  # Install using `pip install ".[dev]"`
    },
    python_requires=">=3.11",
)
