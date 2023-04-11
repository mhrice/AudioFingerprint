from pathlib import Path
from setuptools import setup, find_packages

NAME = "Fingerprint"
DESCRIPTION = ""
URL = ""
EMAIL = "m.rice@se22.qmul.ac.uk"
AUTHOR = "Matthew Rice"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.0.1"

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=[
        "torch>=1.11.0",
        "torchaudio>=0.13.0",
        "torchvision",
        "numpy",
        "pytorch-lightning",
        "numba",
        "wandb",
        "librosa",
        "auraloss",
        "pyloudnorm",
        "matplotlib",
    ],
    include_package_data=True,
    license="Apache License 2.0",
    classifiers=[
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
    ],
)
