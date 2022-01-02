from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "boltzmanngen/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="boltzmanngen",
    version=version,
    description="BoltzmannGen is an open-source code that reimplements the Boltzmann Generators paper by Frank Noe.",
    download_url="https://github.com/Daniangio/paper_boltzmann_generators",
    author="Frank Noe, Daniele Angioletti",
    python_requires=">=3.8",
    packages=find_packages(include=["boltzmanngen", "boltzmanngen.*"]),
    install_requires=[
        "numpy",
        "ase",
        "tqdm",
        "wandb",
        "biopandas",
        "torch@https://download.pytorch.org/whl/cu111/torch-1.9.1%2Bcu111-cp38-cp38-linux_x86_64.whl",
        "e3nn>=0.3.5,<0.5.0",
        "pyyaml",
        "contextlib2;python_version<'3.7'",  # backport of nullcontext
        "typing_extensions;python_version<'3.8'",  # backport of Final
        "scikit_learn",  # for GaussianProcess for per-species statistics
    ],
    zip_safe=True,
)
