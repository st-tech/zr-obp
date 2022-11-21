from os import path
import sys

from setuptools import find_packages
from setuptools import setup

from obp.version import __version__


here = path.abspath(path.dirname(__file__))
sys.path.insert(0, path.join(here, "obp"))

print("version")
print(__version__)

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

package_data_list = ["obp/policy/conf/prior_bts.yaml", "obp/dataset/obd"]

setup(
    name="obp",
    version=__version__,
    description="Open Bandit Pipeline: a python library for bandit algorithms and off-policy evaluation",
    url="https://github.com/st-tech/zr-obp",
    author="Yuta Saito",
    author_email="open-bandit-project@googlegroups.com",
    keywords=["bandit algorithms", "off-policy evaluation"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "matplotlib>=3.4.3",
        "mypy-extensions>=0.4.3",
        "numpy>=1.21.2",
        "pandas>=1.3.2",
        "pyyaml>=6.0.0",
        "seaborn>=0.11.2",
        "scikit-learn>=1.0.2",
        "scipy>=1.7.3",
        "torch>=1.9.0",
        "tqdm>=4.62.2",
        "pyieoe>=0.1.1",
        "pingouin>=0.4.0",
    ],
    license="Apache License",
    packages=find_packages(
        exclude=["benchmark", "docs", "examples", "obd", "tests", "slides"]
    ),
    package_data={"obp": package_data_list},
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
    ],
)
