from obp.version import __version__
from setuptools import setup, find_packages
from os import path
import sys

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
        "matplotlib>=3.2.2",
        "mypy-extensions>=0.4.3",
        "numpy>=1.18.1",
        "pandas>=0.25.1",
        "pyyaml>=5.1",
        "seaborn>=0.10.1",
        "scikit-learn>=0.23.1",
        "scipy>=1.4.1",
        "torch>=1.7.1",
        "tqdm>=4.41.1",
    ],
    license="Apache License",
    packages=find_packages(
        exclude=["benchmark", "docs", "examples", "obd", "tests", "slides"]
    ),
    package_data={"obp": package_data_list},
    include_package_data=True,
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
    ],
)
