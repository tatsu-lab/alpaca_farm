# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re

import setuptools


def parse_requirements(filename):
    logging.warning(f"Reading requirements from {filename}")
    with open(filename, "r") as file:
        lines = [line.strip() for line in file]
    return [line for line in lines if line and not line.startswith("#")]


here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, "src", "alpaca_farm", "__init__.py")) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find `__version__`.")

# Must use absolute path here to ensure `python3 -m build` runs when porting package to pypi.
install_requires = parse_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt"))

setuptools.setup(
    name="alpaca_farm",
    version=version,
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
        "full": [
            # Training efficiency.
            "flash-attn",
            "apex",
            "deepspeed",
            # Plotting and visualization.
            "benepar",
            "spacy",
            "spacy_fastlang",
            "plotly",
            "mapply",
        ],
        "dev": {
            "pre-commit>=3.2.0",
            "black>=23.1.0",
            "isort",
        },
    },
    python_requires=">=3.10",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
