import os
import re

import setuptools

here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, "src", "alpaca_farm", "__init__.py")) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find `__version__`.")

setuptools.setup(
    name="alpaca_farm",
    version=version,
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    install_requires=[
        "einops",
        "nltk",
        "accelerate>=0.18.0",
        "tabulate",
        "transformers>=4.26.0",
        "statsmodels",
        "black>=23.1.0",
        "tiktoken>=0.3.2",
        "markdown",
        "scikit-learn",
        "rouge_score",
        "sentencepiece",
        "pre-commit>=3.2.0",
        "deepspeed",
        "pandas",
    ],
    extras_require={
        "full": [
            # Efficiency.
            "flash-attn" "apex",
            # Plotting and visualization.
            "benepar",
            "spacy",
            "spacy_fastlang",
            "plotly",
            "mapply",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
