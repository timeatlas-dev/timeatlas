import setuptools
from pathlib import Path

with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


setuptools.setup(
    name="timeatlas",
    version="0.0.3",
    author="Frédéric Montet",
    author_email="frederic.montet@hefr.ch",
    description="A time series data manipulation tool for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=setuptools.find_packages('src', exclude='tests'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=read_requirements('requirements/src.txt'),
)
