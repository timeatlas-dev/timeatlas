import setuptools
import subprocess
from pathlib import Path

with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


def get_branch():
    try:
        return subprocess.check_output(['git', 'branch', '--show-current'])\
                .decode('ascii').strip()
    except Exception:
        return None


def get_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])\
            .decode('ascii').strip()
    except Exception:
        return 'unknown'


def create_version():
    version = open('version.txt', 'r').read().strip()
    commit = get_commit()
    branch = get_branch()
    if branch == "develop" or branch is None:
        version += '.dev' + commit
    elif branch == "master":
        version
    return version


setuptools.setup(
    name="timeatlas",
    version=create_version(),
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
