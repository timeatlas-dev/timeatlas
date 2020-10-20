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


def get_commit_time(commit):
    try:
        return subprocess.check_output(
            ['git', 'show', '-s', '--format="%ct"', commit]) \
            .decode('ascii').strip()[1:-1]
    except Exception:
        return 'unknown'


def create_version():
    version = open('version.txt', 'r').read().strip()
    branch = get_branch()
    if branch == "develop" or branch is None:
        commit_time = get_commit_time(get_commit())
        version = "{}{}{}".format(version, '.dev', commit_time)
    elif branch == "master":
        version
    return version

def get_extras():
    extras = {
        'torch': ['torch==1.6.*'],
        'prophet': ['fbprophet==0.*'],
        'all': ['torch==1.6.*', 'fbprophet==0.*']
    }

    return extras

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
    extra_require=get_extras(),
)
