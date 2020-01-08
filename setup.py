import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="timeatlas",
    version="0.0.1",
    author="Frédéric Montet",
    author_email="frederic.montet@hefr.ch",
    description="A Toolbox to analyze time series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'bbdata-python',
        'matplotlib',
        'pandas',
        'tqdm'
    ]
)
