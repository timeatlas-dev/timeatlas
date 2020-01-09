import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="timeatlas",
    version="0.0.1",
    author="Frédéric Montet",
    author_email="frederic.montet@hefr.ch",
    description="A toolbox to handle, analyze and predict time series data.",
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
    install_requires=[
        'bbdata>=0.4.3',
        'matplotlib',
        'pandas',
        'tqdm'
    ]
)
