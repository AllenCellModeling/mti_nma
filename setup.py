#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

test_requirements = [
    "codecov",
    "flake8",
    "black",
    "pytest",
    "pytest-cov",
    "pytest-raises",
]

setup_requirements = [
    "pytest-runner",
]

dev_requirements = [
    "bumpversion>=0.5.3",
    "coverage>=5.0a4",
    "flake8>=3.7.7",
    "ipython>=7.5.0",
    "m2r>=0.2.1",
    "pytest>=4.3.0",
    "pytest-cov==2.6.1",
    "pytest-raises>=0.10",
    "pytest-runner>=4.4",
    "Sphinx>=2.0.0b1",
    "sphinx_rtd_theme>=0.1.2",
    "tox>=3.5.2",
    "twine>=1.13.0",
    "wheel>=0.33.1",
]

interactive_requirements = [
    "altair",
    "jupyterlab",
    "matplotlib"
]

requirements = [
    "aicsimageio",
    "aicsshparam",
    "bokeh",
    "datastep>=0.1.5",
    "dask[bag]",
    "dask_jobqueue",
    "docutils<0.16",  # needed for botocore (quilt dependency)
    "fire",
    "lkaccess",
    "msgpack==0.6.2",  # needed to resolve dep conflict for prefect
    "numpy",
    "numpy-stl",
    "pandas",
    "pyshtools",
    "python-dateutil<=2.8.0",  # need <=2.8.0 for quilt3 in step
    "scikit-image",
    "seaborn",
    "vtk",
]

extra_requirements = {
    "test": test_requirements,
    "setup": setup_requirements,
    "dev": dev_requirements,
    "interactive": interactive_requirements,
    "all": [
        *requirements,
        *test_requirements,
        *setup_requirements,
        *dev_requirements,
        *interactive_requirements
    ]
}

setup(
    author="Julie Cass",
    author_email="juliec@alleninstitute.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Workflow for normal mode analysis",
    entry_points={"console_scripts": [
        "mti_nma=mti_nma.bin.cli:cli",
        "color_vertices=mti_nma.bin.color_vertices:main"
    ]},
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="mti_nma",
    name="mti_nma",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.6",
    setup_requires=setup_requirements,
    test_suite="mti_nma/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/jcass11/mti_nma",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.1.0",
    zip_safe=False,
)
