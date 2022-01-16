from setuptools import find_packages, setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="easypred",
    version="0.1.1",
    description="Easily store, assess and compare predictions obtained through Machine Learning models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FilippoPisello/EasyPred",
    project_urls={
        "Documentation": "https://readthedocs.org/projects/easypred/",
    },
    author="Filippo Pisello",
    author_email="filippo.pisello@live.it",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude="tests"),
    include_package_data=True,
    install_requires=["numpy>=1.10", "pandas>=1.1.0", "matplotlib>=3"],
    python_requires=">=3.7",
)
