import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="id",
    version="1.0",
    author="Mehrab Mahdian",
    author_email="mehrab.mahdian@taltech.ee",
    description="id",
    long_description="A python module for inverse design in electrospinning",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "xgboost",
        "scikit-learn",
        "seaborn"
    ],
    python_requires='>=3.8',
)