import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eSpinID",
    version="1.0.0",

    author="Mehrab Mahdian",
    author_email="mehrab.mahdian@taltech.ee",

    description="An XGBoost and PSO framework for the inverse design and targeted diameter control of electrospun nanofibers.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/taltechloc/cogni-e-spin-inverse-design.git",

    packages=setuptools.find_packages(),

    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "scipy",
        "scikit-optimize",
        "shap",
        "matplotlib",
        "seaborn"
    ],

    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Materials Science",
        "Topic :: Scientific/Engineering :: Optimization",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)