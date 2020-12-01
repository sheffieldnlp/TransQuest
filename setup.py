from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="transquest",
    version="0.1.0",
    author="Tharindu Ranasinghe",
    author_email="rhtdranasinghe@gmail.com",
    description="Transformer based translation quality estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sheffieldnlp/TransQuest",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "tqdm", "transformers", "scipy", "scikit-learn", "pandas", "tokenizers", "datasets",],
    entry_points={"console_scripts": ["tq-run-word=transquest_cli.run_wordlevel:main",]},
)
