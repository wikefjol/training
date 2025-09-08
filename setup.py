from setuptools import setup, find_packages

setup(
    name="fungal-classification-training",
    version="0.1.0",
    description="Fungal DNA sequence classification - training pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=5.4.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ]
    },
)