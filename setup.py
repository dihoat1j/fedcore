from setuptools import setup, find_packages

setup(
    name="fedcore",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
    ],
    author="FedCore Maintainers",
    description="A modular federated learning framework",
    python_requires=">=3.8",
)
