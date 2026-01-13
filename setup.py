from setuptools import setup, find_packages

setup(
    name="torchdimviz",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",  # 2.0+ is required for robust Dispatcher support
        "tabulate",
        "rich"           # For colored terminal output
    ],
)