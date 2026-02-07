"""
Setup configuration for MHC Forward Pre operator implementations.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name="mhc-ops",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="流形约束超连接 (MHC) 前置算子的多种实现",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/mhc-ops",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "triton": [
            "triton>=2.0.0",
        ],
        "tilelang": [
            "tilelang>=0.1.0",
            "tvm>=0.14.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pandas>=2.0.0",
        ],
        "all": [
            "triton>=2.0.0",
            "tilelang>=0.1.0",
            "tvm>=0.14.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pandas>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mhc-benchmark=test.benchmark:main",
            "mhc-quick-test=test.quick_test:main",
            "mhc-test=test.test_implementations:main",
        ],
    },
)
