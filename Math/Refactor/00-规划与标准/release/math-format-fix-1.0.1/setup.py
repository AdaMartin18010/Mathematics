#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="math-format-fix",
    version="1.0.1",
    author="数学格式修复项目团队",
    author_email="support@math-format-fix.com",
    description="专业的数学文档格式处理工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/math-format-fix/project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Markup",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "coverage>=5.0.0",
            "pylint>=2.0.0",
            "flake8>=3.8.0",
            "bandit>=1.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "math-format-fix=数学格式修复命令行工具:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
