"""
Setup script for MozhiGPT - Tamil-First ChatGPT
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mozhigpt",
    version="1.0.0",
    author="MozhiGPT Team",
    author_email="your.email@example.com",
    description="Tamil-first ChatGPT with natural language understanding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/MozhiGPT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Researchers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Tamil",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
        ],
        "gpu": [
            "torch>=2.1.0+cu118",
        ],
        "speech": [
            "openai-whisper>=20230918",
            "TTS>=0.20.0",
        ],
        "rag": [
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.4",
            "chromadb>=0.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "mozhigpt-train=train:main",
            "mozhigpt-serve=api.main:main",
            "mozhigpt-chat=inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tokenizers": ["*.json"],
        "models": ["*"],
    },
    keywords="tamil, chatgpt, nlp, ai, tamil-ai, conversational-ai, language-model",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/MozhiGPT/issues",
        "Source": "https://github.com/yourusername/MozhiGPT",
        "Documentation": "https://github.com/yourusername/MozhiGPT/blob/main/README.md",
    },
)
