import os
from setuptools import setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"

setup(
    name="flasheval",
    author="Lin Zhao",
    description="FlashEval: Towards Fast and Accurate Evaluation of Text-to-image Diffusion Generative Models",
    python_requires=">=3.8",
    py_modules=[],
    install_requires=[
        "accelerate==0.29.3",
        "astunparse>=1.6.3",
        "attrs>=22.1.0",
        "brotlipy>=0.7.0",
        "clip",
        "cycler>=0.12.1",
        "diffusers>=0.24.0",
        "dnspython>=2.2.1",
        "exceptiongroup>=1.0.4",
        "expecttest>=0.1.4",
        "fonttools>=4.47.0",
        "ftfy>=6.1.3",
        "future>=0.18.2",
        "h11>=0.14.0",
        "huggingface-hub>=0.13.4",
        "hypothesis>=6.61.0",
        "kiwisolver>=1.4.5",
        "matplotlib>=3.8.2",
        "mkl-fft>=1.3.1",
        "mkl-service>=2.4.0",
        "mpmath>=1.2.1",
        "outcome>=1.3.0.post0",
        "packaging>=23.2",
        "Pillow>=9.3.0",
        "pyparsing>=3.1.1",
        "python-dateutil>=2.8.2",
        "python-etcd>=0.4.5",
        "PyYAML>=6.0",
        "regex>=2023.10.3",
        "selenium>=4.16.0",
        "sniffio>=1.3.0",
        "sortedcontainers>=2.4.0",
        "sympy>=1.11.1",
        "tokenizers>=0.13.3",
        "transformers>=4.27.4",
        "trio>=0.23.2",
        "trio-websocket>=0.11.1",
        "types-dataclasses>=0.6.6",
        "timm",
        "fairscale",
        "wsproto>=1.2.0"
    ],
    include_package_data=True,
)