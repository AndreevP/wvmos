from setuptools import setup, find_packages

setup(
    name="wvmos",
    version="1.0",
    packages=find_packages(),

    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'librosa',
        'torch>=1.7.0',
        'transformers>=4.12.5',
        'pytorch-lightning>=1.5.5'
    ],
    include_package_data=True
)