from setuptools import setup, find_packages

setup(
    name="document_pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "qdrant-client",
        "tqdm",
        "pytest",
    ],
)
