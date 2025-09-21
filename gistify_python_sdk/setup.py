from setuptools import setup, find_packages

setup(
    name="gistify",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["requests"],
    author="Gistify Team",
    author_email="contact@gistify.com",
    description="A Python SDK for the Gistify API.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gistify/gistify-python-sdk",
)
