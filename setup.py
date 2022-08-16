from setuptools import setup, find_packages

setup(
    name="comet-rl",
    version="0.0.1",
    author="Siddharth Mehta",
    author_email="siddharthm@comet.ml",
    description="Provides a gym wrapper for Autologging episode rewards and lenghts to Comet's EM tool",
    packages=find_packages("comet_rl"),
    install_requires=['gym', 'comet-ml']
)