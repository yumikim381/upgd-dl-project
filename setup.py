from setuptools import setup, find_packages

setup(
    name="UPGD_DL_Project",
    url="https://github.com/upgd-dl-project",
    author="Constantin Pinkl, Laura Schulz, Tilman de Lanversin, Yumi Kim",
    packages=find_packages(exclude=["tests*"]),
    install_requires=['backpack-for-pytorch==1.3.0', 'HesScale==1.0.0'],
)
