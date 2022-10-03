from setuptools import setup, find_packages

setup(
    name="lbforaging",
    version="1.1.1",
    description="Level Based Foraging Environment",
    author="Filippos Christianos",
    url="https://github.com/semitable/lb-foraging",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=["numpy>=1.23.1", "gym>=0.26.1", "pyglet", "pettingzoo>=1.21.0"],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
