from setuptools import setup, find_packages

setup(
    name="lbforaging",
    version="1.0.7",
    description="Level Based Foraging Environment",
    author="Filippos Christianos",
    url="https://github.com/semitable/lb-foraging",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=["numpy", "gym"],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
