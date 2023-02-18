import os
import re

from setuptools import find_packages
from setuptools import setup

ROOT_DIR = os.path.dirname(__file__)

with open(os.path.join(ROOT_DIR, "../README.md"), "r") as fh:
    long_description = fh.read()

extras = {}
test_deps = ["pytest"]

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
all_deps = all_deps + test_deps
extras["all"] = all_deps


def find_version(*filepath: str):
    # Extract version information from filepath
    with open(os.path.join(ROOT_DIR, *filepath)) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

setup(
    name='adept',
    version='0,0.0dev0',
    packages=find_packages(),
    url='github.com/shadedtech/adept',
    license='MIT',
    author='ShadedTech',
    author_email='',
    description='',
    install_requires=[],
    test_requires=test_deps,
    extras_require=extras,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
