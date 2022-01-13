#
# Copyright 2021 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="piquasso",
    version="0.7.0",
    packages=find_packages(exclude=["tests.*", "tests", "scripts", "scripts.*"]),
    maintainer="Budapest Quantum Computing Group",
    maintainer_email="kolarovszki@inf.elte.hu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Budapest-Quantum-Computing-Group/piquasso",
    keywords=["python", "piquasso"],
    install_requires=[
        'theboss==1.0.6; python_version >= "3.7"',
        'numpy>=1.19.5; python_version >= "3.7"',
        'scipy>=1.5.4; python_version >= "3.7"',
        "quantum-blackbird==0.3.0",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
    ],
    license="Apache License 2.0.",
)
