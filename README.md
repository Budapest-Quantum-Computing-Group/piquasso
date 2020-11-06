# Development Guide

[[_TOC_]]

## Development requirements

First and foremost, a `python3.6` is needed to be installed on your machine.
For the case of having a system python with version >=3.7, `pyenv` is recommended.

The `eigen3` C++ library is needed to be installed for the
`thewalrus` (dependency of `strawberryfields`).

The deb package is named `libeigen3-dev`, so issue
```
sudo apt-get install libeigen3-dev
```
on a machine running Fedora/CentOS (the rpm package is named `eigen3-devel`).

`python3-venv` is important for the python virtual environment, install it on linux with:
```
sudo apt install python3-venv
```

Additionally, this project uses `tox` to manage virtualenvs, install it with:
``` 
pip install tox
```

For packaging, `poetry` is used.
Install it with the below command:
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```
`poetry` can be installed with `pip` as well, but the recommended way is to install it with the above command. More info [here](https://python-poetry.org/docs/#installation).

The last step is to run the below command to download all the python dependencies (more info [here](https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock)):
```
poetry install
```

## Running a program with poetry
A python program can be run inside poetries virtual environment with the below command
```
poetry run python <python file>
```
Or the virtual environment can be activated explicitly by running:
```
source `poetry env info --path`/bin/activate
```

## Testing

Run tests with
```
tox
```

## How to contribute?

We plan out and track all Piquasso issues through GitLab *Issues*. Feel free
to browse around and pick a sympathetic one to work on. Once you have it,
then please:

1. Follow the [feature branch workflow][1]. In the commit message please
refrence the issue with `#issue-num`.
2. Issue a *merge request*. The Git server will report back the URL for this
after the `push`. You are also welcome to check out the merge requests and add
comments to others' work there.
3. Anytime during the process you can add more changes to your new branch
and `push` anew. If you do this, any approvals will be reset though, but its
fine as the approval concerns the whole of the merge request, so your peers
have to re-evaluate your work.
4. Every request needs to be accepted by at least one other developer. Once
you have it, you can *merge* your change. If your change consists of
multiple commits, please consider checking the *Squash commit* check-box for
GitLab to make a single commit---this improves readability of *master*
branch history.

[1]: https://docs.gitlab.com/ee/gitlab-basics/feature_branch_workflow.html


## Publishing

### Register any repository

Create a [personal access
token](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html)
with at least an `api` permission. Use this access token, to register a
repository in poetry:
```
poetry config repositories.<REPOSITORY-NAME> https://gitlab.inf.elte.hu/api/v4/projects/73/packages/pypi

poetry config pypi-token.<REPOSITORY-NAME> <ACCESS-TOKEN>
```

To publish a package, use
```
poetry publish --build --repository <REPOSITORY-NAME>
```

### Register a private GitLab PyPi repository

Configure a repository named `gitlab` in poetry:
```
poetry config repositories.gitlab https://gitlab.inf.elte.hu/api/v4/projects/73/packages/pypi

poetry config pypi-token.gitlab <ACCESS-TOKEN>
```

To publish a package, use
```
poetry publish --build --repository gitlab
```

References:
- https://python-poetry.org/docs/repositories/#adding-a-repository
- https://python-poetry.org/docs/libraries/#publishing-to-a-private-repository

### Manual publish with GitLab

Automatic publishing don't work with GitLab, since we bump (increment) the
version of the project by hand, and the package repository doesn't overwrite
existing versions (rightfully so!).

**Note**:
- The documentation uses `twine`, but `poetry publish` works similarly.
- Accessing the registry is safer using CI/CD environment variables, see:
https://gitlab.inf.elte.hu/wigner-rcp-quantum-computing-and-information-group/piquasso/-/settings/ci_cd

#### How to publish

1. Create a version bump commit (which increments the version in
   `piquasso/__init__.py` and in `pyproject.toml`.
2. Create a merge request from this commit.
3. When merged, tag the current `origin/master` with `git tag -a <version>`,
   where `<version>` could be e.g. `0.1.1`.
4. Push the tag with `git push --tags`.
5. Go to `CI/CD->Pipelines->Run Pipeline`, select the previously-made git tag,
   then click on `Run Pipeline`.
6. Visit `Packages & Registries->Package Registry` and verify, that there is a
   package under the `PyPI` tab with the newly created version.

References:
- https://gitlab.inf.elte.hu/help/user/packages/pypi_repository/index.md

## Benchmarking

We can run benckmarks with various pennylane devices, for more details see:
https://github.com/XanaduAI/pennylane/blob/master/benchmark/README.rst
For instance:
```
cd pennylane/benchmark/
poetry run python benchmark.py -d default.qubit time bm_entangling_layers
```
### Notes:
Unfortunatelly running `benchmark.py` with `bm_mutable_rotations` or
`bm_mutable_complicated_params` will not work. There is a bug in both of these
in their `__init__()` function: they are expected to accept a named parameter
`qnode_type` but they don't. If you add the parameter by hand it will work
fine.

**The benchmarks, as of now, only work for qubit operations!**

## Generating documentation

Generate the `html` documentation with the below command from the docs folder:
```
poetry run make html_all
```
The generated documentation is available in the docs/_build folder.

To generate the web page and open it in the default browser run `poetry run make open`

The generated documentation can also be downloaded form the gitlab pipeline more info [here](https://docs.gitlab.com/ee/ci/pipelines/job_artifacts.html).
