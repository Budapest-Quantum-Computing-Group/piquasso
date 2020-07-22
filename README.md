Í# Development Guide

## Development requirements

First and foremost, a `python3.6` is needed to be installed on your machine.
For the case of having a system python with version >=3.7, `pyenv` is recommended.

The `eigen3` C++ library is needed to be installed for the
`thewalrus` (dependency of `strawberryfields`).

The deb package is named `libeigen3-dev`, so issue
```
sudo apt-get install libeigen3-dev
```
on a machine running Ubuntu/Debian (the rpm package is named `eigen3-devel`).

Additionally, this project uses `tox` to manage virtualenvs, install it with
```
pip install tox
```

For packaging, `poetry` is used:
```
pip install poetry
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
