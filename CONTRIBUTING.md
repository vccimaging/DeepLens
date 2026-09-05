# Contributing Guide

Thank you for your interest in contributing to DeepLens!

All contributors are expected to follow our [Code of Conduct](./CODE_OF_CONDUCT.md).

## Contributor License Agreement (CLA)

All pull requests to the DeepLens project require contributors to sign a Contributor License Agreement (CLA). This process is automated through [CLA Assistant](https://cla-assistant.io/), which will prompt you to sign the CLA when you submit your first pull request. You can review the [DeepLens-CLA](https://gist.github.com/singer-yang/b2e4214a12a220899ed682d9c24f575b).

## How to Contribute

We welcome contributions in various forms, including but not limited to:
- Reporting bugs
- Submitting pull requests with bug fixes or new features
- Improving documentation
- Adding new examples or tutorials

If you plan to work on a major feature, please open an issue to discuss your ideas with the maintainers first.

## Developer Installation

DeepLens is primarily a Pytorch project. To set up your development environment, please follow the "How to use" section in the [README.md](./README.md) to create a conda environment and install the necessary dependencies.

A quick summary of the steps:
```
# Create and activate a conda environment
conda env create -f environment.yml -n deeplens_env
conda activate deeplens_env
```
or
```
conda create --name deeplens_env python=3.12
conda activate deeplens_env
pip install -r requirements.txt
```

## Code Formatting

Ruff is the canonical Python formatter and linter. Install the development tools with:

```bash
python -m pip install -e ".[dev]"
```

Apply safe lint fixes before formatting only the Python files you changed:

```bash
ruff check --fix path/to/changed.py test/test_changed.py
ruff format path/to/changed.py test/test_changed.py
```

Before submitting a change, run the repository-wide read-only gates:

```bash
ruff check .
ruff format --check .
```

Do not use `--unsafe-fixes` without reviewing the behavior change it proposes.

## Contribution Opportunities

A great place to start looking for contribution ideas is the project's issue tracker on GitHub. You can check out the [open questions project board](https://github.com/users/singer-yang/projects/2) mentioned in the README.

## Proposing Major Changes

For substantial changes to the codebase, it is a good idea to open an issue to propose your change. This allows for discussion with the maintainers and community before you invest significant time in implementation. This helps ensure your contribution aligns with the project's goals and is more likely to be accepted.
