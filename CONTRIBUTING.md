# Contributing to presto-query-predictor

Thanks for your interest in **presto-query-predictor**. The project goal is to introduce machine
learning techniques to the Presto ecosystem and to help gain insights of the Presto system.

## Getting Started

We recommend you follow the [readme](https://github.com/prestodb/presto-query-predictor/blob/main/README.md)
first to get familiar with the project. Contributors would need to install essential dependencies
by running the following commands.

``` bash
pip3 install -e .  # Installs the presto-query-predictor package locally
pip3 install -r requirements.txt  # Installs dependencies
```

The unit tests are located in the `tests` folder. We suggest using `pytest` to run all tests.

## Contributions

We welcome contributions from everyone. Contributions to presto-query-predictor generally
should follow the same guidelines as [Presto's](https://github.com/prestodb/presto/blob/master/CONTRIBUTING.md).

## Code Style

The project leverages [Black](https://github.com/psf/black) as the code formatter and 
[reorder-python-imports](https://github.com/asottile/reorder_python_imports) to format imports.
Black defaults to 88 characters per line (10% over 80), while this project still uses 80 characters
per line. We recommend contributors to run the following commands before submitting pull requests.

```bash
black [changed-file].py --line-length 79
reorder-python-imports [changed-file].py
```

## Maintainers
In addition to PrestoDB committers, this project is also maintained by the individuals below,
who have committer rights to this repository:

* [Chunxu Tang](https://github.com/chunxutang)
* [Beinan Wang](https://github.com/beinan)
