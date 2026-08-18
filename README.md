# disca

Design docs: [docs/architecture.md](docs/architecture.md) · [docs/bridge.md](docs/bridge.md)

## White Paper: The Disca Specification

The description and formal specification of the Disca protocol.

Built with XeLaTex.

### Setup

#### Pre-commit

Install the [`pre-commit` CLI tool](https://pre-commit.com/), available via brew on macOS:

```sh
brew install pre-commit
```

Install the pre-commit hooks:

```sh
pre-commit install
```

This hook is necessary to ensure that the document is properly formatted and spell-checked.

#### XeLaTex

Install XeLaTex, available via brew on macOS:

```sh
brew install basictex
```

### Build

```sh
make
```

### Clean

```sh
make clean
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.