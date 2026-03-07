# Contributing to NeuroScript

Thanks for your interest in contributing to NeuroScript! This guide will help you get started.

## Getting Started

### Prerequisites

- Rust toolchain (1.70+)
- Python 3.8+ with PyTorch (for runtime testing)

### Setup

```bash
git clone https://github.com/severeon/neuroscript-rs.git
cd neuroscript-rs
cargo build --release
pip install -e .  # Install Python runtime
```

### Verify

```bash
cargo test                           # Run all tests
./target/release/neuroscript --help  # Check CLI
```

## Development Workflow

1. Fork the repository and create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Run `cargo test` and `cargo check` before pushing
4. Open a pull request against `main`

### Key Commands

```bash
cargo check                          # Fast syntax check
cargo test                           # All unit tests
cargo test --test integration_tests  # Snapshot tests
cargo insta review                   # Review snapshot changes
./test_examples.sh                   # Parse all .ns files
```

## Project Structure

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation, module organization, and development guidelines. (This file also serves as configuration for AI coding assistants such as Claude Code — human contributors can skip the agent-workflow sections and focus on the architecture and build command reference.)

## What to Work On

Check [open issues](https://github.com/severeon/neuroscript-rs/issues) for tasks labeled `good first issue` or `help wanted`. Issues are categorized by component:

- `parser` — Grammar and parsing
- `validator` — Validation passes
- `shape-system` — Shape inference and algebra
- `codegen` — Code generation
- `tooling` — Developer tools (LSP, extensions)
- `tech-debt` — Refactoring opportunities

## Code Style

- Follow standard Rust conventions (`cargo fmt`, `cargo clippy`)
- Use `thiserror` for error types and `miette` for user-facing diagnostics
- Collect all validation errors rather than failing fast
- Add tests for new features — snapshot tests for codegen, unit tests for logic

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
