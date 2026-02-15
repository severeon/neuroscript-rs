---
sidebar_position: 5
title: CLI Reference
description: Complete reference for all package management commands
---
All package management commands are subcommands of the `neuroscript` CLI.

## `neuroscript init`

Initialize a new NeuroScript package.

```bash
neuroscript init <NAME> [OPTIONS]
```

### Arguments [init]

| Argument | Description |
| ---------- | ------------- |
| `<NAME>` | Package name (lowercase, alphanumeric + hyphens) |

### Options [init]

| Option | Description |
| -------- | ------------- |
| `--author <AUTHOR>` | Author in `"Name <email>"` format |
| `--license <LICENSE>` | License identifier (e.g., `MIT`, `Apache-2.0`) |
| `--version <VERSION>` | Initial version (default: `0.1.0`) |
| `--bin` | Include an `examples/` directory |
| `--path <DIR>` | Create in a specific directory |

### Example [init]

```bash
neuroscript init my-neurons \
  --author "Your Name <you@example.com>" \
  --license MIT \
  --bin
```

---

## `neuroscript add`

Add a dependency to the current package.

```bash
neuroscript add <NAME> [OPTIONS]
```

### Arguments [add]

| Argument | Description |
| ---------- | ------------- |
| `<NAME>` | Dependency name |

### Options [add]

| Option | Description |
| -------- | ------------- |
| `--version <VERSION>` | Version constraint (e.g., `"^1.0"`, `"0.3"`) |
| `--git <URL>` | Git repository URL |
| `--branch <BRANCH>` | Git branch to track |
| `--tag <TAG>` | Git tag to pin |
| `--rev <REV>` | Git commit hash to pin |
| `--path <PATH>` | Local filesystem path |

### Examples [add]

```bash
# Git dependency with branch
neuroscript add blocks --git "https://github.com/user/blocks.git" --branch main

# Local path dependency
neuroscript add utils --path ../utils

# Version constraint (future registry)
neuroscript add core --version "^1.0"
```

---

## `neuroscript fetch`

Fetch all declared dependencies and generate a lockfile.

```bash
neuroscript fetch [OPTIONS]
```

### Options [fetch]

| Option | Description |
| -------- | ------------- |
| `--verbose` | Show detailed fetch progress |
| `--update` | Update dependencies to latest compatible versions |

### Example [fetch]

```bash
neuroscript fetch --verbose
```

---

## `neuroscript keygen`

Generate an Ed25519 signing keypair.

```bash
neuroscript keygen <NAME>
```

### Arguments [keygen]

| Argument | Description |
| ---------- | ------------- |
| `<NAME>` | Key name (used as filename prefix) |

Keys are stored in `~/.neuroscript/keys/`:

- `<NAME>.key` - Private key (restricted permissions)
- `<NAME>.pub` - Public key

### Example [keygen]

```bash
neuroscript keygen my-package
```

---

## `neuroscript publish`

Compute checksums, sign the package, and update `Axon.toml` with security metadata.

```bash
neuroscript publish [OPTIONS]
```

### Options [publish]

| Option | Description |
| -------- | ------------- |
| `--no-sign` | Compute checksums only, skip signing |
| `--key <PATH>` | Path to Ed25519 private key |
| `--verbose` | Show detailed output |

### Example [publish]

```bash
neuroscript publish --verbose
neuroscript publish --no-sign
neuroscript publish --key ~/.neuroscript/keys/custom.key
```

---

## `neuroscript verify`

Verify package checksums and signature.

```bash
neuroscript verify [OPTIONS]
```

### Options [verify]

| Option | Description |
| -------- | ------------- |
| `--verbose` | Show per-file verification details |

### Example [verify]

```bash
neuroscript verify --verbose
```

---

## `neuroscript list` (Package Flags)

List available neurons from various sources.

```bash
neuroscript list [FILE] [OPTIONS]
```

### Arguments [list]

| Argument | Description |
| ---------- | ------------- |
| `[FILE]` | NeuroScript file to list neurons from (optional with `--stdlib`/`--available`) |

### Package-Related Options [list]

| Option | Description |
| -------- | ------------- |
| `--stdlib` | List all primitives and stdlib composites |
| `--package <NAME>` | List neurons from a specific fetched dependency |
| `--available` | List everything: stdlib + all dependencies |
| `--verbose` | Show connection details |

### Examples [list]

```bash
# List primitives and standard library neurons
neuroscript list --stdlib

# List neurons from a specific dependency
neuroscript list --package attention-blocks

# List all available neurons (stdlib + all deps)
neuroscript list --available

# List neurons in a file
neuroscript list src/model.ns --verbose
```

---

## `neuroscript validate` / `neuroscript compile` (Dependency Flags)

Both `validate` and `compile` automatically load dependencies from `Axon.lock` when present.

### Dependency Options [validate]

| Option | Description |
| -------- | ------------- |
| `--no-deps` | Skip dependency loading |

### Examples [validate]

```bash
# Compile with dependencies loaded automatically
neuroscript compile src/model.ns

# Compile without loading dependencies
neuroscript compile src/model.ns --no-deps

# Validate with dependency neurons in scope
neuroscript validate src/model.ns
```

For the full list of compiler options (including `--bundle`, `--no-optimize`, and others), see the [Compiler Reference](/docs/compiler).
