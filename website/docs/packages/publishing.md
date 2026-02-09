---
sidebar_position: 4
title: Publishing and Security
description: Sign packages and verify integrity with Ed25519 and SHA-256
---

# Publishing and Security

NeuroScript packages include cryptographic security features to ensure integrity and authenticity. Packages are signed with Ed25519 keys and verified with SHA-256 checksums.

## Security Overview

Every published package gets:

- **SHA-256 checksums** for each `.ns` source file and an overall package checksum
- **Ed25519 signature** over the package checksum, proving publisher identity
- **Automatic verification** when consumers fetch the package

## Generating Keys

Before publishing, generate an Ed25519 keypair:

```bash
neuroscript keygen my-package
```

```
✓ Generated Ed25519 keypair 'my-package'
  Private key: ~/.neuroscript/keys/my-package.key
  Public key:  ~/.neuroscript/keys/my-package.pub
```

Keys are stored in `~/.neuroscript/keys/`. Private keys are created with restricted permissions (0600 on Unix).

## Publishing a Package

The `publish` command computes checksums, signs the package, and updates `Axon.toml` with security metadata:

```bash
neuroscript publish
```

```
Publishing my-package v0.1.0
  Computed checksums for 3 files
  Overall checksum: sha256:93cd5af901067ee...
  Signing with key: ~/.neuroscript/keys/my-package.key
✓ Package prepared for distribution
```

### Options

```bash
# Checksums only (no signature)
neuroscript publish --no-sign

# Use a specific key file
neuroscript publish --key /path/to/my.key

# See detailed output
neuroscript publish --verbose
```

### What Publish Does

1. Scans all `.ns` files in `src/`
2. Computes SHA-256 checksum for each file
3. Computes an overall package checksum
4. Signs the overall checksum with your Ed25519 private key
5. Writes the security metadata into `Axon.toml`:

```toml
[security]
publisher-key = "ED25519:29da85daa608..."
signature = "ED25519:a1b2c3d4e5f6..."
checksum = "sha256:93cd5af901067..."

[security.checksums]
"src/attention.ns" = "sha256:4128bddc..."
"src/projection.ns" = "sha256:7f3a2b1c..."
```

After publishing, commit and push your updated `Axon.toml` to make the signed package available.

## Verifying a Package

Verify the integrity and authenticity of a package:

```bash
neuroscript verify
```

```
✓ File checksums: all 3 files verified
✓ Overall checksum: valid
✓ Signature: valid

✓ Package verification passed
```

Use `--verbose` for per-file details:

```bash
neuroscript verify --verbose
```

### What Verify Checks

1. Each `.ns` file matches its recorded SHA-256 checksum
2. The overall checksum matches the combined file checksums
3. The Ed25519 signature is valid for the publisher key

### Verification on Fetch

When you `neuroscript fetch` a git dependency that includes security metadata in its `Axon.toml`, checksums are **automatically verified**. Checksum mismatches produce a hard error (fetch fails). Signature mismatches produce a warning but allow the fetch to proceed.

## Security Formats

| Field | Format | Description |
|-------|--------|-------------|
| Publisher key | `ED25519:<64-hex>` | 32-byte Ed25519 public key |
| Signature | `ED25519:<128-hex>` | 64-byte Ed25519 signature |
| Checksum | `sha256:<64-hex>` | 32-byte SHA-256 hash |
| Key files | Raw hex | 32-byte seed (`.key`) or public key (`.pub`) |

## Workflow: Publish and Consume

### Publisher Side

```bash
# Create and develop the package
neuroscript init my-neurons
cd my-neurons
# ... write neurons in src/ ...

# Generate keys and publish
neuroscript keygen my-neurons
neuroscript publish

# Push to git
git add -A && git commit -m "v0.1.0"
git push origin main
```

### Consumer Side

```bash
# Add and fetch the dependency
neuroscript add my-neurons \
  --git "https://github.com/user/my-neurons.git"
neuroscript fetch

# Checksums are verified automatically during fetch
# Use the neurons in your project
neuroscript compile src/model.ns
```

## Best Practices

- **Commit your lockfile** (`Axon.lock`) to version control for reproducible builds
- **Generate keys per package** rather than using a single global key
- **Never share private keys** (`.key` files) - only the public key is embedded in `Axon.toml`
- **Verify after cloning** - run `neuroscript verify` when working with packages from untrusted sources
- **Use tags for releases** - pin dependencies to git tags rather than branches for stability
