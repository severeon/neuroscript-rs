//! Package initialization (`axon init` command)
//!
//! Creates a new NeuroScript package with default structure.

use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during package initialization
#[derive(Debug, Error)]
pub enum InitError {
    #[error("Failed to create directory: {0}")]
    DirCreationError(std::io::Error),

    #[error("Failed to write file: {0}")]
    FileWriteError(std::io::Error),

    #[error("Package already exists at: {0}")]
    AlreadyExists(PathBuf),

    #[error("Invalid package name: {0}")]
    InvalidName(String),
}

/// Options for initializing a new package
#[derive(Debug, Clone)]
pub struct InitOptions {
    /// Package name
    pub name: String,

    /// Directory to create package in (defaults to current directory)
    pub path: Option<PathBuf>,

    /// Package version
    pub version: String,

    /// Author name and email
    pub author: Option<String>,

    /// License
    pub license: Option<String>,

    /// Create a binary package (with examples)
    pub bin: bool,
}

impl Default for InitOptions {
    fn default() -> Self {
        Self {
            name: String::new(),
            path: None,
            version: "0.1.0".to_string(),
            author: None,
            license: Some("MIT".to_string()),
            bin: false,
        }
    }
}

/// Initialize a new NeuroScript package
pub fn init_package(options: &InitOptions) -> Result<PathBuf, InitError> {
    // Determine package directory
    let package_dir = if let Some(path) = &options.path {
        path.clone()
    } else {
        PathBuf::from(&options.name)
    };

    // Check if directory already exists
    if package_dir.exists() {
        let axon_toml = package_dir.join("Axon.toml");
        if axon_toml.exists() {
            return Err(InitError::AlreadyExists(package_dir));
        }
    }

    // Create directory structure
    create_directory_structure(&package_dir)?;

    // Generate files
    create_axon_toml(&package_dir, options)?;
    create_readme(&package_dir, options)?;
    create_gitignore(&package_dir)?;

    // Create src directory with example neuron
    create_src_directory(&package_dir, options)?;

    // Create examples directory if binary package
    if options.bin {
        create_examples_directory(&package_dir)?;
    }

    Ok(package_dir)
}

/// Create the directory structure for a new package
fn create_directory_structure(package_dir: &Path) -> Result<(), InitError> {
    fs::create_dir_all(package_dir).map_err(InitError::DirCreationError)?;
    fs::create_dir_all(package_dir.join("src")).map_err(InitError::DirCreationError)?;

    Ok(())
}

/// Generate Axon.toml file
fn create_axon_toml(package_dir: &Path, options: &InitOptions) -> Result<(), InitError> {
    let mut content = format!(
        r#"[package]
name = "{}"
version = "{}"
"#,
        options.name, options.version
    );

    if let Some(author) = &options.author {
        content.push_str(&format!("authors = [\"{}\"]\n", author));
    }

    if let Some(license) = &options.license {
        content.push_str(&format!("license = \"{}\"\n", license));
    }

    content.push_str(&format!(
        r#"description = "A NeuroScript package"

# List the neurons this package provides
neurons = []

# Dependencies on other NeuroScript packages
[dependencies]
# Example: core-primitives = "1.0.0"

# Python runtime requirements
[python-runtime]
requires = ["torch>=2.0"]
"#
    ));

    let path = package_dir.join("Axon.toml");
    fs::write(&path, content).map_err(InitError::FileWriteError)?;

    Ok(())
}

/// Generate README.md file
fn create_readme(package_dir: &Path, options: &InitOptions) -> Result<(), InitError> {
    let content = format!(
        r#"# {}

A NeuroScript package for neural architecture composition.

## Overview

TODO: Add description

## Usage

```neuroscript
use {}

// Your neurons here
```

## Installation

```bash
axon add {}
```

## Development

Build the package:
```bash
axon build
```

Run tests:
```bash
axon test
```

## License

{}
"#,
        options.name,
        options.name,
        options.name,
        options.license.as_deref().unwrap_or("MIT")
    );

    let path = package_dir.join("README.md");
    fs::write(&path, content).map_err(InitError::FileWriteError)?;

    Ok(())
}

/// Generate .gitignore file
fn create_gitignore(package_dir: &Path) -> Result<(), InitError> {
    let content = r#"# NeuroScript build artifacts
/target/
Axon.lock

# Python artifacts
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
"#;

    let path = package_dir.join(".gitignore");
    fs::write(&path, content).map_err(InitError::FileWriteError)?;

    Ok(())
}

/// Create src directory with example neuron
fn create_src_directory(package_dir: &Path, options: &InitOptions) -> Result<(), InitError> {
    // Create a simple example neuron
    let neuron_name = to_pascal_case(&options.name);
    let content = format!(
        r#"// Example neuron definition
// Replace this with your actual neuron implementation

neuron {}:
  in input: [*shape]
  out output: [*shape]
  graph:
    input -> Identity() -> output
"#,
        neuron_name
    );

    let filename = format!("{}.ns", options.name.replace('-', "_"));
    let path = package_dir.join("src").join(filename);
    fs::write(&path, content).map_err(InitError::FileWriteError)?;

    Ok(())
}

/// Create examples directory with sample usage
fn create_examples_directory(package_dir: &Path) -> Result<(), InitError> {
    fs::create_dir_all(package_dir.join("examples")).map_err(InitError::DirCreationError)?;

    let content = r#"""
Example usage of this NeuroScript package.

This file demonstrates how to use the neurons defined in this package.
"""

import torch
from neuroscript_runtime import load_module

# Load the generated module
# MyNeuron = load_module("path/to/generated/module.py")

# Example usage
# model = MyNeuron()
# x = torch.randn(1, 512)
# y = model(x)
# print(f"Output shape: {y.shape}")
"#;

    let path = package_dir.join("examples").join("usage.py");
    fs::write(&path, content).map_err(InitError::FileWriteError)?;

    Ok(())
}

/// Convert kebab-case to PascalCase
fn to_pascal_case(s: &str) -> String {
    s.split('-')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("my-neuron"), "MyNeuron");
        assert_eq!(to_pascal_case("attention-block"), "AttentionBlock");
        assert_eq!(to_pascal_case("simple"), "Simple");
    }
}
