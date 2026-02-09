//! Package manifest (Axon.toml) parsing and types
//!
//! Defines the structure of Axon.toml files and provides parsing logic.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur when working with manifests
#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("Failed to read manifest file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse manifest: {0}")]
    ParseError(#[from] toml::de::Error),

    #[error("Invalid manifest: {0}")]
    ValidationError(String),

    #[error("Manifest not found at path: {0}")]
    NotFound(PathBuf),
}

/// The root structure of an Axon.toml file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub package: PackageMetadata,

    #[serde(default)]
    pub neurons: Vec<String>,

    #[serde(default)]
    pub dependencies: HashMap<String, Dependency>,

    #[serde(rename = "python-runtime")]
    #[serde(default)]
    pub python_runtime: Option<PythonRuntime>,

    #[serde(default)]
    pub security: Option<Security>,
}

/// Package metadata section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageMetadata {
    pub name: String,
    pub version: String,

    #[serde(default)]
    pub authors: Vec<String>,

    #[serde(default)]
    pub license: Option<String>,

    #[serde(default)]
    pub description: Option<String>,

    #[serde(default)]
    pub repository: Option<String>,

    #[serde(default)]
    pub homepage: Option<String>,

    #[serde(default)]
    pub documentation: Option<String>,

    #[serde(default)]
    pub keywords: Vec<String>,

    #[serde(default)]
    pub categories: Vec<String>,
}

/// Dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dependency {
    /// Simple version string: `core-primitives = "1.2.0"`
    Simple(String),

    /// Detailed specification
    Detailed(DependencyDetail),
}

/// Detailed dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyDetail {
    /// Version requirement (e.g., "^1.2.0", ">=1.0,<2.0")
    #[serde(default)]
    pub version: Option<String>,

    /// Git repository URL
    #[serde(default)]
    pub git: Option<String>,

    /// Git branch
    #[serde(default)]
    pub branch: Option<String>,

    /// Git tag
    #[serde(default)]
    pub tag: Option<String>,

    /// Git revision (commit hash)
    #[serde(default)]
    pub rev: Option<String>,

    /// Local filesystem path (for development)
    #[serde(default)]
    pub path: Option<PathBuf>,

    /// Whether this is an optional dependency
    #[serde(default)]
    pub optional: bool,
}

/// Python runtime requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonRuntime {
    /// Python package requirements (e.g., ["torch>=2.0", "einops>=0.6"])
    #[serde(default)]
    pub requires: Vec<String>,
}

/// Security metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Security {
    /// Publisher's public key fingerprint (ED25519:<hex>)
    #[serde(rename = "publisher-key")]
    #[serde(default)]
    pub publisher_key: Option<String>,

    /// Package signature (ED25519:<hex>)
    #[serde(default)]
    pub signature: Option<String>,

    /// Overall package checksum (sha256:<hex>)
    #[serde(default)]
    pub checksum: Option<String>,

    /// Per-file checksums (BTreeMap for deterministic serialization)
    #[serde(default)]
    pub checksums: BTreeMap<String, String>,
}

impl Manifest {
    /// Load a manifest from a file path
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, ManifestError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(ManifestError::NotFound(path.to_path_buf()));
        }

        let contents = std::fs::read_to_string(path)?;
        Self::from_str(&contents)
    }

    /// Parse a manifest from a string
    pub fn from_str(contents: &str) -> Result<Self, ManifestError> {
        let manifest: Manifest = toml::from_str(contents)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validate the manifest structure
    pub fn validate(&self) -> Result<(), ManifestError> {
        // Validate package name
        if self.package.name.is_empty() {
            return Err(ManifestError::ValidationError(
                "Package name cannot be empty".to_string(),
            ));
        }

        if !is_valid_package_name(&self.package.name) {
            return Err(ManifestError::ValidationError(
                format!("Invalid package name '{}': must contain only lowercase letters, numbers, and hyphens", self.package.name),
            ));
        }

        // Validate version
        if semver::Version::parse(&self.package.version).is_err() {
            return Err(ManifestError::ValidationError(
                format!("Invalid version '{}': must be valid semver", self.package.version),
            ));
        }

        // Validate neuron names
        for neuron in &self.neurons {
            if neuron.is_empty() {
                return Err(ManifestError::ValidationError(
                    "Neuron name cannot be empty".to_string(),
                ));
            }
        }

        // Validate dependencies
        for (name, dep) in &self.dependencies {
            if name.is_empty() {
                return Err(ManifestError::ValidationError(
                    "Dependency name cannot be empty".to_string(),
                ));
            }

            // Validate version requirement if present
            if let Some(version) = dep.version_req() {
                if semver::VersionReq::parse(version).is_err() {
                    return Err(ManifestError::ValidationError(
                        format!("Invalid version requirement '{}' for dependency '{}'", version, name),
                    ));
                }
            }

            // Validate that git deps have at least one ref
            if let Dependency::Detailed(detail) = dep {
                if detail.git.is_some() && detail.path.is_some() {
                    return Err(ManifestError::ValidationError(
                        format!("Dependency '{}' cannot specify both git and path", name),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Find the Axon.toml file starting from a directory and searching upwards
    pub fn find_in_directory<P: AsRef<Path>>(start: P) -> Result<PathBuf, ManifestError> {
        let mut current = start.as_ref().to_path_buf();

        loop {
            let manifest_path = current.join("Axon.toml");
            if manifest_path.exists() {
                return Ok(manifest_path);
            }

            // Try parent directory
            if let Some(parent) = current.parent() {
                current = parent.to_path_buf();
            } else {
                return Err(ManifestError::NotFound(start.as_ref().to_path_buf()));
            }
        }
    }
}

impl Dependency {
    /// Get the version requirement string if present
    pub fn version_req(&self) -> Option<&str> {
        match self {
            Dependency::Simple(version) => Some(version.as_str()),
            Dependency::Detailed(detail) => detail.version.as_deref(),
        }
    }

    /// Check if this is a git dependency
    pub fn is_git(&self) -> bool {
        match self {
            Dependency::Simple(_) => false,
            Dependency::Detailed(detail) => detail.git.is_some(),
        }
    }

    /// Check if this is a path dependency
    pub fn is_path(&self) -> bool {
        match self {
            Dependency::Simple(_) => false,
            Dependency::Detailed(detail) => detail.path.is_some(),
        }
    }
}

/// Validate a package name according to NeuroScript conventions
/// Package names must:
/// - Be lowercase
/// - Contain only letters, numbers, and hyphens
/// - Not start or end with a hyphen
/// - Not contain consecutive hyphens
fn is_valid_package_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    // Check first and last character
    if name.starts_with('-') || name.ends_with('-') {
        return false;
    }

    // Check for consecutive hyphens
    if name.contains("--") {
        return false;
    }

    // Check all characters
    name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_manifest() {
        let toml = r#"
            [package]
            name = "test-package"
            version = "0.1.0"
        "#;

        let manifest = Manifest::from_str(toml).unwrap();
        assert_eq!(manifest.package.name, "test-package");
        assert_eq!(manifest.package.version, "0.1.0");
        assert!(manifest.neurons.is_empty());
        assert!(manifest.dependencies.is_empty());
    }

    #[test]
    fn test_parse_full_manifest() {
        let toml = r#"
neurons = ["MultiHeadAttention", "ScaledDotProduct"]

[package]
name = "attention-mechanisms"
version = "0.1.0"
authors = ["Test Author <test@example.com>"]
license = "MIT"
description = "Self-attention neurons"
repository = "https://github.com/user/attention"

[dependencies]
core-primitives = "1.2.0"
residual-blocks = { version = "0.3", git = "https://github.com/org/residual" }
local-dev = { path = "../local-dev" }

[python-runtime]
requires = ["torch>=2.0", "einops>=0.6"]

[security]
publisher-key = "ED25519:abc123"
checksum = "sha256:def456"
        "#;

        let manifest = Manifest::from_str(toml).unwrap();
        assert_eq!(manifest.package.name, "attention-mechanisms");
        assert_eq!(manifest.neurons.len(), 2);
        assert_eq!(manifest.dependencies.len(), 3);
        assert!(manifest.python_runtime.is_some());
        assert!(manifest.security.is_some());
    }

    #[test]
    fn test_invalid_package_name() {
        let toml = r#"
            [package]
            name = "Invalid_Name"
            version = "0.1.0"
        "#;

        assert!(Manifest::from_str(toml).is_err());
    }

    #[test]
    fn test_invalid_version() {
        let toml = r#"
            [package]
            name = "test-package"
            version = "not-a-version"
        "#;

        assert!(Manifest::from_str(toml).is_err());
    }

    #[test]
    fn test_valid_package_names() {
        assert!(is_valid_package_name("core-primitives"));
        assert!(is_valid_package_name("attention123"));
        assert!(is_valid_package_name("my-cool-neurons"));

        assert!(!is_valid_package_name("Invalid_Name"));
        assert!(!is_valid_package_name("-starts-with-hyphen"));
        assert!(!is_valid_package_name("ends-with-hyphen-"));
        assert!(!is_valid_package_name("double--hyphen"));
        assert!(!is_valid_package_name(""));
    }
}
