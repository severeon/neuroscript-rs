//! Lockfile (Axon.lock) management
//!
//! Provides functionality for generating, parsing, and managing Axon.lock files
//! which pin exact dependency versions for reproducible builds.

use crate::package::manifest::Manifest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur when working with lockfiles
#[derive(Debug, Error)]
pub enum LockfileError {
    #[error("Failed to read lockfile: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse lockfile: {0}")]
    ParseError(#[from] toml::de::Error),

    #[error("Failed to serialize lockfile: {0}")]
    SerializeError(#[from] toml::ser::Error),

    #[error("Lockfile not found at path: {0}")]
    NotFound(PathBuf),
}

/// The root structure of an Axon.lock file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lockfile {
    /// Version of the lockfile format
    #[serde(default = "default_version")]
    pub version: u32,

    /// All resolved packages with exact versions
    #[serde(rename = "package")]
    pub packages: Vec<LockedPackage>,
}

fn default_version() -> u32 {
    1
}

/// A single locked package with exact version and source
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LockedPackage {
    /// Package name
    pub name: String,

    /// Exact version (no ranges, e.g., "1.2.3")
    pub version: String,

    /// Where this package comes from
    pub source: PackageSource,

    /// Optional checksum for verification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,

    /// Dependencies of this package (for graph traversal)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<String>,
}

/// Source of a package (registry, git, or path)
///
/// Serialized as a prefixed string:
///   - `"registry+https://..."` for registry sources
///   - `"git+https://...?rev=abc123"` for git sources
///   - `"path+/some/path"` for local path sources
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PackageSource {
    /// From a registry (e.g., "registry+https://axons.neuroscript.org")
    Registry(String),

    /// From a git repository with revision
    Git {
        url: String,
        rev: String,
    },

    /// From a local path (for development)
    Path(PathBuf),
}

impl Serialize for PackageSource {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let s = match self {
            PackageSource::Registry(url) => {
                if url.starts_with("registry+") {
                    url.clone()
                } else {
                    format!("registry+{}", url)
                }
            }
            PackageSource::Git { url, rev } => format!("git+{}?rev={}", url, rev),
            PackageSource::Path(path) => format!("path+{}", path.display()),
        };
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for PackageSource {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;

        if let Some(rest) = s.strip_prefix("path+") {
            Ok(PackageSource::Path(PathBuf::from(rest)))
        } else if let Some(rest) = s.strip_prefix("git+") {
            // Parse "git+url?rev=hash"
            if let Some((url, query)) = rest.split_once('?') {
                let rev = query
                    .strip_prefix("rev=")
                    .unwrap_or(query)
                    .to_string();
                Ok(PackageSource::Git {
                    url: url.to_string(),
                    rev,
                })
            } else {
                Ok(PackageSource::Git {
                    url: rest.to_string(),
                    rev: String::new(),
                })
            }
        } else if let Some(rest) = s.strip_prefix("registry+") {
            Ok(PackageSource::Registry(format!("registry+{}", rest)))
        } else {
            // Legacy: bare path (no prefix) — treat as path if it looks like a filesystem path
            if s.starts_with('/') || s.starts_with('.') || s.contains('/') && !s.contains("://") {
                Ok(PackageSource::Path(PathBuf::from(&s)))
            } else {
                // Assume registry
                Ok(PackageSource::Registry(s))
            }
        }
    }
}

impl Lockfile {
    /// Create a new empty lockfile
    pub fn new() -> Self {
        Self {
            version: 1,
            packages: Vec::new(),
        }
    }

    /// Load a lockfile from a file path
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, LockfileError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(LockfileError::NotFound(path.to_path_buf()));
        }

        let contents = std::fs::read_to_string(path)?;
        Self::from_str(&contents)
    }

    /// Parse a lockfile from a string
    pub fn from_str(contents: &str) -> Result<Self, LockfileError> {
        let lockfile: Lockfile = toml::from_str(contents)?;
        Ok(lockfile)
    }

    /// Save the lockfile to a file path
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), LockfileError> {
        let contents = self.to_string()?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Serialize the lockfile to a TOML string
    pub fn to_string(&self) -> Result<String, LockfileError> {
        // Add header comment
        let mut output = String::from("# This file is @generated by neuroscript\n");
        output.push_str("# It is not intended for manual editing\n\n");

        // Serialize packages
        let toml = toml::to_string_pretty(self)?;
        output.push_str(&toml);

        Ok(output)
    }

    /// Add a package to the lockfile
    pub fn add_package(&mut self, package: LockedPackage) {
        // Remove any existing package with the same name and version
        self.packages.retain(|p| !(p.name == package.name && p.version == package.version));
        self.packages.push(package);
    }

    /// Find a package in the lockfile by name
    pub fn find_package(&self, name: &str) -> Option<&LockedPackage> {
        self.packages.iter().find(|p| p.name == name)
    }

    /// Find a package by name and version
    pub fn find_exact(&self, name: &str, version: &str) -> Option<&LockedPackage> {
        self.packages.iter().find(|p| p.name == name && p.version == version)
    }

    /// Check if the lockfile is up-to-date with the manifest
    /// Returns true if all manifest dependencies are in the lockfile with compatible versions
    pub fn is_up_to_date(&self, manifest: &Manifest) -> bool {
        for (dep_name, dep_spec) in &manifest.dependencies {
            if let Some(version_req) = dep_spec.version_req() {
                // Parse version requirement
                let req = match semver::VersionReq::parse(version_req) {
                    Ok(r) => r,
                    Err(_) => return false,
                };

                // Find in lockfile
                match self.find_package(dep_name) {
                    Some(locked) => {
                        // Check if locked version satisfies requirement
                        let locked_version = match semver::Version::parse(&locked.version) {
                            Ok(v) => v,
                            Err(_) => return false,
                        };

                        if !req.matches(&locked_version) {
                            return false;
                        }
                    }
                    None => return false,
                }
            }
        }

        true
    }

    /// Get all packages in dependency order (topological sort)
    pub fn dependency_order(&self) -> Vec<&LockedPackage> {
        let mut visited = std::collections::HashSet::new();
        let mut order = Vec::new();
        let pkg_map: HashMap<&str, &LockedPackage> =
            self.packages.iter().map(|p| (p.name.as_str(), p)).collect();

        fn visit<'a>(
            name: &str,
            pkg_map: &HashMap<&str, &'a LockedPackage>,
            visited: &mut std::collections::HashSet<String>,
            order: &mut Vec<&'a LockedPackage>,
        ) {
            if visited.contains(name) {
                return;
            }
            visited.insert(name.to_string());

            if let Some(pkg) = pkg_map.get(name) {
                // Visit dependencies first
                for dep in &pkg.dependencies {
                    visit(dep, pkg_map, visited, order);
                }
                order.push(*pkg);
            }
        }

        for pkg in &self.packages {
            visit(&pkg.name, &pkg_map, &mut visited, &mut order);
        }

        order
    }
}

impl Default for Lockfile {
    fn default() -> Self {
        Self::new()
    }
}

impl LockedPackage {
    /// Create a new locked package from registry
    pub fn from_registry(name: String, version: String, registry: &str) -> Self {
        Self {
            name,
            version,
            source: PackageSource::Registry(format!("registry+{}", registry)),
            checksum: None,
            dependencies: Vec::new(),
        }
    }

    /// Create a new locked package from git
    pub fn from_git(name: String, version: String, url: String, rev: String) -> Self {
        Self {
            name,
            version,
            source: PackageSource::Git { url, rev },
            checksum: None,
            dependencies: Vec::new(),
        }
    }

    /// Create a new locked package from local path
    pub fn from_path(name: String, version: String, path: PathBuf) -> Self {
        Self {
            name,
            version,
            source: PackageSource::Path(path),
            checksum: None,
            dependencies: Vec::new(),
        }
    }
}

impl std::fmt::Display for PackageSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackageSource::Registry(url) => write!(f, "{}", url),
            PackageSource::Git { url, rev } => write!(f, "git+{}?rev={}", url, rev),
            PackageSource::Path(path) => write!(f, "path+{}", path.display()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lockfile_serialization() {
        let mut lockfile = Lockfile::new();

        lockfile.add_package(LockedPackage::from_registry(
            "core-primitives".to_string(),
            "1.2.3".to_string(),
            "https://axons.neuroscript.org",
        ));

        lockfile.add_package(LockedPackage::from_git(
            "attention-blocks".to_string(),
            "0.3.1".to_string(),
            "https://github.com/org/attention".to_string(),
            "abc123".to_string(),
        ));

        let serialized = lockfile.to_string().unwrap();
        assert!(serialized.contains("core-primitives"));
        assert!(serialized.contains("1.2.3"));
        assert!(serialized.contains("attention-blocks"));

        // Round-trip
        let parsed = Lockfile::from_str(&serialized).unwrap();
        assert_eq!(parsed.packages.len(), 2);
    }

    #[test]
    fn test_find_package() {
        let mut lockfile = Lockfile::new();
        lockfile.add_package(LockedPackage::from_registry(
            "test-package".to_string(),
            "1.0.0".to_string(),
            "https://example.com",
        ));

        assert!(lockfile.find_package("test-package").is_some());
        assert!(lockfile.find_package("nonexistent").is_none());
    }

    #[test]
    fn test_dependency_order() {
        let mut lockfile = Lockfile::new();

        let mut pkg_a = LockedPackage::from_registry("a".to_string(), "1.0.0".to_string(), "");
        pkg_a.dependencies = vec!["b".to_string()];

        let pkg_b = LockedPackage::from_registry("b".to_string(), "1.0.0".to_string(), "");

        lockfile.add_package(pkg_a);
        lockfile.add_package(pkg_b);

        let order = lockfile.dependency_order();
        assert_eq!(order.len(), 2);
        // b should come before a
        assert_eq!(order[0].name, "b");
        assert_eq!(order[1].name, "a");
    }
}
