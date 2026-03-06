//! Dependency resolution
//!
//! Resolves dependency graphs using semantic versioning constraints.

use crate::package::{Lockfile, LockedPackage, Manifest, PackageSource};
use semver::{Version, VersionReq};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors that can occur during dependency resolution
#[derive(Debug, Error)]
pub enum ResolverError {
    #[error("Failed to parse version requirement '{req}' for package '{package}': {source}")]
    InvalidVersionReq {
        package: String,
        req: String,
        source: semver::Error,
    },

    #[error("Failed to parse version '{version}' for package '{package}': {source}")]
    InvalidVersion {
        package: String,
        version: String,
        source: semver::Error,
    },

    #[error("No version of '{package}' satisfies requirement '{req}'")]
    NoMatchingVersion { package: String, req: String },

    #[error("Dependency conflict for package '{package}'")]
    Conflict {
        package: String,
        requirements: Vec<String>,
    },

    #[error("Circular dependency detected: {}", .cycle.join(" -> "))]
    CircularDependency { cycle: Vec<String> },

    #[error("Package '{0}' not found in any registry")]
    PackageNotFound(String),
}

/// Dependency resolver that handles version constraints
pub struct Resolver {
    /// Available package versions (package_name -> versions)
    available_versions: HashMap<String, Vec<AvailablePackage>>,

    /// Currently resolved packages
    resolved: HashMap<String, ResolvedPackage>,
}

/// A package version available for installation
#[derive(Debug, Clone)]
pub struct AvailablePackage {
    pub name: String,
    pub version: Version,
    pub source: PackageSource,
    pub dependencies: HashMap<String, VersionReq>,
}

/// A resolved package with exact version
#[derive(Debug, Clone)]
struct ResolvedPackage {
    name: String,
    version: Version,
    source: PackageSource,
    dependencies: Vec<String>,
}

impl Resolver {
    /// Create a new resolver
    pub fn new() -> Self {
        Self {
            available_versions: HashMap::new(),
            resolved: HashMap::new(),
        }
    }

    /// Register available package versions
    pub fn add_available_package(&mut self, package: AvailablePackage) {
        self.available_versions
            .entry(package.name.clone())
            .or_insert_with(Vec::new)
            .push(package);
    }

    /// Resolve dependencies from a manifest
    pub fn resolve(&mut self, manifest: &Manifest) -> Result<Lockfile, ResolverError> {
        // Start with root dependencies
        let mut to_resolve: Vec<(String, VersionReq)> = Vec::new();

        for (name, dep) in &manifest.dependencies {
            if let Some(version_req_str) = dep.version_req() {
                let version_req = VersionReq::parse(version_req_str).map_err(|e| {
                    ResolverError::InvalidVersionReq {
                        package: name.clone(),
                        req: version_req_str.to_string(),
                        source: e,
                    }
                })?;

                to_resolve.push((name.clone(), version_req));
            } else if dep.is_path() {
                // Path dependencies - resolve from local path
                // For now, we'll skip these in resolution (handled separately)
                continue;
            } else if dep.is_git() {
                // Git dependencies - will be handled by fetcher
                continue;
            }
        }

        // Resolve dependencies recursively
        let mut visited = HashSet::new();
        for (name, req) in to_resolve {
            self.resolve_recursive(&name, &req, &mut visited)?;
        }

        // Convert resolved packages to lockfile
        let mut lockfile = Lockfile::new();
        for (_, resolved) in &self.resolved {
            let locked = LockedPackage {
                name: resolved.name.clone(),
                version: resolved.version.to_string(),
                source: resolved.source.clone(),
                checksum: None,
                dependencies: resolved.dependencies.clone(),
            };
            lockfile.add_package(locked);
        }

        Ok(lockfile)
    }

    /// Recursively resolve a single package
    fn resolve_recursive(
        &mut self,
        name: &str,
        req: &VersionReq,
        visited: &mut HashSet<String>,
    ) -> Result<(), ResolverError> {
        // Check for circular dependencies — build full cycle path
        if visited.contains(name) {
            let mut cycle: Vec<String> = visited.iter().cloned().collect();
            cycle.push(name.to_string());
            return Err(ResolverError::CircularDependency { cycle });
        }

        // Check if already resolved
        if let Some(resolved) = self.resolved.get(name) {
            // Verify the resolved version matches the requirement
            if !req.matches(&resolved.version) {
                return Err(ResolverError::Conflict {
                    package: name.to_string(),
                    requirements: vec![req.to_string(), resolved.version.to_string()],
                });
            }
            return Ok(());
        }

        visited.insert(name.to_string());

        // Find matching versions
        let available = self
            .available_versions
            .get(name)
            .ok_or_else(|| ResolverError::PackageNotFound(name.to_string()))?;

        // Filter versions matching the requirement
        let mut matching: Vec<&AvailablePackage> = available
            .iter()
            .filter(|pkg| req.matches(&pkg.version))
            .collect();

        if matching.is_empty() {
            return Err(ResolverError::NoMatchingVersion {
                package: name.to_string(),
                req: req.to_string(),
            });
        }

        // Sort by version (highest first) - prefer latest compatible version
        matching.sort_by(|a, b| b.version.cmp(&a.version));

        // Pick the highest matching version and clone what we need
        let selected = matching[0];
        let selected_name = selected.name.clone();
        let selected_version = selected.version.clone();
        let selected_source = selected.source.clone();
        let selected_deps = selected.dependencies.clone();

        // Resolve dependencies of selected version
        for (dep_name, dep_req) in &selected_deps {
            self.resolve_recursive(dep_name, dep_req, visited)?;
        }

        // Add to resolved set
        let resolved = ResolvedPackage {
            name: selected_name,
            version: selected_version,
            source: selected_source,
            dependencies: selected_deps.keys().cloned().collect(),
        };

        self.resolved.insert(name.to_string(), resolved);
        visited.remove(name);

        Ok(())
    }

    /// Get the resolved version for a package
    pub fn get_resolved_version(&self, name: &str) -> Option<&Version> {
        self.resolved.get(name).map(|r| &r.version)
    }
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::package::Dependency;

    #[test]
    fn test_simple_resolution() {
        let mut resolver = Resolver::new();

        // Add available packages
        resolver.add_available_package(AvailablePackage {
            name: "pkg-a".to_string(),
            version: Version::parse("1.0.0").unwrap(),
            source: PackageSource::Registry("test".to_string()),
            dependencies: HashMap::new(),
        });

        resolver.add_available_package(AvailablePackage {
            name: "pkg-a".to_string(),
            version: Version::parse("1.5.0").unwrap(),
            source: PackageSource::Registry("test".to_string()),
            dependencies: HashMap::new(),
        });

        // Create manifest
        let mut manifest = Manifest {
            package: crate::package::PackageMetadata {
                name: "test".to_string(),
                version: "0.1.0".to_string(),
                authors: vec![],
                license: None,
                description: None,
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec![],
                categories: vec![],
            },
            neurons: vec![],
            dependencies: HashMap::new(),
            python_runtime: None,
            security: None,
        };

        manifest
            .dependencies
            .insert("pkg-a".to_string(), Dependency::Simple("^1.0".to_string()));

        // Resolve
        let lockfile = resolver.resolve(&manifest).unwrap();

        // Should pick 1.5.0 (highest matching version)
        let pkg = lockfile.find_package("pkg-a").unwrap();
        assert_eq!(pkg.version, "1.5.0");
    }

    #[test]
    fn test_transitive_dependencies() {
        let mut resolver = Resolver::new();

        // pkg-a depends on pkg-b
        let mut deps_a = HashMap::new();
        deps_a.insert("pkg-b".to_string(), VersionReq::parse("^1.0").unwrap());

        resolver.add_available_package(AvailablePackage {
            name: "pkg-a".to_string(),
            version: Version::parse("1.0.0").unwrap(),
            source: PackageSource::Registry("test".to_string()),
            dependencies: deps_a,
        });

        // pkg-b has no dependencies
        resolver.add_available_package(AvailablePackage {
            name: "pkg-b".to_string(),
            version: Version::parse("1.0.0").unwrap(),
            source: PackageSource::Registry("test".to_string()),
            dependencies: HashMap::new(),
        });

        let mut manifest = Manifest {
            package: crate::package::PackageMetadata {
                name: "test".to_string(),
                version: "0.1.0".to_string(),
                authors: vec![],
                license: None,
                description: None,
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec![],
                categories: vec![],
            },
            neurons: vec![],
            dependencies: HashMap::new(),
            python_runtime: None,
            security: None,
        };

        manifest
            .dependencies
            .insert("pkg-a".to_string(), Dependency::Simple("^1.0".to_string()));

        // Resolve
        let lockfile = resolver.resolve(&manifest).unwrap();

        // Should have both packages
        assert!(lockfile.find_package("pkg-a").is_some());
        assert!(lockfile.find_package("pkg-b").is_some());
    }

    #[test]
    fn test_version_conflict() {
        let mut resolver = Resolver::new();

        resolver.add_available_package(AvailablePackage {
            name: "pkg-a".to_string(),
            version: Version::parse("1.0.0").unwrap(),
            source: PackageSource::Registry("test".to_string()),
            dependencies: HashMap::new(),
        });

        let mut manifest = Manifest {
            package: crate::package::PackageMetadata {
                name: "test".to_string(),
                version: "0.1.0".to_string(),
                authors: vec![],
                license: None,
                description: None,
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec![],
                categories: vec![],
            },
            neurons: vec![],
            dependencies: HashMap::new(),
            python_runtime: None,
            security: None,
        };

        // Request version that doesn't exist
        manifest
            .dependencies
            .insert("pkg-a".to_string(), Dependency::Simple("^2.0".to_string()));

        // Should fail
        assert!(resolver.resolve(&manifest).is_err());
    }
}
