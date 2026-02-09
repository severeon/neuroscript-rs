//! Dependency loading for NeuroScript packages
//!
//! Bridges the gap between fetched dependencies (on disk) and the compilation
//! pipeline. Loads neuron definitions from fetched packages, validates use
//! statements, and merges everything into a single Program for compilation.

use crate::package::lockfile::{Lockfile, LockedPackage, PackageSource};
use crate::package::manifest::Manifest;
use crate::{parse, stdlib, NeuronDef, Program, UseStmt, ValidationError};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during dependency loading
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("Failed to read lockfile: {0}")]
    LockfileError(#[from] crate::package::lockfile::LockfileError),

    #[error("Failed to read manifest for package '{name}': {source}")]
    ManifestError {
        name: String,
        source: crate::package::manifest::ManifestError,
    },

    #[error("Failed to read file '{path}': {source}")]
    IoError { path: String, source: std::io::Error },

    #[error("Failed to parse '{path}' in package '{package}': {message}")]
    ParseError {
        package: String,
        path: String,
        message: String,
    },

    #[error("Package '{name}' not found on disk at '{path}'")]
    PackageNotFound { name: String, path: String },

    #[error("Neuron name conflict: '{neuron}' is defined in both '{package_a}' and '{package_b}'")]
    NeuronConflict {
        neuron: String,
        package_a: String,
        package_b: String,
    },

    #[error("Unknown package '{pkg_source}' in use statement (not declared in Axon.toml dependencies)")]
    UnknownPackage { pkg_source: String },

    #[error("Neuron '{neuron}' is not exported by package '{package}' (exported: {exported})")]
    NeuronNotExported {
        neuron: String,
        package: String,
        exported: String,
    },

    #[error("Could not resolve package '{name}' to a disk location")]
    UnresolvablePath { name: String },
}

/// A loaded package with its parsed neuron definitions
#[derive(Debug, Clone)]
pub struct LoadedPackage {
    /// Package name (from Axon.toml)
    pub name: String,
    /// Package version
    pub version: String,
    /// Path on disk where the package lives
    pub path: PathBuf,
    /// All neurons parsed from the package's .ns files
    pub all_neurons: HashMap<String, NeuronDef>,
    /// Exported neurons (filtered by Axon.toml `neurons` list)
    /// If `neurons` list is empty in manifest, all neurons are exported
    pub exported_neurons: HashMap<String, NeuronDef>,
}

/// Context holding all loaded dependencies, ready for merging
#[derive(Debug, Clone)]
pub struct DependencyContext {
    /// Loaded packages in dependency order (leaves first)
    pub packages: Vec<LoadedPackage>,
    /// Quick lookup: package name -> index in packages vec
    pub package_index: HashMap<String, usize>,
}

impl DependencyContext {
    /// Get a loaded package by name
    pub fn get_package(&self, name: &str) -> Option<&LoadedPackage> {
        self.package_index
            .get(name)
            .and_then(|&idx| self.packages.get(idx))
    }

    /// Get all exported neuron names across all packages
    pub fn all_exported_neurons(&self) -> HashMap<String, &str> {
        let mut result = HashMap::new();
        for pkg in &self.packages {
            for name in pkg.exported_neurons.keys() {
                result.insert(name.clone(), pkg.name.as_str());
            }
        }
        result
    }
}

/// Hash a string using SHA-256 (same algorithm as Registry::hash_string)
fn hash_string(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    format!("{:x}", hasher.finalize())[..16].to_string()
}

/// Recursively collect all .ns files from a directory
fn collect_ns_files(dir: &Path) -> Result<Vec<PathBuf>, LoadError> {
    let mut files = Vec::new();
    if !dir.exists() {
        return Ok(files);
    }
    let entries = fs::read_dir(dir).map_err(|e| LoadError::IoError {
        path: dir.display().to_string(),
        source: e,
    })?;
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            files.extend(collect_ns_files(&path)?);
        } else if path.extension().and_then(|s| s.to_str()) == Some("ns") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

/// Load a single package from its on-disk location.
///
/// Parses all .ns files under `src/` (or root if no src/ exists),
/// then filters exported neurons by the manifest's `neurons` list.
pub fn load_package(name: &str, path: &Path) -> Result<LoadedPackage, LoadError> {
    // Load manifest
    let manifest_path = path.join("Axon.toml");
    let manifest = Manifest::from_path(&manifest_path).map_err(|e| LoadError::ManifestError {
        name: name.to_string(),
        source: e,
    })?;

    // Find .ns files: prefer src/ directory, fall back to root
    let src_dir = path.join("src");
    let search_dir = if src_dir.exists() && src_dir.is_dir() {
        src_dir
    } else {
        path.to_path_buf()
    };

    let ns_files = collect_ns_files(&search_dir)?;

    // Parse all .ns files
    let mut all_neurons: HashMap<String, NeuronDef> = HashMap::new();
    for ns_path in &ns_files {
        let source = fs::read_to_string(ns_path).map_err(|e| LoadError::IoError {
            path: ns_path.display().to_string(),
            source: e,
        })?;

        let program = parse(&source).map_err(|e| LoadError::ParseError {
            package: name.to_string(),
            path: ns_path.display().to_string(),
            message: format!("{}", e),
        })?;

        for (neuron_name, neuron_def) in program.neurons {
            all_neurons.insert(neuron_name, neuron_def);
        }
    }

    // Filter exported neurons
    let exported_neurons = if manifest.neurons.is_empty() {
        // All neurons are exported
        all_neurons.clone()
    } else {
        let mut exported = HashMap::new();
        for export_name in &manifest.neurons {
            if let Some(neuron) = all_neurons.get(export_name) {
                exported.insert(export_name.clone(), neuron.clone());
            }
        }
        exported
    };

    Ok(LoadedPackage {
        name: manifest.package.name,
        version: manifest.package.version,
        path: path.to_path_buf(),
        all_neurons,
        exported_neurons,
    })
}

/// Resolve a locked package to its on-disk path.
fn resolve_package_path(locked: &LockedPackage) -> Result<PathBuf, LoadError> {
    match &locked.source {
        PackageSource::Path(p) => {
            let abs = if p.is_absolute() {
                p.clone()
            } else {
                std::env::current_dir()
                    .map_err(|e| LoadError::IoError {
                        path: ".".to_string(),
                        source: e,
                    })?
                    .join(p)
            };
            if abs.exists() {
                Ok(abs)
            } else {
                Err(LoadError::PackageNotFound {
                    name: locked.name.clone(),
                    path: abs.display().to_string(),
                })
            }
        }
        PackageSource::Git { url, .. } => {
            let url_hash = hash_string(url);
            let home = dirs::home_dir().ok_or_else(|| LoadError::UnresolvablePath {
                name: locked.name.clone(),
            })?;
            let checkout_dir = home.join(".neuroscript").join("git").join(&url_hash);
            if checkout_dir.exists() {
                Ok(checkout_dir)
            } else {
                Err(LoadError::PackageNotFound {
                    name: locked.name.clone(),
                    path: checkout_dir.display().to_string(),
                })
            }
        }
        PackageSource::Registry(_) => Err(LoadError::UnresolvablePath {
            name: locked.name.clone(),
        }),
    }
}

/// Load all dependencies from an Axon.lock file.
///
/// Loads packages in topological (dependency) order so that transitive
/// dependencies are available when validating later packages.
pub fn load_dependencies(lockfile_path: &Path) -> Result<DependencyContext, LoadError> {
    let lockfile = Lockfile::from_path(lockfile_path)?;
    let ordered = lockfile.dependency_order();

    let mut packages = Vec::new();
    let mut package_index = HashMap::new();

    for locked in ordered {
        let pkg_path = resolve_package_path(locked)?;
        let loaded = load_package(&locked.name, &pkg_path)?;
        let idx = packages.len();
        package_index.insert(loaded.name.clone(), idx);
        packages.push(loaded);
    }

    Ok(DependencyContext {
        packages,
        package_index,
    })
}

/// Resolve a single `use` statement against loaded packages.
///
/// Returns the list of neuron names that should be imported.
/// Skips `source == "core"` since that refers to StdlibRegistry primitives.
pub fn resolve_use_stmt(
    use_stmt: &UseStmt,
    dep_ctx: &DependencyContext,
) -> Result<Vec<String>, LoadError> {
    // "core" refers to stdlib primitives, not a fetched package
    if use_stmt.source == "core" {
        return Ok(Vec::new());
    }

    let pkg = dep_ctx
        .get_package(&use_stmt.source)
        .ok_or_else(|| LoadError::UnknownPackage {
            pkg_source: use_stmt.source.clone(),
        })?;

    // Check if the last path component is a wildcard
    let is_wildcard = use_stmt
        .path
        .last()
        .map(|s| s == "*")
        .unwrap_or(false);

    if is_wildcard {
        // Import all exported neurons from the package
        Ok(pkg.exported_neurons.keys().cloned().collect())
    } else {
        // Import a specific neuron by the last path component
        let neuron_name = use_stmt
            .path
            .last()
            .cloned()
            .unwrap_or_default();

        if neuron_name.is_empty() {
            return Ok(Vec::new());
        }

        if pkg.exported_neurons.contains_key(&neuron_name) {
            Ok(vec![neuron_name])
        } else {
            let exported: Vec<String> = pkg.exported_neurons.keys().cloned().collect();
            Err(LoadError::NeuronNotExported {
                neuron: neuron_name,
                package: pkg.name.clone(),
                exported: if exported.is_empty() {
                    "none".to_string()
                } else {
                    exported.join(", ")
                },
            })
        }
    }
}

/// Validate all `use` statements in a program against loaded packages.
///
/// Returns a list of validation errors for unresolvable imports.
/// Skips `source == "core"` (stdlib primitives).
pub fn validate_use_stmts(
    program: &Program,
    dep_ctx: &DependencyContext,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for use_stmt in &program.uses {
        if use_stmt.source == "core" {
            continue;
        }

        match resolve_use_stmt(use_stmt, dep_ctx) {
            Ok(_) => {}
            Err(e) => {
                errors.push(ValidationError::UseError {
                    message: format!("{}", e),
                });
            }
        }
    }

    errors
}

/// Check for neuron name conflicts between packages.
fn check_inter_package_conflicts(
    dep_ctx: &DependencyContext,
) -> Result<(), LoadError> {
    let mut neuron_to_package: HashMap<String, String> = HashMap::new();

    for pkg in &dep_ctx.packages {
        for neuron_name in pkg.exported_neurons.keys() {
            if let Some(existing_pkg) = neuron_to_package.get(neuron_name) {
                return Err(LoadError::NeuronConflict {
                    neuron: neuron_name.clone(),
                    package_a: existing_pkg.clone(),
                    package_b: pkg.name.clone(),
                });
            }
            neuron_to_package.insert(neuron_name.clone(), pkg.name.clone());
        }
    }

    Ok(())
}

/// Merge all dependency neurons, stdlib, and user program into a single Program.
///
/// Precedence (later overrides earlier):
/// 1. Dependency neurons (in dependency order)
/// 2. Standard library neurons
/// 3. User-defined neurons
pub fn merge_all(
    dep_ctx: &DependencyContext,
    stdlib_program: Program,
    user_program: Program,
) -> Result<Program, LoadError> {
    // Check for inter-package conflicts first
    check_inter_package_conflicts(dep_ctx)?;

    // Start with dependency neurons
    let mut merged_neurons: HashMap<String, NeuronDef> = HashMap::new();

    for pkg in &dep_ctx.packages {
        for (name, neuron) in &pkg.exported_neurons {
            merged_neurons.insert(name.clone(), neuron.clone());
        }
    }

    // Layer stdlib on top
    for (name, neuron) in stdlib_program.neurons {
        merged_neurons.insert(name, neuron);
    }

    // Layer user neurons on top (highest precedence)
    let mut merged_uses = Vec::new();
    merged_uses.extend(user_program.uses);

    let mut merged_globals = stdlib_program.globals;
    merged_globals.extend(user_program.globals);

    for (name, neuron) in user_program.neurons {
        merged_neurons.insert(name, neuron);
    }

    Ok(Program {
        uses: merged_uses,
        globals: merged_globals,
        neurons: merged_neurons,
    })
}

/// Convenience function: merge deps + stdlib + user without a DependencyContext
/// (when no deps are loaded, equivalent to stdlib::merge_programs)
pub fn merge_with_deps(
    dep_ctx: Option<&DependencyContext>,
    stdlib_program: Program,
    user_program: Program,
) -> Result<Program, LoadError> {
    match dep_ctx {
        Some(ctx) => merge_all(ctx, stdlib_program, user_program),
        None => Ok(stdlib::merge_programs(stdlib_program, user_program)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_package(dir: &Path, name: &str, neurons: &[&str], ns_content: &str) {
        // Create Axon.toml
        // neurons must come before [package] section to be a top-level key
        let neurons_toml: Vec<String> = neurons.iter().map(|n| format!("\"{}\"", n)).collect();
        let manifest = format!(
            r#"neurons = [{}]

[package]
name = "{}"
version = "0.1.0"
"#,
            neurons_toml.join(", "),
            name,
        );
        fs::write(dir.join("Axon.toml"), manifest).unwrap();

        // Create src/ directory with .ns file
        let src_dir = dir.join("src");
        fs::create_dir_all(&src_dir).unwrap();
        fs::write(src_dir.join(format!("{}.ns", name)), ns_content).unwrap();
    }

    #[test]
    fn test_load_package_basic() {
        let temp = TempDir::new().unwrap();
        let pkg_dir = temp.path().join("test-pkg");
        fs::create_dir_all(&pkg_dir).unwrap();

        create_test_package(
            &pkg_dir,
            "test-pkg",
            &["TestNeuron"],
            r#"
neuron TestNeuron(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#,
        );

        let loaded = load_package("test-pkg", &pkg_dir).unwrap();
        assert_eq!(loaded.name, "test-pkg");
        assert_eq!(loaded.exported_neurons.len(), 1);
        assert!(loaded.exported_neurons.contains_key("TestNeuron"));
    }

    #[test]
    fn test_load_package_all_exported_when_empty_neurons_list() {
        let temp = TempDir::new().unwrap();
        let pkg_dir = temp.path().join("all-pkg");
        fs::create_dir_all(&pkg_dir).unwrap();

        // Empty neurons list -> all exported
        create_test_package(
            &pkg_dir,
            "all-pkg",
            &[],
            r#"
neuron Alpha(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out

neuron Beta(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#,
        );

        let loaded = load_package("all-pkg", &pkg_dir).unwrap();
        assert_eq!(loaded.exported_neurons.len(), 2);
        assert!(loaded.exported_neurons.contains_key("Alpha"));
        assert!(loaded.exported_neurons.contains_key("Beta"));
    }

    #[test]
    fn test_load_package_filtered_exports() {
        let temp = TempDir::new().unwrap();
        let pkg_dir = temp.path().join("filtered-pkg");
        fs::create_dir_all(&pkg_dir).unwrap();

        create_test_package(
            &pkg_dir,
            "filtered-pkg",
            &["Public"],
            r#"
neuron Public(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out

neuron Internal(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#,
        );

        let loaded = load_package("filtered-pkg", &pkg_dir).unwrap();
        assert_eq!(loaded.exported_neurons.len(), 1);
        assert!(loaded.exported_neurons.contains_key("Public"));
        assert!(!loaded.exported_neurons.contains_key("Internal"));
        // But all_neurons should have both
        assert_eq!(loaded.all_neurons.len(), 2);
    }

    #[test]
    fn test_resolve_use_stmt_core_skipped() {
        let ctx = DependencyContext {
            packages: vec![],
            package_index: HashMap::new(),
        };

        let use_stmt = UseStmt {
            source: "core".to_string(),
            path: vec!["nn".to_string(), "*".to_string()],
        };

        let result = resolve_use_stmt(&use_stmt, &ctx).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_use_stmt_wildcard() {
        let temp = TempDir::new().unwrap();
        let pkg_dir = temp.path().join("my-pkg");
        fs::create_dir_all(&pkg_dir).unwrap();

        create_test_package(
            &pkg_dir,
            "my-pkg",
            &["NeuronA", "NeuronB"],
            r#"
neuron NeuronA(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out

neuron NeuronB(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#,
        );

        let loaded = load_package("my-pkg", &pkg_dir).unwrap();
        let ctx = DependencyContext {
            packages: vec![loaded],
            package_index: HashMap::from([("my-pkg".to_string(), 0)]),
        };

        let use_stmt = UseStmt {
            source: "my-pkg".to_string(),
            path: vec!["*".to_string()],
        };

        let mut result = resolve_use_stmt(&use_stmt, &ctx).unwrap();
        result.sort();
        assert_eq!(result, vec!["NeuronA".to_string(), "NeuronB".to_string()]);
    }

    #[test]
    fn test_resolve_use_stmt_specific_neuron() {
        let temp = TempDir::new().unwrap();
        let pkg_dir = temp.path().join("my-pkg");
        fs::create_dir_all(&pkg_dir).unwrap();

        create_test_package(
            &pkg_dir,
            "my-pkg",
            &["NeuronA"],
            r#"
neuron NeuronA(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#,
        );

        let loaded = load_package("my-pkg", &pkg_dir).unwrap();
        let ctx = DependencyContext {
            packages: vec![loaded],
            package_index: HashMap::from([("my-pkg".to_string(), 0)]),
        };

        let use_stmt = UseStmt {
            source: "my-pkg".to_string(),
            path: vec!["NeuronA".to_string()],
        };

        let result = resolve_use_stmt(&use_stmt, &ctx).unwrap();
        assert_eq!(result, vec!["NeuronA".to_string()]);
    }

    #[test]
    fn test_resolve_use_stmt_not_exported() {
        let temp = TempDir::new().unwrap();
        let pkg_dir = temp.path().join("my-pkg");
        fs::create_dir_all(&pkg_dir).unwrap();

        create_test_package(
            &pkg_dir,
            "my-pkg",
            &["NeuronA"],
            r#"
neuron NeuronA(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#,
        );

        let loaded = load_package("my-pkg", &pkg_dir).unwrap();
        let ctx = DependencyContext {
            packages: vec![loaded],
            package_index: HashMap::from([("my-pkg".to_string(), 0)]),
        };

        let use_stmt = UseStmt {
            source: "my-pkg".to_string(),
            path: vec!["NonExistent".to_string()],
        };

        let result = resolve_use_stmt(&use_stmt, &ctx);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, LoadError::NeuronNotExported { .. }));
    }

    #[test]
    fn test_unknown_package_in_use() {
        let ctx = DependencyContext {
            packages: vec![],
            package_index: HashMap::new(),
        };

        let use_stmt = UseStmt {
            source: "nonexistent-pkg".to_string(),
            path: vec!["*".to_string()],
        };

        let result = resolve_use_stmt(&use_stmt, &ctx);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LoadError::UnknownPackage { .. }));
    }

    #[test]
    fn test_inter_package_conflict_detection() {
        let temp = TempDir::new().unwrap();

        // Create two packages that both export "Conflict"
        let pkg_a_dir = temp.path().join("pkg-a");
        fs::create_dir_all(&pkg_a_dir).unwrap();
        create_test_package(
            &pkg_a_dir,
            "pkg-a",
            &["Conflict"],
            r#"
neuron Conflict(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#,
        );

        let pkg_b_dir = temp.path().join("pkg-b");
        fs::create_dir_all(&pkg_b_dir).unwrap();
        create_test_package(
            &pkg_b_dir,
            "pkg-b",
            &["Conflict"],
            r#"
neuron Conflict(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#,
        );

        let loaded_a = load_package("pkg-a", &pkg_a_dir).unwrap();
        let loaded_b = load_package("pkg-b", &pkg_b_dir).unwrap();

        let ctx = DependencyContext {
            packages: vec![loaded_a, loaded_b],
            package_index: HashMap::from([
                ("pkg-a".to_string(), 0),
                ("pkg-b".to_string(), 1),
            ]),
        };

        let result = check_inter_package_conflicts(&ctx);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LoadError::NeuronConflict { .. }));
    }

    #[test]
    fn test_merge_all_precedence() {
        let temp = TempDir::new().unwrap();
        let pkg_dir = temp.path().join("dep-pkg");
        fs::create_dir_all(&pkg_dir).unwrap();

        create_test_package(
            &pkg_dir,
            "dep-pkg",
            &["DepNeuron"],
            r#"
neuron DepNeuron(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#,
        );

        let loaded = load_package("dep-pkg", &pkg_dir).unwrap();
        let ctx = DependencyContext {
            packages: vec![loaded],
            package_index: HashMap::from([("dep-pkg".to_string(), 0)]),
        };

        let stdlib = Program {
            uses: vec![],
            globals: vec![],
            neurons: HashMap::new(),
        };

        let user = Program {
            uses: vec![],
            globals: vec![],
            neurons: HashMap::new(),
        };

        let merged = merge_all(&ctx, stdlib, user).unwrap();
        assert!(merged.neurons.contains_key("DepNeuron"));
    }

    #[test]
    fn test_validate_use_stmts_with_errors() {
        let ctx = DependencyContext {
            packages: vec![],
            package_index: HashMap::new(),
        };

        let program = Program {
            uses: vec![
                UseStmt {
                    source: "core".to_string(),
                    path: vec!["nn".to_string(), "*".to_string()],
                },
                UseStmt {
                    source: "missing-pkg".to_string(),
                    path: vec!["*".to_string()],
                },
            ],
            globals: vec![],
            neurons: HashMap::new(),
        };

        let errors = validate_use_stmts(&program, &ctx);
        // core should be skipped, missing-pkg should error
        assert_eq!(errors.len(), 1);
        assert!(matches!(&errors[0], ValidationError::UseError { .. }));
    }
}
