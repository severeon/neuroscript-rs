//! Package registry and cache management
//!
//! Handles fetching packages from git repositories and managing the local cache.

use crate::package::{Dependency, Manifest};
use git2::{build::RepoBuilder, FetchOptions, Repository};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during registry operations
#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("Failed to clone repository: {0}")]
    CloneError(#[from] git2::Error),

    #[error("Failed to read directory: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Package not found: {0}")]
    PackageNotFound(String),

    #[error("Invalid package structure: {0}")]
    InvalidStructure(String),

    #[error("Manifest error: {0}")]
    ManifestError(#[from] crate::package::ManifestError),

    #[error("Cache directory not found")]
    CacheNotFound,

    #[error("Security verification failed for package '{name}': {reason}")]
    SecurityVerificationFailed { name: String, reason: String },
}

/// Package registry and cache manager
pub struct Registry {
    /// Path to the cache directory (~/.neuroscript)
    cache_root: PathBuf,
}

impl Registry {
    /// Create a new registry with default cache location
    pub fn new() -> Result<Self, RegistryError> {
        let cache_root = Self::default_cache_dir()?;
        Ok(Self { cache_root })
    }

    /// Create a registry with a custom cache location
    pub fn with_cache_dir(cache_root: PathBuf) -> Self {
        Self { cache_root }
    }

    /// Get the default cache directory (~/.neuroscript)
    pub fn default_cache_dir() -> Result<PathBuf, RegistryError> {
        let home = dirs::home_dir().ok_or(RegistryError::CacheNotFound)?;
        Ok(home.join(".neuroscript"))
    }

    /// Initialize the cache directory structure
    pub fn init(&self) -> Result<(), RegistryError> {
        // Create directory structure:
        // ~/.neuroscript/
        // ├── registry/
        // │   ├── index/       # Git repo with package metadata
        // │   └── cache/       # Downloaded package contents
        // └── git/             # Git checkouts for git dependencies

        fs::create_dir_all(self.cache_root.join("registry/index"))?;
        fs::create_dir_all(self.cache_root.join("registry/cache"))?;
        fs::create_dir_all(self.cache_root.join("git"))?;

        Ok(())
    }

    /// Fetch a git dependency
    pub fn fetch_git(
        &self,
        name: &str,
        url: &str,
        branch: Option<&str>,
        tag: Option<&str>,
        rev: Option<&str>,
    ) -> Result<PathBuf, RegistryError> {
        // Create a unique directory name based on URL hash
        let url_hash = Self::hash_string(url);
        let checkout_dir = self.cache_root.join("git").join(&url_hash);

        // Clone or update repository
        let repo = if checkout_dir.exists() {
            // Repository exists, try to open it
            Repository::open(&checkout_dir)?
        } else {
            // Clone the repository
            println!("Cloning {} ...", url);
            RepoBuilder::new()
                .fetch_options(Self::fetch_options())
                .clone(url, &checkout_dir)?
        };

        // Checkout the requested ref
        if let Some(rev) = rev {
            // Checkout specific revision
            let oid = git2::Oid::from_str(rev)
                .map_err(|e| git2::Error::from_str(&format!("Invalid revision: {}", e)))?;
            let commit = repo.find_commit(oid)?;
            repo.checkout_tree(commit.as_object(), None)?;
            repo.set_head_detached(oid)?;
        } else if let Some(tag) = tag {
            // Checkout tag
            let refname = format!("refs/tags/{}", tag);
            let reference = repo.find_reference(&refname)?;
            repo.checkout_tree(reference.peel_to_tree()?.as_object(), None)?;
            repo.set_head(&refname)?;
        } else if let Some(branch) = branch {
            // Checkout branch
            let refname = format!("refs/remotes/origin/{}", branch);
            let reference = repo.find_reference(&refname)?;
            repo.checkout_tree(reference.peel_to_tree()?.as_object(), None)?;
            let branch_refname = format!("refs/heads/{}", branch);
            repo.set_head(&branch_refname)?;
        } else {
            // Checkout default branch (usually main or master)
            let head = repo.head()?;
            repo.checkout_tree(head.peel_to_tree()?.as_object(), None)?;
        }

        // Verify package structure
        let manifest_path = checkout_dir.join("Axon.toml");
        if !manifest_path.exists() {
            return Err(RegistryError::InvalidStructure(format!(
                "No Axon.toml found in git repository for package '{}'",
                name
            )));
        }

        // Verify package name matches
        let manifest = Manifest::from_path(&manifest_path)?;
        if manifest.package.name != name {
            return Err(RegistryError::InvalidStructure(format!(
                "Package name mismatch: expected '{}', found '{}'",
                name, manifest.package.name
            )));
        }

        // Verify security checksums if present
        if let Some(security) = &manifest.security {
            if !security.checksums.is_empty() {
                match crate::package::security::verify_package(&checkout_dir, security) {
                    Ok(report) => {
                        if !report.checksums_valid {
                            return Err(RegistryError::SecurityVerificationFailed {
                                name: name.to_string(),
                                reason: format!(
                                    "Checksum mismatch for files: {}",
                                    report.failed_files.join(", ")
                                ),
                            });
                        }
                        if let Some(false) = report.signature_valid {
                            return Err(RegistryError::SecurityVerificationFailed {
                                name: name.to_string(),
                                reason: "Ed25519 signature verification failed".to_string(),
                            });
                        }
                        if report.signature_valid == Some(true) {
                            println!("  ✓ Signature verified for '{}'", name);
                        }
                    }
                    Err(e) => {
                        return Err(RegistryError::SecurityVerificationFailed {
                            name: name.to_string(),
                            reason: format!("verification error: {}", e),
                        });
                    }
                }
            }
        }

        Ok(checkout_dir)
    }

    /// Fetch a path dependency (just verify it exists)
    pub fn resolve_path(&self, name: &str, path: &Path) -> Result<PathBuf, RegistryError> {
        let abs_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };

        if !abs_path.exists() {
            return Err(RegistryError::PackageNotFound(name.to_string()));
        }

        let manifest_path = abs_path.join("Axon.toml");
        if !manifest_path.exists() {
            return Err(RegistryError::InvalidStructure(format!(
                "No Axon.toml found at path '{}'",
                abs_path.display()
            )));
        }

        // Verify package name
        let manifest = Manifest::from_path(&manifest_path)?;
        if manifest.package.name != name {
            return Err(RegistryError::InvalidStructure(format!(
                "Package name mismatch: expected '{}', found '{}'",
                name, manifest.package.name
            )));
        }

        Ok(abs_path)
    }

    /// Fetch all dependencies from a manifest
    pub fn fetch_dependencies(&self, manifest: &Manifest) -> Result<Vec<(String, PathBuf)>, RegistryError> {
        let mut fetched = Vec::new();

        for (name, dep) in &manifest.dependencies {
            let path = match dep {
                Dependency::Simple(_version) => {
                    // For simple version dependencies, we'd fetch from central registry
                    // For now, skip these (Phase 3)
                    println!("Skipping registry dependency: {} (not yet implemented)", name);
                    continue;
                }
                Dependency::Detailed(detail) => {
                    if let Some(git_url) = &detail.git {
                        // Git dependency
                        self.fetch_git(
                            name,
                            git_url,
                            detail.branch.as_deref(),
                            detail.tag.as_deref(),
                            detail.rev.as_deref(),
                        )?
                    } else if let Some(path) = &detail.path {
                        // Path dependency
                        self.resolve_path(name, path)?
                    } else if detail.version.is_some() {
                        // Registry dependency with detailed spec
                        println!("Skipping registry dependency: {} (not yet implemented)", name);
                        continue;
                    } else {
                        return Err(RegistryError::InvalidStructure(format!(
                            "Dependency '{}' has no source (git, path, or version)",
                            name
                        )));
                    }
                }
            };

            fetched.push((name.clone(), path));
        }

        Ok(fetched)
    }

    /// Clear the cache
    pub fn clear_cache(&self) -> Result<(), RegistryError> {
        if self.cache_root.exists() {
            fs::remove_dir_all(&self.cache_root)?;
        }
        self.init()?;
        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Result<CacheStats, RegistryError> {
        let mut stats = CacheStats::default();

        // Count git checkouts
        let git_dir = self.cache_root.join("git");
        if git_dir.exists() {
            for entry in fs::read_dir(git_dir)? {
                let entry = entry?;
                if entry.path().is_dir() {
                    stats.git_checkouts += 1;
                    stats.total_size += Self::dir_size(&entry.path())?;
                }
            }
        }

        // Count cached packages
        let cache_dir = self.cache_root.join("registry/cache");
        if cache_dir.exists() {
            for entry in fs::read_dir(cache_dir)? {
                let entry = entry?;
                if entry.path().is_dir() {
                    stats.cached_packages += 1;
                    stats.total_size += Self::dir_size(&entry.path())?;
                }
            }
        }

        Ok(stats)
    }

    /// Hash a string using SHA-256
    fn hash_string(s: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(s.as_bytes());
        format!("{:x}", hasher.finalize())[..16].to_string()
    }

    /// Get fetch options for git operations
    fn fetch_options() -> FetchOptions<'static> {
        let fo = FetchOptions::new();
        // Set callbacks for authentication if needed in the future
        fo
    }

    /// Calculate size of a directory recursively
    fn dir_size(path: &Path) -> Result<u64, std::io::Error> {
        let mut size = 0;
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    size += fs::metadata(&path)?.len();
                } else if path.is_dir() {
                    size += Self::dir_size(&path)?;
                }
            }
        }
        Ok(size)
    }
}

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStats {
    pub git_checkouts: usize,
    pub cached_packages: usize,
    pub total_size: u64,
}

impl CacheStats {
    /// Format size in human-readable format
    pub fn format_size(&self) -> String {
        let size = self.total_size as f64;
        if size < 1024.0 {
            format!("{} B", size)
        } else if size < 1024.0 * 1024.0 {
            format!("{:.2} KB", size / 1024.0)
        } else if size < 1024.0 * 1024.0 * 1024.0 {
            format!("{:.2} MB", size / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", size / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_registry_init() {
        let temp_dir = TempDir::new().unwrap();
        let registry = Registry::with_cache_dir(temp_dir.path().to_path_buf());

        registry.init().unwrap();

        assert!(temp_dir.path().join("registry/index").exists());
        assert!(temp_dir.path().join("registry/cache").exists());
        assert!(temp_dir.path().join("git").exists());
    }

    #[test]
    fn test_hash_string() {
        let hash1 = Registry::hash_string("https://github.com/user/repo");
        let hash2 = Registry::hash_string("https://github.com/user/repo");
        let hash3 = Registry::hash_string("https://github.com/other/repo");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 16); // We truncate to 16 chars
    }
}
