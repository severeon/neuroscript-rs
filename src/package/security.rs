//! Package security: Ed25519 signing, SHA-256 checksums, and verification
//!
//! Provides cryptographic signing and verification for NeuroScript packages.
//! Packages are signed with Ed25519 keys and verified using SHA-256 checksums.

use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::package::manifest::{Manifest, Security};

/// Errors that can occur during security operations
#[derive(Debug, Error)]
pub enum SecurityError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid key format: {0}")]
    InvalidKeyFormat(String),

    #[error("Invalid signature format: {0}")]
    InvalidSignatureFormat(String),

    #[error("Checksum mismatch for {file}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        file: String,
        expected: String,
        actual: String,
    },

    #[error("Hex decode error: {0}")]
    HexDecode(#[from] hex::FromHexError),

    #[error("Ed25519 error: {0}")]
    Ed25519(#[from] ed25519_dalek::SignatureError),

    #[error("Manifest error: {0}")]
    Manifest(#[from] crate::package::ManifestError),

    #[error("No security metadata in manifest")]
    NoSecurityMetadata,

    #[error("TOML serialization error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),
}

/// Result of verifying a package
#[derive(Debug)]
pub struct VerificationReport {
    /// Whether all file checksums match
    pub checksums_valid: bool,
    /// Whether the overall checksum matches
    pub overall_checksum_valid: bool,
    /// Whether the signature is valid (None if no signature present)
    pub signature_valid: Option<bool>,
    /// Files that failed checksum verification
    pub failed_files: Vec<String>,
    /// Files that were added (not in manifest)
    pub extra_files: Vec<String>,
    /// Files that were removed (in manifest but missing)
    pub missing_files: Vec<String>,
}

impl VerificationReport {
    /// Whether the package passes all checks
    pub fn is_valid(&self) -> bool {
        self.checksums_valid
            && self.overall_checksum_valid
            && self.signature_valid.unwrap_or(true)
            && self.extra_files.is_empty()
            && self.missing_files.is_empty()
    }
}

// =============================================================================
// Key Management
// =============================================================================

/// Generate a new Ed25519 keypair
pub fn generate_keypair() -> (SigningKey, VerifyingKey) {
    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key = signing_key.verifying_key();
    (signing_key, verifying_key)
}

/// Save a keypair to disk at `~/.neuroscript/keys/{name}.key` and `.pub`
pub fn save_keypair(name: &str, signing_key: &SigningKey) -> Result<(PathBuf, PathBuf), SecurityError> {
    let keys_dir = keys_directory()?;
    fs::create_dir_all(&keys_dir)?;

    let key_path = keys_dir.join(format!("{}.key", name));
    let pub_path = keys_dir.join(format!("{}.pub", name));

    // Save private key as hex-encoded seed (32 bytes)
    let seed = signing_key.to_bytes();
    fs::write(&key_path, hex::encode(seed))?;

    // Save public key as hex-encoded bytes (32 bytes)
    let verifying_key = signing_key.verifying_key();
    fs::write(&pub_path, hex::encode(verifying_key.to_bytes()))?;

    // Set restrictive permissions on private key (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        fs::set_permissions(&key_path, perms)?;
    }

    Ok((key_path, pub_path))
}

/// Load a signing key from a file (hex-encoded 32-byte seed)
pub fn load_signing_key(path: &Path) -> Result<SigningKey, SecurityError> {
    let contents = fs::read_to_string(path)?.trim().to_string();
    let bytes = hex::decode(&contents)?;

    if bytes.len() != 32 {
        return Err(SecurityError::InvalidKeyFormat(format!(
            "Expected 32-byte seed, got {} bytes",
            bytes.len()
        )));
    }

    let mut seed = [0u8; 32];
    seed.copy_from_slice(&bytes);
    Ok(SigningKey::from_bytes(&seed))
}

/// Parse a publisher key string in "ED25519:<hex>" format
pub fn parse_publisher_key(s: &str) -> Result<VerifyingKey, SecurityError> {
    let hex_str = s
        .strip_prefix("ED25519:")
        .ok_or_else(|| SecurityError::InvalidKeyFormat(
            "Publisher key must start with 'ED25519:'".to_string(),
        ))?;

    let bytes = hex::decode(hex_str)?;
    if bytes.len() != 32 {
        return Err(SecurityError::InvalidKeyFormat(format!(
            "Expected 32-byte public key, got {} bytes",
            bytes.len()
        )));
    }

    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(&bytes);
    Ok(VerifyingKey::from_bytes(&key_bytes)?)
}

/// Format a verifying key as "ED25519:<hex>"
pub fn format_publisher_key(key: &VerifyingKey) -> String {
    format!("ED25519:{}", hex::encode(key.to_bytes()))
}

/// Get the keys directory (~/.neuroscript/keys/)
fn keys_directory() -> Result<PathBuf, SecurityError> {
    let home = dirs::home_dir().ok_or_else(|| {
        SecurityError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Home directory not found",
        ))
    })?;
    Ok(home.join(".neuroscript").join("keys"))
}

/// Try to auto-discover a signing key for a package name
pub fn discover_signing_key(package_name: &str) -> Result<Option<PathBuf>, SecurityError> {
    let keys_dir = keys_directory()?;
    let key_path = keys_dir.join(format!("{}.key", package_name));

    if key_path.exists() {
        Ok(Some(key_path))
    } else {
        Ok(None)
    }
}

// =============================================================================
// Checksums
// =============================================================================

/// Compute SHA-256 checksums for all .ns files in a package directory
pub fn compute_checksums(package_dir: &Path) -> Result<BTreeMap<String, String>, SecurityError> {
    let src_dir = package_dir.join("src");
    let mut checksums = BTreeMap::new();

    if src_dir.exists() {
        collect_checksums(&src_dir, package_dir, &mut checksums)?;
    }

    // Also check for .ns files directly in the package root
    if let Ok(entries) = fs::read_dir(package_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |e| e == "ns") {
                let relative = path
                    .strip_prefix(package_dir)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string();
                let hash = hash_file(&path)?;
                checksums.insert(relative, format!("sha256:{}", hash));
            }
        }
    }

    Ok(checksums)
}

/// Recursively collect checksums for .ns files
fn collect_checksums(
    dir: &Path,
    base: &Path,
    checksums: &mut BTreeMap<String, String>,
) -> Result<(), SecurityError> {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_checksums(&path, base, checksums)?;
            } else if path.extension().map_or(false, |e| e == "ns") {
                let relative = path
                    .strip_prefix(base)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string();
                let hash = hash_file(&path)?;
                checksums.insert(relative, format!("sha256:{}", hash));
            }
        }
    }
    Ok(())
}

/// Compute SHA-256 hash of a file, returning hex string
fn hash_file(path: &Path) -> Result<String, SecurityError> {
    let contents = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&contents);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute an overall checksum from per-file checksums (deterministic)
pub fn compute_overall_checksum(checksums: &BTreeMap<String, String>) -> String {
    let mut hasher = Sha256::new();
    // BTreeMap iteration is deterministic (sorted by key)
    for (path, hash) in checksums {
        hasher.update(format!("{}:{}\n", path, hash).as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

// =============================================================================
// Signing & Verification
// =============================================================================

/// Sign a checksum string with an Ed25519 signing key
pub fn sign_checksum(checksum: &str, signing_key: &SigningKey) -> String {
    let signature = signing_key.sign(checksum.as_bytes());
    format!("ED25519:{}", hex::encode(signature.to_bytes()))
}

/// Verify a signature against a checksum and public key
pub fn verify_signature(
    checksum: &str,
    signature_str: &str,
    verifying_key: &VerifyingKey,
) -> Result<bool, SecurityError> {
    let sig_hex = signature_str
        .strip_prefix("ED25519:")
        .ok_or_else(|| SecurityError::InvalidSignatureFormat(
            "Signature must start with 'ED25519:'".to_string(),
        ))?;

    let sig_bytes = hex::decode(sig_hex)?;
    if sig_bytes.len() != 64 {
        return Err(SecurityError::InvalidSignatureFormat(format!(
            "Expected 64-byte signature, got {} bytes",
            sig_bytes.len()
        )));
    }

    let mut bytes = [0u8; 64];
    bytes.copy_from_slice(&sig_bytes);
    let signature = ed25519_dalek::Signature::from_bytes(&bytes);

    match verifying_key.verify(checksum.as_bytes(), &signature) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Verify an entire package directory against its security metadata
pub fn verify_package(
    package_dir: &Path,
    security: &Security,
) -> Result<VerificationReport, SecurityError> {
    let mut report = VerificationReport {
        checksums_valid: true,
        overall_checksum_valid: true,
        signature_valid: None,
        failed_files: Vec::new(),
        extra_files: Vec::new(),
        missing_files: Vec::new(),
    };

    // Recompute file checksums
    let actual_checksums = compute_checksums(package_dir)?;

    // Compare per-file checksums
    for (path, expected_hash) in &security.checksums {
        match actual_checksums.get(path) {
            Some(actual_hash) => {
                if actual_hash != expected_hash {
                    report.checksums_valid = false;
                    report.failed_files.push(path.clone());
                }
            }
            None => {
                report.checksums_valid = false;
                report.missing_files.push(path.clone());
            }
        }
    }

    // Check for extra files not in manifest
    for path in actual_checksums.keys() {
        if !security.checksums.contains_key(path) {
            report.extra_files.push(path.clone());
        }
    }

    // Verify overall checksum
    if let Some(expected_overall) = &security.checksum {
        let actual_overall = compute_overall_checksum(&actual_checksums);
        if &actual_overall != expected_overall {
            report.overall_checksum_valid = false;
        }
    }

    // Verify signature if present
    if let (Some(signature), Some(publisher_key)) = (&security.signature, &security.publisher_key) {
        if let Some(checksum) = &security.checksum {
            let verifying_key = parse_publisher_key(publisher_key)?;
            let valid = verify_signature(checksum, signature, &verifying_key)?;
            report.signature_valid = Some(valid);
        }
    }

    Ok(report)
}

/// Update the [security] section of an Axon.toml file
pub fn update_manifest_security(
    manifest_path: &Path,
    security: Security,
) -> Result<(), SecurityError> {
    let mut manifest = Manifest::from_path(manifest_path)?;
    manifest.security = Some(security);

    let toml_string = toml::to_string_pretty(&manifest)?;
    fs::write(manifest_path, toml_string)?;

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_and_load_keypair() {
        let temp_dir = TempDir::new().unwrap();
        let (signing_key, verifying_key) = generate_keypair();

        // Save key manually to temp dir
        let key_path = temp_dir.path().join("test.key");
        let seed = signing_key.to_bytes();
        fs::write(&key_path, hex::encode(seed)).unwrap();

        // Load it back
        let loaded = load_signing_key(&key_path).unwrap();
        assert_eq!(loaded.verifying_key(), verifying_key);
    }

    #[test]
    fn test_format_parse_publisher_key() {
        let (_, verifying_key) = generate_keypair();
        let formatted = format_publisher_key(&verifying_key);
        assert!(formatted.starts_with("ED25519:"));

        let parsed = parse_publisher_key(&formatted).unwrap();
        assert_eq!(parsed, verifying_key);
    }

    #[test]
    fn test_invalid_publisher_key_prefix() {
        let result = parse_publisher_key("RSA:abcdef");
        assert!(result.is_err());
    }

    #[test]
    fn test_sign_and_verify() {
        let (signing_key, verifying_key) = generate_keypair();
        let checksum = "sha256:abcdef1234567890";

        let signature = sign_checksum(checksum, &signing_key);
        assert!(signature.starts_with("ED25519:"));

        let valid = verify_signature(checksum, &signature, &verifying_key).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_verify_wrong_key() {
        let (signing_key, _) = generate_keypair();
        let (_, wrong_key) = generate_keypair();
        let checksum = "sha256:abcdef1234567890";

        let signature = sign_checksum(checksum, &signing_key);
        let valid = verify_signature(checksum, &signature, &wrong_key).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_verify_tampered_checksum() {
        let (signing_key, verifying_key) = generate_keypair();
        let checksum = "sha256:abcdef1234567890";

        let signature = sign_checksum(checksum, &signing_key);
        let valid =
            verify_signature("sha256:tampered0000000000", &signature, &verifying_key).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_compute_checksums() {
        let temp_dir = TempDir::new().unwrap();

        // Create src/ with .ns files
        let src = temp_dir.path().join("src");
        fs::create_dir(&src).unwrap();
        fs::write(src.join("neuron.ns"), "neuron Test:\n  in: [*]\n  out: [*]\n  impl: core,nn/Test\n").unwrap();

        let checksums = compute_checksums(temp_dir.path()).unwrap();
        assert_eq!(checksums.len(), 1);
        assert!(checksums.contains_key("src/neuron.ns"));
        assert!(checksums["src/neuron.ns"].starts_with("sha256:"));
    }

    #[test]
    fn test_checksums_deterministic() {
        let temp_dir = TempDir::new().unwrap();
        let src = temp_dir.path().join("src");
        fs::create_dir(&src).unwrap();
        fs::write(src.join("a.ns"), "content a").unwrap();
        fs::write(src.join("b.ns"), "content b").unwrap();

        let checksums1 = compute_checksums(temp_dir.path()).unwrap();
        let checksums2 = compute_checksums(temp_dir.path()).unwrap();

        assert_eq!(checksums1, checksums2);

        let overall1 = compute_overall_checksum(&checksums1);
        let overall2 = compute_overall_checksum(&checksums2);
        assert_eq!(overall1, overall2);
    }

    #[test]
    fn test_verify_package_valid() {
        let temp_dir = TempDir::new().unwrap();
        let src = temp_dir.path().join("src");
        fs::create_dir(&src).unwrap();
        fs::write(src.join("test.ns"), "neuron Test:\n  in: [*]\n  out: [*]\n  impl: core,nn/Test\n").unwrap();

        let checksums = compute_checksums(temp_dir.path()).unwrap();
        let overall = compute_overall_checksum(&checksums);

        let security = Security {
            publisher_key: None,
            signature: None,
            checksum: Some(overall),
            checksums,
        };

        let report = verify_package(temp_dir.path(), &security).unwrap();
        assert!(report.is_valid());
    }

    #[test]
    fn test_verify_package_tampered() {
        let temp_dir = TempDir::new().unwrap();
        let src = temp_dir.path().join("src");
        fs::create_dir(&src).unwrap();
        fs::write(src.join("test.ns"), "original content").unwrap();

        let checksums = compute_checksums(temp_dir.path()).unwrap();
        let overall = compute_overall_checksum(&checksums);

        let security = Security {
            publisher_key: None,
            signature: None,
            checksum: Some(overall),
            checksums,
        };

        // Tamper with the file
        fs::write(src.join("test.ns"), "tampered content").unwrap();

        let report = verify_package(temp_dir.path(), &security).unwrap();
        assert!(!report.is_valid());
        assert!(!report.checksums_valid);
        assert_eq!(report.failed_files.len(), 1);
    }

    #[test]
    fn test_full_sign_verify_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let src = temp_dir.path().join("src");
        fs::create_dir(&src).unwrap();
        fs::write(src.join("test.ns"), "neuron content").unwrap();

        // Generate keys
        let (signing_key, verifying_key) = generate_keypair();

        // Compute checksums and sign
        let checksums = compute_checksums(temp_dir.path()).unwrap();
        let overall = compute_overall_checksum(&checksums);
        let signature = sign_checksum(&overall, &signing_key);

        let security = Security {
            publisher_key: Some(format_publisher_key(&verifying_key)),
            signature: Some(signature),
            checksum: Some(overall),
            checksums,
        };

        // Verify
        let report = verify_package(temp_dir.path(), &security).unwrap();
        assert!(report.is_valid());
        assert_eq!(report.signature_valid, Some(true));
    }
}
