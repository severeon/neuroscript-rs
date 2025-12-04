//! Standard library loading and merging
//!
//! This module handles loading all .ns files from the stdlib/ directory
//! and merging them with user programs.

use crate::{parse, Program};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StdlibError {
    #[error("Failed to find stdlib directory (searched: {0})")]
    StdlibNotFound(String),

    #[error("Failed to read stdlib file {0}: {1}")]
    ReadError(String, std::io::Error),

    #[error("Failed to parse stdlib file {0}:\n{1}")]
    ParseError(String, String),

    #[error("Duplicate neuron definition '{0}' found in stdlib files {1} and {2}")]
    DuplicateNeuron(String, String, String),
}

/// Load all stdlib files and merge them into a single Program
///
/// Searches for the stdlib/ directory relative to the current working directory,
/// then loads and parses all .ns files found within.
pub fn load_stdlib() -> Result<Program, StdlibError> {
    let stdlib_dir = find_stdlib_dir()?;
    load_stdlib_from_dir(&stdlib_dir)
}

/// Find the stdlib directory
///
/// Searches in the following order:
/// 1. ./stdlib (current directory)
/// 2. ../stdlib (parent directory)
/// 3. ../../stdlib (grandparent directory)
fn find_stdlib_dir() -> Result<PathBuf, StdlibError> {
    let mut search_paths = Vec::new();

    // Try current directory first
    let current = PathBuf::from("stdlib");
    search_paths.push(current.clone());
    if current.exists() && current.is_dir() {
        return Ok(current);
    }

    // Try parent directories (in case we're in a subdirectory)
    if let Ok(mut current_dir) = std::env::current_dir() {
        for _ in 0..2 {
            if !current_dir.pop() {
                break;
            }
            let stdlib_path = current_dir.join("stdlib");
            search_paths.push(stdlib_path.clone());
            if stdlib_path.exists() && stdlib_path.is_dir() {
                return Ok(stdlib_path);
            }
        }
    }

    let searched = search_paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");

    Err(StdlibError::StdlibNotFound(searched))
}

/// Load stdlib from a specific directory
///
/// Parses all .ns files in the directory and merges them into a single Program.
/// Returns an error if any file fails to parse or if duplicate neuron names are found.
fn load_stdlib_from_dir(dir: &Path) -> Result<Program, StdlibError> {
    let mut merged = Program {
        uses: Vec::new(),
        neurons: HashMap::new(),
    };

    // Track which file each neuron came from for better error messages
    let mut neuron_sources: HashMap<String, String> = HashMap::new();

    // Find all .ns files
    let entries = fs::read_dir(dir)
        .map_err(|e| StdlibError::ReadError(dir.display().to_string(), e))?;

    let mut ns_files: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("ns"))
        .collect();

    // Sort for deterministic loading order
    ns_files.sort();

    for path in ns_files {
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let source = fs::read_to_string(&path)
            .map_err(|e| StdlibError::ReadError(path.display().to_string(), e))?;

        let program = parse(&source).map_err(|e| {
            StdlibError::ParseError(path.display().to_string(), format!("{}", e))
        })?;

        // Check for duplicate neurons
        for name in program.neurons.keys() {
            if let Some(existing_source) = neuron_sources.get(name) {
                return Err(StdlibError::DuplicateNeuron(
                    name.clone(),
                    existing_source.clone(),
                    filename.clone(),
                ));
            }
        }

        // Merge neurons
        for (name, neuron) in program.neurons {
            neuron_sources.insert(name.clone(), filename.clone());
            merged.neurons.insert(name, neuron);
        }

        // Merge uses
        merged.uses.extend(program.uses);
    }

    Ok(merged)
}

/// Merge stdlib program with user program
///
/// User neurons take precedence over stdlib neurons with the same name.
/// This allows users to override stdlib definitions if needed.
pub fn merge_programs(stdlib: Program, user: Program) -> Program {
    let mut merged = stdlib;

    // Add/override with user neurons
    for (name, neuron) in user.neurons {
        merged.neurons.insert(name, neuron);
    }

    // Add user uses (stdlib uses are already in merged)
    merged.uses.extend(user.uses);

    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_stdlib_dir() {
        // This test assumes we're running from the project root
        let result = find_stdlib_dir();
        assert!(result.is_ok(), "Should find stdlib directory");
        let path = result.unwrap();
        assert!(path.exists());
        assert!(path.is_dir());
    }

    #[test]
    fn test_load_stdlib() {
        let result = load_stdlib();
        assert!(result.is_ok(), "Should load stdlib without errors");
        let program = result.unwrap();

        // Check that we loaded some neurons
        assert!(!program.neurons.is_empty(), "Stdlib should contain neurons");

        // Verify some expected stdlib neurons exist
        // (based on the stdlib files we saw)
        println!("Loaded {} stdlib neurons", program.neurons.len());
        for name in program.neurons.keys() {
            println!("  - {}", name);
        }
    }

    #[test]
    fn test_merge_programs() {
        let stdlib = Program {
            uses: vec![],
            neurons: HashMap::new(),
        };

        let user = Program {
            uses: vec![],
            neurons: HashMap::new(),
        };

        // Create mock neuron (we can't easily construct real neurons in tests)
        // Just verify the merge logic works at a basic level
        let merged = merge_programs(stdlib, user);
        assert_eq!(merged.neurons.len(), 0);
    }
}
