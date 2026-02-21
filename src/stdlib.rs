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
/// Parses all .ns files in the directory and its subdirectories, merging them into a single Program.
/// Returns an error if any file fails to parse or if duplicate neuron names are found.
fn load_stdlib_from_dir(dir: &Path) -> Result<Program, StdlibError> {
    let mut merged = Program {
        uses: Vec::new(),
        globals: Vec::new(),
        neurons: HashMap::new(),
    };

    // Track which file each neuron came from for better error messages
    let mut neuron_sources: HashMap<String, String> = HashMap::new();

    // Collect .ns files from directory and subdirectories
    let mut ns_files: Vec<PathBuf> = Vec::new();
    collect_ns_files(dir, &mut ns_files)?;

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

        let program = parse(&source)
            .map_err(|e| StdlibError::ParseError(path.display().to_string(), format!("{}", e)))?;

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

/// Recursively collect all .ns files from a directory and its subdirectories
fn collect_ns_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), StdlibError> {
    let entries =
        fs::read_dir(dir).map_err(|e| StdlibError::ReadError(dir.display().to_string(), e))?;

    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            // Recursively collect from subdirectories
            collect_ns_files(&path, files)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("ns") {
            files.push(path);
        }
    }

    Ok(())
}

/// Load stdlib from sources embedded at compile time.
///
/// This is used in environments without filesystem access (e.g., WASM).
/// All stdlib .ns files are included via `include_str!()` at compile time.
pub fn load_stdlib_embedded() -> Result<Program, StdlibError> {
    const STDLIB_SOURCES: &[(&str, &str)] = &[
        // Composite neurons
        ("BottleneckBlock.ns", include_str!("../stdlib/BottleneckBlock.ns")),
        ("Conformer.ns", include_str!("../stdlib/Conformer.ns")),
        ("ConvNeXtBlock.ns", include_str!("../stdlib/ConvNeXtBlock.ns")),
        ("CrossAttention.ns", include_str!("../stdlib/CrossAttention.ns")),
        ("DenseBlock.ns", include_str!("../stdlib/DenseBlock.ns")),
        ("DenseConnection.ns", include_str!("../stdlib/DenseConnection.ns")),
        ("Expert.ns", include_str!("../stdlib/Expert.ns")),
        ("FusedMBConv.ns", include_str!("../stdlib/FusedMBConv.ns")),
        ("FFN.ns", include_str!("../stdlib/FFN.ns")),
        ("GatedFFN.ns", include_str!("../stdlib/GatedFFN.ns")),
        ("GeGLU.ns", include_str!("../stdlib/GeGLU.ns")),
        ("GLU.ns", include_str!("../stdlib/GLU.ns")),
        ("GroupedQueryAttention.ns", include_str!("../stdlib/GroupedQueryAttention.ns")),
        ("HighwayConnection.ns", include_str!("../stdlib/HighwayConnection.ns")),
        ("InceptionBlock.ns", include_str!("../stdlib/InceptionBlock.ns")),
        ("MBConvBlock.ns", include_str!("../stdlib/MBConvBlock.ns")),
        ("MetaNeurons.ns", include_str!("../stdlib/MetaNeurons.ns")),
        ("MultiHeadAttention.ns", include_str!("../stdlib/MultiHeadAttention.ns")),
        ("MultiQueryAttention.ns", include_str!("../stdlib/MultiQueryAttention.ns")),
        ("PatchEmbedding.ns", include_str!("../stdlib/PatchEmbedding.ns")),
        ("PostNormResidual.ns", include_str!("../stdlib/PostNormResidual.ns")),
        ("PreNormResidual.ns", include_str!("../stdlib/PreNormResidual.ns")),
        ("RelativePositionBias.ns", include_str!("../stdlib/RelativePositionBias.ns")),
        ("Residual.ns", include_str!("../stdlib/Residual.ns")),
        ("ResNetBasicBlock.ns", include_str!("../stdlib/ResNetBasicBlock.ns")),
        ("ResNeXtBlock.ns", include_str!("../stdlib/ResNeXtBlock.ns")),
        ("SEBlock.ns", include_str!("../stdlib/SEBlock.ns")),
        ("SwiGLU.ns", include_str!("../stdlib/SwiGLU.ns")),
        ("TransformerBlock.ns", include_str!("../stdlib/TransformerBlock.ns")),
        ("TransformerDecoderBlock.ns", include_str!("../stdlib/TransformerDecoderBlock.ns")),
        ("TransformerEncoderBlock.ns", include_str!("../stdlib/TransformerEncoderBlock.ns")),
        ("TransformerStack.ns", include_str!("../stdlib/TransformerStack.ns")),
        ("ViTBlock.ns", include_str!("../stdlib/ViTBlock.ns")),
        ("WaveNetBlock.ns", include_str!("../stdlib/WaveNetBlock.ns")),
        // Primitives
        ("primitives/AdaptiveAvgPool.ns", include_str!("../stdlib/primitives/AdaptiveAvgPool.ns")),
        ("primitives/AdaptiveMaxPool.ns", include_str!("../stdlib/primitives/AdaptiveMaxPool.ns")),
        ("primitives/Add.ns", include_str!("../stdlib/primitives/Add.ns")),
        ("primitives/AvgPool.ns", include_str!("../stdlib/primitives/AvgPool.ns")),
        ("primitives/BatchNorm.ns", include_str!("../stdlib/primitives/BatchNorm.ns")),
        ("primitives/Bias.ns", include_str!("../stdlib/primitives/Bias.ns")),
        ("primitives/Concat.ns", include_str!("../stdlib/primitives/Concat.ns")),
        ("primitives/ConstScale.ns", include_str!("../stdlib/primitives/ConstScale.ns")),
        ("primitives/Conv1d.ns", include_str!("../stdlib/primitives/Conv1d.ns")),
        ("primitives/Conv2d.ns", include_str!("../stdlib/primitives/Conv2d.ns")),
        ("primitives/Conv3d.ns", include_str!("../stdlib/primitives/Conv3d.ns")),
        ("primitives/DepthwiseConv.ns", include_str!("../stdlib/primitives/DepthwiseConv.ns")),
        ("primitives/DropConnect.ns", include_str!("../stdlib/primitives/DropConnect.ns")),
        ("primitives/DropPath.ns", include_str!("../stdlib/primitives/DropPath.ns")),
        ("primitives/Dropout.ns", include_str!("../stdlib/primitives/Dropout.ns")),
        ("primitives/ELU.ns", include_str!("../stdlib/primitives/ELU.ns")),
        ("primitives/Einsum.ns", include_str!("../stdlib/primitives/Einsum.ns")),
        ("primitives/Embedding.ns", include_str!("../stdlib/primitives/Embedding.ns")),
        ("primitives/Flatten.ns", include_str!("../stdlib/primitives/Flatten.ns")),
        ("primitives/Fork.ns", include_str!("../stdlib/primitives/Fork.ns")),
        ("primitives/Fork3.ns", include_str!("../stdlib/primitives/Fork3.ns")),
        ("primitives/GELU.ns", include_str!("../stdlib/primitives/GELU.ns")),
        ("primitives/GlobalAvgPool.ns", include_str!("../stdlib/primitives/GlobalAvgPool.ns")),
        ("primitives/GlobalMaxPool.ns", include_str!("../stdlib/primitives/GlobalMaxPool.ns")),
        ("primitives/GroupNorm.ns", include_str!("../stdlib/primitives/GroupNorm.ns")),
        ("primitives/Identity.ns", include_str!("../stdlib/primitives/Identity.ns")),
        ("primitives/InstanceNorm.ns", include_str!("../stdlib/primitives/InstanceNorm.ns")),
        ("primitives/LayerNorm.ns", include_str!("../stdlib/primitives/LayerNorm.ns")),
        ("primitives/LearnedPositionalEmbedding.ns", include_str!("../stdlib/primitives/LearnedPositionalEmbedding.ns")),
        ("primitives/Linear.ns", include_str!("../stdlib/primitives/Linear.ns")),
        ("primitives/MatMul.ns", include_str!("../stdlib/primitives/MatMul.ns")),
        ("primitives/MaxPool.ns", include_str!("../stdlib/primitives/MaxPool.ns")),
        ("primitives/Mish.ns", include_str!("../stdlib/primitives/Mish.ns")),
        ("primitives/MultiHeadSelfAttention.ns", include_str!("../stdlib/primitives/MultiHeadSelfAttention.ns")),
        ("primitives/Multiply.ns", include_str!("../stdlib/primitives/Multiply.ns")),
        ("primitives/PReLU.ns", include_str!("../stdlib/primitives/PReLU.ns")),
        ("primitives/Pad.ns", include_str!("../stdlib/primitives/Pad.ns")),
        ("primitives/PositionalEncoding.ns", include_str!("../stdlib/primitives/PositionalEncoding.ns")),
        ("primitives/ReLU.ns", include_str!("../stdlib/primitives/ReLU.ns")),
        ("primitives/RMSNorm.ns", include_str!("../stdlib/primitives/RMSNorm.ns")),
        ("primitives/Reshape.ns", include_str!("../stdlib/primitives/Reshape.ns")),
        ("primitives/RotaryEmbedding.ns", include_str!("../stdlib/primitives/RotaryEmbedding.ns")),
        ("primitives/Scale.ns", include_str!("../stdlib/primitives/Scale.ns")),
        ("primitives/ScaledDotProductAttention.ns", include_str!("../stdlib/primitives/ScaledDotProductAttention.ns")),
        ("primitives/SeparableConv.ns", include_str!("../stdlib/primitives/SeparableConv.ns")),
        ("primitives/SiLU.ns", include_str!("../stdlib/primitives/SiLU.ns")),
        ("primitives/Sigmoid.ns", include_str!("../stdlib/primitives/Sigmoid.ns")),
        ("primitives/Slice.ns", include_str!("../stdlib/primitives/Slice.ns")),
        ("primitives/Softmax.ns", include_str!("../stdlib/primitives/Softmax.ns")),
        ("primitives/Split.ns", include_str!("../stdlib/primitives/Split.ns")),
        ("primitives/Tanh.ns", include_str!("../stdlib/primitives/Tanh.ns")),
        ("primitives/Transpose.ns", include_str!("../stdlib/primitives/Transpose.ns")),
        ("primitives/TransposedConv.ns", include_str!("../stdlib/primitives/TransposedConv.ns")),
    ];

    let mut merged = Program {
        uses: Vec::new(),
        globals: Vec::new(),
        neurons: HashMap::new(),
    };

    let mut neuron_sources: HashMap<String, String> = HashMap::new();

    for (filename, source) in STDLIB_SOURCES {
        let program = parse(source)
            .map_err(|e| StdlibError::ParseError(filename.to_string(), format!("{}", e)))?;

        for name in program.neurons.keys() {
            if let Some(existing_source) = neuron_sources.get(name) {
                return Err(StdlibError::DuplicateNeuron(
                    name.clone(),
                    existing_source.clone(),
                    filename.to_string(),
                ));
            }
        }

        for (name, neuron) in program.neurons {
            neuron_sources.insert(name.clone(), filename.to_string());
            merged.neurons.insert(name, neuron);
        }

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
    fn test_load_stdlib_embedded() {
        let result = load_stdlib_embedded();
        assert!(result.is_ok(), "Should load embedded stdlib without errors: {:?}", result.err());
        let program = result.unwrap();

        // Should contain neurons from all stdlib files
        assert!(!program.neurons.is_empty(), "Embedded stdlib should contain neurons");

        // Verify key neurons from both composite and primitive files exist
        let expected_composites = ["FFN", "Residual", "TransformerBlock", "CrossAttention"];
        for name in expected_composites {
            assert!(
                program.neurons.contains_key(name),
                "Missing expected composite neuron: {}",
                name
            );
        }

        let expected_primitives = ["Linear", "LayerNorm", "Dropout", "ReLU", "Concat", "Softmax"];
        for name in expected_primitives {
            assert!(
                program.neurons.contains_key(name),
                "Missing expected primitive neuron: {}",
                name
            );
        }
    }

    #[test]
    fn test_embedded_matches_filesystem() {
        // Both loading methods should produce the same set of neurons
        let fs_program = load_stdlib().expect("filesystem stdlib should load");
        let embedded_program = load_stdlib_embedded().expect("embedded stdlib should load");

        let mut fs_names: Vec<_> = fs_program.neurons.keys().cloned().collect();
        let mut embedded_names: Vec<_> = embedded_program.neurons.keys().cloned().collect();
        fs_names.sort();
        embedded_names.sort();

        assert_eq!(
            fs_names, embedded_names,
            "Embedded and filesystem stdlib should contain the same neurons"
        );
    }

    #[test]
    fn test_merge_programs_user_overrides_stdlib() {
        use crate::interfaces::{NeuronDef, NeuronBody, ImplRef, Port, Shape, Dim};

        let make_neuron = |impl_source: &str| NeuronDef {
            name: "TestNeuron".to_string(),
            params: vec![],
            inputs: vec![Port {
                name: "default".to_string(),
                shape: Shape { dims: vec![Dim::Wildcard] },
                variadic: false,
            }],
            outputs: vec![Port {
                name: "default".to_string(),
                shape: Shape { dims: vec![Dim::Wildcard] },
                variadic: false,
            }],
            body: NeuronBody::Primitive(ImplRef::Source {
                source: impl_source.to_string(),
                path: "Test".to_string(),
            }),
            max_cycle_depth: None,
            doc: None,
        };

        let mut stdlib = Program {
            uses: vec![],
            globals: vec![],
            neurons: HashMap::new(),
        };
        stdlib.neurons.insert("TestNeuron".to_string(), make_neuron("stdlib_source"));

        let mut user = Program {
            uses: vec![],
            globals: vec![],
            neurons: HashMap::new(),
        };
        user.neurons.insert("TestNeuron".to_string(), make_neuron("user_source"));

        let merged = merge_programs(stdlib, user);
        assert_eq!(merged.neurons.len(), 1);

        // User neuron should override stdlib neuron
        let neuron = merged.neurons.get("TestNeuron").unwrap();
        if let NeuronBody::Primitive(ImplRef::Source { source, .. }) = &neuron.body {
            assert_eq!(source, "user_source", "User neuron should override stdlib");
        } else {
            panic!("Expected Primitive(Source) body");
        }
    }

    #[test]
    fn test_merge_programs_empty() {
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

        let merged = merge_programs(stdlib, user);
        assert_eq!(merged.neurons.len(), 0);
    }
}
