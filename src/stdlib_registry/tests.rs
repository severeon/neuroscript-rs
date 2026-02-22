use crate::{ImplRef, StdlibRegistry};

#[test]
fn test_registry_basics() {
    let registry = StdlibRegistry::new();

    // Check Level 0 primitives are registered
    assert!(registry.contains("Linear"));
    assert!(registry.contains("GELU"));
    assert!(registry.contains("Dropout"));
    assert!(registry.contains("LayerNorm"));
    assert!(registry.contains("Embedding"));

    // Check non-existent primitive
    assert!(!registry.contains("NonExistent"));

    // Check we have the expected number of primitives
    assert_eq!(registry.len(), 65); // 65 primitives registered

    // Check lookup works
    let linear = registry.lookup("Linear").unwrap();
    match linear {
        ImplRef::Source { source, path } => {
            assert_eq!(path, "Linear");
            assert_eq!(source, "neuroscript_runtime.primitives.linear");
            assert_eq!(
                linear.full_name(),
                "neuroscript_runtime.primitives.linear.Linear"
            );
        }
        _ => panic!("Expected Source variant"),
    }
}

#[test]
fn test_impl_ref() {
    let impl_ref = ImplRef::with_desc(
        "neuroscript_runtime.primitives.linear",
        "Linear",
        "Dense layer",
    );

    assert_eq!(
        impl_ref.full_name(),
        "neuroscript_runtime.primitives.linear.Linear"
    );
    assert_eq!(
        impl_ref.import_statement(),
        "from neuroscript_runtime.primitives.linear import Linear"
    );
}

#[test]
fn test_generate_imports() {
    let registry = StdlibRegistry::new();

    let used = vec![
        "Linear".to_string(),
        "GELU".to_string(),
        "Dropout".to_string(),
    ];

    let imports = registry.generate_imports(&used);

    assert_eq!(imports.len(), 3);
    assert!(
        imports.contains(&"from neuroscript_runtime.primitives.linear import Linear".to_string())
    );
    assert!(imports
        .contains(&"from neuroscript_runtime.primitives.activations import GELU".to_string()));
    assert!(imports.contains(
        &"from neuroscript_runtime.primitives.regularization import Dropout".to_string()
    ));
}

#[test]
fn test_all_primitives() {
    let registry = StdlibRegistry::new();
    let primitives = registry.primitives();

    // Should be sorted
    assert_eq!(primitives[0], "ALiBi");
    assert!(primitives.contains(&"Linear".to_string()));
    assert!(primitives.contains(&"GELU".to_string()));

    // All primitives should have valid impl refs
    for name in primitives {
        let impl_ref = registry.lookup(&name).unwrap();
        match impl_ref {
            ImplRef::Source { source, path } => {
                assert!(!source.is_empty());
                assert!(!path.is_empty());
            }
            ImplRef::External { .. } => {
                // External impl refs are also valid
            }
        }
    }
}
