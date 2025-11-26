/// Example: Pattern Matching for Tensor/Neuron Shapes
///
/// This demonstrates how the shape_algebra pattern matching can be used
/// for neuron and tensor shape validation.

use neuroscript::shape_algebra::{Shape, Pattern, PatToken};

fn main() {
    println!("=== Shape Algebra Pattern Matching Examples ===\n");

    // Example 1: Batch dimension extraction
    println!("1. Batch Dimension Pattern: [batch, ...]");
    let batch_pattern = Pattern::from_tokens(vec![PatToken::Any, PatToken::Rest]);
    
    let image_batch = Shape::new(vec![32, 3, 224, 224]);
    if let Some(captured) = batch_pattern.matches_and_capture(&image_batch, true) {
        println!("   ✓ Matched shape [32, 3, 224, 224]");
        println!("   Captured batch size: {}\n", captured[0]);
    }

    // Example 2: Attention head pattern
    println!("2. Multi-Head Attention Pattern: [B, H, SeqLen, D]");
    let attention_pattern = Pattern::from_tokens(vec![
        PatToken::Any,  // Batch
        PatToken::Any,  // Heads
        PatToken::Any,  // Sequence length
        PatToken::Any   // Head dimension
    ]);
    
    let attention_tensor = Shape::new(vec![16, 12, 512, 64]);
    if let Some(captured) = attention_pattern.matches_and_capture(&attention_tensor, true) {
        println!("   ✓ Matched shape [16, 12, 512, 64]");
        println!("   Batch: {}, Heads: {}, SeqLen: {}, HeadDim: {}\n", 
                 captured[0], captured[1], captured[2], captured[3]);
    }

    // Example 3: Fixed spatial size
    println!("3. Fixed Spatial Pattern: [*, 224, 224, *]");
    let fixed_spatial = Pattern::from_tokens(vec![
        PatToken::Any,
        PatToken::Lit(224),
        PatToken::Lit(224),
        PatToken::Any
    ]);
    
    let imagenet_batch = Shape::new(vec![32, 224, 224, 3]);
    if fixed_spatial.matches(&imagenet_batch) {
        println!("   ✓ Matched ImageNet-sized batch [32, 224, 224, 3]\n");
    }
    
    let wrong_size = Shape::new(vec![32, 128, 128, 3]);
    if !fixed_spatial.matches(&wrong_size) {
        println!("   ✗ Rejected wrong size [32, 128, 128, 3]\n");
    }

    // Example 4: Conv kernel pattern
    println!("4. 3x3 Conv Kernel Pattern: [*, *, 3, 3]");
    let kernel_pattern = Pattern::from_tokens(vec![
        PatToken::Any,
        PatToken::Any,
        PatToken::Lit(3),
        PatToken::Lit(3)
    ]);
    
    let conv_weights = Shape::new(vec![64, 128, 3, 3]);
    if let Some(captured) = kernel_pattern.matches_and_capture(&conv_weights, true) {
        println!("   ✓ Matched 3x3 conv kernel [64, 128, 3, 3]");
        println!("   Out channels: {}, In channels: {}\n", captured[0], captured[1]);
    }

    // Example 5: Broadcasting check
    println!("5. Broadcasting Compatibility");
    let tensor_a = Shape::new(vec![32, 64, 56, 56]);
    let tensor_b = Shape::new(vec![1, 64, 1, 1]);
    
    if neuroscript::shape_algebra::broadcastable(&tensor_a, &tensor_b) {
        println!("   ✓ [32, 64, 56, 56] and [1, 64, 1, 1] are broadcastable");
        println!("   (Can be used in ResNet residual connections)\n");
    }

    // Example 6: Reshape compatibility
    println!("6. Reshape Compatibility");
    let flat = Shape::new(vec![32, 784]);
    let spatial = Shape::new(vec![32, 28, 28]);
    
    if neuroscript::shape_algebra::reshapeable(&flat, &spatial) {
        println!("   ✓ [32, 784] can be reshaped to [32, 28, 28]");
        println!("   (MNIST batch: 32 x 28x28 images)\n");
    }

    println!("=== All Examples Completed Successfully ===");
}
