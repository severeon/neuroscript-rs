#!/bin/bash

# Script to generate documentation from NeuroScript files
# This processes all .ns files with doc comments and generates markdown for Docusaurus

set -e  # Exit on error

# Colors for output
GREEN='\033[0.32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building neuroscript-doc binary...${NC}"
cargo build --release --bin neuroscript-doc

DOC_BINARY="./target/release/neuroscript-doc"

echo -e "${BLUE}Generating primitive documentation...${NC}"

# Generate docs for primitives
PRIMITIVE_FILES=(stdlib/**/*.ns)

PRIMITIVES_GENERATED=0

for file in "${PRIMITIVE_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Processing $file..."
        $DOC_BINARY --input "$file" --output website/docs/primitives --category primitives
        PRIMITIVES_GENERATED=$((PRIMITIVES_GENERATED + 1))
    else
        echo "  Warning: $file not found, skipping..."
    fi
done

echo -e "${GREEN}✓ Generated $PRIMITIVES_GENERATED primitive documentation pages${NC}"

# TODO: Add stdlib documentation generation when stdlib neurons have doc comments
# echo -e "${BLUE}Generating stdlib documentation...${NC}"
# STDLIB_FILES=(
#     "stdlib/FFN.ns"
#     "stdlib/Residual.ns"
#     etc...
# )

echo -e "${GREEN}✓ Documentation generation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. cd website"
echo "  2. npm install"
echo "  3. npm start"
