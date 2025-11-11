import numpy as np
import os

# Configuration
MATRIX_SIZES = [64, 128, 256, 512, 1024, 2048]
SPARSE_LEVELS = [0, 50, 75, 90, 95]
OUTPUT_DIR = 'matrices'
SEED = 42 

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use numpy Generator for reproducibility
rng = np.random.default_rng(SEED)

for size in MATRIX_SIZES:
    for sparse in SPARSE_LEVELS:
        for matrix_label in ['A', 'B']:
            # Generate dense random matrix with integers 0-9
            matrix = rng.integers(low=0, high=10, size=(size, size), dtype=np.int32)

            # Apply sparsity
            if sparse > 0:
                mask = rng.random((size, size)) < (sparse / 100)
                matrix[mask] = 0

            # Create filename
            filename = f'{matrix_label}_{size}_{sparse}.bin'
            filepath = os.path.join(OUTPUT_DIR, filename)

            # Save matrix in binary format
            matrix.tofile(filepath)

            print(f'Saved {filepath}')