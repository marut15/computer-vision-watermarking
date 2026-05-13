import json
import numpy as np

def create_split(metadata_path, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Create stratified split ensuring all 256 IDs appear in train set.
    """
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    # Group by ID
    id_groups = {}
    for i, entry in enumerate(data):
        id_int = entry['id_int']
        if id_int not in id_groups:
            id_groups[id_int] = []
        id_groups[id_int].append(i)
    
    # Verify all IDs present
    assert len(id_groups) == 256, f"Expected 256 IDs, got {len(id_groups)}"
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    np.random.seed(seed)
    
    for id_int, indices in id_groups.items():
        # Each ID has 10 samples (one per prompt)
        assert len(indices) == 10, f"ID {id_int} has {len(indices)} samples"
        
        # Shuffle
        indices_copy = indices.copy()
        np.random.shuffle(indices_copy)
        
        # Split: 8 train, 1 val, 1 test
        train_indices.extend(indices_copy[:8])
        val_indices.append(indices_copy[8])
        test_indices.append(indices_copy[9])
    
    print(f"Train: {len(train_indices)} samples")  # 2048
    print(f"Val: {len(val_indices)} samples")      # 256
    print(f"Test: {len(test_indices)} samples")    # 256
    
    # Save splits
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    with open('splits.json', 'w') as f:
        json.dump(splits, f, indent=2)
    
    return splits

if __name__ == '__main__':
    import os
    from pathlib import Path
    _dr = Path(os.environ.get("PROJECT_DATA_ROOT", Path(__file__).resolve().parents[3] / "data" / "computer-vision-watermarking"))
    splits = create_split(str(_dr / "watermark_encoding/data/metadata.json"))
    
    # Verify no overlap
    assert len(set(splits['train']) & set(splits['val'])) == 0
    assert len(set(splits['train']) & set(splits['test'])) == 0
    assert len(set(splits['val']) & set(splits['test'])) == 0
    print("✓ No data leakage")
    print("✓ Splits saved to splits.json")

