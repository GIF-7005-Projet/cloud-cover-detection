from pathlib import Path


def get_X_paths(features_folder_path: Path = Path('../../data/final/public/train_features/')) -> list[list[Path]]:
    
    paths = []
    
    for folder in Path(features_folder_path).iterdir():
        if folder.is_dir():
            X_features_paths = []
            
            for image_path in folder.iterdir():
                if image_path.is_file() and image_path.suffix == '.tif':
                    X_features_paths.append(image_path)
        
        paths.append(sorted(X_features_paths))

    return sorted(paths)


def get_y_paths(labels_folder_path: Path = Path('../../data/final/public/train_labels/')) -> list[list[Path]]:
    
    paths = []
    
    for file in Path(labels_folder_path).iterdir():
        if file.is_file() and file.suffix == '.tif':
            paths.append(file)
    
    return sorted(paths)
