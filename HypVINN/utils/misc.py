
def create_expand_output_directory(root_path):
    import os
    paths = [root_path,
             os.path.join(root_path, 'mri', 'transforms'),
             os.path.join(root_path, 'stats'),
             os.path.join(root_path, 'qc_snapshots')]
    for path in paths:
        if path is not None and not os.path.exists(path):
            os.makedirs(path)
