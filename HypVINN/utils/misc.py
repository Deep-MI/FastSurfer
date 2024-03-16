
def create_expand_output_directory(args):
    import os
    root_path = args.out_dir
    paths = [root_path,
             os.path.join(root_path, 'mri', 'transforms'),
             os.path.join(root_path, 'stats')]
    if args.qc_snapshots:
        paths.append(os.path.join(root_path, 'qc_snapshots'))

    for path in paths:
        if path is not None and not os.path.exists(path):
            os.makedirs(path)
