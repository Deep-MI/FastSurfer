from os.path import join, split, splitext

from HypVINN.config.hypvinn import get_cfg_hypvinn


def get_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    """
    # Setup cfg.
    cfg = get_cfg_hypvinn()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.LOG_DIR = args.LOG_dir

    cfg_file_name = splitext(split(args.cfg_file)[1])[0]
    cfg.LOG_DIR = join(cfg.LOG_DIR, cfg_file_name)

    return cfg

def load_config(cfg_file):
    # setup base
    cfg = get_cfg_hypvinn()
    cfg.EXPR_NUM = None
    cfg.SUMMARY_PATH = ""
    cfg.CONFIG_LOG_PATH = ""
    cfg.TRAIN.RESUME_EXPR_NUM = None
    # Overwrite with stored arguments
    cfg.merge_from_file(cfg_file)
    return cfg