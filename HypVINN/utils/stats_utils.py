import os
from HypVINN.config.hypvinn_files import HYPVINN_LUT


def compute_stats(orig_path,prediction_path,save_dir,threads,hypo_lut=HYPVINN_LUT):
    from collections import namedtuple
    import multiprocessing
    from FastSurferCNN.segstats import main
    from HypVINN.config.hypvinn_files import HYPVINN_STATS_NAME

    args = namedtuple('ArgNamespace', ['normfile', 'i', 'o', 'excludedid',
                                        'ids', 'merged_labels','robust',
                                        'threads','patch_size','device','lut','allow_root'])

    args.normfile = orig_path
    args.segfile = prediction_path
    args.segstatsfile = os.path.join(save_dir, HYPVINN_STATS_NAME)
    args.excludeid = [0]
    args.ids = None
    args.merged_labels = []
    args.robust = None
    args.threads = threads
    args.patch_size = 32
    args.device = "auto"
    args.lut = hypo_lut
    args.allow_root = False
    flag = main(args)

    return flag




