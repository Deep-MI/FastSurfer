from HypVINN.utils.img_processing_utils import N4_bias_correct,t1_to_t2_registration
from FastSurferCNN.utils import logging
import time


LOGGER = logging.get_logger(__name__)



def run_hypo_preproc(args):
    load_bc = time.time()
    bc_done = False
    if args.bias_field_correction:
        LOGGER.info("N4 Bias field correction step...")
        args.t1, args.t2 = N4_bias_correct(t1_path=args.t1, t2_path=args.t2, mode=args.mode, out_dir=args.out_dir,
                                           threads=args.threads)
        LOGGER.info("N4 Bias field correction finish in {:0.4f} seconds".format(time.time() - load_bc))
        bc_done = True
    else:
        LOGGER.info(
            "Warning: No bias field correction step, this step is recommended to compenstate for partial voluming.\n "
            "This should result in more accurate volumes. "
            "Omit this message if input images are already bias field corrected")
    LOGGER.info('----' * 30)

    if args.mode == 'multi':
        if args.registration:
            load_res = time.time()
            LOGGER.info("Registering T1 to T2 ...")
            args.in_t2 = t1_to_t2_registration(t1_path=args.t1, t2_path=args.t2, out_dir=args.out_dir,
                                               registration_type=args.reg_type,bc_status=bc_done)
            LOGGER.info("Registration finish in {:0.4f} seconds".format(time.time() - load_res))
        else:
            LOGGER.info(
                "Warning: No registration step, registering T1w and T2w is required when running the multi-modal input mode.\n "
                "No register images can generate wrong predictions. Omit this message if input images are already registered.")

        LOGGER.info('----' * 30)

    return args