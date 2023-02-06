import os

from FastSurferCNN.utils import logging

LOGGER = logging.getLogger(__name__)


def assert_no_root() -> bool:
    """Checks whether the user is the root user and raises an error message is so"""

    if os.name == 'posix' and os.getuid() == 0:
        import sys, __main__
        sys.exit(
            """----------------------------
            ERROR: You are trying to run 'run_prediction.py' as root. We advice to avoid running 
            FastSurfer as root, because it will lead to files and folders created as root.
            If you are running FastSurfer in a docker container, you can specify the user with 
            '-u $(id -u):$(id -g)' (see https://docs.docker.com/engine/reference/run/#user).
            If you want to force running as root, you may pass --allow_root to %s.
            """ % os.path.basename(__main__.__file__))
    return True


def handle_cuda_memory_exception(exception: RuntimeError, exit_on_out_of_memory: bool = True) -> bool:
    if not isinstance(exception, RuntimeError):
        return False
    message = exception.args[0]
    if message.startswith("CUDA out of memory. "):
        LOGGER.critical("ERROR - INSUFFICIENT GPU MEMORY")
        LOGGER.info("The memory requirements exceeds the available GPU memory, try using a smaller batch size "
                    "(--batch_size <int>) and/or view aggregation on the cpu (--viewagg_device 'cpu')."
                    "Note: View Aggregation on the GPU is particularly memory-hungry at approx. 5 GB for standard "
                    "256x256x256 images.")
        memory_message = message[message.find("(") + 1:message.find(")")]
        LOGGER.info(f"Using {memory_message}.")
        return True
    else:
        return False
