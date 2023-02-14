
# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# IMPORTS
import os
from concurrent.futures import Executor, Future
from typing import List, Union, TypeVar, Callable, Iterable, Any, Iterator, Optional

import torch

from FastSurferCNN.utils import logging, parser_defaults

LOGGER = logging.getLogger(__name__)


def find_device(device: Union[torch.device, str] = "auto", flag_name:str = "device", min_memory: int = 0) -> torch.device:
    """Create a device object from the device string passed, including detection of devices if device is not defined
    or "auto". """
    logger = logging.get_logger(__name__ + ".auto_device")
    # if specific device is requested, check and stop if not available:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        logger.info(f"cuda not available, try switching to cpu: --{flag_name} cpu")
        raise ValueError(f"--{flag_name} cuda not available, try --{flag_name} cpu !")
    if str(device) == "mps" and not torch.backends.mps.is_available():
        logger.info(f"mps not available, try switching to cpu: --{flag_name} cpu")
        raise ValueError(f"--{flag_name} mps not available, try --{flag_name} cpu !")
    # If auto detect:
    if str(device) == "auto" or not device:
        # 1st check cuda / also finds AMD ROCm
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device = torch.device(device)

    if device.type == "cuda" and min_memory > 0:
        dev_num = torch.cuda.current_device() if device.index is None else device.index
        total_gpu_memory = torch.cuda.get_device_properties(dev_num).__getattribute__("total_memory")
        if total_gpu_memory < min_memory:
            giga = 1024 ** 3
            logger.info(f"Found {total_gpu_memory/giga:.1f} GB GPU memory, but {min_memory/giga:.f} GB was required.")
            device = torch.device("cpu")

    # Define device and transfer model
    logger.info(f"Using {flag_name}: {device}")
    return device


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


def removesuffix(string: str, suffix: str) -> str:
    """Similar to string.removesuffix in PY3.9+, removes a suffix from a string."""
    import sys
    if sys.version_info.minor >= 9:
        # removesuffix is a Python3.9 feature
        return string.removesuffix(suffix)
    else:
        return string[:-len(suffix)] if len(suffix) > 0 and string.endswith(suffix) else string


class SubjectDirectory:
    _orig_name: str
    _copy_orig_name: str
    _conf_name: str
    _segfile: str
    _aparc_aseg_segfile: str
    _main_segfile: str
    _subject_dir: str
    _id: str

    def __init__(self, **kwargs):
        """
        Create a subject, supports generic attributes. Some well integrated attributes arguments include:

        Args:
            id (int): the subject id
            native_t1_file (str): the *absolute* path to the native (and source t1 file).
            conformed (str): the absolute path (or relative to the subject directory `folder`)
            aparc_aseg_segile (str): the absolute (or relative to
        """
        for k, v in kwargs.items():
            setattr(self, "_" + k, v)

    def filename_in_subject_folder(self, filepath: str) -> str:
        return filepath if os.path.isabs(filepath) else os.path.join(self.subject_dir, self._id, filepath)

    def fileexists_in_subject_folder(self, filepath: str) -> bool:
        return os.path.exists(self.filename_in_subject_folder(filepath))

    def fileexists_by_attribute(self, attr_name: str) -> bool:
        return self.fileexists_in_subject_folder(self.get_attribute(attr_name))

    @property
    def subject_dir(self) -> str:
        assert hasattr(self, '_subject_dir') or "The folder attribute has not been set!"
        return self._subject_dir

    @subject_dir.setter
    def subject_dir(self, _folder: str):
        self._subject_dir = _folder

    @property
    def id(self) -> str:
        assert hasattr(self, '_id') or "The id attribute has not been set!"
        return self._id

    @id.setter
    def id(self, _id: str):
        self._id = _id

    @property
    def orig_name(self) -> str:
        """This will typically try to return absolute path, if the native_t1_file is a relative path, it will be
        interpreted as relative to folder."""
        assert hasattr(self, '_orig_name') or "The orig_name attribute has not been set!"
        return self._orig_name

    @orig_name.setter
    def orig_name(self, _orig_name: str):
        self._orig_name = _orig_name

    @property
    def copy_orig_name(self) -> str:
        """This will typically try to return absolute path, if the copy_orig_t1_file is a relative path, it will be
        interpreted as relative to folder."""
        assert hasattr(self, '_copy_orig_name') or "The copy_orig_name attribute has not been set!"
        return self._copy_orig_name

    @copy_orig_name.setter
    def copy_orig_name(self, _copy_orig_name: str):
        self._copy_orig_name = _copy_orig_name

    @property
    def conf_name(self) -> str:
        """This will typically try to return absolute path, if the conformed_t1_file is a relative path, it will be
        interpreted as relative to folder."""
        assert hasattr(self, '_conf_name') or "The conf_name attribute has not been set!"
        return self.filename_in_subject_folder(self._conf_name)

    @conf_name.setter
    def conf_name(self, _conf_name: str):
        self._conf_name = _conf_name

    @property
    def segfile(self) -> str:
        """This will typically try to return absolute path, if the segfile is a relative path, it will be
        interpreted as relative to folder."""
        assert hasattr(self, '_segfile') or "The _segfile attribute has not been set!"
        return self.filename_in_subject_folder(self._segfile)

    @segfile.setter
    def segfile(self, _segfile: str):
        self._segfile = _segfile

    @property
    def aparc_aseg_segfile(self) -> str:
        """This will typically try to return absolute path, if the aparc_aseg_segfile is a relative path, it will be
        interpreted as relative to folder."""
        assert hasattr(self, '_segfile') or "The aparc_aseg_segfile attribute has not been set!"
        return self.filename_in_subject_folder(self._aparc_aseg_segfile)

    @aparc_aseg_segfile.setter
    def aparc_aseg_segfile(self, _aparc_aseg_segfile: str):
        self._aparc_aseg_segfile = _aparc_aseg_segfile

    @property
    def main_segfile(self) -> str:
        """This will typically try to return absolute path, if the main_segfile is a relative path, it will be
        interpreted as relative to folder."""
        assert hasattr(self, '_main_segfile') or "The main_segfile attribute has not been set!"
        return self.filename_in_subject_folder(self._main_segfile)

    @main_segfile.setter
    def main_segfile(self, _main_segfile: str):
        self._main_segfile = _main_segfile

    def can_resolve_filename(self, filename: str) -> bool:
        """Whether we can resolve the file name."""
        return os.path.isabs(filename) or self._subject_dir is not None

    def can_resolve_attribute(self, attr_name: str) -> bool:
        return self.can_resolve_filename(self.get_attribute(attr_name))

    def has_attribute(self, attr_name: str) -> bool:
        return getattr(self, '_' + attr_name, None) is not None

    def get_attribute(self, attr_name: str):
        if not self.has_attribute(attr_name):
            raise AttributeError(f"The subject has no attribute named {attr_name}.")
        return getattr(self, '_' + attr_name)


class SubjectList:
    _subjects: List[str]
    _orig_name_: str
    _conf_name_: str
    _segfile_: str

    DEFAULT_FLAGS = {k: v(dict) for k, v in parser_defaults.ALL_FLAGS.items()}

    def __init__(self, args, **assign):
        """Create an iterate-able list of subjects from the arguments passed.

        Args:
            args: The Namespace object (object with attributes to define parameters) with the following 'required'
                definitions.
                orig_name (str): the path to the input t1 file.
                conf_name (str): the path to the conformed t1 file.
                segfile (str): the path to the main output file.
            out_filename: the attribute name of the file that is the main segmentation output of this module
                (default: "segfile").

        There are three modes of operation:

        If the {csv_file[flag]} flag is passed (in args), read subjects from a subject list file
            The subject listfile is a textfile with one subject per line, where each line can be an absolute or relative
            path. If they are relative paths, they are interpreted as relative to args.in_dir, so args.in_dir needs to
            be defined. Paths can either point to subject directories (file is path + the t1 image name in
            args.orig_name) or directly to the t1 image.

        Else, if the {in_dir[flag]} flag is passed (in_dir in args), list the folder contents of in_dir
            The search in pattern can be specified by the flag {tag[flag]} (args.tag), which is {tag[default]} (all
            files and folders) by default.
            For containing objects that are folders, these folders are interpreted as subject directories and the t1
            image is loaded according to the (necessarily relative) {t1[flag]} (args.orig_name), which defaults to
            {t1[default]}. The folder name is treated as the subject id, if no {sid[flag]} is passed (args.sid).
            For the containing objects that are files, these files are interpreted are loaded as to-be analyzed data.

        Finally, if an absolute path is specified with the {t1[flag]} flag, only this specific file is processed
            (args.orig_name).

        If no {sid[flag]} is passed (args.sid), subject ids are extracted from the subject details (excluding
            potentially added relative paths). Suffixes can be removed from this by use of {remove_suffix[flag]}
            (args.remove_suffix) including file extensions or subfolders (e.g. `--tag */anat --remove_suffix /anat` or
            `--tag *_t1.nii.gz --remove_suffix _t1.nii.gz`).

        Raises:
            RuntimeError: For invalid configurations.
            RuntimeError: When using {sid[flag]} with multiple subjects.
        """

        # Check input and output options
        if args.in_dir is None and args.csv_file is None and not os.path.isabs(args.orig_name):
            raise RuntimeError(('One of the following three options has to be passed {in_dir[flag]}, '
                                '{csv_file[flag]} or {t1[flag]} with an absolute file path. Please specify the data '
                                'input directory, the subject list file or the full path to input '
                                'volume').format(**self.DEFAULT_FLAGS))
        assign.setdefault('segfile', 'segfile')
        assign.setdefault('orig_name', 'orig_name')
        assign.setdefault('conf_name', 'conf_name')

        self.__attr_assign = assign
        for subject_attribute, args_attribute in assign.items():
            setattr(self, '_' + subject_attribute + '_', getattr(args, args_attribute))

        self._out_segfile = getattr(self, '_segfile_', None)
        if self._out_segfile is None:
            raise RuntimeError(f"The segmentation output file is not set, it should be either "
                               f"'segfile' (which gets populated from args.segfile), or a keyword argument to "
                               f"__init__, e.g. `SubjectList(args, subseg='subseg_param', out_filename='subseg')`.")

        if args.out_dir is None and not os.path.isabs(self._out_segfile):
            raise RuntimeError(('Please specify, where the segmentation output should be stored by either the '
                                '{sd[flag]} flag (output subject directory, this can be same as input directory) or an '
                                'absolute path to the {aparc_aseg_segfile[flag]} output segmentation volume.'
                                ).format(**self.DEFAULT_FLAGS))
        self._out_dir = args.out_dir

        # 1. are we doing a csv file of subjects
        if args.csv_file is not None:
            with open(args.csv_file, "r") as s_dirs:
                self._subjects = [line.strip() for line in s_dirs.readlines()]
            if any(not os.path.isabs(d) for d in self._subjects):
                msg = f'At least one path in {args.csv_file} was relative, but the '
                if args.in_dir is None:
                    raise RuntimeError(msg + '{in_dir[flag]} flag was not passed.'.format(**self.DEFAULT_FLAGS))
                elif not os.path.isdir(args.in_dir):
                    raise RuntimeError(msg + f'input directory {args.in_dir} does not exist.')
                self._subjects = [os.path.join(args.in_dir, d) if os.path.isabs(d) else d for d in self._subjects]
            self._num_subjects = len(self._subjects)
            LOGGER.info(f"Analyzing all {self._num_subjects} subjects from csv_file {args.csv_file}.")

        # 2. do we search in a directory
        elif hasattr(args, 'search_tag'):
            if not os.path.isdir(args.in_dir):
                raise RuntimeError(f'The input directory {args.in_dir} does not exist.')
            from glob import glob
            search_tag = args.search_tag
            if not os.path.isabs(search_tag):
                search_tag = os.path.join(args.in_dir, search_tag)
            self._subjects = glob(search_tag)
            self._num_subjects = len(self._subjects)
            LOGGER.info(f"Analyzing all {self._num_subjects} subjects from in_dir {args.in_dir} with search pattern "
                        f"{search_tag}.")

        # 3. are we doing a single file (absolute path to the file)
        elif os.path.isabs(self._orig_name_):
            LOGGER.info('Single subject with absolute file path for input.')
            if not os.path.isfile(self._orig_name_):
                raise RuntimeError(f"The input file {self._orig_name_} does not exist (is not a file).")
            if self._out_dir is None:
                sid = ""
                if os.path.isabs(self._out_segfile):
                    # try to extract the subject directory from the absolute out filename by, containing folder is 'mri'
                    # or the subject id
                    out_dirname = os.path.dirname(self._out_segfile)
                    parent_dir = os.path.basename(out_dirname)
                    if parent_dir == 'mri':
                        LOGGER.info(f'No subjects directory specified, but the parent directory of the output file '
                                    f"{self._out_segfile} is 'mri', so we are assuming this is the 'mri' folder in "
                                    f"the subject directory.")
                        self._out_dir, sid = os.path.split(os.path.dirname(out_dirname))
                        self._out_segfile = os.path.join("mri", os.path.basename(self._out_segfile))
                    elif parent_dir == getattr(args, "sid", ""):
                        LOGGER.info(f'No subjects directory specified, but the parent directory of the output file '
                                    f"{self._out_segfile} is the subject id, so we are assuming this is the subject "
                                    f"directory.")
                        self._out_dir, sid = os.path.split(out_dirname)
                        self._out_segfile = os.path.basename(self._out_segfile)

                not_abs = lambda subj_attr: not os.path.isabs(getattr(self, f'_{subj_attr}_'))
                if getattr(args, "sid", "") in [None, ""]:
                    args.sid = sid
                elif getattr(args, "sid", "") != sid and any(map(not_abs, self.__attr_assign.keys())):
                    relative_files = filter(not_abs, self.__attr_assign.keys())
                    relative_files = [self.DEFAULT_FLAGS[self.__attr_assign[k]["flag"]] + ": " + getattr(self, f'_{k}_') for k in relative_files]
                    raise RuntimeError("Could not extract the subject id from the command line and the output file, "
                                       "'{0}' while at the same time, not all output files are absolute. Try passing "
                                       "the subjects directory {sd[flag]}, or absolute paths for {1}.".format(
                        self._segfile_, ", ".join(relative_files), **self.DEFAULT_FLAGS))

            self._subjects = [self._orig_name_]
            self._num_subjects = 1
            LOGGER.info("Analyzing single subject {}".format(self._orig_name_))
        else:
            from FastSurferCNN.utils.parser_defaults import ALL_FLAGS
            flags = {f: a(dict) for f, a in ALL_FLAGS}
            raise RuntimeError("Could not identify how to find images to segment. Options are:\n1. Provide a text "
                               "file with one subject directory or image file per line via {csv_file[flag]};\n2. "
                               "Provide a search pattern to search for subject directories or images via {tag[flag]};\n"
                               "3. specify an absolute path for relevant files, specifically the t1 file via "
                               "{t1[flag]}, but ideally also for expected output files such as the segmentation "
                               "output file.\n Note also, that the input directory (specified via {in_dir[flag]}) "
                               "will be used as the base path for relative file paths of input files.".format(**flags))

        self._remove_suffix = args.remove_suffix
        if self._num_subjects > 1:
            if getattr(args, "sid", "") not in ["", None]:
                raise RuntimeError("The usage of {sid[flag]} with multiple subjects is undefined."
                                   .format(**self.DEFAULT_FLAGS))
            if self._remove_suffix == '':
                all_subject_files = self.are_all_subject_files()
                common_suffix = self.get_common_suffix()
                if all_subject_files and common_suffix != '':
                    LOGGER.info(f"We detected that the subjects are a list of files with the common suffix "
                                f"{common_suffix} in the subject name. You can remove trailing parts of the filename "
                                f"such as file extensions and other suffixes by passing "
                                f"{self.DEFAULT_FLAGS['remove_suffix']['flag']} <suffix>, e.g. "
                                f"{self.DEFAULT_FLAGS['remove_suffix']['flag']} '{common_suffix}'.")
            if os.path.isabs(self._out_segfile):
                raise RuntimeError(f'An absolute path was passed for the output segmentation {self._out_segfile}, '
                                   f'but more than one input image fits the input definition.')

        self._sid = getattr(args, "sid", "")

    __init__.__doc__ = __init__.__doc__.format(**DEFAULT_FLAGS)

    def __len__(self) -> int:
        return self._num_subjects

    def make_subjects_dir(self):
        LOGGER.info("Output will be stored in: {}".format(self._out_dir))

        if self._out_dir is not None and not os.path.exists(self._out_dir):
            LOGGER.info("Output directory does not exist. Creating it now...")
            os.makedirs(self._out_dir)

    def __getitem__(self, item: int) -> SubjectDirectory:
        if item < 0 or item >= self._num_subjects:
            raise IndexError(f"The index {item} is out of bounds for the subject list.")

        # subject is always an absolute path (or relative to the working directory) ... of the input file
        subject = self._subjects[item]

        # Set subject and load orig
        special_rules = ["orig_name"]
        subject_parameters = {v: getattr(self, f"_{v}_") for v in self.__attr_assign.keys() if v not in special_rules}

        return SubjectDirectory(subject_dir=self._out_dir,
                                id=os.path.basename(removesuffix(subject, self._remove_suffix)) if self._sid is None else self._sid,
                                orig_name=subject if os.path.isfile(subject) else os.path.join(subject, self._orig_name_),
                                **subject_parameters)

    def get_common_suffix(self) -> str:
        """Finds, if all entries in the subject list share a common suffix"""
        suffix = self._subjects[0]
        for subj in self._subjects[1:]:
            if subj.endswith(suffix):
                continue
            for i in range(1 - len(suffix), 1):
                if i == 0:
                    return ''
                if subj[i:] == suffix[i:]:
                    suffix = suffix[i:]
                    break
        return suffix

    def are_all_subject_files(self):
        """Checks if all entries in subjects are actually files. This is performed asynchronously internally."""
        from asyncio import run, gather

        async def is_file(path):
            return os.path.isfile(path)

        async def check_files(files):
            return await gather(*[is_file(s) for s in files])

        return all(run(check_files(self._subjects)))


_T = TypeVar("_T")


class NoParallelExecutor(Executor):
    def map(self, fn: Callable[..., _T], *iterables: Iterable[Any], timeout: Optional[float] = None, chunksize: int = -1) -> Iterator[_T]:
        return map(fn, *iterables)

    def submit(self, __fn: Callable[..., _T], *args, **kwargs) -> 'Future[_T]':
        f = Future()
        try:
            f.set_result(__fn(*args, **kwargs))
        except Exception as e:
            f.set_exception(e)
        return f
