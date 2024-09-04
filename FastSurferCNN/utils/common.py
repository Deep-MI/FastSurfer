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
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future
from pathlib import Path
from typing import (
    Any,
    TypeVar,
)

import torch

from FastSurferCNN.utils import logging, parser_defaults

__all__ = [
    "assert_no_root",
    "find_device",
    "handle_cuda_memory_exception",
    "iterate",
    "SerialExecutor",
    "pipeline",
    "SubjectList",
    "SubjectDirectory",
]

from FastSurferCNN.utils.parser_defaults import SubjectDirectoryConfig

LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")
_Ti = TypeVar("_Ti")


def find_device(
    device: torch.device | str = "auto",
    flag_name: str = "device",
    min_memory: int = 0,
) -> torch.device:
    """
    Create a device object from the device string passed.

    Includes detection of devices if device is not defined or "auto".

    Parameters
    ----------
    device : torch.device, str
        The device to search for and test following pytorch device naming
        conventions, e.g. 'cuda:0', 'cpu', etc. (default: 'auto').
    flag_name : str
        Name of the corresponding flag for error messages (default: 'device').
    min_memory : int
        The minimum memory in bytes required for cuda-devices to
        be valid (default: 0, works always).

    Returns
    -------
    device: torch.device
        The torch.device object.
    """
    logger = logging.get_logger(__name__ + ".auto_device")
    # if specific device is requested, check and stop if not available:
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    msg = None
    if str(device).startswith("cuda") and not has_cuda:
        msg = f"cuda not available, try switching to cpu: --{flag_name} cpu"
    if str(device) == "mps" and not has_mps:
        msg = f"mps not available, try switching to cpu: --{flag_name} cpu"
    if msg is not None:
        logger.info(msg)
        raise ValueError(msg)
    # If auto detect:
    if str(device) == "auto" or not device:
        # 1st check cuda / also finds AMD ROCm, then mps, finally cpu
        device = "cuda" if has_cuda else "mps" if has_mps else "cpu"

    device = torch.device(device)

    if device.type == "cuda" and min_memory > 0:
        dev_num = torch.cuda.current_device() if device.index is None else device.index
        total_gpu_memory = torch.cuda.get_device_properties(dev_num).__getattribute__(
            "total_memory"
        )
        if total_gpu_memory < min_memory:
            giga = 1024**3
            logger.info(
                f"Found {total_gpu_memory/giga:.1f} GB GPU memory, but "
                f"{min_memory/giga:.1f} GB was required."
            )
            device = torch.device("cpu")

    # Define device and transfer model
    logger.info(f"Using {flag_name}: {device}")
    return device


def assert_no_root() -> bool:
    """
    Check whether the user is the root user and raises an error message is so.

    Returns
    -------
    bool
        Whether the user is root or not.
    """
    if os.name == "posix" and os.getuid() == 0:
        import sys

        import __main__

        sys.exit(
            """----------------------------
            ERROR
            You are trying to run '{0}' as root. We advice to avoid running FastSurfer 
            as root, because it will lead to files and folders created as root.
            If you are running FastSurfer in a docker container, you can specify the 
            user with '-u $(id -u):$(id -g)' 
            (see https://docs.docker.com/engine/reference/run/#user).
            If you want to force running as root, you may pass --allow_root to {0}.
            """.format(
                os.path.basename(__main__.__file__)
            )
        )
    return True


def handle_cuda_memory_exception(exception: BaseException) -> bool:
    """
    Handle CUDA out of memory exception and print a help text.

    Parameters
    ----------
    exception : builtins.BaseException
        Received exception.

    Returns
    -------
    bool
        Whether the exception was a RuntimeError caused by Cuda out memory.
    """
    if not isinstance(exception, RuntimeError):
        return False
    message = exception.args[0]
    if message.startswith("CUDA out of memory. "):
        LOGGER.critical("ERROR - INSUFFICIENT GPU MEMORY")
        LOGGER.info(
            "The memory requirements exceeds the available GPU memory, try using a "
            "smaller batch size (--batch_size <int>) and/or view aggregation on the "
            "cpu (--viewagg_device 'cpu')."
        )
        LOGGER.info(
            "Note: View Aggregation on the GPU is particularly memory-hungry at "
            "approx. 5 GB for standard 256x256x256 images."
        )
        memory_message = message[message.find("(") + 1:message.find(")")]
        LOGGER.info(f"Using {memory_message}.")
        return True
    else:
        return False


def pipeline(
    pool: Executor,
    func: Callable[[_Ti], _T],
    iterable: Iterable[_Ti],
    *,
    pipeline_size: int = 1,
) -> Iterator[tuple[_Ti, _T]]:
    """
    Pipeline a function to be executed in the pool.

    Analogous to iterate, but run func in a different
    thread for the next element while the current element is returned.

    Parameters
    ----------
    pool : Executor
        Thread pool executor for parallel execution.
    func : callable
        Function to use.
    iterable : Iterable
        Iterable containing input elements.
    pipeline_size : int, default=1
        Size of the processing pipeline.

    Yields
    ------
    element : _Ti
        Elements
    _T
        Results of func corresponding to element: func(element).
    """
    # do pipeline loading the next element
    from collections import deque

    futures_queue = deque()
    import itertools

    for i, element in zip(itertools.count(-pipeline_size), iterable):
        # pre-load next element/data
        futures_queue.append((element, pool.submit(func, element)))
        if i >= 0:
            element, future = futures_queue.popleft()
            yield element, future.result()
    while len(futures_queue) > 0:
        element, future = futures_queue.popleft()
        yield element, future.result()


def iterate(
    pool: Executor, func: Callable[[_Ti], _T], iterable: Iterable[_Ti],
) -> Iterator[tuple[_Ti, _T]]:
    """
    Iterate over iterable, yield pairs of elements and func(element).

    Parameters
    ----------
    pool : Executor
        The Executor object (dummy object to have a common API with pipeline).
    func : callable
        Function to use.
    iterable : Iterable
        Iterable to draw objects to process with func from.

    Yields
    ------
    element : _Ti
        Elements
    _T
        Results of func corresponding to element: func(element).
    """
    for element in iterable:
        yield element, func(element)


class SubjectDirectory:
    """
    Represent a subject directory.
    """

    _orig_name: str
    _copy_orig_name: str
    _conf_name: str
    _segfile: str
    _asegdkt_segfile: str
    _main_segfile: str
    _subject_dir: str
    _id: str

    def __init__(self, **kwargs):
        """
        Create a subject, supports generic attributes.

        Parameters
        ----------
        id : str
            The subject id.
        orig_name : str
            Relative or absolute filename of the orig filename.
        conf_name : str
            Relative or absolute filename of the conformed filename.
        segfile : str
            Relative or absolute filename of the segmentation filename.
        main_segfile : str
            Relative or absolute filename of the main segmentation filename.
        asegdkt_segfile : str
            Relative or absolute filename of the aparc+aseg segmentation filename.
        subject_dir : Path
            Path to the subjects directory (containing subject folders).
        """
        for k, v in kwargs.items():
            if k == "subject_dir":
                v = Path(v)
            setattr(self, "_" + k, v)

    def filename_in_subject_folder(self, filepath: str | Path) -> Path:
        """
        Return the full path to the file.

        Parameters
        ----------
        filepath : str, Path
            Absolute to the file or name of the file.

        Returns
        -------
        Path
            Path to the file.
        """
        if Path(filepath).is_absolute():
            return Path(filepath)
        else:
            return self.subject_dir / self._id / filepath

    def filename_by_attribute(self, attr_name: str) -> Path:
        """
        Retrieve a filename based on the provided attribute name.

        Parameters
        ----------
        attr_name : str
            The name of the attribute associated with the desired filename.

        Returns
        -------
        Path
            The filename corresponding to the provided attribute name.
        """
        return self.filename_in_subject_folder(self.get_attribute(attr_name))

    def fileexists_in_subject_folder(self, filepath: str | Path) -> bool:
        """
        Check if file exists in the subject folder.

        Parameters
        ----------
        filepath : Path, str
            Path to the file.

        Returns
        -------
        bool
            Whether the file exists or not.
        """
        return self.filename_in_subject_folder(filepath).exists()

    def fileexists_by_attribute(self, attr_name: str) -> bool:
        """
        Check if a file exists based on the provided attribute name.

        Parameters
        ----------
        attr_name : str
            The name of the attribute associated with the file existence check.

        Returns
        -------
        bool
            Whether the file exists or not.
        """
        return self.fileexists_in_subject_folder(self.get_attribute(attr_name))

    @property
    def subject_dir(self) -> Path:
        """
        Gets the subject directory name.

        Returns
        -------
        Path
            The set subject directory.
        """
        assert hasattr(self, "_subject_dir") or "The folder attribute has not been set!"
        return Path(self._subject_dir)

    @subject_dir.setter
    def subject_dir(self, _folder: str | Path):
        """
        Set the subject directory name.

        Parameters
        ----------
        _folder : str, Path
            The subject directory.
        """
        self._subject_dir = _folder

    @property
    def id(self) -> str:
        """
        Get the id.

        Returns
        -------
        str
            The id.
        """
        assert hasattr(self, "_id") or "The id attribute has not been set!"
        return self._id

    @id.setter
    def id(self, _id: str):
        """
        Set the id.

        Parameters
        ----------
        _id : str
            The id.
        """
        self._id = _id

    @property
    def orig_name(self) -> str:
        """
        Try to return absolute path.

        If the native_t1_file is a relative path, it will be
        interpreted as relative to folder.

        Returns
        -------
        str
            The orig name.
        """
        assert (
            hasattr(self, "_orig_name") or "The orig_name attribute has not been set!"
        )
        return self._orig_name

    @orig_name.setter
    def orig_name(self, _orig_name: str):
        """
        Set the orig name.

        Parameters
        ----------
        _orig_name : str
            The orig name.
        """
        self._orig_name = _orig_name

    @property
    def copy_orig_name(self) -> Path:
        """
        Try to return absolute path.

        If the copy_orig_t1_file is a relative path, it will be
        interpreted as relative to folder.

        Returns
        -------
        Path
            The copy of orig name.
        """
        assert (
            hasattr(self, "_copy_orig_name")
            or "The copy_orig_name attribute has not been set!"
        )
        return self.filename_in_subject_folder(self._copy_orig_name)

    @copy_orig_name.setter
    def copy_orig_name(self, _copy_orig_name: str):
        """
        Set the copy of orig name.

        Parameters
        ----------
        _copy_orig_name : str
            The copy of the orig name.

        Returns
        -------
        str
            Original name.
        """
        self._copy_orig_name = _copy_orig_name

    @property
    def conf_name(self) -> Path:
        """
        Try to return absolute path.

        If the conformed_t1_file is a relative path, it will be
        interpreted as relative to folder.

        Returns
        -------
        Path
            The path to the conformed image file.
        """
        assert (
            hasattr(self, "_conf_name") or "The conf_name attribute has not been set!"
        )
        return self.filename_in_subject_folder(self._conf_name)

    @conf_name.setter
    def conf_name(self, _conf_name: str):
        """
        Set the path to the conformed image.

        Parameters
        ----------
        _conf_name : str
            Path to the conformed image.

        """
        self._conf_name = _conf_name

    @property
    def segfile(self) -> Path:
        """
        Try to return absolute path.

        If the segfile is a relative path, it will be interpreted as relative to folder.

        Returns
        -------
        Path
            Path to the segfile.
        """
        assert hasattr(self, "_segfile") or "The _segfile attribute has not been set!"
        return self.filename_in_subject_folder(self._segfile)

    @segfile.setter
    def segfile(self, _segfile: str):
        """
        Set segfile.

        Parameters
        ----------
        _segfile : str
            Path to the segmentation file.
        """
        self._segfile = _segfile

    @property
    def asegdkt_segfile(self) -> Path:
        """
        Try to return absolute path.

        If the asegdkt_segfile is a relative path, it will be
        interpreted as relative to folder.

        Returns
        -------
        Path
            Path to segmentation file.
        """
        assert (
            hasattr(self, "_segfile")
            or "The asegdkt_segfile attribute has not been set!"
        )
        return self.filename_in_subject_folder(self._asegdkt_segfile)

    @asegdkt_segfile.setter
    def asegdkt_segfile(self, _asegdkt_segfile: str | Path):
        """
        Set path to segmentation file.

        Parameters
        ----------
        _asegdkt_segfile : Path, str
            Path to segmentation file.
        """
        self._asegdkt_segfile = str(_asegdkt_segfile)

    @property
    def main_segfile(self) -> Path:
        """
        Try to return absolute path.

        If the main_segfile is a relative path, it will be
        interpreted as relative to folder.

        Returns
        -------
        Path
            Path to the main segfile.

        """
        assert (
            hasattr(self, "_main_segfile")
            or "The main_segfile attribute has not been set!"
        )
        return self.filename_in_subject_folder(self._main_segfile)

    @main_segfile.setter
    def main_segfile(self, _main_segfile: str):
        """
        Set the main segfile.

        Parameters
        ----------
        _main_segfile : str
            Path to the main_segfile.
        """
        self._main_segfile = _main_segfile

    def can_resolve_filename(self, filename: str) -> bool:
        """
        Check whether we can resolve the file name.

        Parameters
        ----------
        filename : str
            Name of the filename to check.

        Returns
        -------
        bool
            Whether we can resolve the file name.
        """
        return os.path.isabs(filename) or self._subject_dir is not None

    def can_resolve_attribute(self, attr_name: str) -> bool:
        """
        Check whether we can resolve the attribute.

        Parameters
        ----------
        attr_name : str
            Name of the attribute to check.

        Returns
        -------
        bool
            Whether we can resolve the attribute.
        """
        return self.can_resolve_filename(self.get_attribute(attr_name))

    def has_attribute(self, attr_name: str) -> bool:
        """
        Check if the attribute is set.

        Parameters
        ----------
        attr_name : str
            Name of the attribute to check.

        Returns
        -------
        bool
            Whether the attribute exists or not.
        """
        return getattr(self, "_" + attr_name, None) is not None

    def get_attribute(self, attr_name: str) -> str | Path:
        """
        Give the requested attribute.

        Parameters
        ----------
        attr_name : str
            Name of the attribute to return.

        Returns
        -------
        str, Path
            The value of the requested attribute.

        Raises
        ------
        AttributeError
            If the subject has no attribute with the given name.
        """
        if not self.has_attribute(attr_name):
            raise AttributeError(f"The subject has no attribute named {attr_name}.")
        return getattr(self, "_" + attr_name)


class SubjectList:
    """
    Represent a list of subjects.
    """

    _subjects: list[Path]
    _orig_name_: str
    _conf_name_: str
    _segfile_: str
    _flags: dict[str, dict]

    DEFAULT_FLAGS = {k: v(dict) for k, v in parser_defaults.ALL_FLAGS.items()}

    def __init__(
            self,
            args: SubjectDirectoryConfig,
            flags: dict[str, dict] | None = None,
            **assign,
    ):
        """
        Create an iterate-able list of subjects from the arguments passed.

        There are three modes of operation:
        If args has a non-empty csv_file attribute (cf. {csv_file[flag]} flag),
            read subjects from a subject list file:  The subject listfile is a
            textfile with one subject per line, where each line can be an absolute
            or relative path. If they are relative paths, they are interpreted as
            relative to args.in_dir, so args.in_dir needs to be defined. Paths can
            either point to subject directories (file is path + the t1 image name in
            args.orig_name) or directly to the t1 image.
        Else, if args has a non-empty in_dir attribute (c.f. {in_dir[flag]} flag),
            list the folder contents of in_dir: The search pattern can be specified
            by the search_tag attribute of args (cf. {tag[flag]} flag), which is
            {tag[default]} (all files and folders) by default.
            For containing objects that are folders, these folders are interpreted
            as subject directories and the t1 image is loaded according to the
            (necessarily relative) {t1[flag]} (args.orig_name), which defaults to
            {t1[default]}. The folder name is treated as the subject id, if no
            {sid[flag]} is passed (args.sid).
            For the containing objects that are files, these files are interpreted
            are loaded as to-be analyzed data.
        Finally, if an absolute path is specified with the orig_name attribute of
            args (cf. {t1[flag]}), only this specific file is processed.

        Parameters
        ----------
        args : SubjectDirectoryConfig
            The namedtuple/Namespace object (object with attributes to define
            parameters) with the following 'required' definitions.
            - orig_name (str): the path to the input t1 file.
            - conf_name (str): the path to the conformed t1 file.
            - segfile (str): the path to the main output file.
            - in_dir (str) or csv_file (str), if orig_name is not an absolute path.
            If args is passed without a sid attribute (cf. {sid[flag]}), subject ids are
            extracted from the subject details (excluding potentially added relative
            paths). Suffixes can be removed from this by use of the remove_suffix
            attribute of args (cf. {remove_suffix[flag]}) including file extensions or
            subfolders (e.g. `{tag[flag]} */anat {remove_suffix[flag]} /anat` or
            `{tag[flag]} *_t1.nii.gz {remove_suffix[flag]} _t1.nii.gz`).
        flags : dict[str, Dict], optional
            dictionary of flags used to generate args (used to populate messages).
            Default: `SubjectList.DEFAULT_FLAGS`, which get initialized from
            `FastSurferCNN.utils.parser_defaults.ALL_FLAGS`.
        **assign :

        Raises
        ------
        RuntimeError
            For invalid configurations, e.g. no 'in_dir', 'csv_file', or absolute
            'orig_name'.
        RuntimeError
            When using {sid[flag]} with multiple subjects.
        """
        # populate _flags with DEFAULT_FLAGS
        self._flags = flags.copy() if flags is not None else {}
        for flag, default in self.DEFAULT_FLAGS.items():
            self._flags.setdefault(flag, default)

        # Check input and output options
        if (
            getattr(args, "in_dir", None) is None
            and getattr(args, "csv_file", None) is None
            and not os.path.isabs(getattr(args, "orig_name", "undefined"))
        ):
            msg = (
                "One of the following three options has to be passed {in_dir[flag]}, "
                "{csv_file[flag]} or {t1[flag]} with an absolute file path. Please "
                "specify the data input directory, the subject list file or the full "
                "path to input volume"
            )
            raise RuntimeError(msg.format(**self._flags))
        assign.setdefault("segfile", "segfile")
        assign.setdefault("orig_name", "orig_name")
        assign.setdefault("conf_name", "conf_name")

        self.__attr_assign = assign
        for subject_attribute, args_attribute in assign.items():
            if not hasattr(args, args_attribute):
                raise ValueError(
                    f"You have defined {args_attribute} as a attribute of `args` via "
                    f"keyword argument to SubjectList.__init__ or {args_attribute} is "
                    f"required, but `args` does not have {args_attribute} as an "
                    f"attribute."
                )
            setattr(self, "_" + subject_attribute + "_", getattr(args, args_attribute))

        self._out_segfile = getattr(self, "_segfile_", None)
        if self._out_segfile is None:
            raise RuntimeError(
                "The segmentation output file is not set, it should be either "
                "'segfile' (which gets populated from args.segfile), or a keyword "
                "argument to __init__, e.g. `SubjectList(args, subseg='subseg_param', "
                "out_filename='subseg')`."
            )

        # if out_dir is not set, fall back to in_dir by default
        self._out_dir = getattr(args, "out_dir", None) or getattr(args, "in_dir", None)
        if self._out_dir in [None, ""] and not os.path.isabs(self._out_segfile):
            msg = (
                "Please specify, where the segmentation output should be stored by "
                "either the {sd[flag]} flag (output subject directory, this can be "
                "same as input directory) or an absolute path to the "
                "{asegdkt_segfile[flag]} output segmentation volume."
            )
            raise RuntimeError(msg.format(**self._flags))

        # 1. are we doing a csv file of subjects
        if args.csv_file is not None:
            with open(args.csv_file) as s_dirs:
                self._subjects = [Path(line.strip()) for line in s_dirs.readlines()]
            if any(not d.is_absolute() for d in self._subjects):
                msg = f"At least one path in {args.csv_file} was relative, but the "
                if args.in_dir is None:
                    raise RuntimeError(
                        "{}in_dir was not in args (no {in_dir[flag]} flag).".format(
                            msg, **self._flags
                        )
                    )
                elif not os.path.isdir(args.in_dir):
                    raise RuntimeError(
                        msg + f"input directory {args.in_dir} does not exist."
                    )
                base = Path(args.in_dir)
                self._subjects = [
                    base / d if not d.is_absolute() else d
                    for d in self._subjects
                ]
            self._num_subjects = len(self._subjects)
            LOGGER.info(
                f"Analyzing all {self._num_subjects} subjects from csv_file "
                f"{args.csv_file}."
            )

        # 2. are we doing a single file (absolute path to the file)
        elif (orig_name := Path(self._orig_name_)).is_absolute():
            LOGGER.info("Single subject with absolute file path for input.")
            if not orig_name.is_file():
                raise RuntimeError(
                    f"The input file {orig_name} does not exist (is not a file)."
                )
            if self._out_dir is None:
                sid = ""
                if (out_segfile := Path(self._out_segfile)).is_absolute():
                    # try to extract the subject directory from the absolute out
                    # filename by, containing folder is 'mri' or the subject id
                    out_dirname = out_segfile.parent
                    parent_dir = out_dirname.name
                    msg = (
                        f"No subjects directory specified, but the parent directory "
                        f"of the output file {out_segfile} is"
                    )
                    if parent_dir == "mri":
                        LOGGER.info(
                            f"{msg} 'mri', so we are assuming this is the 'mri' folder "
                            f"in the subject directory."
                        )
                        self._out_dir = out_dirname.parent.parent
                        sid = out_dirname.parent.name
                        self._out_segfile = "mri/" + out_segfile.name
                    elif parent_dir == getattr(args, "sid", ""):
                        LOGGER.info(
                            f"{msg} the subject id, so we are assuming this is the "
                            f"subject directory."
                        )
                        self._out_dir = out_dirname.parent
                        sid = out_dirname.name
                        self._out_segfile = out_segfile.name

                def _not_abs(subj_attr):
                    return not os.path.isabs(getattr(self, f"_{subj_attr}_"))

                if getattr(args, "sid", "") in (None, ""):
                    args.sid = sid
                elif getattr(args, "sid", "") != sid and any(
                    map(_not_abs, self.__attr_assign.keys())
                ):
                    relative_files = [
                        f"args.{k} (cf. self._flags[v['flag']]): {getattr(self, f'_{k}_')}"
                        for k, v in self.__attr_assign.items()
                        if _not_abs(k)
                    ]
                    msg = (
                        "Could not extract the subject id from the command line and "
                        "the output file '{0}', while at the same time, not all output "
                        "files are absolute. Try passing the subjects directory in "
                        "args (c.f. {sd[flag]}), or absolute paths for {1}.".format(
                            self._segfile_, ", ".join(relative_files), **self._flags
                        )
                    )
                    raise RuntimeError(msg)

            self._subjects = [Path(self._orig_name_)]
            self._num_subjects = 1
            LOGGER.info(f"Analyzing single subject {self._orig_name_}")
        # 3. do we search in a directory
        elif getattr(args, "search_tag", None) is not None:
            search_tag = args.search_tag
            _in_dir = getattr(args, "in_dir", None)
            if not Path(search_tag).is_absolute() and _in_dir:
                base = Path(_in_dir)
                if not base.is_dir():
                    raise RuntimeError(f"The input directory {base} does not exist.")
                where = f"in_dir {base}"
            else:
                base = Path.cwd()
                where = f"the working directory {Path.cwd()}"

            self._subjects = list(base.glob(search_tag))
            self._num_subjects = len(self._subjects)
            LOGGER.info(
                f"Analyzing all {self._num_subjects} subjects from {where} with search "
                f"pattern {search_tag}."
            )

        else:
            msg = (
                "Could not identify how to find images to segment. Options are:\n1. "
                "Provide a text file with one subject directory or image file per line "
                "via args.csv (cf. {csv_file[flag]});\n2. specify an absolute path for "
                "relevant files, specifically the t1 file via args.orig_name (cf. "
                "{t1[flag]}), but ideally also for expected output files such as the "
                "segmentation output file,\n 3. provide a search pattern to search "
                "for subject directories or images via args.search_tag (c.f. "
                "{tag[flag]}).\n Note also, that the input directory (specified via "
                "{in_dir[flag]}) will be used as the base path for relative file paths "
                "of input files."
            )
            raise RuntimeError(msg.format(**self._flags))

        self._remove_suffix = getattr(args, "remove_suffix", "")
        if self._num_subjects > 1:
            if getattr(args, "sid", "") not in ["", None]:
                msg = (
                    "The usage of args.sid (cf. {sid[flag]}) with multiple subjects is "
                    "undefined."
                )
                raise RuntimeError(msg.format(**self._flags))
            if self._remove_suffix == "":
                all_subject_files = self.are_all_subject_files()
                common_suffix = self.get_common_suffix()
                msg = (
                    "We detected that the subjects share the common suffix {0} in the "
                    "subject name. You can remove trailing parts of the filename such "
                    "as file extensions and/or other characters by passing this suffix "
                    "in args.remove_suffix (cf. {remove_suffix[flag]} <suffix>, e.g. "
                    "{remove_suffix[flag]} '{0}'."
                )
                if all_subject_files and common_suffix != "":
                    LOGGER.info(msg.format(common_suffix, **self._flags))
                if os.path.isabs(self._out_segfile):
                    raise RuntimeError(
                        f"An absolute path was passed for the output segmentation "
                        f"{self._out_segfile}, but more than one input image fits the "
                        f"input definition."
                    )

        self._sid = getattr(args, "sid", "")

    __init__.__doc__ = __init__.__doc__.format(**DEFAULT_FLAGS)

    @property
    def flags(self) -> dict[str, dict]:
        """
        Give the flags.

        Returns
        -------
        dict[str, dict]
            Flags.
        """
        return self._flags

    def __len__(self) -> int:
        """
        Give length of subject list.

        Returns
        -------
        int
            Number of subjects.
        """
        return self._num_subjects

    def make_subjects_dir(self):
        """
        Try to create the subject directory.
        """
        if self._out_dir is None:
            LOGGER.info(
                "No Subjects directory found, absolute paths for filenames are "
                "required."
            )
            return

        LOGGER.info(f"Output will be stored in Subjects Directory: {self._out_dir}")

        if not os.path.exists(self._out_dir):
            LOGGER.info("Output directory does not exist. Creating it now...")
            os.makedirs(self._out_dir)

    def __getitem__(self, item: int | str) -> SubjectDirectory:
        """
        Return a SubjectDirectory object for the i-th subject (if item is an int) or for
        the subject with name/folder (if item is a str).

        Parameters
        ----------
        item : int, str
            The index or name of the subject.
            If integer, it is treated as an index and corresponding subject is returned.
            If string, it is treated as the subject.

        Returns
        -------
        SubjectDirectory
            A SubjectDirectory object corresponding to the provided index or name.
        """
        if isinstance(item, int):
            if item < 0 or item >= self._num_subjects:
                raise IndexError(
                    f"The index {item} is out of bounds for the subject list."
                )

            # subject is always an absolute path (or relative to the working directory)
            # ... of the input file
            subject = self._subjects[item]
            sid = (
                Path(str(subject).removesuffix(self._remove_suffix)).name
                if self._sid is None
                else self._sid
            )
        elif isinstance(item, str):
            subject = Path(item)
            sid = item
        else:
            raise TypeError("Invalid type of the item, must be int or str.")

        # Set subject and load orig
        special_rules = ["orig_name"]
        subject_parameters = {
            v: getattr(self, f"_{v}_")
            for v in self.__attr_assign.keys()
            if v not in special_rules
        }
        orig_name = (
            subject
            if subject.is_file()
            else subject / self._orig_name_
        )
        return SubjectDirectory(
            subject_dir=self._out_dir,
            id=sid,
            orig_name=orig_name,
            **subject_parameters,
        )

    def get_common_suffix(self) -> str:
        """
        Find common suffix, if all entries in the subject list share a common suffix.

        Returns
        -------
        str
            The suffix the entries share.
        """
        suffix = self._subjects[0]
        for subject_path in self._subjects[1:]:
            subj = str(subject_path)
            if subj.endswith(suffix):
                continue
            for i in range(1 - len(suffix), 1):
                if i == 0:
                    return ""
                if subj[i:] == suffix[i:]:
                    suffix = suffix[i:]
                    break
        return suffix

    def are_all_subject_files(self):
        """
        Check if all entries in subjects are actually files.

        This is performed asynchronously internally.
        """
        from concurrent.futures import ThreadPoolExecutor

        def is_file(p: Path):
            return p.is_file()
        with ThreadPoolExecutor(len(self._subjects)) as pool:
            return all(pool.map(is_file, self._subjects))


class SerialExecutor(Executor):
    """
    Represent a serial executor.
    """

    def map(
        self,
        fn: Callable[..., _T],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = -1,
    ) -> Iterator[_T]:
        """
        The map function.

        Parameters
        ----------
        fn : Callable[..., _T]
            A callable function to be applied to the items in the iterables.
        *iterables : Iterable[Any]
            One or more iterable objects.
        timeout : Optional[float]
            Maximum number of seconds to wait for a result. Default is None.
        chunksize : int
            The size of the chunks, default value is -1.

        Returns
        -------
        Iterator[_T]
            An iterator that yields the results of applying 'fn' to the items of
            'iterables'.
        """
        return map(fn, *iterables)

    def submit(self, __fn: Callable[..., _T], *args, **kwargs) -> "Future[_T]":
        """
        A callable function that returns a Future representing the result.

        Parameters
        ----------
        __fn : Callable[..., _T]
            A callable function to be executed.
        *args :
            Potential arguments to be passed to the callable function.
        **kwargs :
            Keyword arguments to be passed to the callable function.

        Returns
        -------
        "Future[_T]"
            A Future object representing the execution result of the callable function.
        """
        f = Future()
        try:
            f.set_result(__fn(*args, **kwargs))
        except Exception as e:
            f.set_exception(e)
        return f
