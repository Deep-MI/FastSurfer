## Start with Docker pytorch base

FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04
ENV LANG=C.UTF-8
ARG PYTHON_VERSION=3.6

# Install custom libraries + freesurfer 6.0 (and necessary dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         vim \
         wget \
         ca-certificates \
         bzip2 \
         libx11-6 \
         libjpeg-dev \
         libpng-dev \
         bc \
         tar \
         zip \
         gawk \
         tcsh \
         time \
         libgomp1 \
         libglu1-mesa \
	 libglu1-mesa-dev \
	 perl-modules && \
         apt-get clean && \
         rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
	 wget -qO- https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.1/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1.tar.gz | tar zxv --no-same-owner -C /opt \
    	--exclude='freesurfer/trctrain' \
    	--exclude='freesurfer/subjects/fsaverage_sym' \
    	--exclude='freesurfer/subjects/fsaverage3' \
    	--exclude='freesurfer/subjects/fsaverage4' \
    	--exclude='freesurfer/subjects/fsaverage5' \
    	--exclude='freesurfer/subjects/fsaverage6' \
    	--exclude='freesurfer/subjects/cvs_avg35' \
    	--exclude='freesurfer/subjects/cvs_avg35_inMNI152' \
    	--exclude='freesurfer/subjects/bert' \
    	--exclude='freesurfer/subjects/V1_average' \
    	--exclude='freesurfer/average/mult-comp-cor' \
    	--exclude='freesurfer/lib/cuda' \
    	--exclude='freesurfer/lib/qt'

# Install miniconda and needed python packages (for FastSurferCNN)
RUN wget -qO ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION python-dateutil pyyaml numpy scipy matplotlib h5py scikit-image && \
     /opt/conda/bin/conda install -y -c pytorch cudatoolkit=10.0 "pytorch=1.2.0=py3.6_cuda10.0.130_cudnn7.6.2_0" torchvision=0.4.0 && \
     /opt/conda/bin/conda install -y -c conda-forge scikit-sparse nibabel=2.5.1 pillow=8.3.2 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# install pip related dependencies
RUN python$PYTHON_VERSION -m pip install -U git+https://github.com/Deep-MI/LaPy.git#egg=lapy

# Add FreeSurfer Environment variables (.license file needed, alternatively export FS_LICENSE=path/to/license)
ENV OS=Linux \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA= \
    SUBJECTS_DIR=/opt/freesurfer/subjects \
    FSF_OUTPUT_FORMAT=nii.gz \
    MNI_DIR=/opt/freesurfer/mni \
    LOCAL_DIR=/opt/freesurfer/local \
    FREESURFER_HOME=/opt/freesurfer \
    FSFAST_HOME=/opt/freesurfer/fsfast \
    MINC_BIN_DIR=/opt/freesurfer/mni/bin \
    MINC_LIB_DIR=/opt/freesurfer/mni/lib \
    MNI_DATAPATH=/opt/freesurfer/mni/data \
    FMRI_ANALYSIS_DIR=/opt/freesurfer/fsfast \
    PERL5LIB=/opt/freesurfer/mni/lib/perl5/5.8.5 \
    MNI_PERL5LIB=/opt/freesurfer/mni/lib/perl5/5.8.5 \
    PYTHONUNBUFFERED=0 \
    PATH=/opt/freesurfer/bin:/opt/freesurfer/fsfast/bin:/opt/freesurfer/tktools:/opt/freesurfer/mni/bin:$PATH


# Add FastSurfer (copy application code) to docker image
COPY . /fastsurfer/
ENV FASTSURFER_HOME=/fastsurfer
WORKDIR "/fastsurfer"


ENTRYPOINT ["./run_fastsurfer.sh"]
CMD ["--help"]
