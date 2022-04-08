## Start with Docker pytorch base

FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04
ENV LANG=C.UTF-8
ARG PYTHON_VERSION=3.8
ENV DEBIAN_FRONTEND=noninteractive

# get fastsurfer into docker (needed for some install scripts)
COPY . /fastsurfer/

# Install custom libraries, freesurfer dependencies
# Install miniconda python packages, pip packages and freesurfer
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget \
      git \
      tcsh \
      time \
      bc \
      gawk \
      ca-certificates \
      libgomp1 \
      upx \
      file && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 
    
RUN wget --no-check-certificate -qO ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION python-dateutil=2.8.2 pyyaml=6.0 matplotlib=3.5.1 h5py=3.6.0 scikit-image=0.19.2 && \
     /opt/conda/bin/conda install -y pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=11.3 -c pytorch && \
     /opt/conda/bin/conda install -y -c conda-forge pillow=9.0.1 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN python$PYTHON_VERSION -m pip install -U git+https://github.com/Deep-MI/LaPy.git#egg=lapy && \
    python$PYTHON_VERSION -m pip install simpleitk==2.1.1 && \
    python$PYTHON_VERSION -m pip install numpy==1.22.3 && \
    python$PYTHON_VERSION -m pip install scipy==1.8.0 && \
    python$PYTHON_VERSION -m pip install nibabel==3.2.2 && \
    /fastsurfer/Docker/install_fs_pruned.sh /opt --upx

# Add FreeSurfer Environment variables
ENV OS=Linux \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA= \
    SUBJECTS_DIR=/opt/freesurfer/subjects \
    FSF_OUTPUT_FORMAT=nii.gz \
    FREESURFER_HOME=/opt/freesurfer \
    PYTHONUNBUFFERED=0 \
    PATH=/opt/freesurfer/bin:$PATH


# Set FastSurfer env and workdir:
WORKDIR "/fastsurfer"
ENV FASTSURFER_HOME=/fastsurfer
ENTRYPOINT ["./run_fastsurfer.sh"]
CMD ["--help"]
