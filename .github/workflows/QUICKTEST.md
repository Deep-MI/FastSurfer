# FastSurfer Singularity GitHub Actions Workflow

This GitHub Actions workflow is designed to automate the integration testing of new code into the FastSurfer repository using Singularity containers. The workflow is triggered whenever new code is pushed to the repository.

The workflow runs on a self-hosted runner labelled 'ci-gpu' to ensure security.

## Jobs

The workflow consists of several jobs that are executed in sequence:

### Checkout

This job checks out the repository using the `actions/checkout@v2` action.

### Prepare Job

This job sets up the necessary environments for the workflow. It depends on the successful completion of the `checkout` job. The environments set up in this job include:

- Python 3.10, using the `actions/setup-python@v3` action.
- Go, using the `actions/setup-go@v5` action with version `1.13.1`.
- Singularity, using the `eWaterCycle/setup-singularity@v7` action with version `3.8.3`.

### Build Singularity Image

This job builds a Docker image and converts it to a Singularity image. It depends on the successful completion of the `prepare-job`. The Docker image is built using a Python script `Docker/build.py` with the `--device cuda --tag fastsurfer_gpu:cuda` flags. The Docker image is then converted to a Singularity image.

### Run FastSurfer

This job runs FastSurfer on sample MRI data using the Singularity image built in the previous job. It depends on the successful completion of the `build-singularity-image` job. The Singularity container is executed with the `--nv`, `--no-home`, and `--bind` flags to enable GPU access, prevent home directory mounting, and bind the necessary directories respectively. The `FASTSURFER_HOME` environment variable is set to `/fastsurfer-dev` inside the container.

### Test File Existence

This job tests for the existence of certain files after running FastSurfer. It depends on the successful completion of the `run-fastsurfer` job. The test is performed using a Python script `test/test_file_existence.py`.

### Test Error Messages

This job tests for errors in log files after running FastSurfer. It runs on a self-hosted runner labeled `ci-gpu` and depends on the successful completion of both the `run-fastsurfer` and `test-file-existence` jobs. The test is performed using a Python script `test/test_error_messages.py`.

## Usage

To use this workflow, you need to have a self-hosted runner labeled `ci-gpu` set up on your machine. You also need to update the environment variables of the runner, by going to `/home/your_runner/.env` file and adding the following environment variables with the actual paths you want to use.


### Environment variables
`RUNNER_FS_MRI_DATA`: Path to MRI Data

`RUNNER_FS_OUTPUT`: Path to Output directory

`RUNNER_FS_LICENSE`: Path to License directory

`RUNNER_SINGULARITY_IMGS`: Path to where Singularity images should be stored

`RUNNER_FS_OUTPUT_FILES`: Path to output files to be tested

`RUNNER_FS_OUTPUT_LOGS`: Path to output log files to check for errors


Once everything is set up, you can trigger the workflow manually from the GitHub Actions tab in your repository, as well as by pushing code to the repository.
