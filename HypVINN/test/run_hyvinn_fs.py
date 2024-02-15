import subprocess
import shlex
import glob
import os

def run_cmd(cmd):
    """
    execute the comand
    """
    print('#@# Command: ' + cmd + '\n')
    args = shlex.split(cmd)
    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError as e:
        print('ERROR: ' + 'cannot run command')
        # sys.exit(1)
        raise
    print('\n')

data_dir = '/data/UKBiobank/output'
fs_licence = '/fs_license/.license'
t1_prefix = '*orig_nu.mgz'
t2_prefix = None#'*T2*.nii.gz'
reg_type = 'coreg'
mode = 'auto'
batch_size =6
log_prefix = '*deep-seg.log'

no_pre_proc = True
no_bc = False
no_reg = False

subjects = [sub for sub in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, sub))]

os.environ['FS_LICENSE'] = fs_licence


for subject in subjects:
    if t1_prefix:
        t1_image = glob.glob(os.path.join(data_dir, subject,'mri', t1_prefix))
    else:
        t1_image = [None]
    if t2_prefix:
        t2_image = glob.glob(os.path.join(data_dir, subject,'mri', t2_prefix))
    else:
        t2_image = [None]

    seg_log = glob.glob(os.path.join(data_dir, subject, 'scripts', log_prefix))

    if t1_image or t2_image:
        cmd = 'python3 /fastsurfer/HypVINN/run_pipeline.py --sid {} --t1 {} --t2 {} --sd {} --batch_size {} --reg_type {} --seg_log {}'.format(
            subject, t1_image[0], t2_image[0], data_dir, batch_size, reg_type,seg_log[0])
        # print(cmd)
        if no_pre_proc:
            cmd = cmd + ' --no_pre_proc'
        else:
            if no_bc:
                cmd = cmd + ' --no_bc'
            if no_reg:
                cmd = cmd + ' --no_reg'
        run_cmd(cmd)
    else:
        print(f" Subject {subject} image t1 or t2 not found")


