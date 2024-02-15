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

data_dir = '/data/UKBiobank/scans'
out_dir = '/output/UKBiobank/output'
fs_licence = '/fs_license/.license'
t1_prefix = 'T1.nii.gz'

subjects = [sub for sub in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, sub))]


for subject in subjects:
    t1_image = glob.glob(os.path.join(data_dir,subject,t1_prefix))
    if t1_image:
        cmd = '/fastsurfer/run_fastsurfer.sh --t1 {} --sid {} --sd {} --fs_license {} --parallel --3T'.format(t1_image[0],
                                                                                                                subject,out_dir,fs_licence)
        #print(cmd)
        run_cmd(cmd)
    else:
        print(f" Subject {subject} image t1 not found")






