import glob
import os


root_dir = '/groups/ag-reuter/projects/hypothalamus/vinn_version/hypvinn_testsuite/'

folders = ['RS', 'UKBiobank']
t1_prefix = 'T1.nii.gz'

for folder in folders:
    dir = os.path.join(root_dir,folder)
    subjects = [sub for sub in os.listdir(dir) if os.path.isdir(os.path.join(dir, sub))]
    run_cmd = []
    for sub in subjects:
        t1_file = glob.glob(os.path.join(dir,sub,t1_prefix))
        if t1_file:
            cmd = sub + '=/data/' + sub + '/' + t1_file[0].split('/')[-1]
            run_cmd.append(cmd)

    with open(os.path.join(dir,'subject_list.txt'), 'w') as fp:
        for item in run_cmd:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')

