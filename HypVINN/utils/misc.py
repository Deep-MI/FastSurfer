import subprocess
import shlex


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
