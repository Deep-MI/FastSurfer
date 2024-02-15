import os
def get_hypinn_mode_config(args):

    mode = args.mode

    if mode == 'auto':
        if hasattr(args, 't1') and hasattr(args, 't2'):
            if os.path.isfile(args.t1) and os.path.isfile(args.t2):
                args.mode = 'multi'
            elif os.path.isfile(args.t1):
                args.mode ='t1'
                args.t2 = None
            elif os.path.isfile(args.t2):
                args.mode = 't2'
                args.t1 = None
            else:
                raise FileNotFoundError('No t1 or t2 image found')
        elif hasattr(args, 't1'):
            args.mode = 't1'
            args.t2 = None
        else:
            args.mode = 't2'
            args.t1 = None
    elif mode == 'multi':
        pass
    elif mode == 't1':
        args.t2 = None
    elif mode == 't2':
        args.t1 = None

    return args



