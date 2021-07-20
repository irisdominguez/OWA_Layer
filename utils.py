from errno import EEXIST
import os
    
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    try:
        os.makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else: raise