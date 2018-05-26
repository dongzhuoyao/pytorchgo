# Author: Tao Hu <taohu620@gmail.com>
import sys

def set_debugger_org_frc():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)