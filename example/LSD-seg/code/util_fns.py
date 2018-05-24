import os
import os.path as osp


def get_log_dir(log_dir, model_name, lr, opt):
    
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
        
    # load config
    name = 'MODEL-%s_CFG-%s_LR_%.8f' % (model_name, opt, lr)
    
    # create out
    log_dir = osp.join(log_dir, name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir





def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.uniform_(-0.01, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


