# Author: Tao Hu <taohu620@gmail.com>

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, base_lr, i_iter, total_iter, power=0.9):
    lr = lr_poly(base_lr, i_iter, total_iter, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

    return lr