import argparse

def isNone(v):
    return v.lower() in ('none')

def isInt(v):
    return v.isdigit() or (v.startswith('-') and v[1:].isdigit())

def isTrue(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')

def isFalse(v):
    return v.lower() in ('no', 'false', 'f', 'n', '0')

def str2int(v):
    if isNone(v):
        return None
    elif isInt(v):
        return int(v)
    else:
        raise argparse.ArgumentTypeError('Integer value expected.')

def str2bool(v):
    if isNone(v):
        return None
    elif isTrue(v):
        return True
    elif isFalse(v):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')