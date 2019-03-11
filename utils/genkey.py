from hashlib import blake2b

def genkey(value, length = 8):
    """
    >>> ' '.join(genkey(i) for i in  range(5))
    '0UAqFt42 i0VpEq24 76dfZeT3 oHwLM35E ogyjesdg'
    """
    if not isinstance(value, str):
        raise ValueError('Expected `value` to be `str`.')

    return blake2b(value.encode('utf-8'), digest_size=4).hexdigest()
