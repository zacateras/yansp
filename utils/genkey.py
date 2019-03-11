import random
import string

def genkey(value, length = 8, chars = string.ascii_letters + string.digits):
    """
    Return a string of `length` characters chosen pseudo-randomly from
    `chars` using `value` as the seed.

    >>> ' '.join(genkey(i) for i in  range(5))
    '0UAqFt42 i0VpEq24 76dfZeT3 oHwLM35E ogyjesdg'
    """
    generator = random.Random()
    generator.seed(value)

    return ''.join(generator.choice(chars) for _ in range(length))
