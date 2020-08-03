from collections import namedtuple

Genotype = namedtuple('Genotype', 'hpf_select normal normal_concat reduce reduce_concat')


PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'conv_3x3',
    'conv_5x5',
    'sep_conv_3x3',
]

PC_DARTS_steganalysis = Genotype(hpf_select=[28, 29, 17, 0, 20, 11, 21, 8, 1, 15, 23, 6, 2, 3, 4, 19], normal=[('conv_3x3', 0), ('conv_5x5', 1)], normal_concat=range(2, 3), reduce=[('conv_5x5', 1), ('skip_connect', 0)], reduce_concat=range(2, 3))