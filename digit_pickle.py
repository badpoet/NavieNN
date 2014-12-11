__author__ = 'badpoet'

import struct
import cPickle
import numpy as np

def prepare_data(fn):
    f = open(fn, "rb")
    bin_data = f.read()
    f.close()
    i = 0
    obj_data = []

    while (i < len(bin_data)):
        this_len,  = struct.unpack('h', bin_data[i : i + 2])
        bitmap = ""
        for each in bin_data[i + 6 : i + this_len]:
            k, = struct.unpack('B', each)
            s = ""
            cnt = 8
            while cnt > 0:
                s = str(k % 2) + s
                k /= 2
                cnt -= 1
            bitmap += s
        idx, = struct.unpack('B', bin_data[i + 3])
        idx = int(chr(idx))
        obj_data.append({
            "in": bitmap,
            "out": idx
        })
        i += this_len

    print "data ready"
    return obj_data

f = open("digit/digit_train.dat", "wb")
cPickle.dump(prepare_data("digit/OURDIG.PNT"), f)
f = open("digit/digit_test.dat", "wb")
cPickle.dump(prepare_data("digit/COLTEST.PNT"), f)
