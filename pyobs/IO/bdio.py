#################################################################################
#
# bdio.py: binary file format based on the bdio library
# Copyright (C) 2021 Mattia Bruno
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
#################################################################################

# The BDIO file format consists in 1 header at the beginning of the file and
# several records. See https://github.com/to-ko/bdio
# The observables are stored in binary records with user info = 2. All details on
# such records can be found here
# https://ific.uv.es/~alramos/docs/ADerrors/tutorial/#BDIO-Native-format

import numpy
import pyobs
import datetime
import hashlib
from sys import byteorder


class BINARY_TYPES:
    INT32 = numpy.dtype("<i4")
    INT64 = numpy.dtype("<i8")
    FLOAT32 = numpy.dtype("<f4")
    FLOAT64 = numpy.dtype("<f8")


class BDIO_CONSTANTS:
    BDIO_BIN_GENERIC = 0
    BDIO_ASC_EXEC = 1
    BDIO_BIN_INT32BE = 2
    BDIO_BIN_INT32LE = 3
    BDIO_BIN_INT64BE = 4
    BDIO_BIN_INT64LE = 5
    BDIO_BIN_F32BE = 6
    BDIO_BIN_F32LE = 7
    BDIO_BIN_F64BE = 8
    BDIO_BIN_F64LE = 9
    BDIO_ASC_GENERIC = 10
    BDIO_ASC_XML = 11


bdio_const = BDIO_CONSTANTS
dtypes = BINARY_TYPES
little = byteorder == "little"


class binary_file:
    def __init__(self, fname, mode):
        self.mode = mode
        pyobs.assertion(mode == "r", "only read mode supported")
        self.fname = fname
        self.pos = 0
        if mode == "r":
            with open(fname, "rb") as f:
                f.seek(0, 2)
                self.len = f.tell()
        else:
            self.len = 0

    def eof(self):
        return self.pos >= self.len

    def skip(self, n):
        self.pos += n

    def read_binary(self, n):
        with open(self.fname, "rb") as f:
            f.seek(self.pos)
            bb = f.read(n)
            self.skip(n)
        return bb

    def read(self, dtype, n=1, force_array=False):
        pyobs.assertion(self.mode == "r", "Cannot read in write mode")
        sz = dtype.itemsize * n
        bb = self.read_binary(sz)
        if (n == 1) and (not force_array):
            out = numpy.frombuffer(bb, dtype)[0]
        else:
            out = numpy.frombuffer(bb, dtype).reshape((n,))
        return out if little else out.byteswap()

    def read_str(self, arg):
        if isinstance(arg, str):
            out = [""]
            while out[-1] != arg:
                out += [self.read_binary(1).decode("utf-8")]
            return "".join(out[:-1])
        elif isinstance(arg, (int, numpy.int32, numpy.int64)):
            return self.read_binary(arg).decode("utf-8")
        return ""


class bdio_file(binary_file):
    def __init__(self, fname, mode):
        super().__init__(fname, mode)
        self.BDIO_MAGIC = 2147209342
        self.BDIO_HASH_MAGIC_S = 1515784845
        self.header = {}
        self.records = []

    def parse(self):
        self.read_bdio_header()
        while not self.eof():
            self.read_record()

    def read_bdio_header(self):
        hdr = self.read(dtypes.INT32, 5)
        pyobs.assertion(hdr[0] == self.BDIO_MAGIC, "Not a bdio file")
        rlen = hdr[1] & int("00000fff", 16)
        self.header["version"] = (hdr[1] & int("ffff0000", 16)) >> 16
        self.header["dirinfo"] = [
            (hdr[2] & int("ffc00000", 16)) >> 22,
            (hdr[2] & int("003fffff", 16)),
        ]
        self.header["ctime"] = datetime.datetime.fromtimestamp(hdr[3])
        self.header["mtime"] = datetime.datetime.fromtimestamp(hdr[4])

        info = self.read_str(rlen - 12).split("\0")
        self.header["cuser"] = info[0]
        self.header["muser"] = info[1]
        self.header["chost"] = info[2]
        self.header["mhost"] = info[3]
        self.header["info"] = info[4]

    def read_record(self):
        hdr = int(self.read(dtypes.INT32, 1))
        islong = (hdr & int("00000008", 16)) >> 3
        if islong:
            self.skip(-4)
            lhdr = int.from_bytes(self.read_binary(8), "little")
            rfmt = (lhdr & int("00000000000000f0", 16)) >> 4
            ruinfo = (lhdr & int("0000000000000f00", 16)) >> 8
            rlen = (lhdr & int("fffffffffffff000", 16)) >> 12
        else:
            rfmt = (hdr & int("000000f0", 16)) >> 4
            ruinfo = (hdr & int("00000f00", 16)) >> 8
            rlen = (hdr & int("fffff000", 16)) >> 12
        dd = {
            "pos": self.pos,
            "uinfo": ruinfo,
            "fmt": rfmt,
            "len": rlen,
            "islong": islong,
            "content": self.parse_record_content(ruinfo, rfmt, rlen),
        }
        self.records += [dd]

    def parse_record_content(self, uinfo, fmt, rlen):
        if fmt == bdio_const.BDIO_BIN_GENERIC:
            if uinfo == 7:
                pyobs.assertion(rlen == 20, "Wrong MD5 record")
                pyobs.assertion(
                    self.read(dtypes.INT32) == self.BDIO_HASH_MAGIC_S,
                    "Failed checking HASH_MAGIC",
                )
                md5 = [
                    "%02hX" % int.from_bytes(self.read_binary(1), "little")
                    for _ in range(16)
                ]
                return "".join(md5)
            else:
                return hashlib.md5(self.read_binary(rlen)).hexdigest().upper()
        elif fmt == bdio_const.BDIO_ASC_GENERIC:
            return self.read_str(rlen)
        elif fmt == bdio_const.BDIO_BIN_F64BE:
            if little:
                return self.read(
                    dtypes.FLOAT64, rlen // dtypes.FLOAT64.itemsize
                ).byteswap()
            else:
                return self.read(dtypes.FLOAT64, rlen // dtypes.FLOAT64.itemsize)
        elif fmt == bdio_const.BDIO_BIN_F64LE:
            if not little:
                return self.read(
                    dtypes.FLOAT64, rlen // dtypes.FLOAT64.itemsize
                ).byteswap()
            else:
                return self.read(dtypes.FLOAT64, rlen // dtypes.FLOAT64.itemsize)
        self.skip(rlen)
        return None

    def __str__(self):
        out = f'\nBDIO File {self.header["info"]}\n'
        out += f'File created by {self.header["cuser"]} on {self.header["ctime"]} at {self.header["chost"]} \n'
        out += f'File modified by {self.header["muser"]} on {self.header["mtime"]} at {self.header["mhost"]} \n'
        for r in self.records:
            out += f'\nuinfo = {r["uinfo"]} ; bytes = {r["len"]} ; islong = {r["islong"]} \n'
            out += f'\n{r["content"]}\n'
        return out


def decode_bdio_observable(f):
    res = pyobs.observable()
    res.set_mean(f.read(dtypes.FLOAT64))

    neid = f.read(dtypes.INT32)
    f.skip(dtypes.INT32.itemsize * neid)  # ndata
    nrep = f.read(dtypes.INT32, neid, True)
    vrep = []
    for i in range(neid):
        vrep.append(f.read(dtypes.INT32, nrep[i], True))  # meas per replica
    f.skip(dtypes.INT32.itemsize * neid)

    # nt, zeros, fours
    f.skip((dtypes.INT32.itemsize + dtypes.FLOAT64.itemsize * 2) * neid)

    deltas = []
    for i in range(neid):
        for j in range(nrep[i]):
            deltas += [f.read(dtypes.FLOAT64, vrep[i][j], True)]

    res.description = f.read_str("\0")

    all_ename = []
    for i in range(neid):
        # numeric id
        f.skip(dtypes.INT32.itemsize)
        all_ename += [f.read_str("\0")]

    k = 0
    for i in range(neid):
        # numeric id
        f.skip(dtypes.INT32.itemsize)
        ename = all_ename[i]
        rname = [f.read_str("\0") for _ in range(nrep[i])]
        for j in range(nrep[i]):
            if vrep[i][j] > 1:
                icnfg = [int(idx) for idx in f.read(dtypes.INT32, vrep[i][j])]
                tag = f"{ename}:{rname[j]}"
                res.delta[tag] = pyobs.core.data.delta([0], icnfg)
                res.delta[tag].delta[0, :] = deltas[k][:]
            else:
                icnfg = f.read(dtypes.INT32)
                assert icnfg == 1
                res.cdata[ename] = pyobs.core.cdata.cdata(deltas[k] ** 2, [0])
            k += 1

    res.ename_from_delta()
    pyobs.memory.update(res)
    return res


def load(fname):
    f = bdio_file(fname, "r")
    f.parse()

    content = []
    obs = []
    for i in range(len(f.records)):
        r = f.records[i]
        if (r["uinfo"] == 7) and (
            f.records[i - 1]["fmt"] == bdio_const.BDIO_BIN_GENERIC
        ):
            pyobs.assertion(
                f.records[i - 1]["content"] == r["content"],
                "MD5 checksums do not match",
            )
        elif r["uinfo"] == 2:
            ff = binary_file(fname, "r")
            ff.skip(r["pos"])
            obs.append(decode_bdio_observable(ff))
        elif r["uinfo"] == 1:
            content.append(r["content"])

    return {
        "file_content": content,
        "observables": obs,
        "bdio_file": f,
    }
