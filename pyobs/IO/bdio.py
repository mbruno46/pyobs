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
import os, pwd
import time, datetime
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


def get_bdio_const(data):
    if isinstance(data, str):
        return bdio_const.BDIO_ASC_GENERIC
    elif isinstance(data, (bytearray, bytes)):
        return bdio_const.BDIO_BIN_GENERIC
    elif isinstance(data, numpy.ndarray):
        #         if data.dtype == dtypes.INT32:
        #             return bdio_const.BDIO_BIN_INT32LE
        if data.dtype == dtypes.FLOAT64:
            return bdio_const.BDIO_BIN_F64LE
    raise pyobs.PyobsError("data format not supported")


def md5_hash(buf):
    return hashlib.md5(buf).hexdigest().upper()


class binary_file:
    def __init__(self, fname, mode):
        self.mode = mode
        self.fname = fname
        self.pos = 0
        if mode == "r":
            with open(fname, "rb") as f:
                f.seek(0, 2)
                self.len = f.tell()
        elif mode == "w":
            f = open(fname, "wb")
            f.close()
            self.len = 0
        else:  # pragma: no cover
            raise pyobs.PyobsError("mode not understood or supported")
        self.buf = bytearray()

    def eof(self):
        return self.pos >= self.len

    def skip(self, n):
        self.pos += n

    def read_binary(self, n):
        pyobs.assertion(self.mode == "r", "Read mode")
        with open(self.fname, "rb") as f:
            f.seek(self.pos)
            bb = f.read(n)
            self.skip(n)
        return bb

    def write_binary(self, bb):
        pyobs.assertion(self.mode == "w", "Write mode")
        with open(self.fname, "ab") as f:
            f.seek(self.pos)
            self.skip(f.write(bb))
        return bb

    def read(self, dtype, n=1, force_array=False, data_little=True):
        pyobs.assertion(self.mode == "r", "Cannot read in write mode")
        sz = dtype.itemsize * n
        bb = self.read_binary(sz)
        if (n == 1) and (not force_array):
            out = numpy.frombuffer(bb, dtype)[0]
        else:
            out = numpy.frombuffer(bb, dtype).reshape((n,))
        return out if (little == data_little) else out.byteswap()

    def read_str(self, arg):
        if isinstance(arg, str):
            out = [""]
            while out[-1] != arg:
                out += [self.read_binary(1).decode("utf-8")]
            return "".join(out[:-1])
        elif isinstance(arg, (int, numpy.int32, numpy.int64)):
            return self.read_binary(arg).decode("utf-8")
        return ""

    def reset_encoder(self):
        self.buf = bytearray()

    def encode(self, data, dt):
        array = numpy.array(data).astype(dt)
        if little:
            self.buf += array.tobytes("C")
        else:
            self.buf += array.byteswap().tobytes("C")

    def encode_str(self, string):
        if string[-1] != "\0":
            string += "\0"
        self.buf += str.encode(string)

    def flush(self):
        self.write_binary(self.buf)


class bdio_file(binary_file):
    def __init__(self, fname, mode):
        super().__init__(fname, mode)
        self.BDIO_MAGIC = 2147209342
        self.BDIO_HASH_MAGIC_S = 1515784845
        self.BDIO_VERSION = 1
        self.BDIO_MAX_RECORD_LENGTH = 1048575
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
        pyobs.assertion(self.header["version"] == self.BDIO_VERSION, "Not a bdio file")
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

    def write_bdio_header(self, protocol):
        info = [pwd.getpwuid(os.getuid())[0] + "\0"] * 2
        info += [os.uname()[1] + "\0"] * 2
        info += [protocol[0:3505] + "\0"]
        rlen = 20 + sum([len(i) for i in info])

        minpadding = 2 * 33 + 2 * 256 + 12 - rlen
        minpadding += (4 - (rlen + minpadding) % 4) % 4
        info += ["\0"] * minpadding
        rlen += minpadding

        hdr = numpy.zeros((5,), dtype=dtypes.INT32)
        hdr[0] = self.BDIO_MAGIC
        hdr[1] = ((self.BDIO_VERSION & int("0xffff", 16)) << 16) | (
            (rlen - 8) & int("0x00000fff", 16)
        )
        hdr[2] = ((0 & int("0x3ff", 16)) << 22) | (0 & int("0x3fffff", 16))
        hdr[3] = time.time()
        hdr[4] = hdr[3]

        self.reset_encoder()
        self.encode(hdr, dtypes.INT32)
        self.encode_str("".join(info))
        self.flush()

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

    def write_record(self, data, uinfo):
        if isinstance(data, str):
            bb = str.encode(data)
        elif isinstance(data, (bytearray, bytes)):
            bb = data
        elif isinstance(data, numpy.ndarray):
            if little:
                bb = data.tobytes("C")
            else:
                bb = data.byteswap().tobytes("C")
        rlen = len(bb)
        islong = rlen > self.BDIO_MAX_RECORD_LENGTH
        fmt = get_bdio_const(data)
        self.reset_encoder()
        if islong:
            hdr = (
                int("0x0000000000000001", 16)
                | int(" 0x0000000000000008", 16)
                | ((fmt) << 4)
                | ((uinfo) << 8)
                | (rlen << 12)
            )
            self.encode(hdr, dtypes.INT64)
        else:
            hdr = int("00000001", 16) | ((fmt) << 4) | ((uinfo) << 8) | (rlen << 12)
            self.encode(hdr, dtypes.INT32)
        self.buf += bb
        self.flush()

    def write_md5_record(self, md5):
        self.reset_encoder()
        self.encode(self.BDIO_HASH_MAGIC_S, dtypes.INT32)
        self.buf += md5
        self.write_record(self.buf, 7)

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
                return md5_hash(self.read_binary(rlen))
        elif fmt == bdio_const.BDIO_ASC_GENERIC:
            return self.read_str(rlen)
        elif fmt == bdio_const.BDIO_BIN_F64BE:
            return self.read(
                dtypes.FLOAT64, rlen // dtypes.FLOAT64.itemsize, data_little=False
            )
        elif fmt == bdio_const.BDIO_BIN_F64LE:
            return self.read(
                dtypes.FLOAT64, rlen // dtypes.FLOAT64.itemsize, data_little=True
            )
        else:  # pragma: no cover
            raise pyobs.PyobsError("bdio error: format not supported")

    def __str__(self):
        out = f'\nBDIO File {self.header["info"]}\n'
        out += f'File created by {self.header["cuser"]} on {self.header["ctime"]} at {self.header["chost"]} \n'
        out += f'File modified by {self.header["muser"]} on {self.header["mtime"]} at {self.header["mhost"]} \n'
        for r in self.records:
            out += f'\nuinfo = {r["uinfo"]} ; bytes = {r["len"]} ; islong = {r["islong"]} \n'
            out += f'\t{r["content"]}\n'
        return out


def decode_bdio_observable(f, info):
    res = pyobs.observable()
    res.www = [info["cuser"], info["chost"], info["ctime"]]
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
                res.cdata[rname[j]] = pyobs.core.cdata.cdata([1.0], [0])
                res.cdata[rname[j]].grad[0, 0] = deltas[k]
            k += 1

    res.ename_from_delta()
    pyobs.memory.update(res)
    return res


def encode_bdio_observable(f, obs):
    pyobs.assertion(obs.size == 1, "bdio format supports only scalar observables")
    cd = {}
    for key in obs.cdata:
        tmp = obs.cdata[key].cholesky()
        cd[key] = tmp.grad[0, :]
    cnames = list(cd.keys())

    f.reset_encoder()
    f.encode(obs.mean, dtypes.FLOAT64)
    neid = len(obs.ename) + len(cnames)
    f.encode(neid, dtypes.INT32)

    nrep = [0] * neid
    vrep = []
    ndata = [0] * neid
    nt = [0] * neid
    for key in obs.delta:
        en = key.split(":")[0]
        i = obs.ename.index(en)
        ndata[i] += obs.delta[key].n
        nrep[i] += 1
        vrep += [obs.delta[key].n]
        if obs.delta[key].n // 2 > nt[i]:
            nt[i] = obs.delta[key].n // 2

    k = len(obs.ename)
    for key in cd:
        i = k + cnames.index(key)
        ndata[i] += 1
        nrep[i] += len(cd[key])
        vrep += [1] * len(cd[key])
        nt[i] = 0

    f.encode(ndata, dtypes.INT32)
    f.encode(nrep, dtypes.INT32)
    f.encode(vrep, dtypes.INT32)
    f.encode(range(neid), dtypes.INT32)
    f.encode(nt, dtypes.INT32)
    f.encode([0.0] * neid, dtypes.FLOAT64)
    f.encode([4.0] * neid, dtypes.FLOAT64)

    for key in obs.delta:
        f.encode(obs.delta[key].delta, dtypes.FLOAT64)
    for key in cd:
        f.encode(cd[key], dtypes.FLOAT64)

    f.encode_str(obs.description)

    k = 0
    for en in obs.ename + cnames:
        f.encode(k, dtypes.INT32)
        f.encode_str(en)
        k += 1

    k = 0
    for en in obs.ename:
        f.encode(k, dtypes.INT32)
        for key in obs.delta:
            h = key.split(":")
            if h[0] == en:
                f.encode_str(h[1])
        for key in obs.delta:
            h = key.split(":")
            if h[0] == en:
                f.encode(obs.delta[key].idx, dtypes.INT32)
        k += 1

    for key in cd:
        f.encode(k, dtypes.INT32)
        for n in range(len(cd[key])):
            f.encode_str(f"{key}_{n}")
        for n in range(len(cd[key])):
            f.encode(1, dtypes.INT32)
        k += 1


def load(fname):
    f = bdio_file(fname, "r")
    f.parse()

    out = []
    for i in range(len(f.records)):
        r = f.records[i]
        if r["uinfo"] == 2:
            ff = binary_file(fname, "r")
            ff.skip(r["pos"])
            out.append(decode_bdio_observable(ff, f.header))
        elif r["uinfo"] == 1:
            out.append(r["content"])
        elif (r["uinfo"] == 7) and (
            f.records[i - 1]["fmt"] == bdio_const.BDIO_BIN_GENERIC
        ):
            pyobs.assertion(
                f.records[i - 1]["content"] == r["content"],
                "MD5 checksums do not match",
            )
        elif r["uinfo"] == 8:
            out.append(r["content"])

    return out + [f]


def save(fname, *args):
    f = bdio_file(fname, "w")
    f.write_bdio_header("prova1")

    pyobs.assertion(len(args) > 1, "")
    f.write_record(args[0], 1)
    for a in args[1:]:
        if isinstance(a, pyobs.observable):
            pyobs.assertion(
                a.size == 1, "Only single observables can be stored in bdio format"
            )
            encode_bdio_observable(f, a)
            md5 = hashlib.md5(f.buf).digest()
            f.write_record(f.buf, 2)
            f.write_md5_record(md5)
        else:
            f.write_record(a, 8)
    f.write_record("end", 1)
