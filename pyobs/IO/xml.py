#################################################################################
#
# xml.py: plain text file format based on XML
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

# This file format is defined by the obs-tools MATLAB package developed by R. Sommer
# at DESY Zeuthen

import xml.etree.ElementTree as ET
import gzip
import pyobs
import numpy
import re
import json


def import_data(string):
    return json.loads("[" + ",".join(string.split()) + "]")

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        else:
            elem.tail += i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def dict2xml(parent, d):
    for key in d:
        el = ET.SubElement(parent, key)
        if type(d[key]) is dict:
            dict2xml(el, d[key])
        else:
            if type(d[key]) is list:
                el.text = str(d[key][0])
                el.tail = str(d[key][1])
            else:
                el.text = str(d[key])

def load(fname):
    def check(condition):
        pyobs.assertion(condition, "XML file format not supported")

    def import_array(arr):
        check(arr[0].tag == "id")
        tag = arr[0].text.strip()
        check(arr[1].tag == "layout")
        tmp = import_data(arr[1].tail)

        l = arr[1].text.strip()
        m = re.search(r"(\d+)\s+(\w?)\s*(\w?)(\d+)\s*(\w?)", l)

        if m.group(2) == "i" and m.group(3) == "f":
            nc = int(m.group(1))
            na = int(m.group(4))
            _dat = []
            mask = []
            for a in range(na):
                h = numpy.unique(tmp[1 + a :: na + 1])
                if len(h) == 1 and numpy.all(h == 0.0):
                    continue
                mask += [a]
                _dat += [numpy.array(tmp[1 + a :: na + 1])]
            check(len(tmp[0 :: na + 1]) == nc)
            return [tag, tmp[0 :: na + 1], mask, _dat]
        elif m.group(2) == "" and m.group(3) == "" and m.group(5) == "f":
            sh = (int(m.group(1)), int(m.group(4)))
            return numpy.reshape(tmp, sh)
        elif m.group(2) == "f" and m.group(3) == "" and m.group(5) == "":
            sh = (int(m.group(1)), int(m.group(4)))
            return numpy.reshape(tmp, sh)
        else:  # pragma: no cover
            check(False)

    def import_rdata(rd):
        rname, icnfg, mask, dat = import_array(rd)
        delta = pyobs.core.data.delta(mask, icnfg)
        for i in range(len(dat)):
            delta.delta[i, :] = dat[i]
        return rname, delta

    def import_cdata(cd):
        check(cd[0].tag == "id")
        check(cd[1][0].text.strip() == "cov")
        cov = import_array(cd[1])
        grad = import_array(cd[2]).T
        cdata = pyobs.core.cdata.cdata(cov, range(grad.shape[0]))
        check(cdata.grad.shape == grad.shape)
        cdata.grad[:, :] = grad[:, :]
        return [cd[0].text, cdata]

    xml = ET.fromstring(gzip.open(fname, "r").read())
    check(xml.tag == "OBSERVABLES")
    check(xml[0].tag == "SCHEMA")
    for i, key, val in zip([0, 1], ["NAME", "VERSION"], ["lattobs", "1.0"]):
        check((xml[0][i].tag == key) and (xml[0][i].text.strip() == val))

    check(xml[1].tag == "origin")
    www = [xml[1][i] for i in [2, 0, 1]]
    check(xml[1][3].tag == "tool")
    for i, key, val in zip([0, 1], ["name", "version"], ["obs2mxtree", "1"]):
        check((xml[1][3][i].tag == key) and (xml[1][3][i].text.strip() == val))

    check(xml[2].tag == "dobs")
    dobs = xml[2]
    res = pyobs.observable(
        description=dobs[2].text.strip()
        + " ; "
        + dobs[1].text.strip()
        + " ; "
        + dobs[0].text.strip()
    )
    res.www = www

    check(dobs[3].tag == "array")
    res.set_mean(import_array(dobs[3]))

    check(dobs[4].tag == "ne")
    ne = int(dobs[4].text.strip())
    check(dobs[5].tag == "nc")
    nc = int(dobs[5].text.strip())

    for k in range(6, len(list(dobs))):
        if dobs[k].tag == "edata":
            check(dobs[k][0].tag == "enstag")
            ename = dobs[k][0].text.strip()

            check(dobs[k][1].tag == "nr")
            R = int(dobs[k][1].text.strip())
            for i in range(2, 2 + R):
                rname, delta = import_rdata(dobs[k][i])
                res.delta[f"{ename}:{rname}"] = delta
        elif dobs[k].tag == "cdata":
            cname, cdata = import_cdata(dobs[k])
            res.cdata[cname] = cdata
        else:  # pragma: no cover
            check(False)

    res.ename_from_delta()
    check(len(res.ename) == ne)
    check(len(res.cdata) == nc)
    return res


def save(fname, obs):
    pyobs.assertion(len(obs.shape)==1,"Only 1-D observables supported")
            
    def delta2array(key, delta):
        pyobs.assertion(len(delta.mask) == obs.size, 'Observables with sub-masks not supported')
        sh = f'{delta.n} i f{len(delta.mask)}'
        dat = ''
        for _idx, _row in zip(delta.idx, delta.delta.T):
            dat += f'\n{_idx:d} ' + ' '.join(str(f) for f in _row)
        return {
            'id': key,
            'layout': [sh, dat[1:]]
        }
    
    root = ET.Element('OBSERVABLES')
    dict2xml(root, {'SCHEMA': {'NAME': 'lattobs', 'VERSION': '1.0',}})
    dict2xml(root, {'origin': {
        'who': obs.www[0],
        'date': obs.www[2],
        'host': obs.www[1],
        'tool': {
            'name': 'obs2mxtree',
            'version': '1',
        }
    }})
    dict2xml(root, {'dobs': {
        'spec': 'dobs v1.0',
        'origin': 'unknown',
        'name': obs.description,
        'array': {
            'id': 'val',
            'layout': [f'{obs.shape[0]} f1', ' '.join([str(m) for m in obs.mean])]
        },
        'ne': len(obs.ename),
        'nc': len(obs.cdata),
    }})
    
    def write_edata(ename):
        xml_edata = ET.SubElement(root.find('./dobs'), 'edata')
        dict2xml(xml_edata, {'enstag': ename})
        nr = 0

        arrays = []
        for key in obs.delta:
            en, rn = key.split(':')
            if en==ename:
                nr += 1
                arrays += [delta2array(rn, obs.delta[key])]
                
        dict2xml(xml_edata, {'nr': nr})
        for a in arrays:
            dict2xml(xml_edata, {'array': a})
        
    for en in obs.ename:
        write_edata(en)

    pyobs.assertion(len(obs.cdata)==0, "cdata not yet supported")
    
    indent(root)
    if fname[-7:]!='.xml.gz':
        fname += '.xml.gz'
    f = gzip.open(fname, 'wb')
    ET.ElementTree(root).write(f, xml_declaration=True)
    f.close()
