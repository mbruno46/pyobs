import numpy
import xml.etree.ElementTree as ET

def read_array(root):
    dd = {}
    for child in root:
        dd[child.tag] = child.text
    dd['data'] = child.tail
    return dd

def read_edata(root):
	dd = {'rdata':[]}
	for child in root:
		if (child.tag=="array"):
			dd['rdata'].append( read_array(child) )
		else:
			dd[child.tag] = child.text
			print child.tag, child.text
	return dd

def read_xml(fname):
    xml = {'edata':[]}
    if (fname[-4:]=='.xml'):
        tree = ET.parse(fname)
        root = tree.getroot()
    else:
        root = ET.fromstring(fname)

    for child in root:
        if (child.tag=="SCHEMA"):
            for subchild in child:
                print child.tag, subchild.tag, subchild.text
        elif (child.tag=="origin"):
            for subchild in child:
                print child.tag, subchild.tag, subchild.text
        elif (child.tag=="dobs"):
            for subchild in child:
                if (subchild.tag=="name"):
                    xml[subchild.tag] = subchild.text
                elif (subchild.tag=="ne"):
                    xml[subchild.tag] = subchild.text
                elif (subchild.tag=="nc"):
                    xml[subchild.tag] = subchild.text
                elif (subchild.tag=="array"):
                    tmp = read_array(subchild)
                    xml['mean'] = tmp['data']
                elif (subchild.tag=="edata"):
                    tmp = read_edata(subchild)
                    xml['edata'].append(tmp)
                else:
                    print child.tag, subchild.tag, subchild.text
            break
    return xml
