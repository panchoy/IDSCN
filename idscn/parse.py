import argparse as argp
import re


def parse_name(path, tp='0'):
    type = ['[group_name]', '[cova_name]', '[region_name]']
    group_name = []
    cova_name = []
    region_name = []
    reg = '[()a-zA-Z\\d_-]+'
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            while line and line.strip() == '':
                line = f.readline()
            if line.strip() in type:
                param = line.strip()
                if param == '[group_name]':
                    if tp == '0':
                        hc = []
                        pa = []
                        for i in range(2):
                            line = f.readline()
                            while line and line.strip() == '':
                                line = f.readline()
                            if line:
                                arr = re.findall(reg, line.strip())
                                if arr[0] == 'HC':
                                    hc = arr[1:]
                                if arr[0] == 'PA':
                                    pa = arr[1:]
                        group_name = [hc, pa]
                    else:
                        line = f.readline()
                        while line and line.strip() == '':
                            line = f.readline()
                        if line:
                            group_name = re.findall(reg, line.strip())
                    line = f.readline()
                else:
                    arr = []
                    line = f.readline()
                    while line:
                        if line.strip() and line.strip()[0] == '[' and line.strip()[-1] == ']':
                            break
                        arr.append(line)
                        line = f.readline()
                    if param[1] == 'c':
                        cova_name = re.findall(reg, ','.join(arr).strip())
                    else:
                        region_name = re.findall(reg, ','.join(arr).strip())
            else:
                print('illegal parameter {} !'.format(line.strip()))
    return group_name, cova_name, region_name


def parse():
    parser = argp.ArgumentParser()
    parser.add_argument('-m', '--mode', help="mode of process the data")
    parser.add_argument('-i', '--input', help='path of input data')
    parser.add_argument('-o', '--output', help='path of output data')
    parser.add_argument('--name', help='path of txt stores the group/cova/region name')
    parser.add_argument('--group_index', type=int, help='index of column about the group name')
    parser.add_argument('-t', '--type', help='type control')
    parser.add_argument('-n', '--number', help='number of selected edges')
    parser.add_argument('--plot', help='plot the histogram of sorted edges', action='store_true')
    parser.add_argument('--fdr', help='fdr correction', action='store_true')
    parser.add_argument('--n_perm', help='number of permutations', type=int, default=1000)
    return parser
