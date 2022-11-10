import os
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import scipy.stats as sps
import statsmodels.stats.multitest as smsm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def generate_dataset(filepath, outpath, group_name, group_index, cova_name, region_name=None, tp='0'):
    """
    This method will generate the control group and standard patient data (covariates in front
    and regions behind) from raw data.

    |<--                  raw data                     -->|                   |<--cova-->|   |<--    regions   -->|
    subject    name    age    sex    region1    region2 ...         subject    age    sex    region1    region2 ...
    sub-001    Amy     18     Female (number)   (number)     --->   sub-001    18     Female (number)   (number)
    sub-002    Bob     20     Male   (number)   (number)            sub-002    20     Male   (number)   (number)
    sub-003    Chris   30     Female (number)   (number)            sub-003    30     Female (number)   (number)

    :param filepath: str
           path of raw data file. The source file is .csv type.
    :param outpath: str
           path to the generated files.
    :param group_name: string or list
           group name of control group data or patient.
           if tp == 0, params must be a list with two elements, the first element (list or string)
           is for control group and the second (list or string) is for patients.
           if tp != 0, params is a list or string.
    :param group_index: integer
           index of column to separate control group data and patient data.
    :param cova_name: list
           name list of covariates.
    :param region_name: list
           name list of regions.
    :param tp: integer
           type of generate mode.
           0 -> generate both control group and patient data.
           1 -> only generate control group.
           2 -> only generate patient data.
    :return:
    """

    # check the parameters
    assert isinstance(filepath, str), 'filepath must be a string.'
    assert filepath.strip().split('.')[-1] == 'csv', 'input file must be a .csv file.'
    assert os.path.exists(filepath), 'input file is not exist.'
    assert isinstance(outpath, str), 'outpath must be a string.'
    assert isinstance(cova_name, list), 'cova_name must be a list.'
    for cova in cova_name:
        assert isinstance(cova, str), 'items in cova_name must be string.'
    assert region_name is None or isinstance(region_name, list), 'region_name must be a list.'
    if region_name is not None:
        for region in region_name:
            assert isinstance(region, str), 'items in region_name must be string.'
    assert isinstance(group_name, str) or isinstance(group_name, list), 'params must be a string or list.'
    assert tp in ['0', '1', '2'], 'tp must be an integer, which is 0, 1 or 2.'
    if tp == '0':
        assert len(group_name) == 2, 'length of params must be less than two with tp==0.'
        for group in group_name:
            if isinstance(group, list):
                for g in group:
                    assert isinstance(g, str), 'group name of control case and patient must be string.'
            elif isinstance(group, str):
                pass
            else:
                raise TypeError('group name of control case or patient must be string.')
    else:
        if isinstance(group_name, list):
            for group in group_name:
                assert isinstance(group, str), 'group name of control case and patient must be string.'
    assert isinstance(group_index, int) and group_index > 0, 'group index must be an integer and greater than 0.'

    # start generate
    outPath = outpath
    if outpath[-1] in ['/', '\\']:
        outPath = outpath[:-1]
    if not os.path.exists(outPath) or os.path.isfile(outPath):
        os.makedirs(outPath)
    print('Raw data path is: {}'.format(filepath))
    print('Outpath is: {}'.format(outpath))

    raw = pd.read_csv(filepath, encoding_errors='ignore')
    source = raw[[raw.columns.values[group_index - 1], raw.columns.values[0]] + cova_name + region_name]
    col_sex = 'sex'
    col_group = source.columns.values[0]
    for c in source.columns.values:
        if c.lower() == 'sex':
            col_sex = c
            break
    source = source.dropna(axis=0)
    if source[col_sex].dtype == 'object':
        source[col_sex] = source[col_sex].str.lower()
        source[col_sex] = source[col_sex].replace('female', 1)
        source[col_sex] = source[col_sex].replace('male', 2)
    if tp == '0':
        hc = (source.loc[source[col_group].isin(group_name[0])])[source.columns.values[1:]]
        pa = (source.loc[source[col_group].isin(group_name[1])])[source.columns.values[1:]]
        hc.to_csv(outPath + '/controls.csv', index=False)
        print('Generate {} successfully!'.format(outPath + '/controls.csv'))
        pa.to_csv(outPath + '/patients.csv', index=False)
        print('Generate {} successfully!'.format(outPath + '/patients.csv'))
    elif tp == '1':
        hc = (source.loc[source[col_group].isin(group_name)])[source.columns.values[1:]]
        hc.to_csv(outPath + '/controls.csv', index=False)
        print('Generate {} successfully!'.format(outPath + '/controls.csv'))
    else:
        pa = (source.loc[source[col_group].isin(group_name)])[source.columns.values[1:]]
        pa.to_csv(outPath + '/patients.csv', index=False)
        print('Generate {} successfully!'.format(outPath + '/patients.csv'))
    print('Dataset is generated successfully!')


def read_dataset(filepath, tp, cova, region):
    """
    This method will read the generated dataset of controls or patients.

    :param filepath: str
           path of raw data file. The source file is .csv type.
    :param tp: str
           data type of the dataset, controls or patients,so it must be in ['ctrl','pati'].
    :param c_r: tuple
           total covariates and regions.
           e.g. (2,3) expresses 2 covariates and 3 regions.
    :param cova: list
           name list of selected covariates.
    :param region: list
           name list of selected regions.

    :return: (cova_cols, region_cols, ct) or (pati_subs, cova_cols, region_cols, pa)
    """

    # check the parameters
    assert isinstance(filepath, str), 'filepath must be a string.'
    assert filepath.strip().split('.')[-1] == 'csv', 'input file must be a .csv file.'
    assert os.path.exists(filepath), 'input file is not exist.'
    assert tp in ['ctrl', 'pati'], 'tp must be "ctrl" or "pati".'
    assert isinstance(cova, list), 'cova must be a list.'
    for c in cova:
        assert isinstance(c, str), 'items in cova_name must be string.'
    assert isinstance(region, list), 'region must be a list.'
    for r in region:
        assert isinstance(r, str), 'items in region must be string.'

    # start run
    df = pd.read_csv(filepath, encoding='utf-8', encoding_errors='ignore')
    covas = df[cova]
    regions = df[region]
    if tp == 'ctrl':
        ctrl_covas = covas
        ctrl_regions = regions
        ct = pd.concat([ctrl_covas, ctrl_regions], axis=1)
        return list(ctrl_covas.columns.values), list(ctrl_regions.columns.values), ct
    else:
        pati_subs = list(df.iloc[:, 0].values)
        pa = pd.concat([covas, regions], axis=1)
        return list(pati_subs), list(covas.columns.values), list(regions.columns.values), pa


def PCC(covas, regions, group):
    """
    This method will generate the partial correlation matrix (i.e. PCCn).

    :param covas: list
           name list of covariates.
    :param regions: list
           name list of regions.
    :param group: pandas.core.frame.DataFrame
           group data.

    :return: PCC
    """

    assert isinstance(covas, list), 'covas must be a list.'
    for cova in covas:
        assert isinstance(cova, str), 'items in covas must be string.'
    assert isinstance(regions, list), 'regions must be a list.'
    for region in regions:
        assert isinstance(region, str), 'items in covas must be string.'
    assert isinstance(group, pd.DataFrame), 'ctrl_group must be a pandas.core.frame.DataFrame.'

    pcorr = []
    for r1 in regions:
        pcorr_col = []
        for r2 in regions:
            if r1 != r2:
                results = pg.partial_corr(data=group, x=r1, y=r2, covar=covas)
                pcorr_col.append(results.r.values[0])
            else:
                pcorr_col.append(1.0)
        pcorr.append(pcorr_col)
    return np.array(pcorr).astype(np.float_)


def mix_group(cols, ctrl, pati):
    """
    This method will mix a patient into the control group and return mixed group.

    :param cols:list
           name list of ctrl DataFrame columns.
    :param ctrl:pandas.core.frame.DataFrame
           control group data.
    :param pati:numpy.ndarray
           patient data.
    :return: mixed
    """
    pati = pd.DataFrame(np.array([pati]), columns=cols)
    mixed = pd.concat([ctrl, pati])
    return mixed


def Z_score(PCCn, delta_PCC):
    """
    This method is to calculate Z-score.

    :param PCCn:
    :param delta_PCC:
    :return: Z
    """

    assert isinstance(PCCn, np.ndarray), 'PCCn must be a numpy.ndarray.'
    assert isinstance(delta_PCC, np.ndarray), 'delta_PCC must be a numpy.ndarray.'
    assert PCCn.shape == delta_PCC.shape, 'shape of PCCn and delta_PCC must be equal.'

    n = PCCn.shape[0]
    ones = np.ones(PCCn.shape, dtype=np.float_)
    d = ones - PCCn * PCCn
    for i in range(n):
        d[i][i] = 1
    Z = (n - 1) * delta_PCC / d
    for i in range(n):
        Z[i][i] = 1
    return Z


def IDSCN(inpath, outpath, cova=None, region=None):
    ctrl_path = os.path.normpath(os.path.join(inpath, 'controls.csv'))
    pati_path = os.path.normpath(os.path.join(inpath, 'patients.csv'))
    print('Controls are in {}'.format(ctrl_path))
    print('Patients are in {}'.format(pati_path))
    ctrl = read_dataset(filepath=ctrl_path, tp='ctrl', cova=cova, region=region)
    pati = read_dataset(filepath=pati_path, tp='pati', cova=cova, region=region)
    PCCn = PCC(covas=ctrl[0], regions=ctrl[1], group=ctrl[2])
    if os.path.isdir(outpath):
        l = os.listdir(outpath)
        if len(l) != 0:
            print('Please input a empty directory for outpath!')
            exit(1)
    outPath = outpath
    if outpath[-1] in ['/', '\\']:
        outPath = outpath[:-1]
    if not os.path.exists(outPath) or os.path.isfile(outPath):
        os.makedirs(outPath)
    np.savetxt(outPath + '/regions.txt', np.array(ctrl[1]), delimiter=',', fmt='%s')
    np.savetxt(outPath + '/PCCn.csv', PCCn, delimiter=',')
    print('PCCn done.')
    for sub, p in zip(pati[0], pati[3]):
        mixed_group = mix_group(ctrl[0] + ctrl[1], ctrl[2], p)
        PCCn_1 = PCC(ctrl[0], ctrl[1], mixed_group)
        delta_PCC = PCCn_1 - PCCn
        Z = Z_score(PCCn, delta_PCC)
        if not os.path.exists(outPath + '/' + sub):
            os.mkdir(outPath + '/' + sub)
        np.savetxt(outPath + '/' + sub + '/' + sub + '_PCCn+1.csv', PCCn_1, delimiter=',')
        np.savetxt(outPath + '/' + sub + '/' + sub + '_Z.csv', Z, delimiter=',')
        print('Subject: ', sub, ' done.')
    print("All subjects' PCC are generated successfully!")


def read_matrix(path, tp):
    assert tp in ['pcc', 'z', 'sg'], 'tp must be in ["pcc", "z", "sg"]'
    f = open(path, 'r')
    dtype = np.int_
    if tp in ['pcc', 'z']:
        dtype = np.float_
    m = np.array([line.strip().split(',') for line in f.readlines()]).astype(dtype)
    f.close()
    return m


def P(Z):
    p = sps.norm.sf(abs(Z)) * 2
    shape = p.shape
    correct_P = smsm.fdrcorrection(p.flatten())
    return correct_P[1].reshape(shape)


def draw_signifcant(savepath, count, re_col, plot):
    index_dict = {}
    for i in range(count.shape[0]):
        for j in range(i + 1):
            if count[i][j] in index_dict.keys():
                index_dict[count[i][j]][0] += 1
                index_dict[count[i][j]][1].append((i, j))
            else:
                index_dict[count[i][j]] = [1, [(i, j)]]
    index_tuple = sorted(zip(index_dict.keys(), index_dict.values()), reverse=True)
    if plot:
        name_list = []
        y = []
        for c, locs in index_tuple:
            for loc in locs[1]:
                if c != 0:
                    name_list.append(re_col[loc[0]] + '--' + re_col[loc[1]])
                    y.append(c)
        view_len = int(len(name_list) * 0.1)
        if view_len > 200:
            view_len = 200
        name_list = name_list[:view_len]
        y = y[:len(name_list)]
        plt.figure(figsize=(100, 15))
        plt.bar(range(len(name_list)), y, tick_label=name_list)
        for name, num in zip(range(len(name_list)), y):
            plt.text(name, num, '%d' % num, ha='center')
        plt.title('Sorted Effective Connections')
        plt.xticks(fontsize=14, rotation=315, ha='left')
        plt.yticks(fontsize=20)
        plt.xlabel('Connection', fontsize=20)
        plt.ylabel('Number of significant people', fontsize=20)
        plt.tight_layout()
        plt.savefig(savepath)
        plt.show()
    return index_tuple


def getTopLocs(count, num):
    i = 0
    c = 0
    locs = []
    while c < num:
        c += count[i][1][0]
        locs += count[i][1][1]
        i += 1
    return locs


def subtype(input_dir, outpath, plot=True):
    assert isinstance(input_dir, str), 'input dir must be a string.'
    inputdir = input_dir
    if input_dir[-1] in ['/', '\\']:
        inputdir = input_dir[:-1]
    dirlist = os.listdir(inputdir)
    assert 'regions.txt' in dirlist, 'regions.txt not found.'
    f_re = open(inputdir + '/regions.txt', 'r')
    regions = [line.strip() for line in f_re.readlines()]
    pati = []
    for f in dirlist:
        if os.path.isdir(inputdir + '/' + f):
            pati.append(f)
    significant = None
    for p in pati:
        Z = read_matrix(inputdir + '/' + p + '/' + p + '_Z.csv', tp='z')
        if significant is None:
            significant = np.zeros(Z.shape)
        correct_P = P(Z)
        signi_conn_index = np.argwhere(correct_P < 0.05)
        if signi_conn_index.shape[0] > 0:
            rows, cols = zip(*signi_conn_index)
            significant[rows, cols] = significant[rows, cols] + 1
    sorted_edges = draw_signifcant(inputdir + '/' + inputdir.strip().split('/')[-1] + '.jpg', significant, regions,
                                   plot)
    sg_num = []
    for se in sorted_edges:
        sg_num.append(str(int(se[0])) + '/' + str(se[1][0]))
    print('Significant_count/Edge_num:')
    print(sg_num)
    select_num = int(input('Please input number of selected edges: '))
    selected_edges = getTopLocs(sorted_edges, select_num)
    print('Selected {} edges'.format(len(selected_edges)))
    row, col = zip(*selected_edges)
    cluster_source = []
    for p in pati:
        PCCn_1 = read_matrix(inputdir + '/' + p + '/' + p + '_PCCn+1.csv', tp='pcc')
        if PCCn_1 is not None:
            dist = np.ones((len(selected_edges),)) - PCCn_1[row, col]
            cluster_source.append(dist)
    cluster_source = np.array(cluster_source)
    dist_pred = None
    max_sc = -2
    k_last = 1

    for k in range(2, 6):
        dist_pred_k = KMeans(n_clusters=k, n_init=100).fit_predict(cluster_source.copy())
        sc = silhouette_score(cluster_source.copy(), dist_pred_k)
        if sc > max_sc:
            max_sc = sc
            dist_pred = dist_pred_k
            k_last = k
    clusters = [[] for i in range(k_last)]

    for p, c in zip(pati, dist_pred):
        clusters[c].append(p)
    ret = k_last, clusters, len(selected_edges), max_sc
    c_num = []
    df = pd.DataFrame(data={0: ret[1][0]}, columns=[0])
    c_num.append(str(len(ret[1][0])))
    for i in range(ret[0] - 1):
        tp = pd.DataFrame(data=ret[1][i + 1], columns=[i + 1])
        df = pd.concat([df, tp], axis=1, join='outer')
        c_num.append(str(len(ret[1][i + 1])))
    if not os.path.exists(outpath) or os.path.isfile(outpath):
        os.makedirs(outpath)
    outpath = os.path.normpath(os.path.join(outpath, 'cluster_result.csv'))
    df.to_csv(outpath, index=False)
    with open(outpath, mode='a+') as ff:
        print('Number of connections to cluster: ' + str(ret[2]))
        ff.write('Number of connections to cluster,' + str(ret[2]) + '\n')
        print('Number of cluster: ' + str(ret[0]))
        ff.write('Number of cluster,' + str(ret[0]) + '\n')
        print('Clustering result: ' + '/'.join(c_num))
        ff.write('Clustering result,' + '/'.join(c_num) + '\n')
        print('Silhouette score: ' + str(ret[3]))
        ff.write('Silhouette score,' + str(ret[3]) + '\n')
        ff.close()
    print('Cluster result is saved in {}'.format(outpath))

# if __name__ == '__main__':
#     sourcepath = './source/'
#     cova_name = ['Age', 'Sex', 'ICV']
#     group_name = ['HC', 'MDD']
#     f_ROI = open('E:/Data/IDSCN/ROIselection.csv', 'r')
#     roi_lines = f_ROI.readlines()
#     roi_col = roi_lines[0].strip().split(',')
#     roi_data = np.array([line.strip().split(',') for line in roi_lines[1:]])
#     roi = pd.DataFrame(data=roi_data, columns=roi_col)
#     sel_roi_columns = [col for col in roi.columns.values if col == 'new_altas']
#     for sel_roi in sel_roi_columns:
#         print(sel_roi)
#         outpath = './dataset/' + sel_roi + '/'
#         regions = list(roi.loc[:, sel_roi].drop(roi.loc[:, sel_roi][roi.loc[:, sel_roi] == ''].index).values)
#         # generate_dataset(filepath=sourcepath + 'cova_' + sel_roi + '.csv', outpath=outpath,
#         #                  cova_name=cova_name, params=params, region_name=regions, group_index=2)
#         ctrl_path = outpath + 'controls.csv'
#         pati_path = outpath + 'patients.csv'
#         outpath = './output/' + sel_roi + '/'
#         # IDSCN(ctrl_path=ctrl_path, pati_path=pati_path, outpath=outpath, c_r=(3,), cova=cova_name, region=regions)
#         # ret = subtype(outpath)
#         ct = 100
#         save_dir = './cluster/' + sel_roi + '/'
#         sel_num = [5]
#         for sn in sel_num:
#             max_sc = -1
#             last_res = None
#             last_c_num = None
#             save_path = save_dir + 's-' + str(sn) + '_result.csv'
#             for k in range(ct):
#                 if not os.path.exists(save_dir):
#                     os.makedirs(save_dir)
#                 ret = subtype(outpath, select_num=sn)
#                 c_num = []
#                 df = pd.DataFrame(data={0: ret[1][0]}, columns=[0])
#                 c_num.append(str(len(ret[1][0])))
#                 for i in range(ret[0] - 1):
#                     tp = pd.DataFrame(data=ret[1][i + 1], columns=[i + 1])
#                     df = pd.concat([df, tp], axis=1, join='outer')
#                     c_num.append(str(len(ret[1][i + 1])))
#                 if ret[3] > max_sc:
#                     max_sc = ret[3]
#                     last_res = ret
#                     last_c_num = c_num
#                     df.to_csv(save_path, index=False)
#                 print('s-', sn, '_', k, ' done')
#             with open(save_path, mode='a+') as ff:
#                 ff.write('Number of connections to cluster,' + str(last_res[2]) + '\n')
#                 ff.write('Number of cluster,' + str(last_res[0]) + '\n')
#                 ff.write('Clustering result,' + '/'.join(last_c_num) + '\n')
#                 ff.write('Silhouette score,' + str(last_res[3]) + '\n')
#                 ff.close()
