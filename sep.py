import pandas as pd

filepath = 'cluster/new_altas/s-5_result_HXY20221011.csv'
cluster_path = 'cluster/new_altas/s-5_result.csv'
cluster = [[], []]
savepath = 'cluster/new_altas/edge-5_'
with open(cluster_path, 'r') as fc:
    lines = fc.readlines()
    for i in range(1, 126):
        line_arr = lines[i].strip().split(',')
        for j in range(2):
            if line_arr[j] != '':
                cluster[j].append(line_arr[j])

savefiles = [[], []]
df = pd.read_csv(filepath, encoding='ANSI', encoding_errors='ignore')
for _, line in df.iterrows():
    index = line[0]
    if '(' not in index:
        index_ = 'sub-' + index[:-3] + '0' + index[-3:]
        for j in range(len(cluster)):
            if index_ in cluster[j]:
                savefiles[j].append(_)
    else:
        index_ = index.strip().split('(')
        index_[-1] = index_[-1][:-1]
        for j in range(len(cluster)):
            for k in index_:
                iindex = 'sub-' + k[:-3] + '0' + k[-3:]
                if iindex in cluster[j]:
                    savefiles[j].append(_)
for i in range(len(cluster)):
    save = df.loc[savefiles[i]]
    save.to_csv(savepath + str(i) + '.csv', index=False, encoding='ANSI', errors='ignore')
