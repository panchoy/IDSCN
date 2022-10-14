import os
import pandas as pd
import numpy as np


def to_dataframe(filepath):
    f = open(filepath, 'r')
    f_lines = f.readlines()
    f.close()
    f_col = f_lines[0].strip().split(',')[1:]
    f_lines = np.array([line.strip().split(',') for line in f_lines[1:]]).T
    f_index = list(f_lines[0])
    f_data = f_lines[1:].T
    return pd.DataFrame(data=f_data, columns=f_col, index=f_index)


if __name__ == '__main__':
    source_dir = 'E:/Data/IDSCN/'
    files = os.listdir(source_dir)
    merge_dict = {}
    for file in files:
        if 'aparc' in file:
            name_arr = file.strip().split('_')
            name = name_arr[0] + '_' + name_arr[2]
            if name in merge_dict.keys():
                merge_dict[name].append(file)
            else:
                merge_dict[name] = [file]
        elif 'MDD' in file:
            merge_dict['COV'] = file
        elif 'ROI' in file:
            merge_dict['ROI'] = file
        else:
            merge_dict['VOL'] = file
    df_vol = to_dataframe(source_dir + merge_dict['VOL'])
    df_cov = to_dataframe(source_dir + merge_dict['COV'])
    # for key in merge_dict.keys():
    #     if key not in ['COV', 'ROI', 'VOL']:
    #         df_f1 = to_dataframe(source_dir + merge_dict[key][0])
    #         df_f2 = to_dataframe(source_dir + merge_dict[key][1])
    #         temp = pd.concat([df_f1, df_f2], axis=1, join='inner')
    #         merged = pd.concat([df_vol, temp], axis=1, join='inner')
    #         merged.to_csv('./source/' + key, index=True)
    #         merged_cova = pd.concat([df_cov, merged], axis=1, join='inner')
    #         merged_cova.to_csv('./source/cova_' + key, index=True)
    df_a1 = to_dataframe(source_dir + merge_dict['aparc_area.csv'][0])
    df_a2 = to_dataframe(source_dir + merge_dict['aparc_area.csv'][1])
    temp1 = pd.concat([df_a1, df_a2], axis=1, join='inner')
    df_t1 = to_dataframe(source_dir + merge_dict['aparc_thickness.csv'][0])
    df_t2 = to_dataframe(source_dir + merge_dict['aparc_thickness.csv'][1])
    temp2 = pd.concat([df_t1, df_t2], axis=1, join='inner')
    temp = pd.concat([temp1, temp2], axis=1, join='inner')
    merged = pd.concat([df_vol, temp], axis=1, join='inner')
    merged.to_csv('./source/new_altas.csv', index=True)
    merged_cova = pd.concat([df_cov, merged], axis=1, join='inner')
    merged_cova.to_csv('./source/cova_new_altas.csv', index=True)
