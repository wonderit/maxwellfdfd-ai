#%%

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def search(dirname):
    filenames = os.listdir(dirname)
    return filenames

def parse_result(models):

    data_array = []
    for model_name in models:
        data = dict()
        str_arr = model_name.split('_')
        iteration = int(str_arr[1].replace('it', ''))
        model_num = int(str_arr[2].replace('m',''))
        acc = float(str_arr[3].replace('acc',''))

        # if model_num > 2:
        #     continue
        data['it'] = iteration
        data['m'] = model_num
        data['acc'] = acc
        data_array.append(data)

    df = pd.DataFrame(data_array)
    df = df.sort_values(by=['it'])
    result = df.groupby(['it']).mean()
    result = result.round(4)
    return result
#%% Max diff start
# models_max_diff = search('torch/al_bn_l1_adamw_max_diff0.5_wd0.02_b32_e100_lr0.001_it10_K200/model')
# df_max_diff = parse_result(models_max_diff)
#
# selected_columns = df_max_diff[['r2']]
# result_df = selected_columns.copy()
# result_df.rename(columns={'r2':'max_diff'}, inplace=True)
#%% random
MODEL_ARRAY = [
    {
        'color': 'k',
        'name': 'random',
        'path': 'torch/al_g0_s0_nobn_ce_sgd_random0.5_wd0.0_b32_e100_lr0.001_it5_K200/txt'
    },
    {
        'color': 'b',
        'name': 'max_ce',
        'path': 'torch/al_g0_s0_nobn_ce_sgd_max_ce0.5_wd0.0_b32_e100_lr0.001_it5_K200/txt'
    },

    {
        'color': 'r',
        'name': 'ua_max_ce_sb1',
        'path': 'torch/al_g0_s0_rm_nobn_uaresidual_sb1.0_ce_sgd_max_ce0.5_wd0.0_b32_e100_lr0.001_it5_K200/txt'
    },
    {
        'color': 'orange',
        'name': 'ua_max_ce_sb10',
        'path': 'torch/al_g0_s0_rm_nobn_uaresidual_sb10.0_ce_sgd_max_ce0.5_wd0.0_b32_e100_lr0.001_it5_K200/txt'
    },
{
        'color': 'yellow',
        'name': 'ua_max_ce_sb100',
        'path': 'torch/al_g0_s0_rm_nobn_uaresidual_sb100.0_ce_sgd_max_ce0.5_wd0.0_b32_e100_lr0.001_it5_K200/txt'
    },
{
        'color': 'brown',
        'name': 'ua_max_ce_sb200',
        'path': 'torch/al_g0_s0_rm_nobn_uaresidual_sb200.0_ce_sgd_max_ce0.5_wd0.0_b32_e100_lr0.001_it5_K200/txt'
    },
]

result = pd.DataFrame()

# gca stands for 'get current axis'
ax = plt.gca()

k = 200
result['x'] = list(range(k, 5 * k + 1, k))

cmap = plt.cm.get_cmap("hsv", len(MODEL_ARRAY)+1)
for i, model_info in enumerate(MODEL_ARRAY):
    model_results = search(model_info['path'])
    df_results = parse_result(model_results)
    result[model_info['name']] = df_results['acc']

    result.plot(kind='line', use_index=True, x='x', y=model_info['name'], color=model_info['color'], ax=ax)

plt.xlabel('Number of labeled images')
plt.ylabel('Accuracy')

plt.xticks(result['x'])

plt.savefig('result-mnist-ua-sb-k200.png', dpi=300)
