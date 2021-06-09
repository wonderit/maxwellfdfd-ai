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
        'path': 'torch/al_g0_s7_rs999_nobn_ce_adam_resnet_random0.5_wd0.0001_b128_e100_lr0.001_it10_K1000/txt'
    },
    {
        'color': 'b',
        'name': 'max_ce',
        'path': 'torch/al_g0_s7_nobn_ce_adam_resnet_max_ce0.5_wd0.0001_b128_e100_lr0.001_it10_K1000/txt'
    },

    {
        'color': 'r',
        'name': 'ua_max_ce_0.1',
        'path': 'torch/al_g0_s7_rs999_nobn_ualambda_residual_sb1.0_ual0.1_ce_adam_resnet_max_ce0.5_wd0.0001_b128_e100_lr0.001_it10_K1000/txt'
    },
    {
        'color': 'orange',
        'name': 'ua_max_ce_0.5',
        'path': 'torch/al_g0_s7_rs999_nobn_ualambda_residual_sb1.0_ual0.5_ce_adam_resnet_max_ce0.5_wd0.0001_b128_e100_lr0.001_it10_K1000/txt'
    },

    {
        'color': 'pink',
        'name': 'ua_max_ce_0.9',
        'path': 'torch/al_g0_s7_rs999_nobn_ualambda_residual_sb1.0_ual0.9_ce_adam_resnet_max_ce0.5_wd0.0001_b128_e100_lr0.001_it10_K1000/txt'
    },
    # {
    #     'color': 'm',
    #     'name': 'ua_b10_max',
    #     'path': 'torch/rpo_compare_k400/al_g0_s0_bn_uamultiply_sigmoid_beta10.0_l1_adamw_max_diff0.5_wd0.04_b32_e100_lr0.001_it5_K400/txt'
    # },
    # {
    #     'color': 'y',
    #     'name': 'ua_b100_max',
    #     'path': 'torch/rpo_compare_k400/al_g1_s0_bn_uamultiply_sigmoid_beta100.0_l1_adamw_max_diff0.5_wd0.04_b32_e100_lr0.001_it5_K400/txt'
    # },
    # {
    #     'color': 'b',
    #     'name': 'ua_b10_max_wds',
    #     'path': 'torch/rpo_compare_k400/al_g0_s1_bn_wds_uamultiply_sigmoid_beta10.0_l1_adamw_max_diff0.5_wd0.04_b32_e100_lr0.001_it5_K400/txt'
    # },
    # {
    #     'color': 'yellow',
    #     'name': 'tbr',
    #     'path': 'torch/rpo_compare_k400/al_g0_s1_rm_bn_tbr_upper_bound_lambda0.5_l1_adamw_max_diff0.5_wd0.04_b32_e100_lr0.001_it5_K400/txt'
    # },
    # {
    #     'color': 'cyan',
    #     'name': 'tor',
    #     'path': 'torch/rpo_compare_k400/al_g1_s1_rm_bn_tor_z2.0_lambda0.5_l1_adamw_max_diff0.5_wd0.04_b32_e100_lr0.001_it5_K400/txt'
    # },
    # {
    #     'color': 'brown',
    #     'name': 'ua_add_b10',
    #     'path': 'torch/rpo_compare_k400/al_g1_s0_bn_uaadd_sigmoid_beta10.0_l1_adamw_random0.5_wd0.04_b32_e100_lr0.001_it5_K400/txt'
    # },
]

result = pd.DataFrame()

# gca stands for 'get current axis'
ax = plt.gca()

k = 1000
result['x'] = list(range(k, 10 * k + 1, k))

cmap = plt.cm.get_cmap("hsv", len(MODEL_ARRAY)+1)
for i, model_info in enumerate(MODEL_ARRAY):
    model_results = search(model_info['path'])
    df_results = parse_result(model_results)
    result[model_info['name']] = df_results['acc']

    result.plot(kind='line', use_index=True, x='x', y=model_info['name'], color=model_info['color'], ax=ax)

plt.xlabel('Number of labeled images')
plt.ylabel('Accuracy')

plt.xticks(result['x'])

plt.savefig('result-cifar100.png', dpi=300)
