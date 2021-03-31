import pandas as pd
import os
import matplotlib.pyplot as plt
def search(dirname):
    filenames = os.listdir(dirname)
    return filenames

def parse_result(models):

    data_array = []
    for model_name in models:
        data = dict()
        str_arr = model_name.split('-')
        it_m = str_arr[0].split('_')
        iteration = int(it_m[1].replace('it', ''))
        model_num = int(it_m[2].replace('m',''))
        rmse = float(str_arr[1])
        r2 = float(str_arr[2])

        # if model_num > 2:
        #     continue
        data['it'] = iteration
        data['m'] = model_num
        data['rmse'] = rmse
        data['r2'] = r2
        data_array.append(data)

    df = pd.DataFrame(data_array)
    df = df.sort_values(by=['it'])
    result = df.groupby(['it']).mean()
    result = result.round(4)
    return result
#%% Max diff start
models_max_diff = search('torch/al_ua_l1_max_random_diff0.1_n3_b32_e100_lr0.001_it10_K200/model')
df_max_diff = parse_result(models_max_diff)

selected_columns = df_max_diff[['r2']]
result_df = selected_columns.copy()
result_df.rename(columns={'r2':'max_random_diff0.1'}, inplace=True)

#%% random
models_random = search('torch/al_ua_l1_random0.5_n3_b32_e100_lr0.001_it10_K200/model')
df_random = parse_result(models_random)
result_df['random'] = df_random['r2']

#%% max_random_diff 0.5
models_random = search('torch/al_ua_l1_max_random_diff0.5_n3_b32_e100_lr0.001_it10_K200/model')
df_random = parse_result(models_random)
result_df['max_random_diff'] = df_random['r2']
#%% max_random_diff 0.5
models_random = search('torch/al_ua_l1_max_random_diff0.9_n3_b32_e100_lr0.001_it10_K200/model')
df_random = parse_result(models_random)
result_df['max_random_diff0.9'] = df_random['r2']


#%%


# gca stands for 'get current axis'
ax = plt.gca()

result_df.plot(kind='line',use_index=True,y='max_random_diff0.9', color='blue', ax=ax)
result_df.plot(kind='line',use_index=True,y='random', color='black', ax=ax)
result_df.plot(kind='line',use_index=True,y='max_random_diff', color='red', ax=ax)

result_df.plot(kind='line',use_index=True,y='max_random_diff0.1', color='orange', ax=ax)

plt.xlabel('Iteration')
plt.ylabel('R-squared')

plt.savefig('fig_result-max-random.png', dpi=300)

plt.show()