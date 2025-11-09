import pandas as pd

def read_txt_system_time_digital(n_pallet, directory=None, file_name=None):
    
    if directory is None:
        print('Directory not declared for system time digital')
    
    if file_name is None:
        print('File name not declared for system time digital')
    
    data = pd.read_csv(str(directory) + str(file_name), names=['System Time Digital', 'timelog'], header=None, sep=" ", index_col=False)
    
    data = data[n_pallet+1:]
    data['timelog'] = data['timelog'] - min(data['timelog'])
    data['timelog'] = data['timelog'] + data.iloc[0][0]
    
    return data.dropna()
