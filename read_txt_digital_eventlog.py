import pandas as pd

def read_txt_digital_eventlog(directory=None, file_name=None):
    
    if directory is None:
        print('Directory not declared for digital_eventlog')
    
    if file_name is None:
        print('File name not declared for digital_eventlog')
    
    data = pd.read_csv(str(directory) + str(file_name), names=['timelog', 'activity', 'type'], header=None, sep=" ", index_col=False)
    
    return data.dropna()
