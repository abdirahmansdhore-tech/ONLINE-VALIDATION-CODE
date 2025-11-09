def write_txt_init_pos(data, directory=None, file_name=None):
    
    data = data
    
    if file_name is None:
        print('Routing.txt is not directed to a file_name')
    
    with open(str(directory) + str(file_name), mode="w") as Routing:
         Routing.write(data['location'].to_string(header=False, index=False))       
         Routing.close()
