import os
import numpy as np
from write_txt_init_pos import write_txt_init_pos
from init_position import init_position
from write_txt_processing_time import write_txt_processing_time, write_txt_processing_time_validator
from processing_time import processing_time_Arena, processing_time_Manpy
from read_txt_system_time_digital import read_txt_system_time_digital
from read_txt_digital_eventlog import read_txt_digital_eventlog
from write_txt_init_pos import write_txt_init_pos
from read_txt_digital_final_position import read_txt_digital_final_position

# Assuming DigitalModel is available in your path
from DigitalModel import Digital

def simulator(
        db=None,
        simulator_type=None,
        t_horizon=None,
        p_timesimul_input=None,
        init_pos=None,
        use_type=None,
        n_pallet=None,
        source_type=None,
        synchroniser_id=None
):
    if not db:
        print("ERROR: interface_database not declared")

    if use_type == "logic_validation":
        if simulator_type == "Manpy":
            exec_model_temp = db.queryData("executable_model", "model")
            exec_model = DigitalModel(exec_model_temp, 1)
            results = exec_model.runTraceSimulation(p_timesimul_input, init_pos['location'].tolist())

            digital_eventlog = results['eventlog']
            digital_events = []
            digital_timelog = digital_eventlog[0]
            for i in range(len(digital_timelog)):
                digital_events.append(digital_eventlog[3][i] + digital_eventlog[1][i])

            digital_data = np.column_stack((digital_events, digital_timelog))
            system_time_digital = results['elementList'][0]['results']['system_time_trace'][0]
            final_position = results['final_position']

            return system_time_digital, digital_eventlog, final_position, digital_data
            print('simulation type Manpy Run for logic_validation')

        if simulator_type == "Arena":
            path = "C:/Users/Abdirahman/OneDrive - Politecnico di Milano/thesis/2023_dt_demo/supervisor_class/simulator_class/arena/shadow/"
            write_txt_init_pos(init_pos, path, "Routing.txt")
            write_txt_processing_time(p_timesimul_input, path, "Processing_time_S")
            os.system(r'siman -p C:\Users\Abdirahman\OneDrive - Politecnico di Milano\thesis\2023_dt_demo\supervisor_class\simulator_class\arena\shadow\Arena_model_legofactory.p')

            system_time_digital = read_txt_system_time_digital(n_pallet, path, 'Validation_System_Time.txt')
            digital_eventlog = read_txt_digital_eventlog(path, 'Validation_digital_eventlog.txt')
            end_pos = read_txt_digital_final_position(n_pallet, path, 'Validation_final_position.txt')

            final_position = end_pos

            eventlog_NP = digital_eventlog.to_numpy()
            s = eventlog_NP[:, 2] + eventlog_NP[:, 1].astype(str)
            string_events = s.astype(str)
            time_events_ = eventlog_NP[:, 0]
            time_events = time_events_.astype(float)
            digital_data = np.stack((string_events, time_events), axis=1)

            return system_time_digital, digital_eventlog, final_position, digital_data
            print('simulation type Arena Run for logic_validation')

    if use_type == "input_validation":
        if simulator_type == "Manpy":
            exec_model_temp = db.queryData("executable_model", "model")
            exec_model = DigitalModel(exec_model_temp, 1)
            results = exec_model.runTraceSimulation(p_timesimul_input, init_pos['location'].tolist())

            digital_eventlog = results['eventlog']
            digital_events = []
            digital_timelog = digital_eventlog[0]
            for i in range(len(digital_timelog)):
                digital_events.append(digital_eventlog[3][i] + digital_eventlog[1][i])

            digital_data = np.column_stack((digital_events, digital_timelog))
            system_time_digital = results['elementList'][0]['results']['system_time_trace'][0]
            final_position = results['final_position']

            return system_time_digital, digital_eventlog, final_position, digital_data
            print('simulation type Manpy Run for Input_validation')

        if simulator_type == "Arena":
            path = "C:/Users/Abdirahman/OneDrive - Politecnico di Milano/thesis/2023_dt_demo/supervisor_class/simulator_class/arena/shadow/"
            write_txt_init_pos(init_pos, path, "Routing.txt")
            write_txt_processing_time(p_timesimul_input, path, "Processing_time_S")
            os.system(r'siman -p C:\Users\Abdirahman\OneDrive - Politecnico di Milano\thesis\2023_dt_demo\supervisor_class\simulator_class\arena\shadow\Arena_model_legofactory.p')

            system_time_digital = []
            digital_eventlog = read_txt_digital_eventlog(path, 'Validation_digital_eventlog.txt')
            end_pos = read_txt_digital_final_position(n_pallet, path, 'Validation_final_position.txt')

            final_position = end_pos

            eventlog_NP = digital_eventlog.to_numpy()
            s = eventlog_NP[:, 2] + str(eventlog_NP[:, 1])
            string_events = s.astype(str)
            time_events_ = eventlog_NP[:, 0]
            time_events = time_events_.astype(float)
            digital_data = np.stack((string_events, time_events), axis=1)

            return system_time_digital, digital_eventlog, final_position, digital_data
            print('simulation type Arena Run for Input_validation')

    if use_type == "shadow":
        if simulator_type == "Arena":
            path = "C:/Users/Abdirahman/OneDrive - Politecnico di Milano/thesis/2023_dt_demo/supervisor_class/simulator_class/arena/shadow/"
            eventlog_sample = db.queryData(None, 'eventlog_Arena', t_horizon)
            p_timesimul_input = processing_time_Arena(eventlog_sample)
            init_pos = init_position(source_type, n_pallet, eventlog_sample)
            write_txt_init_pos(init_pos, path, "Routing.txt")
            write_txt_processing_time(p_timesimul_input, path, "Processing_time_S")
            os.system(r'siman -p C:\Users\Abdirahman\OneDrive - Politecnico di Milano\thesis\2023_dt_demo\supervisor_class\simulator_class\arena\shadow\Arena_model_legofactory.p')

            system_time_digital = read_txt_system_time_digital(n_pallet, path, 'Validation_System_Time.txt')
            digital_eventlog = read_txt_digital_eventlog(path, 'Validation_digital_eventlog.txt')
            end_pos = read_txt_digital_final_position(n_pallet, path, 'Validation_final_position.txt')

            db.writeData("initial_position_Arena", "initialization", init_pos)
            db.writeData("final_position_Arena", "initialization", end_pos)
            db.writeData(None, "digital_eventlog_Arena", digital_eventlog)
            db.writeData("system_time_digital_Arena", "digital_perf", system_time_digital)

            print('simulation type Arena Run & KPIs updated')

        if simulator_type == "Manpy":
            eventlog_sample = db.queryData(None, 'eventlog_Arena', t_horizon)
            p_timesimul_input = processing_time_Manpy(eventlog_sample)
            exec_model_temp = db.queryData("executable_model", "model")

            exec_model = DigitalModel(exec_model_temp, 1)
            init_pos = init_position(source_type, n_pallet, eventlog_sample)

            results = exec_model.runTraceSimulation(p_timesimul_input, init_pos['location'].tolist())

            db.writeData("initial_position_Arena", "initialization", init_pos)
            db.writeData("util_sync", "digital_perf_mean", results)
            db.writeData("system_time_digital_Manpy", "digital_perf", results, synchroniser_id)
            db.writeData("interdeparture_time_digital_Manpy", "digital_perf", results, synchroniser_id)
            db.writeData("final_position_manpy", 'initialization', results)
            db.writeData(None, "digital_eventlog_Manpy", results)

            print('simulation type Manpy Run & KPIs updated')
