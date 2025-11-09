import numpy as np 
from Lcss import LCSS 
from DTW import dtw
from MLCSS import MLCSS

class validator():

    def _init_(self, type_validation, threshold):
        self.type_validation = type_validation
        self.threshold = threshold


    def event_v(self, data_real, data_sim, time_real, time_sim, delta, method):
                information_type ="Events"
                if method == 'lcss':
                    ind = lcss(data_real[:],data_sim[:], time_real[:], time_sim[:], delta)/min(len(data_real[:]),len(data_sim[:])) 
                    # the indicator of the lcss method with delta threshold 
                    if ind>=self.threshold:
                          value = 1

                    else:
                          value = 0
                    data = [value, ind, information_type, method]
                return data

    def kpi_v(self, data_real, data_sim, method, epsilon):
                information_type = "KPI"
                if method == "dtw":
                    ind = 1-dtw(data_real[:]/max(max(data_real[:]), max(data_sim[:])), data_sim[:]/max(max(dat_real[:]), max(data_sim[:])))
                    if ind>=self.threshold:
                        value = 1
                    else:
                          value = 0
                    data = [value,ind,information_type, method]
                if method == 'MLCSS':
                    ind = MLCSS(dat_real[:], data_sim[:], epsilon)/max(len(dat_real[:]),len(data_sim[:]))
                    if ind>=self.threshold:
                        value = 1
                    else:
                        value = 0
                    data = [value,ind, information_type,method]
                return data
    def input_trace(self, processing_times_real, dist, param):
          from ecdf import ecdf 
          import scipy.stats
          from scipy.stats import norm
          from scipy.stats import beta
          from scipy.stats import gamma
          from scipy.stats import lognorm
          from scipy.stats import pareto
          from scipy.stats import logistic
          from scipy.stats import rayleigh
          from scipy.stats import uniform
          from scipy.stats import triang



          n_parts = np.size(processing_times_real)
          u_p = np.array([])
          x_p, f_p = ecdf(processing_times_real)
          for ii in range(n_parts):
            u_p = np.append(u_p, f_p[np.asarray(np.where(x_p ==processing_times_real[ii]))])
                          
          """ Distributions"""
        # 1. uniform distribution 
          if dist =="uniform":
            y_p = uniform.ppf(u_p, float(param[0]), float(param[1]))
        # 2. triangular distribution 
          if dist == 'triang':
            y_p = triang.ppf(u_p, float(param[0]), float(param[1]), float(param[2]))
        # 3. normal distribution 
          if dist == 'norm':
            y_p = norm.ppf(u_p, float(param[0]), float(param[1]))
        #4. beta distribution 
          if dist =='beta':
            y_p = beta.ppf(u_p, float(param[0]), float(param[1]), float(param[2]), float(param[3]))
        # 5. gamma distribution 
          if dist == 'gamma':
            y_p = gamma.ppf(u_p, float(param[0]), float(param[1]), float(param[2]))
        # 6. lognormal distribution 
          if dist == 'lognorm':
            y_p = lognorm.ppf(u_p, float(param[0]), float(param[1]), float(param[2]))
        # 7. pareto distribution 
          if dist == 'pareto':
            y_p = pareto.ppf(u_p, float(param[0]), float(param[1]), float(param[2]))
        #8 logistic distribution 
          if dist == 'logistic':
            y_p = logistic.ppf(u_p, float(param[0]), float(param[1]))
        # 9 rayleigh distribution 
          if dist == 'rayleigh':
            y_p = rayleigh.ppf(u_p, float(param[0]), float(param[1]))
        

          if max(y_p) == np.inf:
           pos = np.where(y_p==np.inf)
           pos = np.asarray(pos)
           y_p[pos]= processing_times_real[pos]
          correlated_processing_time = t_p
          return correlated_processing_time

    # adapted from https://github.com/giovannilugaresi/digital_twin_validator/blob/main/class_validator.py
