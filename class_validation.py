import numpy as np

def levenshteinDistance(s1, s2):
    if len(s1) < len(s2):
        return levenshteinDistance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

class Validator:
    def __init__(self, type_validation, threshold):
        self.type_validation = type_validation
        self.threshold = threshold

    def event_v(self, data_real, data_sim):
        ind = 1 - levenshteinDistance(data_real[:], data_sim[:]) / max(len(data_real[:]), len(data_sim[:]))
        if ind >= self.threshold:
            value = 1
        else:
            value = 0
        data = [value, ind]
        return data

    def input_trace(self, processing_times_real, dist, param):
        from ecdf import ecdf
        import scipy.stats
        from scipy.stats import norm, beta, gamma, lognorm, pareto, logistic, rayleigh, uniform, triang

        n_parts = np.size(processing_times_real)
        u_p = np.array([])
        X_p, F_p = ecdf(processing_times_real)
        for ii in range(n_parts):
            u_p = np.append(u_p, F_p[np.asarray(np.where(X_p == processing_times_real[ii]))])

        if dist == 1:
            y_p = uniform.ppf(u_p, param[0], param[1])
        elif dist == 2:
            y_p = triang.ppf(u_p, (param[2] - param[0]) / param[1] - param[0], param[0], param[1] - param[0])
        elif dist == 3:
            y_p = norm.ppf(u_p, param[0], param[1])
        elif dist == 4:
            y_p = beta.ppf(u_p, param[0], param[1]) * (max(processing_times_real) - min(processing_times_real))
        elif dist == 5:
            y_p = gamma.ppf(u_p, param[0], param[1], param[2])
        elif dist == 6:
            y_p = lognorm.ppf(u_p, param[0], param[1], param[2])
        elif dist == 7:
            y_p = pareto.ppf(u_p, param[0], param[1], param[2])
        elif dist == 8:
            y_p = logistic.ppf(u_p, param[0], param[1])
        elif dist == 9:
            y_p = rayleigh.ppf(u_p, param[0], param[1])

        if max(y_p) == np.inf:
            pos = np.where(y_p == np.inf)
            pos = np.asarray(pos)
            y_p[pos] = processing_times_real[pos]

        correlated_processing_time = y_p
        return correlated_processing_time
