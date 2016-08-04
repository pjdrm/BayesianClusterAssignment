'''
Created on 03/08/2016

@author: Mota
'''
from scipy import stats
import numpy as np
import pandas as pd
import eval_metrics
import matplotlib.pyplot as plt


class ClusterAssignmentsState(object):

    def __init__(self, data_path, labels_path, n_clusters, cluster_var, alpha_prior, theta_prior, var_prior):
        self.cluster_var = cluster_var
        self.alpha_prior = alpha_prior
        self.theta_prior = theta_prior
        self.var_prior = var_prior
        
        self.pi = np.random.dirichlet(alpha_prior, size=None)
        self.theta = np.random.normal(self.theta_prior, self.var_prior, n_clusters)
        
        
        self.n_clusters = n_clusters
        self.cluster_ids = range(n_clusters)
        self.data = pd.Series.from_csv(data_path)
        self.true_labels = pd.Series.from_csv(labels_path)
        self.data_size = self.data.shape[0]
        self.z = np.random.randint(0, high = n_clusters, size = self.data_size)
        self.data_sum = np.zeros(self.n_clusters)
        self.cluster_counts = np.zeros(self.n_clusters)
        
        for y_val, cluster_id in zip(self.data, self.z):
            self.data_sum[cluster_id] += y_val
            self.cluster_counts[cluster_id] += 1.0
    
    def log_assignment_score(self, data_id, cluster_id):
        log_pi_k = np.log(self.pi[cluster_id])
        xi = self.data[data_id]
        theta_k = self.theta[cluster_id]
        return log_pi_k + stats.norm.logpdf(xi, theta_k, self.cluster_var) 
        
    def sample_z_i(self, i):
        scores = [self.log_assignment_score(i, cluster_id) for cluster_id in self.cluster_ids]
        scores = np.exp(np.array(scores))
        p = scores / scores.sum()
        return np.random.choice(self.cluster_ids, p=p)
    
    def sample_z(self):
        for data_i in range(self.data_size):
            y_val = self.data[data_i]
            current_z = self.z[data_i]
            self.data_sum[current_z] -= y_val
            self.cluster_counts[current_z] -= 1.0
            
            new_z = self.sample_z_i(data_i)
            self.z[data_i] = new_z
            self.data_sum[new_z] += y_val
            self.cluster_counts[new_z] += 1.0
            
            for k in range(self.n_clusters):
                self.theta[k] = self.data_sum[k] / self.cluster_counts[k]
                
    def sample_pi(self):
        self.pi = np.random.dirichlet([self.cluster_counts[cid] + self.alpha_prior[cid]/self.n_clusters  \
                                        for cid in self.cluster_ids], size=None)
        
    def sample_theta_k(self, k):
        numerator = self.theta_prior / self.var_prior + self.theta[k] * self.cluster_counts[k] / self.cluster_var
        denominator = 1.0 / self.var_prior + self.cluster_counts[k] / self.cluster_var
        posterior_theta_k = numerator / denominator
        posterior_var_k = 1.0 / denominator
        return stats.norm(posterior_theta_k, np.sqrt(posterior_var_k)).rvs()
    
    def sample_theta(self):
        self.theta = [self.sample_theta_k(cid) for cid in self.cluster_ids]
        
    def gibs_sampler(self, n_iter):
        while n_iter > 0:
            print "Gibbs Sampler iter %d" % n_iter
            n_iter -= 1
            self.sample_z()
            self.sample_pi()
            self.sample_theta()
        
        print "F1: %f" % eval_metrics.f_measure(self.true_labels, self.z)
        
    '''
    Colapsed Gibs sampling related code
    '''
    def p_xi_score(self, data_id, cluster_id):
        posterior_var_k = 1.0 / (self.cluster_counts[cluster_id] / self.cluster_var + 1.0 / self.var_prior)
        posterior_theta_k = posterior_var_k * (self.theta_prior / self.var_prior + \
                             self.cluster_counts[cluster_id] * (self.data_sum[cluster_id] / self.cluster_counts[cluster_id]) / self.cluster_var)
        predictive_var = np.sqrt(posterior_var_k + self.cluster_var)
        return stats.norm(posterior_theta_k, predictive_var).logpdf(self.data[data_id])
    
    def log_cluster_assign_score(self, cluster_id):
        return np.log(self.cluster_counts[cluster_id] + self.alpha_prior[cluster_id] / self.n_clusters)
    
    def cluster_assignment_distribution(self, data_id):
        scores = np.zeros(self.n_clusters)
        for cid in self.cluster_ids:
            cid_score = self.p_xi_score(data_id, cid)
            cid_score += self.log_cluster_assign_score(cid)
            scores[cid] = np.exp(cid_score)
        normalization = 1.0 / np.sum(scores)
        scores = scores * normalization
        return scores
    
    def collapsed_gibs_sampler(self, n_iter):
        while n_iter > 0:
            print "Collapsed Gibbs Sampler iter %d" % n_iter
            n_iter -= 1
            for data_i in range(self.data_size):
                y_val = self.data[data_i]
                current_z = self.z[data_i]
                self.data_sum[current_z] -= y_val
                self.cluster_counts[current_z] -= 1.0
                
                scores = self.cluster_assignment_distribution(data_i)
                new_z = np.random.choice(self.cluster_ids, p=scores)
                
                self.z[data_i] = new_z
                self.data_sum[new_z] += y_val
                self.cluster_counts[new_z] += 1.0
        print "F1: %f" % eval_metrics.f_measure(self.true_labels, self.z)             
        
def plot_clusters(cluster_assigment):
    gby = pd.DataFrame({
            'data': cluster_assigment.data, 
            'assignment': cluster_assigment.z}
        ).groupby(by='assignment')['data']
    hist_data = [gby.get_group(cid).tolist() 
                 for cid in gby.groups.keys()]
    plt.hist(hist_data, 
             bins=20,
             histtype='stepfilled', alpha=.5 )
    plt.show()
            

ca = ClusterAssignmentsState("../clusters.csv", "../clusters_labels.csv", 3, 0.01, [10, 10, 10], 0.0, 1.0)
ca.collapsed_gibs_sampler(10)
plot_clusters(ca)