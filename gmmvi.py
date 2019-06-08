"""
@time:2019/1/17 19:07
@email:qy20115549@126.com
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# data set
N = 300
clusters = 3
means = np.array([2, 5, 9])

def generate_data(N, clusters, means):
    data = []
    for i in range(clusters):
        cluster_data = np.random.normal(means[i], 1, N)
        data.append(cluster_data)
    return np.concatenate(np.array(data))

data = generate_data(N, clusters, means)

# plot the data
# fix, ax = plt.subplots(figsize=(12,3))
sns.distplot(data[:N], color='green', rug = True)
sns.distplot(data[N:2*N], color='orange', rug = True)
sns.distplot(data[2*N:], color='red', rug = True)
plt.show()

class Model:
    def __init__(self, data, num_clusters = 3, sigma=1):
        self.data = data
        self.K = num_clusters
        self.n = data.shape[0]
        self.sigma = sigma
        # model parameters(m,s2,phi)-- these are the things CAVI will update to max ELBO
        self.varphi = np.random.dirichlet(np.random.random(self.K), self.n)
        self.m = np.random.randint(low=np.min(self.data), high=np.max(self.data), size=self.K).astype(float)
        self.s2 = np.random.random(self.K)

    def elbo(self): # check derivation for details on this
        p = -np.sum((self.m**2 + self.s2) / (2 * self.sigma**2))
        next_term = -0.5 * np.add.outer(self.data**2, self.m**2 + self.s2)
        next_term -= np.outer(self.data, self.m)
        next_term *= self.varphi
        p += np.sum(next_term)
        q = np.sum(np.log(self.varphi)) - 0.5 * np.sum(np.log(self.s2))
        elbo = p + q
        return elbo
    # parameters updating
    def cavi(self):
        # cavi varphi update
        e1 = np.outer(self.data, self.m)
        #e2 = -0.5 * (self.m**2 )
        e2 = -0.5 * (self.m**2 + self.s2)
        e = e1 + e2[np.newaxis, :]
        self.varphi = np.exp(e) / np.sum(np.exp(e), axis=1)[:, np.newaxis]
        # cavi m update
        self.m = np.sum(self.data[:, np.newaxis] * self.varphi, axis=0)
        self.m /= (1.0 / self.sigma**2 + np.sum(self.varphi, axis=0))
        # cavi s2 update
        self.s2 = 1.0 / (1.0 / self.sigma**2 + np.sum(self.varphi, axis=0))

    def train(self, epsilon=1e-5, iters=100):
        elbo_record = []
        elbo_record.append(self.elbo())

        # use cavi to update elbo until epsilon-convergence
        for i in range(iters):
            self.cavi()
            elbo_record.append(self.elbo())

            # break if past elbos don't differ too much
            if i % 5 == 0:
                print("elbo is: ", elbo_record[i])
            if np.abs(elbo_record[-1] - elbo_record[-2]) <= epsilon:
                print("converged after %d steps!" % i)
                break
        return elbo_record




model = Model(data, clusters)
elbo_record = model.train()

# plot final parameters
assignments = model.varphi.argmax(1)
converged_means = model.m
print("final means are ", sorted(converged_means))
print("model means are ", sorted(means))

fix, ax = plt.subplots(figsize=(12,3))
sns.distplot(data[:N], color='green', rug=True)
sns.distplot(data[N:2*N], color='orange', rug=True)
sns.distplot(data[2*N:], color='red', rug=True)
# plot modelled gaussians
sns.distplot(np.random.normal(converged_means[0], 1, 1000), color = 'black', hist=False)
sns.distplot(np.random.normal(converged_means[1], 1, 1000), color = 'black', hist=False)
sns.distplot(np.random.normal(converged_means[2], 1, 1000), color = 'black', hist=False)
plt.show()
