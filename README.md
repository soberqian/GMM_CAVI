# GMM_CAVI
Coordinate ascent variational inference for a Gaussian mixture model

最近，有不少人询问我博客中的有关高斯混合模型的推理以及编程方面的内容，特写此内容。
博客地址为：https://qianyang-hfut.blog.csdn.net/article/details/86694325


# 代码对应的论文
Blei D M, Kucukelbir A, McAuliffe J D. Variational inference: A review for statisticians[J]. 
Journal of the American Statistical Association, 2017, 112(518): 859-877.

由概率图模型大佬Blei D M在顶刊《Journal of the American Statistical Association》发表的论文。

# 高斯混合模型生成过程
如下为，高斯混合模型的生成过程：
![高斯混合模型](https://img-blog.csdnimg.cn/20190608082614293.png#pic_center)

# 变分推理
可以参考我的博客：https://qianyang-hfut.blog.csdn.net/article/details/86694325
另外，详细的公式推理，也上上传到该文件夹下面了。

# 算法流程
![算法流程](https://img-blog.csdnimg.cn/20190608090203984.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9xaWFueWFuZy1oZnV0LmJsb2cuY3Nkbi5uZXQ=,size_16,color_FFFFFF,t_70#pic_center)

# 代码
使用的是Python 3，代码如gmmvi.py所示。
```python
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

```
# 运行结果
如下为迭代结果：
![迭代](https://img-blog.csdnimg.cn/20190608084603355.png#pic_center)

真实分布和拟合分布为：
![真实分布和拟合分布](https://img-blog.csdnimg.cn/2019060808464930.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9xaWFueWFuZy1oZnV0LmJsb2cuY3Nkbi5uZXQ=,size_16,color_FFFFFF,t_70#pic_center)
