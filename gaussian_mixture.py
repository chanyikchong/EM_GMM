import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from celluloid import Camera
from datetime import datetime


class GaussianMixture:
    def __init__(self, n_components, reg_cov=1e-6, max_iter=100, stop_dist=1e-2):
        self.n_components = n_components
        self.reg_cov = reg_cov
        self.max_iter = max_iter
        self.stop_dist = stop_dist
        self.features = None
        self.mu = None
        self.sig = None
        self.pi = np.random.uniform(0, 1, self.n_components)
        self.pi /= np.sum(self.pi)

    def fit(self, data, verbose=False):
        start_time = datetime.now()
        n, self.features = data.shape
        self.mu = np.random.randn(self.n_components, self.features)
        sig = np.eye(self.features)
        sig = [np.array([sig])] * self.n_components
        self.sig = np.concatenate(sig)

        old_param = np.concatenate([self.mu.flatten(), self.sig.flatten()])

        fig, axes = plt.subplots(1, 1)
        camera = Camera(fig)
        can_verbose = False

        for i in range(self.max_iter):
            r = self.e_step(data)
            self.m_step(data, r)
            new_param = np.concatenate([self.mu.flatten(), self.sig.flatten()])
            if np.linalg.norm(new_param - old_param) < self.stop_dist:
                break
            old_param = new_param

            if verbose:
                if self.features == 1:
                    can_verbose = True
                    self.one_dim_plot(data, axes, camera, i)
                elif self.features == 2:
                    can_verbose = True
                    self.two_dim_plot(data, axes, fig, camera, i)

        print('Finish fitting in: %s' % (datetime.now() - start_time).total_seconds())
        if can_verbose:
            animation = camera.animate()
            plt.show()

    def one_dim_plot(self, data, axes, camera, step):
        lin = np.array(
            list(np.arange(data.min() - 5, data.max() + 5, np.divide(data.max() - data.min(), 100))))
        axes.hist(data, bins=20, density=True, stacked=True, color='blue')
        t = axes.plot(lin, self.pdf(lin), color='orange')
        axes.legend(t, [f'step {step}'])
        camera.snap()

    def two_dim_plot(self, data, axes, fig, camera, step):
        x = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
        y = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
        xs, ys = np.meshgrid(x, y)
        z = np.dstack((xs, ys))
        z = z.reshape(-1, 2)
        zs = self.pdf(z)
        # for i in range(self.n_components):
        #     zs = stats.multivariate_normal(self.mu[i], self.sig[i] + self.reg_cov).pdf(z)
        zs = zs.reshape(100, 100)
        axes.contour(xs, ys, zs)
        t = axes.scatter(data[:, 0], data[:, 1], c='blue', label=f'step {step}')
        # fig.colorbar()  # draw colorbar
        axes.set_xlim(np.min(x) - 0.3, np.max(x) + 0.3)
        axes.set_ylim(np.min(y) - 0.3, np.max(y) + 0.3)
        # axes.legend(t, [f'step {step}'])
        camera.snap()

    def e_step(self, data):
        n = data.shape[0]
        r = np.zeros((n, self.n_components))
        for i in range(self.n_components):
            sig = self.sig[i] + self.reg_cov * np.eye(self.features)
            pro = stats.multivariate_normal.pdf(data, self.mu[i], sig)
            r[:, i] = self.pi[i] * pro
        r = r / np.sum(r, axis=1, keepdims=True)
        return r

    def m_step(self, data, r):
        sum_rk = np.sum(r, axis=0)
        for i in range(self.n_components):
            self.mu[i] = 1 / sum_rk[i] * np.sum(r[:, i:i + 1] * data, axis=0)  # update mu
            diff = data - self.mu[i, :]
            diff = diff.reshape(-1, self.features, 1)
            sig_square = np.matmul(diff, diff.transpose(0, 2, 1))
            w = 1 / sum_rk[i] * r[:, i]
            s = w.reshape(-1, 1, 1) * sig_square
            s = np.sum(s, axis=0)
            self.sig[i] = 1 / 2 * (s + s.transpose())  # make sure the matrix is symmetric
        self.pi = sum_rk / np.sum(r)

    def log_pdf(self, x):
        return np.log(self.pdf(x))

    def pdf(self, x):
        pro = np.zeros(x.shape[0])
        for i in range(self.n_components):
            sig = self.sig[i] + self.reg_cov * np.eye(self.features)
            pro += self.pi[i] * stats.multivariate_normal.pdf(x, self.mu[i], sig)
        return pro


if __name__ == '__main__':
    n_component = 4

    x1 = np.random.normal(3, 1, (100, 2))
    x2 = np.random.normal(3, 1, (100, 2))
    x3 = np.random.normal(-3, 1, (100, 2))
    x4 = np.random.normal(-3, 1, (100, 2))
    x = np.concatenate([x1, x2], axis=-1)
    y = np.concatenate([x3, x4], axis=-1)
    x = np.concatenate([x, y], axis=0)
    model = GaussianMixture(n_component)
    model.fit(x, verbose=True)
    print('finish')
