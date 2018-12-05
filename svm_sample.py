import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def make_data(n):
    a = np.array([4*i/(n-1) for i in range(n)])
    x_p_1 = a*np.cos(a) + np.random.rand(n)
    x_p_2 = a*np.sin(a) + np.random.rand(n)
    x_p = np.vstack([x_p_1, x_p_2])
    y_p = np.ones_like(x_p_1)
    x_m_1 = (a+np.pi)*np.cos(a) + np.random.rand(n)
    x_m_2 = (a+np.pi)*np.sin(a) + np.random.rand(n)
    x_m = np.vstack([x_m_1, x_m_2])
    y_m = -np.ones_like(x_m_1)
    X = np.vstack([x_p.T, x_m.T])
    Y = np.hstack([y_p.T, y_m.T])
    perm = np.random.permutation(n*2)
    return X[perm], Y[perm]
    
def f_theta(X, theta, h=1):
    K = np.exp(-np.sum(((X[None,:,:] - X[:,None,:]))**2, axis=2)/(2*h*h))
    return np.sum(theta * K, axis=1)

def f_theta_test(test_X, X, theta, h=1):
    K = np.exp(-np.sum(((X[None,:,:] - test_X[:,None,:]))**2, axis=2)/(2*h*h))
    return np.sum(theta * K, axis=1)

def f(X, Y, theta, C=0.1, h=1):
    K = np.exp(-np.sum(((X[None,:,:] - X[:,None,:]))**2, axis=2)/(2*h*h))
    return C * np.sum(np.fmax(1-f_theta(X, theta)*Y, 0)) + theta@K@theta
    
def nabla_f(X, Y, theta, C=0.1, h=1):
    K = np.exp(-np.sum(((X[None,:,:] - X[:,None,:]))**2, axis=2)/(2*h*h))
    t = 1-f_theta(X, theta)*Y
    return C*np.sum(- Y * (t > 0)*K, axis=1) + 2*K@theta

def run():
    N = 50
    alpha = 0.5
    beta = 0.9
    eps = 0.5
    
    X, Y = make_data(N)
    theta = np.random.rand(N*2)
    
    for _ in range(80):
        while f(X, Y, theta - eps*nabla_f(X, Y, theta)) - f(X, Y, theta) > \
                    - alpha * eps * np.sum(nabla_f(X,Y,theta)**2):
            eps = eps * beta
        theta = theta - eps*nabla_f(X, Y, theta)
    plot_result(X, Y, theta)
    return

def plot_result(X, Y, theta):
    plt.scatter(X[:,0][Y>0], X[:,1][Y>0], c='red')
    plt.scatter(X[:,0][Y<0], X[:,1][Y<0], c='blue')
    plt.grid(True)

    pred_Y = f_theta(X, theta)
    x1_mesh = np.arange(-7, 7, 0.1)
    mesh_X = np.array(np.meshgrid(x1_mesh, x1_mesh)).transpose(1,2,0)
    pred_Y = f_theta_test(mesh_X.reshape(-1,2), X, theta).reshape((len(x1_mesh), len(x1_mesh)))
    pred_Y = np.where(pred_Y > 0, -1, 1)
    plt.contourf(x1_mesh, x1_mesh, pred_Y, alpha=0.4, \
                 cmap=ListedColormap(('red', 'blue')))
    plt.xlim(-7, 5)
    plt.savefig('result_svm.png')
    
if __name__ == '__main__':
    run()
