import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

link = [(15, 21),  (16, 20),  (18, 20),  (3, 7),  (14, 16),  (11, 23),  (6, 8),  (15, 17),
 (16, 22), (4, 5), (5, 6), (12, 24), (23, 24), (0, 1), (9, 10), (1, 2), (0, 4), (11, 13),
 (15, 19), (16, 18), (12, 14), (17, 19), (2, 3), (11, 12), (13, 15)]

class Graph():
    def __init__(self, link, node):
        self.link = link
        self.nodes = node
        self.get_adjacency_matrix()
        self.get_degree_matrix()
        self.get_identity_matrix()
        self.get_laplacian_matrix()
        self.get_normalized_variant_matrix()
    def get_adjacency_matrix(self):
        A = np.zeros((self.nodes, self.nodes))
        for (x, y) in self.link:
            A[x, y] = 1
            A[y, x] = 1
        self.A = A
    def get_identity_matrix(self):
        I = np.identity(self.nodes)
        self.I = I
    def get_degree_matrix(self):
        D = np.diag(np.sum(self.A, axis = 0))
        self.D = D
    def get_laplacian_matrix(self):
        L = self.D - self.A
        L_norm = np.dot(np.sqrt(linalg.inv(self.D)), \
                        np.dot(L, linalg.inv(self.D)))
        self.L = L_norm
    def get_normalized_variant_matrix(self):
        A2 = np.dot(np.sqrt(linalg.inv(self.D + self.I)), \
                        np.dot(self.I + self.A, linalg.inv(self.D + self.I)))
        self.A2 = A2

if __name__ == '__main__':
    a = Graph(link = link, node = 25)
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.set_title("Ma tran A")
    ax1.matshow(a.A)
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.set_title("Ma tran D")
    ax2.matshow(a.D)
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.set_title("Ma tran I")
    ax3.matshow(a.I)
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.set_title("Ma tran L")
    ax4.matshow(a.L)
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.set_title("Ma tran A2")
    ax5.matshow(a.A2)
    plt.subplots_adjust(wspace=0.8,
                        hspace=0.8)
    plt.show()



