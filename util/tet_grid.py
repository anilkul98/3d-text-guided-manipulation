import numpy as np
import torch


class Grid:
    """
    A 3D regular grid data structure.
    This data structure only defines functionality for handling indices, coordinates and retrieving nodal values.
    The internal representation of the grid is a flat array. The convention is used that i is the running index.
    Hence, for cache friendly access end users should access nodal values
    in lexicographic increasing order of (i,j,k). No internal testing is done for proper access all methods should
    be considered unsafe in the sense that there are no bounds testing on input values and no testing if the value
    array has been created.
    """

    def __init__(self, min_coord, max_coord, I, J, K, device):
        """
        This constructor creates an empty grid instance. It will initialize all internal member values necessary
        for supporting index and coordinate methods. However, the actual value array is defined to None.

        :param min_coord: The spatial 3D coordinates of the lower-left-back node. The one with smallest lexicographic
                          index value.
        :param max_coord: The spatial 3D coordinates of the upper-right-front node. The one with largest lexicographic
                          index value.
        :param I: The total number of nodes along the x-axis.
        :param J: The total number of nodes along the y-axis.
        :param K: The total number of nodes along the z-axis.
        """
        if np.any(min_coord > max_coord):
            raise ValueError()
        if I < 2:
            raise ValueError()
        if J < 2:
            raise ValueError()
        if K < 2:
            raise ValueError()
        self.min_coord = np.array(min_coord).ravel()
        self.max_coord = np.array(max_coord).ravel()
        self.I = I  # Number of nodes along x axis
        self.J = J  # Number of nodes along y axis
        self.K = K  # Number of nodes along z axis
        res = np.array([I - 1, J - 1, K - 1], dtype=np.float64)
        dims = self.max_coord - self.min_coord
        self.spacing = np.divide(dims, res)
        self.values = None
        self.device = device
        self.v = torch.from_numpy(self.get_all_coords()).to(device).unsqueeze(0)
        self.T = torch.from_numpy(self.get_tetrahedron_array()).cuda().long()
        
    def get_node_coord(self, i, j, k):
        """
        This method computes the 3D spatial coordinates of the grid node with 3D indices (i,j,k). The method
        does not do any bounds testing. Hence if (i,j,k) are outside the valid range then one will get the
        coordinates of a "virtual" node that does not exist in the grid.
        :param i:  The node index along the x-axis.
        :param j:  The node index along the y-axis.
        :param k:  The node index along the z-axis.
        :return: The 3D spatial coordinates of the node with index (i,j,k).
        """
        # z = self.min_coord[2] + self.spacing[2] * k
        # y = self.min_coord[1] + self.spacing[1] * j
        # x = self.min_coord[0] + self.spacing[0] * i
        # return V3.make(x, y, z)
        return self.min_coord + np.multiply(
            self.spacing, np.array([i, j, k], dtype=np.float64)
        )
    def get_all_coords(self):
        v = []
        for i in range(self.I):
            for j in range(self.J):
                for k in range(self.K):
                    v.append(self.get_node_coord(i,j,k))
                    
        return np.asarray(v)
    
    def get_tetrahedron_array(self):
        """
        Generates a tetrahedron array  that fills a cubic cell block structure with I,J,K nodes. The fill pattern is
        generated to minimize bias by flipping the fill pattern.
        :param I: Number of nodes along the x-axis
        :param J: Number of nodes along the y-axis
        :param K: Number of nodes along the z-axis
        :return: A 5(I-1)(J-1)(K-1)-by-4 array. Each row corresponds to one tetrahedon and columns correspond to nodal
                 indices of that tetrahedron.
        """
        I, J, K = self.I, self.J, self.K
        N = (I - 1) * (J - 1) * (K - 1) * 5
        T = np.zeros((N, 4), dtype=np.int32)
        n = 0
        for k in range(K - 1):
            for j in range(J - 1):
                for i in range(I - 1):
                    i000 = (k * J + j) * I + i
                    i001 = (k * J + j) * I + (i + 1)
                    i010 = (k * J + (j + 1)) * I + i
                    i011 = (k * J + (j + 1)) * I + (i + 1)
                    i100 = ((k + 1) * J + j) * I + i
                    i101 = ((k + 1) * J + j) * I + (i + 1)
                    i110 = ((k + 1) * J + (j + 1)) * I + i
                    i111 = ((k + 1) * J + (j + 1)) * I + (i + 1)
                    #
                    #       i110 *----------------------* i111
                    #           /|                     /|
                    #          / |                    / |
                    #         /  |                   /  |
                    #        /   |                  /   |
                    #       /    |                 /    |
                    # i100 *----------------------* i101|
                    #      |i010 *----------------|-----*  i011
                    #      |    /                 |     /
                    #      |   /                  |    /
                    #      |  /                   |   /
                    #      | /                    |  /
                    #      |/                     | /
                    #      *----------------------*
                    #    i000                   i001
                    #
                    flip = (i + j + k) % 2 == 1
                    if flip:
                        T[n, :] = (i000, i001, i010, i100)
                        T[n + 1, :] = (i010, i001, i011, i111)
                        T[n + 2, :] = (i100, i110, i111, i010)
                        T[n + 3, :] = (i100, i111, i101, i001)
                        T[n + 4, :] = (i010, i111, i100, i001)
                    else:
                        T[n, :] = (i000, i001, i011, i101)
                        T[n + 1, :] = (i000, i011, i010, i110)
                        T[n + 2, :] = (i100, i110, i101, i000)
                        T[n + 3, :] = (i101, i110, i111, i011)
                        T[n + 4, :] = (i000, i011, i110, i101)
                    n += 5
        return T