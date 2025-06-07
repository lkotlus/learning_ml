import numpy as np
import numpy.typing as npt

class KM:
    def __init__(self, k: int) -> None:
        """
        Take a Python course.

        Args:
            k: integer storing the value of k for the algorithm.
        """

        self.k = k
        self.means = np.empty((0, 0))
        self.centroids = np.empty(k)
        self.clusters = np.empty(k)
        self.dataset = np.empty((0, 0))

    
    def get_closest(self, cluster: npt.NDArray[np.floating], cv: npt.NDArray[np.floating]) -> int:
        """
        Gets the new centroid of a cluster.

        Args:
            cluster: 2-dimensional numpy array contianing all vectors in the cluster.
            cv: comparison vector

        Returns:
            The index of the vector in the cluster closest to the comparison vector
        """

        # np.linalg.norm(v1 - v2) returns the distance between v1 and v2
        centroid_dist = (0, np.linalg.norm(cluster[0] - cv)) 

        for v in range(len(cluster)):
            dist = np.linalg.norm(cluster[v] - cv)  
            if (dist < centroid_dist[1]):
                centroid_dist = (v, dist)

        # Only return the vector, the distance isn't required
        return centroid_dist[0]


    def get_clusters(self) -> None:
        """
        Generates clusters based on centroids and calculates their means.
        """

        sums = np.empty((self.k, 0))

        self.clusters = np.array([np.empty((0, 0)) for _ in range(self.k)])

        for v in self.dataset:
            index = self.get_closest(self.centroids, v)

            self.clusters[index] = np.append(self.clusters[index], v)
            sums[index] += v

        for i in range(len(sums)):
            self.means[i] = sums[i]/len(self.clusters[i])


    def train(self, dataset: npt.NDArray[np.floating]) -> None:
        """
        Goes through the entire training process

        Args:
            dataset: a 2-dimensional numpy array storing all vectors that the model will be trained on.
        """
