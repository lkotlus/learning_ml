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
        self.dimensions = 0
        self.centroids = np.empty(k)

    
    def _get_closest(self, cv: npt.NDArray[np.floating], cluster: npt.NDArray[np.floating] | None = None) -> int:
        """
        Classifies a vector.

        Args:
            cluster: If called within the object, 2-dimensional numpy array contianing all vectors in the cluster.
                If called by the user, it's set to self.centroids for _get_closestionof the data.
            cv: Classification vector, the vector that is being classified.

        Returns:
            The index of the vector in the cluster closest to the comparison vector. This is the same as the label.
        """

        # This is what happens if the user uses the _get_closest method
        if (cluster is None):
            cluster = self.centroids

        # np.linalg.norm(v1 - v2) returns the distance between v1 and v2
        centroid_dist = (0, np.linalg.norm(cluster[0] - cv)) 

        for v in range(len(cluster)):
            dist = np.linalg.norm(cluster[v] - cv)  
            if (dist < centroid_dist[1]):
                centroid_dist = (v, dist)

        # Only return the index, the distance isn't required
        return centroid_dist[0]


    def _get_clusters(self, dataset: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """
        Generates clusters based on centroids and calculates their means.

        Args:
            dataset: an numpy array containing the training data

        Returns:
            The clusters that were found along with the means that were calculated.
        """

        sums = np.empty((self.k, self.dimensions))
        means = np.empty((self.k, self.dimensions))

        clusters = [[] for _ in range(self.k)]

        for v in dataset:
            index = self._get_closest(v, self.centroids)

            clusters[index].append(v)
            sums[index] += v

        for i in range(len(sums)):
            means[i] = sums[i]/len(clusters[i])

        return (np.array(clusters), means)


    def _calc_centroids(self, clusters: npt.NDArray[np.floating], means: npt.NDArray[np.floating]) -> None:
        """
        Calculates centroids for each cluster from means.

        Args:
            clusters: 2-dimensional numpy array of clusters

        Returns:
            Nothing, centroids are stored in the object.
        """

        for i in range(len(means)):
            self.centroids[i] = clusters[i][self._get_closest(means[i], clusters[i])]


    def train(self, dataset: npt.NDArray[np.floating]) -> None:
        """
        Goes through the entire training process

        Args:
            dataset: a 2-dimensional numpy array storing all vectors that the model will be trained on.
        """

        self.dimensions = dataset.shape[1]
        self.centroids = np.array([dataset[i] for i in range(self.k)])

        clusters, means = self._get_clusters(dataset)
        print(clusters)
        self._calc_centroids(clusters, means)
        print(self.centroids)


    def predict(self, featureset: npt.NDArray[np.floating]) -> npt.NDArray:
        """
        Classifies all vectors in the featureset.
        UNFINISHED

        Args:
            featureset: set of unclassified data

        Returns:
            An array of labels.
        """

        labels = np.empty(len(featureset))

        return labels


if (__name__ == "__main__"):
    model = KM(2)

    model.train(np.array([np.array([1, 1]), np.array([-1, -1]), np.array([-2, -2]), np.array([-3, -3]), np.array([2, 2]), np.array([3, 3])]))
