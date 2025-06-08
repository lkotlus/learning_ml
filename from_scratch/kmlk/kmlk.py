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

    
    def _get_closest(self, cv: npt.NDArray[np.floating], cluster: npt.NDArray[np.floating]) -> int:
        """
        Finds vector in a cluster closest to the comparison vector

        Args:
            cluster: 2-dimensional numpy array contianing all vectors in the cluster
            cv: Comparison vector

        Returns:
            The index of the vector in the cluster closest to the comparison vector. This is the same as the label.
        """
        
        # Small failsafe required
        if (len(cluster) == 0):
            cluster = np.array(cv)

        # np.linalg.norm(v1 - v2) returns the distance between v1 and v2
        vect_dist = (0, np.linalg.norm(cluster[0] - cv)) 

        for v in range(len(cluster)):
            dist = np.linalg.norm(cluster[v] - cv)  
            if (dist < vect_dist[1]):
                vect_dist = (v, dist)

        # Only return the index, the distance isn't required
        return vect_dist[0]
    

    def _get_farthest_avg(self, c_cluster: npt.NDArray[np.floating], cluster: npt.NDArray[np.floating]) -> int:
        """
        Finds the vector with the greatest mean distance from vectors in c_cluster.

        Args:
            cluster: 2-dimensional numpy array contianing all vectors in the cluster
            c_cluster: Comparison cluster

        Returns:
            The index of the vector in the cluster farthest from the comparison cluster average.
        """

        clen = len(c_cluster)

        # np.linalg.norm(v1 - v2) returns the distance between v1 and v2
        dist = 0
        for cv in c_cluster:
            dist += np.linalg.norm(cluster[0] - cv)

        vect_dist = (0, dist/clen) 

        for v in range(len(cluster)):
            dist = 0
            for cv in c_cluster:
                dist += np.linalg.norm(cluster[v] - cv)
            dist /= clen

            if (dist > vect_dist[1]):
                vect_dist = (v, dist)

        # Only return the index, the distance isn't required
        return vect_dist[0]
    

    def _initial_guess(self, dataset: npt.NDArray[np.floating]) -> None:
        """
        Sets the initial centroid guesses
        
        Args:
            dataset: the dataset that the model will be trained on
        """
        
        centroids = []
        avg_vects = np.array(np.zeros(self.dimensions))
        
        for i in range(self.k):
            centroids.append(dataset[self._get_farthest_avg(avg_vects, dataset)])
            
            avg_vects = np.array(centroids)

        self.centroids = np.array(centroids)


    def _get_clusters(self, dataset: npt.NDArray[np.floating]) -> tuple[list, npt.NDArray[np.floating]]:
        """
        Generates clusters based on centroids and calculates their means.

        Args:
            dataset: an numpy array containing the training data

        Returns:
            The clusters that were found along with the means that were calculated.
        """

        sums = np.zeros((self.k, self.dimensions))
        means = np.empty((self.k, self.dimensions))

        clusters = [[] for _ in range(self.k)]

        for v in dataset:
            index = self._get_closest(v, self.centroids)

            clusters[index].append(v)
            sums[index] += v

        for i in range(len(sums)):
            means[i] = sums[i]/len(clusters[i])

        return (clusters, means)


    def _calc_centroids(self, clusters: list, means: npt.NDArray[np.floating]) -> None:
        """
        Calculates centroids for each cluster from means.

        Args:
            clusters: 2-dimensional numpy array of clusters

        Returns:
            Nothing, centroids are stored in the object.
        """

        for i in range(len(means)):
            self.centroids[i] = clusters[i][self._get_closest(means[i], clusters[i])]

    def _compare_means(self, old: npt.NDArray[np.floating], new: npt.NDArray[np.floating]) -> bool:
        """
        Determines if the means have stabilized.

        Args:
            old: old means array
            new: new means array

        Returns:
            True if there have been no changes, False if there have been changes.
        """

        for i in range(len(old)):
            for j in range(len(old[i])):
                if (old[i][j] != new[i][j]):
                    return False

        return True


    def train(self, dataset: npt.NDArray[np.floating]) -> None:
        """
        Goes through the entire training process. By the end, all that is required is to keep the given centroids.

        Args:
            dataset: a 2-dimensional numpy array storing all vectors that the model will be trained on.
        """

        self.dimensions = dataset.shape[1]
        self._initial_guess(dataset)

        clusters, means = self._get_clusters(dataset)
        old_means = np.empty(means.shape)

        while(not self._compare_means(old_means, means)):
            self._calc_centroids(clusters, means)

            old_means = means
            clusters, means = self._get_clusters(dataset)


    def predict(self, featureset: npt.NDArray[np.floating]) -> npt.NDArray:
        """
        Classifies all vectors in the featureset.

        Args:
            featureset: set of unclassified data

        Returns:
            An array of labels.
        """

        labels = np.empty(len(featureset))

        for X in range(len(featureset)):
            labels[X] = self._get_closest(featureset[X], self.centroids)

        return labels
