import numpy as np
from collections import defaultdict
from typing import List
import math

# Object is a wrapper class used in k-means operations
class Object:
    def __init__(self, value):
        """
        Initialize an object with an encapsulated value
        
        - This value must contain a `compare_to` method which returns a float and compares another variable of type `value` with itself
        - This value must contain a `average_with` method which returns an average variable of type `value` of the two provided values
        """
        self.value = value

        # The Object's distance to its nearest center, used for heap removal
        self.nearest_distance = None

        # The index of the nearest center point compared to this object
        self.nearest_center_index = None

        # Center points that this object cannot belong to (as they are already full)
        self.removed_centers = []

    def compare_to(self, obj) -> float:
        """
        Compare this object to another object by referencing the encapsulated value

        :param obj: The object to compare this object to
        """
        return self.value.compare_to(obj.value)

    def average_with(self, objects):
        """
        Find the average of this object with another object by delegating to the encapsulated value
        """
        average_value = self.value.average_with([obj.value for obj in objects])
        return Object(average_value)

# Heap provides a basic heap implementation as a priority queue
class Heap:
    def __init__(self):
        """
        Initialize a new Heap with an empty embedded array
        """
        self.array = []

    def append(self, obj: Object) -> None:
        """
        Put an item into the heap based on its `nearest_distance` attribute

        :param obj: The item to add to the heap
        """
        index = 0

        for i in range(len(self.array)):
            if self.array[i].nearest_distance > obj.nearest_distance:
                index = i
                break
            index = index + 1

        self.array.insert(index, obj)

    def pop(self) -> Object:
        """
        Remove the next item from the heap

        :return: Next item from the heap with the smallest `nearest_distance` attribute
        """
        return self.array.pop(0)

    def isEmpty(self) -> bool:
        """
        Returns status on if the heap is empty

        :return: True if heap is empty, false if not
        """
        return len(self.array) == 0

        

# KMeansVariation class handles performing k-means clustering with clusters of constant sizes and
# customly defined "distance" functions
class KMeansVariation:
    def __init__(self, k: int, max_iter: int = 100, clusters:List[List[Object]]|None = None):
        """
        Initialize the KMeansVariation

        :param k: The number of clusters to use when grouping objects 
        :param max_iter: The maximum number of iterations to use when optimizing the cluster
        """
        self.k = k
        self.max_iter = max_iter

        self.centers = []

        if clusters == None:
            self.clusters = [[] for _ in range(k)]
        else:
            self.cluster_size = len(clusters)
            self.clusters = [[Object(item)] for item in clusters]
            while len(self.clusters) < k:
                self.clusters.append([])

    def fit(self, data: List[any]) -> List[any]:
        """
        Fit provided data into our kmeans clustering

        :param data: Data to fit
        """

        # Convert the data into objects so we can attach attributes to them
        objects = [Object(item) for item in data]

        self.cluster_size = (self.cluster_size + len(objects)) / self.k

        # For our set number of iterations, assign clusters and optimize
        self._initialize_centers(objects)
        for _ in range(self.max_iter):
            self._assign_clusters(objects)
            self._update_centers()
            if not self._optimize_clusters(objects):
                break

        # Return data
        results = []
        for cluster in self.clusters:
            results.append([obj.value for obj in cluster])

        return results

    def _initialize_centers(self, objects: List[Object]) -> None:
        """
        Initialize the centers of the kmeans search by using k-means++ or by providing objects directly

        :param data: data to analyze
        :param centers: centers to provide already
        :param method: what method to use to calculate centers
        """
        # Add the first center randomly from our list of data
        self.centers = [objects[np.random.randint(len(objects))]]

        # Keep adding clusters until we have k clusters. Clusters are added probabalistically such that 
        # clusters further from other centers are more likely to be clusters
        for _ in range(len(self.centers), self.k):
            distances = [min((1 - obj.compare_to(c))/2 for c in self.centers) for obj in objects]
            probs = [d / sum(distances) for d in distances]
            self.centers.append(np.random.choice(objects, p=probs))

    def _assign_clusters(self, objects: List[Object]) -> None:
        """
        Assign objects into k clusters depending on their distance nearest_distances

        :param objects: objects to assign into cluster
        """
        heap = Heap()

        # For each object, find the center it is closest to and by how much, then add it to the heap
        for obj in objects:
            distances = [obj.compare_to(c) for c in self.centers]
            nearest_center_index = np.argmin(distances)

            obj.nearest_distance = distances[nearest_center_index]
            obj.nearest_center_index = nearest_center_index

            heap.append(obj)

        cluster_sizes = [len(cluster) for cluster in self.clusters]
        temp_clusters = [self.clusters[i] for i in range(self.k)]

        # While the heap is not empty
        while not heap.isEmpty():
            # Remove the next object and find it's nearest index
            obj = heap.pop()
            nearest_index = obj.nearest_center_index

            if cluster_sizes[nearest_index] < self.cluster_size:
                # If the cluster this object wants to be in has room, add it
                temp_clusters[nearest_index].append(obj)
                cluster_sizes[nearest_index] += 1
            else:
                # The cluster this object wants to be in has no room, so find the second nearest index and never consider the first index again
                obj.removed_centers.append(nearest_index)
                second_nearest_index = np.argmin([math.inf if i in obj.removed_centers else obj.compare_to(self.centers[i]) for i in range(len(self.centers))])

                # The second nearest cluster has no room, so update the object's inner score and index and add it back to the heap
                obj.nearest_distance = distances[second_nearest_index]
                obj.nearest_center_index = second_nearest_index

                if cluster_sizes[second_nearest_index] < self.cluster_size:
                    # If the second nearest cluster has room, add the object
                    temp_clusters[second_nearest_index].append(obj)
                    cluster_sizes[second_nearest_index] += 1
                else:
                    heap.append(obj)

        # Update the k-means clusters
        self.clusters = temp_clusters

    def _update_centers(self) -> None:
        """
        Update the centers of the clusters now that the clusters are populated by finding averages in the clusters
        """

        # For each cluster that has more than one object, find the average
        for i, cluster in enumerate(self.clusters):
            if len(cluster) > 0:
                self.centers[i] = cluster[0].average_with(cluster[1:])

    def _optimize_clusters(self, objects: List[Object]) -> bool:
        """
        Optimize clusters by swapping objects that yields overall improvement to the system
        """
        proposals = defaultdict(list)
        moved = False

        # For every object in each cluster
        for i, cluster in enumerate(self.clusters):
            for obj in cluster:
                # Find the distances from this object to every center
                distances = [obj.compare_to(c) for c in self.centers]
                current_distance = distances[i]

                # Find the closest index of all centers
                closest_other_index = np.argmin(distances)
                if closest_other_index != i:
                    # If the closest of all centers is not the cluster this object currently belongs in
                    if len(self.clusters[closest_other_index]) < len(objects) // self.k:
                        # Move clusters if the other cluster is not full
                        self.clusters[closest_other_index].append(obj)
                        self.clusters[i].remove(obj)
                        moved = True
                    else:
                        # Append it to a proposal if other cluster is full
                        proposals[closest_other_index].append((obj, distances[closest_other_index] - current_distance, closest_other_index, i))

        # If there were no movements (i.e. every object is perfect or all clusters are full)
        if moved:
            # Loop through each cluster's proposed switches (if there are any)
            for i, cluster in enumerate(self.clusters):
                if proposals[i]:
                    # Find the best swap
                    best_swap = max(proposals[i], key=lambda x: x[1])
                    obj, improvement, other_index, current_index = best_swap

                    # Find the best other object to swap with
                    min_loss = math.inf
                    min_other = None
                    for other in self.clusters[other_index]:
                        loss = self.centers[current_index].compare_to(other)
                        if loss < min_loss:
                            min_loss = loss
                            min_other = other

                    # If we get more improvement than loss, swap the objects
                    if improvement > loss:
                        self.clusters[other_index].remove(min_other)
                        self.clusters[other_index].append(obj)

                        self.clusters[current_index].remove(obj)
                        self.clusters[current_index].append(min_other)

                        moved = True

        return moved

# Example usage
if __name__ == "__main__":
    # Dummy Vec class that implements required methods. A vector is used as encapsulated data for easy and
    # straight-forward implementation
    class Vec:
        def __init__(self, id, vector):
            self.id = id
            self.vector = vector

        def compare_to(self, v) -> float:
            return np.linalg.norm(np.array(self.vector) - np.array(v.vector))

        def average_with(self, vectors):
            vector = np.mean(np.concatenate(([self.vector], [v.vector for v in vectors])), axis=0)
            return Vec(-1, vector)

    # Create sample data and fit by kmeans clustering
    data = [Vec(i, np.random.rand(3)) for i in range(100)]
    kmeans = KMeansVariation(k=50)
    kmeans.initialize_centers(data, method="kmeans++")
    clusters = kmeans.fit(data)

    # Print out clustering results
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i} contains {len(cluster)} objects.")
        for vector in cluster:
            print(vector.id, end=" ")
        print()