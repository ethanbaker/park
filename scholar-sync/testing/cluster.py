from student import Student
import math
import numpy as np

def cluster_pairs(students: list[Student], max_iters=100):
    """
    Cluster a group of students together using k-means
    """

    # Compute the desired cluster size, n/k
    n = len(students)
    k = 2

    num_clusters = n/k

    # Initialize centroids using k-means++
    centroids = []*num_clusters
    centroids[0] = students[math.randint(0, len(students))]

    for i in range(1, num_clusters):
        # Compute the score from each student to the nearest centroid
        distances = [[]*len(centroids)]
        for student in students:
            distances[i].append(centroids[i].compare_to(student))

        # Find the sum of total distance from each centroid
        total_distances = [0]*len(distances[0])
        for count in range(len(total_distances)):
            for centroid in distances:
                total_distances[count] += centroid[count]

        # Convert distances to probabilities
        probabilities = total_distances ** 2
        probabilities /= sum(probabilities)

        next_index = np.random.choice(n, p=probabilities)
        centroids[i] = students[next_index]

    # Sort clusters with an optimal, greedy approach
    clusters = [[c] for c in centroid]
    sort(students, centroids, clusters)

    # Perform optimizing iterations
    for _ in range(max_iters):
        # Compute current cluster means
        cluster_means = []
        for cluster in clusters:
            a = cluster[0].to_vectors()
            b = cluster[1].to_vectors()

            combined = a
            for key in combined:
                combined[key] += b
                combined[key] /= 2

            cluster_means.append(combined)

        # Compute distances to cluster means
        distances = [[] * len(cluster_means)]
        for i in range(len(distances)):
            for j in range(len(students)):
                distances[i][j] = students[j].compare_to_vector(cluster_means[0])

        # Find the best possible cluster mean for each student
        best_assignments = []
        for student in students:
            best_mean = cluster_means[0]
            best_score = math.inf

            for i in range(len(cluster_means)):
                score = student.compare_to_vector(cluster_means[i])
                if score < best_score:
                    best_score = score
                    best_mean = i

            best_assignments.append({
                "index": best_mean,
                "mean": cluster_means[best_mean],
                "score": best_score,
                "student": student,
            })

        # Sort elements based on their current assignment and the best possible assignment
        current_assignments = []
        for i in range(len(cluster_means)):
            cluster = clusters[i]

            for student in cluster:
                # Find the students current and best score
                current_score = cluster[0].compare_to_vector(cluster_means[i])
                best_score = 0
                best_cluster = -1

                for assignment in best_assignments:
                    if assignment.student == student:
                        best_score = assignment.best_score
                        best_cluster = assignment.index

                current_assignments.append({
                    "score": current_score - best_score,
                    "current_cluster": i,
                    "best_cluster": best_cluster,
                })

                
def sort(students: list[Student], centroids: list[Student], clusters: list[list[Student]]):
    # Order students by their distance to their nearest cluster minus distance to the furthest cluster
    ordered = []
    for student in students:
        min_score = math.inf
        min_centroid = None
        max_score = -math.inf

        # Find min and max scores and centroids
        for centroid in centroids:
            s = centroid.compare_to(student)

            if s < min_score:
                s = min_score
                min_centroid = centroid

            if s > max_score:
                s = max_score


        score = max_score - min_score
        ordered.append({
            score: score,
            student: student,
            min_centroid: centroid,
        })

    # Add students to their clusters until the cluster is full
    resort = []
    while len(ordered) > 0:
        min_entry = ordered[0]

        for entry in ordered:
            if entry.score < min_entry.score:
                min_entry = entry

        for cluster in clusters:
            if cluster[0] == min_entry.min_centroid:
                if len(cluster) == 1:
                    cluster.append(min_entry.student)
                    centroids.remove(min_entry.student)
                    ordered.remove(min_entry)
                else:
                    resort.append(min_entry)
                    ordered.remove(min_entry)
        
    if len(resort) == 0:
        return clusters

    return sort(resort, centroids, clusters)
        

        
                

        
