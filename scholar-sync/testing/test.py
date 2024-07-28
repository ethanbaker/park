import student as s
from k_means_constrained import KMeansConstrained

# Example student data
students = [
    s.Student("A", "I love hiking and reading.", "hiking reading", 2023),
    s.Student("B", "I enjoy coding and playing soccer.", "coding soccer", 2022),
    s.Student("C", "I am passionate about photography and painting.", "photography painting", 2024),
    s.Student("D", "I like swimming and playing chess in my free time.", "swimming chess", 2023),
    s.Student("E", "I am interested in astronomy and science fiction.", "astronomy science fiction", 2022),
    s.Student("F", "I enjoy playing guitar and writing poetry.", "guitar poetry", 2024),
    s.Student("G", "I love cooking and exploring new cuisines.", "cooking cuisines", 2023),
    s.Student("H", "I am passionate about environmental conservation.", "environment conservation", 2022),
    s.Student("I", "I enjoy playing basketball and watching movies.", "basketball movies", 2024),
    s.Student("J", "I am interested in computer programming and robotics.", "programming robotics", 2023),
    s.Student("K", "I love traveling and learning about different cultures.", "traveling cultures", 2022),
    s.Student("L", "I enjoy playing piano and reading novels in my free time.", "piano novels", 2024),
    s.Student("M", "I am passionate about social activism and volunteering.", "activism volunteering", 2023),
    s.Student("N", "I love skiing and snowboarding during winter.", "skiing snowboarding", 2022),
    s.Student("O", "I enjoy playing video games and exploring virtual reality.", "video games virtual reality", 2024),
    s.Student("P", "I am interested in history and ancient civilizations.", "history ancient civilizations", 2023),
]

# Number of clusters
n = len(students) // 2

# Run k clusters
km = KMeansConstrained(
    n_clusters=n,
    size_min=2,
    size_max=2,
    random_state=0,
)
cluster_assignments = km.fit_predict([student.to_vector() for student in students])

# Group vectors into clusters
clusters = [[] for _ in range(n)]
for i, cluster_id in enumerate(cluster_assignments):
    clusters[cluster_id].append(students[i].name)

# Now clusters contains the grouped vectors
print("Clusters:")
for cluster in clusters:
    print(cluster)