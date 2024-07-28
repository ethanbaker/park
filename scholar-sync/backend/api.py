from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from k_means_constrained import KMeansConstrained
from student import Student

origins = [
    "https://park.api.ethanbaker.dev",
    "http://localhost:4200",
]

# Create the application
app = FastAPI()
base_path = "/api/v1/"
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(base_path + "pairs")
def create_pairs(students: list[Student]):
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
    return {"pairs": clusters}