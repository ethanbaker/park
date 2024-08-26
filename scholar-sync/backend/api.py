from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from student import StudentModel, Student
from kmeans import KMeansVariation

origins = [
    "https://park.ethanbaker.dev",
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
def create_pairs(models: list[StudentModel]):
    # Turn StudentModels into Students
    mentors = []
    mentees = []
    for m in models:
        student = Student(model=m)

        if m.role == "mentor":
            mentors.append(student)
        else:
            mentees.append(student)

    # Run kmeans on the list of students
    k = len(mentors)
    kmeans = KMeansVariation(k=k, clusters=mentors, max_iter=100)
    clusters = kmeans.fit(mentees)

    # Group students' names into pairs using the clusters
    pairs = []
    for i, cluster in enumerate(clusters):
        pairs.append([student.name for student in cluster])

    # Now clusters contains the grouped vectors
    return {"pairs": pairs}