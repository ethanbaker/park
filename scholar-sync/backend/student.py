import utils
from parameters import weights, methods
from pydantic import BaseModel
from kmeans import KMeansVariation

# StudentModel is used to receive JSON input for an API
class StudentModel(BaseModel):
    role: str
    name: str
    biography: str
    hobbies: str
    class_year: int

# Student class contains complicated behavior for kmeans analysis
class Student():
    def __init__(self, model=None, vectors=None):
        """
        Convert this student representation to an n-dimensional vector when first initialized

        :param model: The student model this student is based from
        """
        # If no model is provided and vectors were, use those instead
        if model == None and vectors != None:
            self.vectors = vectors
            return

        self.name = model.name

        # Convert raw fields to vectors
        biography_vector = utils.string_to_vector(model.biography)
        hobbies_vector = utils.string_to_vector(model.hobbies)
        class_year_vector = utils.num_to_vector(model.class_year)

        # Create a dict holding all vectors
        self.vectors = {
            "biography": biography_vector,
            "hobbies": hobbies_vector,
            "class_year": class_year_vector,
        }

    def compare_to(self, s) -> float:
        """
        Compare two students based on inner vector represetation

        :param s: The student to compare themselves with
        :return: float representation of how close the two students are (0 = identical)
        """
        a = self.to_vectors()
        b = s.to_vectors()

        # Calculate differences in vectors and return their sum
        diffs = [weights[key] * methods[key](a[key], b[key]) for key in a]
        return sum(diffs) 

    def average_with(self, students):
        """
        Find the average student based on this student and other provided students

        :param students: The students to compute this average with
        """
        # Get the base vector
        sum = self.to_vectors()

        # Find the total sum
        for student in students:
            vector = student.to_vectors()
            for key in vector:
                for i in range(len(sum[key])):
                    sum[key][i] += vector[key][i]

        # Divide each key by length for the average
        for key in sum:
            sum[key] = [v / (len(students) + 1) for v in sum[key]]

        return Student(vectors=sum)

    def to_vectors(self):
        """
        Return the vector representation of this student

        :return: vector representation of this student
        """
        return self.vectors


if __name__ == "__main__":
    data = [
        ("Mentor A", "mentor", "I love hiking and reading.", "hiking reading", 2023),
        ("Mentor B", "mentor", "I enjoy coding and playing soccer.", "coding soccer", 2022),
        ("Mentor C", "mentor", "I am passionate about photography and painting.", "photography painting", 2024),
        ("Mentor D", "mentor", "I like swimming and playing chess in my free time.", "swimming chess", 2023),
        ("Mentor E", "mentor", "I am interested in astronomy and science fiction.", "astronomy science fiction", 2022),
        ("Mentor F", "mentor", "I enjoy playing guitar and writing poetry.", "guitar poetry", 2024),
        ("Mentor G", "mentor", "I love cooking and exploring new cuisines.", "cooking cuisines", 2023),
        ("Mentor H", "mentor", "I am passionate about environmental conservation.", "environment conservation", 2022),
        ("Mentee I", "mentee", "I enjoy playing basketball and watching movies.", "basketball movies", 2024),
        ("Mentee J", "mentee", "I am interested in computer programming and robotics.", "programming robotics", 2023),
        ("Mentee K", "mentee", "I love traveling and learning about different cultures.", "traveling cultures", 2022),
        ("Mentee L", "mentee", "I enjoy playing piano and reading novels in my free time.", "piano novels", 2024),
        ("Mentee M", "mentee", "I am passionate about social activism and volunteering.", "activism volunteering", 2023),
        ("Mentee N", "mentee", "I love skiing and snowboarding during winter.", "skiing snowboarding", 2022),
        ("Mentee O", "mentee", "I enjoy playing video games and exploring virtual reality.", "video games virtual reality", 2024),
        ("Mentee P", "mentee", "I am interested in history and ancient civilizations.", "history ancient civilizations", 2023),
    ]

    # Collect student objects
    students = []
    centers = []
    for d in data:
        name, role, biography, hobbies, class_year = d

        model = StudentModel(name=name, biography=biography, hobbies=hobbies, class_year=class_year, role=role)

        student = Student(model)
        students.append(student)

        if role == "mentor":
            centers.append(student)

    for a in students:
        for b in students:
            if a != b:
                print(a.name + " comparing to " + b.name + " = ", a.compare_to(b))

    # Perform kmeans
    kmeans = KMeansVariation(k=len(students)//2)
    kmeans.initialize_centers(students, centers=centers, method="provide")
    clusters = kmeans.fit(students)

    # Print out results
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i} contains {len(cluster)} objects.")
        for vector in cluster:
            print(vector.name, end=" ")
        print()