import utils
from parameters import weights, weights_sum, methods
from pydantic import BaseModel
from kmeans import KMeansVariation
from enums import *

# StudentModel is used to receive JSON input for an API
class StudentModel(BaseModel):
    # Identifying fields
    name: str
    email: str

    # Mentor/Mentee status
    role: str
    mentee_limit: int | None = None

    # Personal fields
    class_year: int
    major: str = ""
    minor: str = ""
    high_school: HighSchoolEnum | None
    lead_conversation: LeadConversationEnum | None = LeadConversationEnum.Neutral
    academic_goals: str = ""
    professional_goals: str = ""
    frequency: FrequencyEnum | None
    involved_off_campus: str = ""
    involved_on_campus: str = ""
    curious: str = ""
    background: str = ""
    gender: str = ""
    description: str = ""
    identities: str = ""


# Student class contains complicated behavior for kmeans analysis
class Student():
    def __init__(self, model: StudentModel|None = None, vectors=None):
        """
        Convert this student representation to an n-dimensional vector when first initialized

        :param model: The student model this student is based from
        """
        # If no model is provided and vectors were, use those instead (for 'cluster average' students)
        if model == None and vectors != None:
            self.vectors = vectors
            return

        # Set static fields that aren't used for comparisons
        self.name = model.name
        self.email = model.email

        self.role = model.role
        self.mentee_limit = model.mentee_limit

        # Convert raw fields to vectors
        class_year_vector = utils.num_to_vector(model.class_year)
        major_vector = utils.string_to_vector(model.major)
        minor_vector = utils.string_to_vector(model.minor)
        high_school_vector = utils.enum_to_vector(model.high_school)
        lead_conversation_vector = utils.enum_to_vector(model.lead_conversation)
        academic_goals_vector = utils.string_to_vector(model.academic_goals)
        professional_goals_vector = utils.string_to_vector(model.professional_goals)
        frequency_vector = utils.enum_to_vector(model.frequency)
        involved_off_campus_vector = utils.string_to_vector(model.involved_off_campus)
        involved_on_campus_vector = utils.string_to_vector(model.involved_on_campus)
        curious_vector = utils.string_to_vector(model.curious)
        background_vector = utils.string_to_vector(model.background)
        gender_vector = utils.string_to_vector(model.gender)
        description_vector = utils.string_to_vector(model.description)
        identities_vector = utils.string_to_vector(model.identities)

        # Create a dict holding all vectors
        self.vectors = {
            "class_year": class_year_vector,
            "major": major_vector,
            "minor": minor_vector,
            "high_school": high_school_vector,
            "lead_conversation": lead_conversation_vector,
            "academic_goals": academic_goals_vector,
            "professional_goals": professional_goals_vector,
            "frequency": frequency_vector,
            "involved_off_campus": involved_off_campus_vector,
            "involved_on_campus": involved_on_campus_vector,
            "curious": curious_vector,
            "background": background_vector,
            "gender": gender_vector,
            "description": description_vector,
            "identities": identities_vector,
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
        diffs = []
        for key in a:
            if len(a[key]) > 0 and len(b[key]) > 0:
                diffs.append(weights[key] * methods[key](a[key], b[key]))

        return sum(diffs) / weights_sum

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
                    if len(vector[key]) > 0 and vector[key][i] != None:
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
    mentors = []
    for d in data:
        name, role, biography, hobbies, class_year = d

        model = StudentModel(
            name=name,
            role=role,
            description=biography,
            curious=hobbies,
            class_year=class_year,
            # Not used
            email="",
            mentee_limit=None,
            major="",
            minor="",
            high_school=0,
            lead_conversation=0,
            academic_goals="",
            professional_goals="",
            frequency=0,
            involved_off_campus="",
            involved_on_campus="",
            gender="",
            identities="",
        )

        student = Student(model)

        if role == "mentor":
            mentors.append(student)
        else:
            students.append(student)

    for a in students:
        for b in students:
            if a != b:
                print(a.name + " comparing to " + b.name + " = ", a.compare_to(b))

    # Perform kmeans
    kmeans = KMeansVariation(k=len(mentors), clusters=mentors)
    clusters = kmeans.fit(students)

    # Print out results
    for i, cluster in enumerate(clusters):
        #print(f"Cluster {i} contains {len(cluster)} objects.")
        for vector in cluster:
            print(vector.name, end=" ")
        print()