import utils
import numpy as np
from pydantic import BaseModel

class Student(BaseModel):
    name: str
    biography: str
    hobbies: str
    class_year: int

    def to_vector(self):
        """
        Convert this student representation to an n-dimensional vector
        """
        biography_vector = utils.string_to_vector(self.biography)
        hobbies_vector = utils.string_to_vector(self.hobbies)
        class_year_vector = utils.num_to_vector(self.class_year)

        # Create a combined vector concatenating all of the attribute vectors
        combined_vector = []
        for vector in [biography_vector, hobbies_vector, class_year_vector]:
            for element in vector:
                combined_vector.append(element)

        return combined_vector