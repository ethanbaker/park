import utils

weights = {
    "biography": 0.5,
    "hobbies": 0.7,
    "class_year": 0.2,
}

methods = {
    "biography": utils.distance,
    "hobbies": utils.distance,
    "class_year": utils.inverse_distance,
}