import comparisons
from enums import *

# Define weights for each field
weights = {
    "class_year": 0.0,
    "major": 10.0,
    "minor": 1.5,
    "high_school": 0.5,
    "lead_conversation": 3.0,
    "academic_goals": 1.0,
    "professional_goals": 1.0,
    "frequency": 3.0,
    "involved_off_campus": 1.5,
    "involved_on_campus": 1.5,
    "curious": 1.0,
    "background": 1.0,
    "gender": 3.0,
    "description": 1.0,
    "identities": 1.0,
}

weights_sum = sum(weights.values())

# Methods used to compare each field
methods = {
    "class_year": comparisons.enum_comparison(4),
    "major": comparisons.text_comparison,
    "minor": comparisons.text_comparison,
    "high_school": comparisons.enum_comparison(len(HighSchoolEnum)),
    "lead_conversation": comparisons.enum_comparison(len(LeadConversationEnum)),
    "academic_goals": comparisons.text_comparison,
    "professional_goals": comparisons.text_comparison,
    "frequency": comparisons.enum_comparison(len(FrequencyEnum)),
    "involved_off_campus": comparisons.text_comparison,
    "involved_on_campus": comparisons.text_comparison,
    "curious": comparisons.text_comparison,
    "background": comparisons.text_comparison,
    "gender": comparisons.text_comparison,
    "description": comparisons.text_comparison,
    "identities": comparisons.text_comparison,
}