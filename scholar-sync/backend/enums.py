from enum import Enum

class HighSchoolEnum(Enum):
    InState = 0
    OutOfState = 1

class LeadConversationEnum(Enum):
    StronglyAgree = 0
    Agree = 1
    Neutral = 2
    Disagree = 3
    StronglyDisagree = 4

class FrequencyEnum(Enum):
    Low = 0
    Medium = 1
    High = 2