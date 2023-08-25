import enum


class RegularisationMode(str, enum.Enum):
    prior = "prior"
    posterior = "posterior"
