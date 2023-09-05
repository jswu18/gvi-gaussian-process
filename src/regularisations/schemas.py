import enum


class RegularisationMode(str, enum.Enum):
    """
    Enum for the regularisation mode.
    """

    prior = "prior"
    posterior = "posterior"
