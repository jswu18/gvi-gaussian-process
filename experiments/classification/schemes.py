import enum


class ClassificationMetricScheme(str, enum.Enum):
    accuracy = "accuracy"
    f1 = "f1"
    precision = "precision"
    recall = "recall"
