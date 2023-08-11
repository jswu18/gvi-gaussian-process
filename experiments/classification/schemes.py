import enum


class MetricScheme(str, enum.Enum):
    accuracy = "accuracy"
    f1 = "f1"
    precision = "precision"
    recall = "recall"
