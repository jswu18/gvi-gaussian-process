from abc import ABC

from pydantic import BaseModel

from src.custom_types import JSON_ENCODERS


class ModuleParameters(BaseModel, ABC):
    class Config:
        json_encoders = JSON_ENCODERS