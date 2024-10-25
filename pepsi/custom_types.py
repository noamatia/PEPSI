from enum import Enum
from typing import Dict
from dataclasses import dataclass


@dataclass
class PointNetModel:
    weight: str
    model: str
    parts: Dict[str, int]


class SPLIT(Enum):
    TEST = "test"
    TRAIN = "train"


class UID_KEY(Enum):
    SOURCE_UID = "source_uid"
    TARGET_UID = "target_uid"


class UTTERANCE_KEY(Enum):
    UTTERANCE_LLAMA3 = "utterance_llama3"
    UTTERANCE_SPELLED = "utterance_spelled"


class SHAPE_CATEGORY(Enum):
    LAMP = "lamp"
    CHAIR = "chair"
    TABLE = "table"


def pepsi_type(value, pepsi_type):
    try:
        return pepsi_type(value)
    except ValueError:
        valid_values = [category.value for category in pepsi_type]
        raise ValueError(f"Invalid pepsi type: {value}. Must be one of {valid_values}")


def split_type(value):
    return pepsi_type(value, SPLIT)


def uid_key_type(value):
    return pepsi_type(value, UID_KEY)


def utterance_key_type(value):
    return pepsi_type(value, UTTERANCE_KEY)


def shape_category_type(value):
    return pepsi_type(value, SHAPE_CATEGORY)
