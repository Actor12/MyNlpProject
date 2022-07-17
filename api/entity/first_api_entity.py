# -*- coding = utf-8 -*-

"""
@Author: wufei
@File: first_api_entity.py
@Time: 2022/7/16 18:21
"""
from typing import List

from pydantic import BaseModel


class TextEntity(BaseModel):
    question: str
    answer: str
    rowKey: str
    embeddingVector: str


class TextEmbeddingEntity(BaseModel):
    textEntities: List[TextEntity]
    vectorType: str
    maxLength: int
