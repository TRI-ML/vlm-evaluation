"""
preprocessing.py

Various preprocessing utilities for handling/cleaning text (questions, punctuation), paths & temporary files, and
general functions for better quality-of-life.
"""
import re


# Lifted from LAVIS (`lavis.processors.blip_processors.py :: BLIPQuestionProcessor.pre_question`)
def process_question(question: str, max_words: int = 128) -> str:
    question = re.sub(r"([.!\"()*#:;~])", "", question.lower())
    question = question.rstrip(" ")

    # Truncate Question
    question_words = question.split(" ")
    if len(question_words) > max_words:
        question = " ".join(question_words[:max_words])

    return question
