import spacy
from spacy.tokens import Doc
import re

import stanza
import spacy_stanza

from typing import List


def spacy_parse(text: str, model="en_core_web_trf") -> list:
    """
    Parse the text using spaCy and return the parsed document.
    
    Args:
        text (str): The text to parse.
        model: The spaCy model to use for parsing. If None, the default model will be used.
    
    Returns:
        doc: As a list of parsed sentences, where each sentence is a spaCy Doc object.
    """

    nlp = spacy.load(model)
    sentences = [s.strip() + " ." if not s.strip().endswith(".") else s.strip()
            for segment in re.split(r'\s*<[ph]>\s*', text)
            for s in re.split(r'(?<=\.)\s+', segment)
            if s.strip() and "@ @ @ @ @ @ @ @ @ @" not in s]
    
    doc = []
    for sentence in sentences:
        parsed_sentence = nlp(sentence)
        if parsed_sentence:
            doc.append(parsed_sentence)

    return doc

def spacy_stanza_parse(text:str) -> List[Doc]:
    """
    Parse the text with stanza wrapped with spaCy
    Args:
        text (str): The text to parse.
    Returns:
        List[Doc]: A list of spaCy Doc objects, each representing a parsed sentence.
    """

    stanza.download('en')
    nlp = spacy_stanza.load_pipeline('en', processors='tokenize,pos,lemma,depparse')
    nlp.add_pipe("conll_formatter", last=True)

    sentences = [s.strip() + " ." if not s.strip().endswith(".") else s.strip()
            for segment in re.split(r'\s*<[ph]>\s*', text)
            for s in re.split(r'(?<=\.)\s+', segment)
            if s.strip() and "@ @ @ @ @ @ @ @ @ @" not in s]
    doc = []
    for sentence in sentences:
        parsed_sentence = nlp(sentence)
        if parsed_sentence:
            doc.append(parsed_sentence)
    
    return doc