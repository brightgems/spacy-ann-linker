import re
from typing import Callable, List, Tuple
import spacy
from spacy.tokens import Doc, Span
from spacy_ann.consts import stopwords
import string


def normalize_text(text):
    # remove special characters
    PUNCTABLE = str.maketrans("", "", '!"#$\()*+,:;<=>?@[\\]^_`{|}~')
    ascii_punc_chars = dict(
        [it for it in PUNCTABLE.items() if chr(it[0])])
    text = text.translate(ascii_punc_chars)
    # remove space if not all english
    if not all([ord(c) < 128 for c in text]):
        text = text.replace(' ', '')
    return text


def get_spans(doc: Doc) -> List[Span]:
    """get all spans from doc"""
    link_spans = list(doc.spans.get('annlink', []))
    return list(doc.ents) + link_spans


def get_span_text(nlp, span):
    """ transform span text by delete redundency words

    Args:
        span ([type]): entity

    Returns:
        span text
    """
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    text = ''.join([w.text for w in span if w.pos_ not in [
                   'PART', 'ADV', 'ADJ']])

    if len(text) > 3:
        if span.label_ in ('ingredient', 'sensorial', 'flavor', 'fragrance'):
            exc_words = stopwords
            text = re.sub('|'.join(exc_words), '', text)
            # replace GPE
            if len(text) > 3:
                doc = nlp.get_pipe('ner')(nlp.make_doc(span.text))
                loc_ents = [ent for ent in doc.ents if ent.label_ in ('GPE')]
                for ent in loc_ents:
                    text = text.replace(ent.text, '')
        elif span.label_ == 'brand' and '/' in text:
            text = text.split('/')[0]
    text = normalize_text(text)
    return text.strip() or span.text


