# coding: utf-8
import regex as re
from typing import Callable, DefaultDict, List, Mapping
from typing import Text
import numpy as np
from spacy.language import Language
from spacy.pipeline.pipe import Pipe
from spacy.tokens.doc import Doc
from spacy.tokens import Span

Span.set_extension('match_', default=None, force=True)
Span.set_extension('labels_', default=[], force=True)

@Language.factory(
    "ann_regex_matcher",
    assigns=["doc.ents", "token.ent_type","span._.match_"],
)
def extract_rule_rela(nlp, name: str, regex: Mapping[str, str],stopwords: List[str]=None,\
    alignment_mode: str='expand',overwrite_ents: bool=False):
    return RegexMatcherPipe(nlp, name, regex=regex,stopwords=stopwords, \
        alignment_mode=alignment_mode,overwrite_ents=overwrite_ents)


class RegexMatcherPipe:
    name = "ann_regex_matcher"

    def __init__(self, nlp, name, regex: Mapping[str, str], stopwords: List[str] = None,
                 alignment_mode: str = 'expand', overwrite_ents=False):
        '''
            regex matcher for spacy pipeline
            Input:
                - nlp: spacy instance
                - regex: regex dictionary, like {"key":regex}
                - alignment: whether make alignment on token if right index of match isn't token boundry
                - align_right: right index need to be token end char
                - stopwords: list of string
                - overwrite_ents: later span can overwrite existing span if matched text contains the other
        '''
        self.nlp = nlp
        self.name = name
        self.alignment_mode = alignment_mode
        self.patterns = {}
        self.stopwords = stopwords or []
        self.overwrite_ents = overwrite_ents
        for k in regex:
            self.patterns[k] = re.compile(regex[k], flags=re.IGNORECASE)
    
    def __call__(self, doc):
        """
         whether align to token boundaries automatically if regex span doesn't map to token boundaries
        """
        seen_tokens = set()
        new_entities = []
        entities = doc.ents

        for k in self.patterns:
            for match in self.patterns[k].finditer(doc.text, concurrent=True):
                # 英文使用subword匹配会产生歧义
                is_ascii = all(ord(c) < 128 for c in match.group())
                alignment_mode = 'strict' if is_ascii else self.alignment_mode
                start_char, end_char = match.span()
                span = doc.char_span(start_char, end_char, label=k, alignment_mode=alignment_mode)
                # ensure span len is bigger than match length
                if span and match.group() and len(span.text) >= len(match.group()):
                    # translate to token position
                    start, end = span.start, span.end
                    span._.set('match_', match.group())
                    span._.labels_.append(k)
                    # avoid overlapping by default
                    # check if any other entities overlap with span and match text does not contain by new span
                    overlap_entities = [
                        e for e in entities if (e.start < end and e.end > start)
                        # if overwrite_ents=True, new span can overwrite existing, like 最近 can replace 最
                        and not(self.overwrite_ents and e._.match_ and e._.match_ in match.group())
                    ]
                    # overwrite if match len is more than old
                    is_contained_by_existing = any([e for e in overlap_entities if
                                                    len(e._.match_) >= len(match.group())])
                    # if is_contained_by_existing:
                    #     print (match.group(), k, overlap_entities[0].text, overlap_entities[0]._.match_, overlap_entities[0].label_)
                    if not is_contained_by_existing:
                        if not all([i in seen_tokens for i in range(start, end)]):
                            # avoid overlap with entities in current pipeline
                            new_entities = [e for e in new_entities if not (e.start < end and e.end > start)]
                            new_entities.append(span)
                            entities = [
                                e for e in entities if not (e.start < end and e.end > start)
                            ]
                            seen_tokens.update(range(start, end))
                            # add existing label to new span
                            for e in overlap_entities:
                                if e.label_ not in span._.labels_:
                                    span._.labels_.append(e.label_)
                    else:
                        # add new lable to existing span
                        for e in overlap_entities:
                            if e.start == start and e.end == end and k not in span._.labels_:
                                e._.labels_.append(k)
        if new_entities:
            doc.ents = entities + new_entities
        return doc
