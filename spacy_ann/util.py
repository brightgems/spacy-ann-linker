import re
from typing import Dict, List, Tuple, Optional
from spacy.tokens import Doc, Span
from .consts import stopwords, country_regions
from .types import AliasCandidate
from dataclasses import dataclass
from collections import Counter
import heapq


def normalize_text(text):
    # remove special characters
    PUNCTABLE = str.maketrans("", "", r'!"#$\()*+,:;<=>?@[\\]^_`{|}~')
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
    text = ''.join([w.text for w in span 
                    if not (
                        w.pos_ in ['PART', 'ADV'] or
                        w.text in country_regions
    )])
    if len(text) == 0:
        return span.text
    elif len(text) > 3:
        if span.label_ in ('ingredient', 'sensorial', 'flavor', 'fragrance'):
            exc_words = stopwords
            text = re.sub('|'.join(exc_words), '', text)
            # replace GPE
            if len(text) > 3:
                doc = nlp.get_pipe('ner')(nlp.make_doc(span.text))
                loc_ents = [ent for ent in doc.ents if ent.label_ in ('ORG', 'GPE')]
                for ent in loc_ents:
                    text = text.replace(ent.text, '')
        elif span.label_ == 'brand' and '/' in text:
            text = text.split('/')[0]
    text = normalize_text(text)
    return text.strip()


@dataclass
class CacheItem:
    candidates: List[AliasCandidate]
    frequency: int = 1
    last_access: float = 0.0

class FrequencyCache:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, CacheItem] = {}
        self._access_count = Counter()
        
    def get(self, key: str) -> Optional[List[AliasCandidate]]:
        """获取缓存项，并更新访问频率"""
        if key in self._cache:
            self._cache[key].frequency += 1
            return self._cache[key].candidates
        return None

    def add(self, key: str, value: List[AliasCandidate]):
        """添加新的缓存项"""
        if len(self._cache) >= self.max_size:
            self._remove_least_frequent()
        self._cache[key] = CacheItem(candidates=value)

    def _remove_least_frequent(self):
        """移除访问频率最低的项"""
        if not self._cache:
            return
        
        # 找到频率最低的项
        min_freq_item = min(self._cache.items(), key=lambda x: x[1].frequency)
        self._cache.pop(min_freq_item[0])

    def get_most_frequent(self, n: int = 10) -> List[Tuple[str, int]]:
        """获取访问频率最高的n个项"""
        return heapq.nlargest(
            n, 
            [(k, v.frequency) for k, v in self._cache.items()],
            key=lambda x: x[1]
        )
