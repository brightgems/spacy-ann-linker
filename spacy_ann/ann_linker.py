# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from pathlib import Path
import re
from typing import Callable, List, Tuple, Dict
import os.path as osp
import itertools as it
import numpy as np
import srsly
from spacy import util
from spacy.pipeline import Pipe
from spacy.kb import InMemoryLookupKB
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy_ann.candidate_generator import CandidateGenerator
from spacy_ann.types import KnowledgeBaseCandidate
from spacy_ann.util import get_spans, get_span_text
from .regex_matcher_pipe import RegexMatcherPipe
    

@Language.factory(
    "ann_linker",
    assigns=["span._.kb_alias"],
    default_config={
        'threshold': 0.7,
        'enable_context_similarity': False,
        'disambiguate': ""
    },
    default_score_weights={
        "ents_f": 1.0,
        "ents_p": 0.0,
        "ents_r": 0.0,
        "ents_per_type": None,
    },
)
def make_ann_linker(
    nlp: Language,
    name: str,
    threshold: float,
    enable_context_similarity: bool,
    disambiguate: str
):
    return AnnLinker(
        nlp,
        name,
        threshold,
        enable_context_similarity,
        disambiguate
    )


class AnnLinker(Pipe):
    """The AnnLinker adds Entity Linking capabilities
    to map NER mentions to KnowledgeBase Aliases or directly to KnowledgeBase Ids
    """

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        """Used in spacy.language.Language when constructing this pipeline.
        Tells spaCy that this pipe requires the nlp object

        nlp (Language): spaCy Language object

        RETURNS (AnnLinker): Initialized AnnLinker.
        """
        return cls(nlp, **cfg)

    def __init__(self, nlp, name="entity_linker", threshold=0.7, enable_context_similarity=False, disambiguate=None):
        """Initialize the AnnLinker

        nlp (Language): spaCy Language object
        """
        Span.set_extension("alias_candidates", default=[], force=True)
        Span.set_extension("kb_candidates", default=[], force=True)

        self.nlp = nlp
        self.name = name
        self.kb = None
        self.cg = None
        self.ent_label_map = {}
        self.threshold = threshold
        self.enable_context_similarity = enable_context_similarity
        self.disambiguate = disambiguate
        if disambiguate and self.ent_label_map:
            self.nlp.add_pipe("ann_regex_matcher", config={"regex": self.get_match_patterns()})
        if not self.nlp.vocab.lookups.has_table("mentions_to_alias_cand"):
            self.nlp.vocab.lookups.add_table("mentions_to_alias_cand")

    def get_match_patterns(self):
        """Get match patterns for regex matcher"""
        def get_patterns(words):
            words = sorted(words, key=len, reverse=True)
            return "|".join([re.escape(w) for w in words])
        
        def is_disambiguate(candidates):
            candidates = [c for c in candidates 
                          if self.ent_label_map.get(c.entity_) == self.disambiguate]
            return len(candidates) > 0
        
        all_aliases = self.kb.get_alias_strings()
        word_items = [e for e in all_aliases if is_disambiguate(self.kb.get_alias_candidates(e))]
        return {self.disambiguate: get_patterns(word_items)}
        
        
    def __call__(self, doc: Doc) -> Doc:
        """Annotate spaCy doc.ents with candidate info.
        Parameters:
          disambiguate(str): If exists, detect phrase use charcter start whith this label
          enable_context_similarity(bool): If  is True, use entity vectors and doc context
        to pick the most likely Candidate

        doc (Doc): spaCy Doc

        RETURNS (Doc): spaCy Doc with updated annotations
        """

        self.require_kb()
        self.require_cg()

        if self.disambiguate:
            self.nlp.select_pipes(disable=["ann_linker", "ner"])
            doc = self.nlp(doc.text)
            self.nlp.select_pipes(enable=["ann_linker", "ner"])
            mentions = doc.ents
            mention_strings = [ent.text for e in mentions]
        else:
            mentions = get_spans(doc)
            mention_strings = [get_span_text(self.nlp, e) for e in mentions]
        batch_candidates = self.cg(mention_strings)
        for ent, alias_candidates in zip(mentions, batch_candidates):
            alias_candidates = [
                ac for ac in alias_candidates if ac.similarity > self.threshold
            ]

            ent._.alias_candidates = alias_candidates

            if len(alias_candidates) == 0:
                continue
            else:
                mentions_table = self.nlp.vocab.lookups.get_table(
                    "mentions_to_alias_cand"
                )
                mentions_table.set(ent.text, alias_candidates[0].alias)

                # return all kb entities of each candidate
                alias_kb_lst = [
                    self.kb.get_alias_candidates(ac.alias) for ac in alias_candidates
                ]
                # flatten to list of candidates
                kba_candidates = list(it.chain(*alias_kb_lst))
                kba_alias_idx = list(
                    it.chain(*[[i] * len(items) for i, items in enumerate(alias_kb_lst)]))
                candicate_similarity = [
                    ac.similarity for ac in alias_candidates
                ]
                if self.enable_context_similarity and ent.has_vector:
                    # create candidate matrix
                    entity_encodings = np.asarray(
                        [c.entity_vector for c in kba_candidates]
                    )
                    doc_vector = doc.vector.T.get() if str(type(doc.vector)).count('cupy') else doc.vector.T
                    candidate_norm = np.linalg.norm(
                        entity_encodings, axis=1)
                    sims = np.dot(entity_encodings, doc_vector) / (
                        (candidate_norm * doc.vector_norm) + 1e-8
                    )
                else:
                    sims = np.zeros(len(kba_candidates))
                kb_candidates = []
                for cand, alias_idx, csim in zip(kba_candidates, kba_alias_idx, sims):
                    asim = candicate_similarity[alias_idx]
                    kb_candidates.append(
                        KnowledgeBaseCandidate(
                            entity=cand.entity_, label=self.ent_label_map.get(
                                cand.entity_, ''),
                            similarity=csim if self.enable_context_similarity and csim > 0 else asim,
                            context_similarity=csim,
                            alias_similarity=asim
                        )
                    )
                
                if self.disambiguate and isinstance(self.disambiguate, str):
                    kb_candidates = [ent for ent in kb_candidates if ent.label.startswith(self.disambiguate)]
                if kb_candidates:
                    # dedup by entity, keep max item for each entity
                    kb_candidates = sorted(kb_candidates, key=lambda x: (
                        x.label, x.similarity), reverse=True)
                    kb_candidates = [list(v)[0] for k, v in it.groupby(
                        kb_candidates, key=lambda x: x.entity)]
                    
                    # sort by similarity
                    kb_candidates = sorted(
                        kb_candidates, key=lambda x: x.similarity, reverse=True)
                    ent._.kb_candidates = kb_candidates

                    # select best candidate as entity
                    best_candidate = kb_candidates[0]
                    for t in ent:
                        t.ent_kb_id_ = best_candidate.entity

        return doc

    def set_kb(self, kb: InMemoryLookupKB):
        """Set the InMemoryLookupKB

        kb (InMemoryLookupKB): spaCy InMemoryLookupKB
        """
        self.kb = kb

    def set_cg(self, cg: CandidateGenerator):
        """Set the CandidateGenerator

        cg (CandidateGenerator): Initialized CandidateGenerator
        """
        self.cg = cg

    def set_entity_lables(self, ent_label_map: Dict[str, str]):
        self.ent_label_map = ent_label_map
        if self.ent_label_map and isinstance(self.ent_label_map, str):
            if self.nlp.has_pipe("ann_regex_matcher"):
                self.nlp.remove_pipe("ann_regex_matcher")
            self.nlp.add_pipe("ann_regex_matcher", config={"regex": self.get_match_patterns()})

    def require_kb(self):
        """Raise an error if the kb is not set.

        RAISES:
            ValueError: kb required
        """
        if getattr(self, "kb", None) in (None, True, False):
            raise ValueError(f"KnowledgeBase `kb` required for {self.name}")

    def require_cg(self):
        """Raise an error if the cg is not set.

        RAISES:
            ValueError: cg required
        """
        if getattr(self, "cg", None) in (None, True, False):
            raise ValueError(
                f"CandidateGenerator `cg` required for {self.name}")

    def from_disk(self, path: Path, **kwargs):
        """Deserialize saved AnnLinker from disk.

        path (Path): directory to deserialize from

        RETURNS (AnnLinker): Initialized AnnLinker
        """
        path = util.ensure_path(path)

        kb = InMemoryLookupKB(self.nlp.vocab, 300)
        kb.from_disk(path / "kb")
        self.set_kb(kb)

        cg = CandidateGenerator().from_disk(path)
        self.set_cg(cg)

        cfg = srsly.read_json(path / "cfg")

        self.threshold = cfg.get("threshold", 0.7)
        self.enable_context_similarity = cfg.get(
            "enable_context_similarity", False)
        if isinstance(cfg.get("disambiguate"), str):
            self.disambiguate = cfg.get("disambiguate")
        if osp.exists(path / "el"):
            self.ent_label_map = srsly.read_json(path / "el")
        return self

    def to_disk(self, path: Path, exclude: Tuple = tuple(), **kwargs):
        """Serialize AnnLinker to disk.

        path (Path): directory to serialize to
        exclude (Tuple, optional): config to exclude. Defaults to tuple().
        """
        path = util.ensure_path(path)
        if not path.exists():
            path.mkdir()

        cfg = {
            "threshold": self.threshold,
            "enable_context_similarity": self.enable_context_similarity,
            "disambiguate": self.disambiguate,
        }
        srsly.write_json(path / "cfg", cfg)

        self.kb.to_disk(path / "kb")
        self.cg.to_disk(path)
        srsly.write_json(path / "el", self.ent_label_map)
