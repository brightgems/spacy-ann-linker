# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from pathlib import Path
import gc
from typing import Callable, List, Tuple, Dict
import os.path as osp
import itertools as it
import srsly
from spacy import util
from spacy.pipeline import Pipe
from spacy.kb import InMemoryLookupKB
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy_ann.candidate_generator import CandidateGenerator
from spacy_ann.llm_disambiguator import LLMDisambiguator
from spacy_ann.types import KnowledgeBaseCandidate
from spacy_ann.util import get_spans, get_span_text, FrequencyCache
from .regex_matcher_pipe import RegexMatcherPipe


# Run GPU/CPU memory cleanup every N documents (avoid overhead of every-doc gc)
_GPU_CLEANUP_INTERVAL = 1000

@Language.factory(
    "ann_linker",
    assigns=["span._.kb_alias"],
    default_config={
        'threshold': 0.7,
        'llm_base_url': "",
        'llm_api_key': "",
        'llm_model': "",
        'llm_context_chars': 2000,
        'llm_cache_size': 10000
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
    llm_base_url: str = "",
    llm_api_key: str = "",
    llm_model: str = "",
    llm_context_chars: int = 2000,
    llm_cache_size: int = 10000,
):
    return AnnLinker(
        nlp,
        name,
        threshold,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_context_chars=llm_context_chars,
        llm_cache_size=llm_cache_size,
    )


class AnnLinker(Pipe):
    """The AnnLinker adds Entity Linking capabilities to map NER mentions
    to KnowledgeBase Aliases or directly to KnowledgeBase Ids.
    """

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        """Used in spacy.language.Language when constructing this pipeline.
        Tells spaCy that this pipe requires the nlp object.

        nlp (Language): spaCy Language object
        **cfg: Configuration keywords

        RETURNS (AnnLinker): Initialized AnnLinker.
        """
        return cls(nlp, **cfg)

    def __init__(self, nlp, name="entity_linker", threshold=0.7,
                 llm_base_url="", llm_api_key="", llm_model="", llm_context_chars=2000,
                 llm_cache_size=10000):
        """Initialize the AnnLinker.

        nlp (Language): spaCy Language object.
        name (str): Pipeline component name.
        threshold (float): Minimum alias similarity to keep a candidate.
        """
        Span.set_extension("alias_candidates", default=[], force=True)
        Span.set_extension("kb_candidates", default=[], force=True)

        self.nlp = nlp
        self.name = name
        self.kb = None
        self.cg = None
        self.ent_label_map = {}
        self.threshold = threshold
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_context_chars = llm_context_chars
        self.llm_cache_size = llm_cache_size
        self.llm_cache = FrequencyCache(max_size=llm_cache_size)
        self.llm_disambiguator = None
        if llm_base_url:
            self.set_llm_disambiguator(
                base_url=llm_base_url,
                api_key=llm_api_key,
                model=llm_model,
                context_chars=llm_context_chars,
            )
        # Cache lightweight pipeline components to avoid repeated get_pipe() lookups
        self._doc_count = 0  # counter for periodic GPU memory cleanup

        if not self.nlp.vocab.lookups.has_table("mentions_to_alias_cand"):
            self.nlp.vocab.lookups.add_table("mentions_to_alias_cand")

    def __call__(self, doc: Doc) -> Doc:
        """Annotate spaCy doc.ents with candidate info.

        In general mode
          - Mentions are collected from ``doc.spans["annlink"]`` or ``doc.ents``

          - No label-based filtering is applied to KB candidates.

        doc (Doc): spaCy Doc

        RETURNS (Doc): spaCy Doc with updated annotations
        """

        self.require_kb()
        self.require_cg()

        mentions = get_spans(doc)
        # when llm is not enabled, we can use the normalized span text from get_span_text() to generate candidates
        if not self.llm_base_url:
            mention_strings = [get_span_text(e) for e in mentions]
        else:
            # When LLM disambiguation is enabled, we need to pass original text
            # (ent.text) to the CandidateGenerator so that it can provide the
            # LLM with context for disambiguation.
            mention_strings = [e.text for e in mentions]
            self.cg.k = 6  # increase k to provide more candidates for LLM disambiguation

        batch_candidates = self.cg(mention_strings)

        for ent, nms_candidates in zip(mentions, batch_candidates):
            alias_candidates = [
                ac for ac in nms_candidates if ac.similarity > self.threshold
            ]
            if len(alias_candidates) == 0 and nms_candidates:
                # find alias use llm
                if self.llm_disambiguator is not None:
                    llm_nms_candidates = [
                        KnowledgeBaseCandidate(
                            entity=ac.alias, label=ent.label_, similarity=ac.similarity
                        )
                        for ac in nms_candidates
                        if not self.ent_label_map or any([kb_cand for kb_cand in self.kb.get_alias_candidates(ac.alias)
                                if self.ent_label_map.get(kb_cand.entity_, '') == ent.label_.lower()])
                    ]
                    # Cache key: mention + label. context (doc.text) and
                    # candidates are intentionally excluded: context varies per
                    # document and candidates are derived from the mention via
                    # NMS, so mention+label captures the deterministic inputs.
                    cache_key = "{}|{}".format(
                        ent.text,
                        ent.label_ or "",
                    )
                    cached_aliases = self.llm_cache.get(cache_key)
                    if cached_aliases is not None:
                        alias_candidates.extend(cached_aliases)
                    else:
                        idx_muti = self.llm_disambiguator.invoke_multi(
                            mention=ent.text,
                            label=ent.label_ or "",
                            context=doc.text,
                            candidates=llm_nms_candidates,
                        )
                        llm_picked = []
                        if idx_muti:
                            for idx in idx_muti:
                                if 1 <= idx <= len(nms_candidates):
                                    best_candidate = nms_candidates[idx - 1].model_copy()
                                    best_candidate.similarity = 1.0
                                    llm_picked.append(best_candidate)
                        alias_candidates.extend(llm_picked)
                        self.llm_cache.add(cache_key, llm_picked)
            ent._.alias_candidates = alias_candidates
            if len(alias_candidates) == 0:
                continue

            mentions_table = self.nlp.vocab.lookups.get_table(
                "mentions_to_alias_cand"
            )
            # Prevent unbounded memory growth in long-running services
            if not hasattr(mentions_table, "_table") or ent.text not in mentions_table._table:
                mentions_table.set(ent.text, alias_candidates[0].alias)

            # Build candidates, filtering by entity ID prefix when NER
            # context label is available. Entity IDs use the format
            # "prefix___name", e.g. "fragrance___橘子" and "ingredient___橘子".
            # When ent.label_ is set, skip candidates whose prefix doesn't
            # match; entities without "___" are always kept.
            ent_label_lower = (ent.label_ or '').lower()

            kb_candidates = []
            for ac in alias_candidates:
                for kb_cand in self.kb.get_alias_candidates(ac.alias):
                    kb_cand_label = self.ent_label_map.get(kb_cand.entity_, '')
                    # Exclude entities whose label doesn't match the NER label, if available.
                    if ent_label_lower and kb_cand_label \
                            and kb_cand_label.lower() != ent_label_lower:
                        continue
                    kb_candidates.append(KnowledgeBaseCandidate(
                        entity=kb_cand.entity_,
                        label=self.ent_label_map.get(kb_cand.entity_, ''),
                        similarity=ac.similarity,
                    ))

            if kb_candidates:
                # sort by similarity
                kb_candidates = sorted(kb_candidates,
                    key=lambda x: (x.label, x.similarity), reverse=True)
                # dedup by entity, keep max item for each entity
                kb_candidates = [list(v)[0] for k, v in it.groupby(
                    kb_candidates, key=lambda x: x.entity)]
                ent._.kb_candidates = kb_candidates

                # Select best candidate as entity.
                exact_match = [c for c in kb_candidates if c.label == ent.label_ and c.similarity == 1]
                if exact_match:
                    best_candidate = exact_match[0]
                else:
                    best_candidate = kb_candidates[0]

                ent.kb_id_ = best_candidate.entity
                for t in ent:
                    t.ent_kb_id_ = best_candidate.entity

        # Periodic GPU memory cleanup: only run every _GPU_CLEANUP_INTERVAL docs.
        # This is far cheaper than per-document gc.collect() and properly
        # releases GPU memory via torch.cuda.empty_cache() / cupy pool.
        self._doc_count += 1
        if self._doc_count % _GPU_CLEANUP_INTERVAL == 0:
            self._release_gpu_memory()
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

    def set_llm_disambiguator(self, base_url, model, api_key="", context_chars=2000):
        """Set an LLMDisambiguator for LLM-based entity disambiguation.

        disambiguator (LLMDisambiguator): Initialized LLMDisambiguator, or
            None to disable LLM disambiguation.
        """
        self.llm_disambiguator = LLMDisambiguator(
                base_url=base_url,
                api_key=api_key,
                model=model,
                context_chars=context_chars,
            )
        self.llm_base_url = base_url
        self.llm_model = model
        self.llm_api_key = api_key
        self.llm_context_chars = context_chars
    
    
    def set_entity_lables(self, ent_label_map: Dict[str, str]):
        self.ent_label_map = ent_label_map


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

    @staticmethod
    def _release_gpu_memory():
        """Release GPU memory back to the driver.
        Detects PyTorch/CuPy backends and calls their memory release functions.
        Only safe to call periodically (e.g., every N documents), not per-document.
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        try:
            import cupy
            pool = cupy.get_default_memory_pool()
            pool.free_all_blocks()
        except (ImportError, AttributeError):
            pass
        gc.collect()

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

        self.llm_base_url = cfg.get("llm_base_url", "")
        self.llm_model = cfg.get("llm_model", "gpt-4o-mini")
        self.llm_api_key = cfg.get("llm_api_key")
        self.llm_context_chars = cfg.get("llm_context_chars", 2000)
        self.llm_cache_size = cfg.get("llm_cache_size", 10000)
        self.llm_cache = FrequencyCache(max_size=self.llm_cache_size)
        if self.llm_base_url:
            self.set_llm_disambiguator(self.llm_base_url, self.llm_model, self.llm_api_key)
        else:
            self.llm_disambiguator = None
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
            "llm_base_url": self.llm_base_url,
            "llm_api_key": self.llm_api_key,
            "llm_model": self.llm_model,
            "llm_context_chars": self.llm_context_chars,
            "llm_cache_size": self.llm_cache_size,
        }
        srsly.write_json(path / "cfg", cfg)

        self.kb.to_disk(path / "kb")
        self.cg.to_disk(path)
        srsly.write_json(path / "el", self.ent_label_map)
