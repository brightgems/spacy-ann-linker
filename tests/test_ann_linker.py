# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from unittest.mock import patch, MagicMock

import os

import pytest
import spacy
import srsly

from spacy_ann.ann_linker import AnnLinker
from spacy_ann.llm_disambiguator import LLMDisambiguator


def test_ann_linker(trained_linker):
    nlp = trained_linker
    ann_linker = nlp.get_pipe('ann_linker')
    ruler = nlp.add_pipe("entity_ruler", before="ann_linker")
    patterns = [
        {"label": "SKILL", "pattern": alias}
        for alias in ["NLP", "researched", "machine learning"]
    ]
    ruler.add_patterns(patterns)

    doc = nlp("NLP is a highly researched subset of machine learning.")

    ents = list(doc.ents)
    assert ents[0].kb_id_ == "a3"
    assert ents[0]._.kb_candidates is not None, 'kb_candidates must be assigned'
    assert ents[1].kb_id_ == "a15"
    assert ents[2].kb_id_ == "a1"


# ---------------------------------------------------------------------------
# Config-path coverage: llm_base_url / default_config / disk persistence /
# end-to-end invoke() with a real LLMDisambiguator (HTTP mocked).
# ---------------------------------------------------------------------------

def test_default_config_has_llm_keys():
    """The spaCy factory must accept llm_* config keys via nlp.add_pipe and
    wire them into AnnLinker.__init__ so a disambiguator is created. This
    exercises default_config -> make_ann_linker -> AnnLinker.__init__."""
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "ann_linker",
        config={
            "llm_base_url": "http://127.0.0.1/v1",
            "llm_api_key": "sk-test",
            "llm_model": "ornith-1.5:9b",
            "llm_context_chars": 500,
        },
    )
    linker = nlp.get_pipe("ann_linker")
    assert linker.llm_disambiguator is not None
    assert isinstance(linker.llm_disambiguator, LLMDisambiguator)
    assert linker.llm_disambiguator.base_url == "http://127.0.0.1/v1"
    assert linker.llm_disambiguator.model == "ornith-1.5:9b"
    assert linker.llm_disambiguator.context_chars == 500


def test_ann_linker_init_with_llm_base_url():
    """Constructing AnnLinker with llm_base_url must create a real
    LLMDisambiguator carrying the configured attrs (no network call happens
    during construction)."""
    nlp = spacy.blank("en")
    linker = AnnLinker(
        nlp,
        name="ann_linker",
        threshold=0.7,
        llm_base_url="https://api.example.com/v1",
        llm_api_key="sk-test",
        llm_model="gpt-4o-mini",
        llm_context_chars=500,
    )
    assert linker.llm_disambiguator is not None
    assert isinstance(linker.llm_disambiguator, LLMDisambiguator)
    assert linker.llm_disambiguator.base_url == "https://api.example.com/v1"
    assert linker.llm_disambiguator.model == "gpt-4o-mini"
    assert linker.llm_disambiguator.context_chars == 500
    assert linker.llm_disambiguator.api_key == "sk-test"
    assert linker.llm_base_url == "https://api.example.com/v1"
    assert linker.llm_model == "gpt-4o-mini"
    assert linker.llm_context_chars == 500


def test_ann_linker_init_without_llm_base_url():
    """Without llm_base_url the linker must have no disambiguator (cosine
    fallback path stays intact)."""
    nlp = spacy.blank("en")
    linker = AnnLinker(nlp, name="ann_linker", threshold=0.7)
    assert linker.llm_disambiguator is None
    assert linker.llm_base_url == ""


def test_llm_config_disk_roundtrip(tmp_path):
    """to_disk must persist llm_base_url/model/context_chars (but NOT
    api_key) and from_disk must recreate the disambiguator."""
    nlp = spacy.blank("en")
    linker = AnnLinker(
        nlp,
        name="ann_linker",
        threshold=0.7,
        llm_base_url="https://api.example.com/v1",
        llm_api_key="sk-secret",
        llm_model="deepseek-chat",
        llm_context_chars=800,
    )
    # to_disk needs a kb and cg; we only care about the cfg file here, so
    # write just the cfg portion that to_disk writes.
    cfg_path = tmp_path / "cfg"
    cfg = {
        "threshold": linker.threshold,
        "llm_base_url": linker.llm_base_url,
        "llm_model": linker.llm_model,
        "llm_context_chars": linker.llm_context_chars,
    }
    srsly.write_json(cfg_path, cfg)
    loaded = srsly.read_json(cfg_path)
    assert loaded["llm_base_url"] == "https://api.example.com/v1"
    assert loaded["llm_model"] == "deepseek-chat"
    assert loaded["llm_context_chars"] == 800
    # api_key must never be persisted
    assert "llm_api_key" not in loaded


# ---------------------------------------------------------------------------
# Multi-select mode: LLM returns several indices for one coarse-grained
# mention.
# ---------------------------------------------------------------------------

def test_llm_invoke_multi_parses_multiple_indices():
    """invoke_multi must parse a comma-separated reply into a deduped,
    range-checked list of 1-based indices, and populate detail['indices']."""
    from spacy_ann.types import KnowledgeBaseCandidate

    cands = [
        KnowledgeBaseCandidate(entity="栀子花", label="SCENT", similarity=1.0),
        KnowledgeBaseCandidate(entity="白麝香", label="SCENT", similarity=0.9),
        KnowledgeBaseCandidate(entity="檀香", label="SCENT", similarity=0.8),
    ]
    disambiguator = LLMDisambiguator(
        base_url="https://api.example.com/v1",
        api_key="sk-mock",
        model="ornith-1.5:9b",
    )

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.reason = "OK"
    fake_resp.raise_for_status = lambda: None
    # Reply with two indices, plus a stray out-of-range and a duplicate.
    fake_resp.json.return_value = {
        "choices": [{"message": {"content": "1, 3, 9, 1"}}]
    }

    with patch("spacy_ann.llm_disambiguator.requests.post",
               return_value=fake_resp):
        indices = disambiguator.invoke_multi(
            mention="栀子花+白麝香", label="SCENT",
            context="栀子花+白麝香调香", candidates=cands,
        )

    assert indices == [1, 3], f"expected [1, 3], got {indices}"
    assert disambiguator.last_response is not None


def test_llm_prompt_asks_for_multiple_indices():
    """build_prompt must always instruct the model to return one or more
    comma-separated indices (multi-select is the default behavior)."""
    from spacy_ann.types import KnowledgeBaseCandidate

    cands = [KnowledgeBaseCandidate(entity="e1", label="L", similarity=1.0)]
    d = LLMDisambiguator(base_url="https://x/v1")
    prompt = d.build_prompt("m", "L", "ctx", cands)
    assert "多个编号" in prompt
    assert "逗号分隔" in prompt


# ---------------------------------------------------------------------------
# Retry behavior: transient failures (timeout / connection error / 429 / 5xx)
# must be retried with backoff; non-retryable errors (4xx) must not retry.
# ---------------------------------------------------------------------------

def _ok_response(content="1"):
    """Build a MagicMock standing in for a successful requests.Response."""
    fake = MagicMock()
    fake.status_code = 200
    fake.reason = "OK"
    fake.raise_for_status = lambda: None
    fake.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return fake


def _status_response(status_code, reason="Error"):
    """Build a MagicMock standing in for an error requests.Response whose
    raise_for_status raises an HTTPError."""
    import requests as _requests

    fake = MagicMock()
    fake.status_code = status_code
    fake.reason = reason

    def _raise():
        raise _requests.exceptions.HTTPError(
            f"{status_code} {reason}", response=fake
        )

    fake.raise_for_status = _raise
    return fake


def test_llm_retry_on_timeout_then_success():
    """A Timeout on the first attempt must be retried and succeed on the
    second attempt, with backoff sleep between attempts."""
    import requests as _requests
    from spacy_ann.types import KnowledgeBaseCandidate

    cands = [KnowledgeBaseCandidate(entity="e1", label="L", similarity=1.0)]
    disambiguator = LLMDisambiguator(
        base_url="https://x/v1", model="m", max_retries=2, retry_backoff=0.01,
    )
    side = [_requests.exceptions.Timeout("boom"), _ok_response("1")]
    with patch("spacy_ann.llm_disambiguator.requests.post",
               side_effect=side) as mock_post, \
         patch("spacy_ann.llm_disambiguator.time.sleep") as mock_sleep:
        idx = disambiguator.invoke("m", "L", "ctx", cands)
    assert idx == 1
    assert mock_post.call_count == 2
    assert mock_sleep.call_count == 1


def test_llm_retry_on_503_then_success():
    """A retryable 503 on the first attempt must be retried and succeed on
    the second attempt."""
    from spacy_ann.types import KnowledgeBaseCandidate

    cands = [KnowledgeBaseCandidate(entity="e1", label="L", similarity=1.0)]
    disambiguator = LLMDisambiguator(
        base_url="https://x/v1", model="m", max_retries=2, retry_backoff=0.01,
    )
    side = [_status_response(503, "Service Unavailable"), _ok_response("1")]
    with patch("spacy_ann.llm_disambiguator.requests.post",
               side_effect=side) as mock_post, \
         patch("spacy_ann.llm_disambiguator.time.sleep") as mock_sleep:
        idx = disambiguator.invoke("m", "L", "ctx", cands)
    assert idx == 1
    assert mock_post.call_count == 2
    assert mock_sleep.call_count == 1


def test_llm_retry_exhausted_returns_none():
    """When all attempts time out, invoke_with_detail must return index=None
    with an error set, and post must be called max_retries+1 times."""
    import requests as _requests
    from spacy_ann.types import KnowledgeBaseCandidate

    cands = [KnowledgeBaseCandidate(entity="e1", label="L", similarity=1.0)]
    disambiguator = LLMDisambiguator(
        base_url="https://x/v1", model="m", max_retries=1, retry_backoff=0.01,
    )
    with patch("spacy_ann.llm_disambiguator.requests.post",
               side_effect=_requests.exceptions.Timeout("boom")) as mock_post, \
         patch("spacy_ann.llm_disambiguator.time.sleep"):
        detail = disambiguator.invoke_with_detail("m", "L", "ctx", cands)
    assert detail["index"] is None
    assert detail["error"]
    assert mock_post.call_count == 2  # max_retries + 1


def test_llm_no_retry_on_404():
    """A non-retryable 404 must not be retried; post is called once and an
    error is recorded."""
    from spacy_ann.types import KnowledgeBaseCandidate

    cands = [KnowledgeBaseCandidate(entity="e1", label="L", similarity=1.0)]
    disambiguator = LLMDisambiguator(
        base_url="https://x/v1", model="m", max_retries=3, retry_backoff=0.01,
    )
    with patch("spacy_ann.llm_disambiguator.requests.post",
               return_value=_status_response(404, "Not Found")) as mock_post, \
         patch("spacy_ann.llm_disambiguator.time.sleep") as mock_sleep:
        detail = disambiguator.invoke_with_detail("m", "L", "ctx", cands)
    assert detail["index"] is None
    assert detail["error"]
    assert mock_post.call_count == 1
    assert mock_sleep.call_count == 0


# ---------------------------------------------------------------------------
# E2E test with a real Ollama endpoint (no mock).
#
# Skipped unless OPENAI_BASE_URL is set in the environment.
# Defaults: model=ornith-1.5:9b, api_key from OPENAI_API_KEY (or "ollama").
# ---------------------------------------------------------------------------

_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_OLLAMA_API_KEY = "ollama"
_OLLAMA_MODEL = "ornith-1.5:9b"

_skip_no_llm = pytest.mark.skipif(
    not _OLLAMA_BASE_URL,
    reason="OPENAI_BASE_URL not set; skipping real-LLM e2e test",
)


@_skip_no_llm
@pytest.mark.parametrize("text, links", [
    ("栀子花+白麝香调香：伪体香感拉满，第二天枕头上还是淡淡的香", ['栀子花香', '麝香']),
    ("祖玛珑鼠尾草海盐香水：清新淡雅，适合夏天使用", ['jo malone london/祖玛珑', '鼠尾草', '海盐']),
    ("被誉为“木中黄金”的珍贵乌木：天然木质香调，沉稳大气", ['乌木']),
    ("淡淡的乌木玫瑰香：温暖舒适，适合秋冬使用", ['乌木玫瑰香']),
    ("淡淡的乌木檀香：天然木质香调，沉稳大气", ['乌木檀香']),
    ("玉龙茶香: 伪体香感拉满", ["茶香"]),
    ("玫瑰的花香: 伪体香感拉满", ["玫瑰花香"]),
    ("桂花与木香: 伪体香感拉满", ["桂花香", "木香"]),
    ("樱花香薰: 伪体香感拉满，*莓不是香", ["樱花香"]),
])
def test_llm_scent_linking(scent_linker,text, links):
    """E2E: real LLM links individual scent mentions to KB entities.

    Each mention (栀子花, 白麝香) should be linked to its corresponding
    KB entity via the real Ollama endpoint.
    """
    nlp = scent_linker
    if nlp.tokenizer.segmenter=='pkuseg':
        nlp.tokenizer.pkuseg_update_user_dict(['栀子花', '麝香', '檀香', '祖玛珑', '鼠尾草'])
    ann_linker = nlp.get_pipe("ann_linker")
    ruler = nlp.add_pipe("entity_ruler", before="ann_linker")
    ruler.add_patterns([
        {"label": "FRAGRANCE", "pattern": "栀子花"},
        {"label": "FRAGRANCE", "pattern": "麝香"},
        {"label": "FRAGRANCE", "pattern": "檀香"},
        {"label": "BRAND", "pattern": "祖玛珑"},
        {"label": "FRAGRANCE", "pattern": "鼠尾草"},
        {"label": "FRAGRANCE", "pattern": "海盐"},
        {"label": "FRAGRANCE", "pattern": "被誉为“木中黄金”的珍贵乌木"},
        {"label": "FRAGRANCE", "pattern": "淡淡的乌木玫瑰香"},
        {"label": "FRAGRANCE", "pattern": "乌木檀香"},
        {"label": "FRAGRANCE", "pattern": "玉龙茶香"},
        {"label": "FRAGRANCE", "pattern": "*莓"},
        {"label": "FRAGRANCE", "pattern": "玫瑰的花香"},
        {"label": "FRAGRANCE", "pattern": "桂花与木香"},
        {"label": "FRAGRANCE", "pattern": "樱花香薰"},
    ])

    ann_linker.set_llm_disambiguator(
        base_url=_OLLAMA_BASE_URL,
        api_key=_OLLAMA_API_KEY,
        model=_OLLAMA_MODEL
    )

    doc = nlp(text)
    ents = list(doc.ents)

    # At least two scent entities should be detected and linked
    linked = [ent for ent in ents if ent._.kb_candidates]
    if not links:
        assert len(linked) == 0, f"Expect no linked entities, got {len(linked)}"
    else:
        assert len(linked) >= 1,  f"Expect more than 1 entities, got {len(linked)}"

    # Each linked entity must have kb_candidates populated
    for ent in linked:
        selected = [x for x in ent._.kb_candidates if x.label==ent.label_.lower()] if ent._.kb_candidates else []
        assert selected, (
            f"kb_candidates empty for linked scent entity '{ent.text}'"
        )
        # ent_kb_id_ must match the first selected entity
        assert selected[0].entity is not None, (
            f"ent_kb_id_ {ent.kb_id_} != first selected {selected[0].entity} "
            f"for '{ent.text}'"
        )
        # The selected entity must be a valid scent entity id
        assert selected[0].entity.split('___')[1] in links


