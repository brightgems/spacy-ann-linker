# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
from pathlib import Path

import pytest
import spacy
import srsly


@pytest.fixture
def entities():
    return list(srsly.read_jsonl("examples/tutorial/data/entities.jsonl"))


@pytest.fixture
def aliases():
    return list(srsly.read_jsonl("examples/tutorial/data/aliases.jsonl"))


@pytest.fixture
def nlp():
    return spacy.load("zh_core_web_md")


@pytest.fixture()
def trained_linker():
    model_path = Path("examples/tutorial/models/ann_linker")
    if not model_path.exists():
        subprocess.run(
            [
                "spacy_ann",
                "create_index",
                "zh_core_web_md",
                "examples/tutorial/data",
                "examples/tutorial/models",
            ]
        )
    # if not TRAINED_LINKER:
    TRAINED_LINKER = spacy.load("examples/tutorial/models/ann_linker")

    return TRAINED_LINKER


@pytest.fixture()
def scent_linker():
    """Build and load a Chinese scent KB model for e2e LLM tests.

    Uses scent entities (栀子花, 白麝香, 檀香, 玫瑰, 茉莉, etc.) defined in
    examples/tutorial/data/scent/.  The model is built once and cached on disk.
    """
    model_path = Path("/code/content_kg/ul_kgminer_zh/ul_entity_link/ann_linker")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path {model_path} does not exist. Please ensure the scent linker model is built and available."
        )
    return spacy.load(model_path)
