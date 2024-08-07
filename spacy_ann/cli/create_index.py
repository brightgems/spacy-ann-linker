# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import spacy
import srsly
import typer
from spacy.kb import InMemoryLookupKB
from spacy_ann.candidate_generator import CandidateGenerator
from wasabi import Printer

INPUT_DIM = 300  # dimension of pretrained input vectors
DESC_WIDTH = 300  # dimension of output entity vectors


def create_index(
    model: str,
    kb_dir: Path,
    output_dir: Path,
    new_model_name: str = "ann_linker",
    cg_threshold: float = 0.8,
    n_iter: int = 5,
    verbose: bool = True,
):

    """Create an AnnLinker based on the Character N-Gram
    TF-IDF vectors for aliases in a InMemoryLookupKB

    model (str): spaCy language model directory or name to load
    kb_dir (Path): path to the directory with kb entities.jsonl and aliases.jsonl files
    output_dir (Path): path to output_dir for spaCy model with ann_linker pipe


    kb File Formats
    
    e.g. entities.jsonl

    {"id": "a1", "description": "Machine learning (ML) is the scientific study of algorithms and statistical models..."}
    {"id": "a2", "description": "ML (\"Meta Language\") is a general-purpose functional programming language. It has roots in Lisp, and has been characterized as \"Lisp with types\"."}

    e.g. aliases.jsonl
    {"alias": "ML", "entities": ["a1", "a2"], "probabilities": [0.5, 0.5]}
    """
    msg = Printer(hide_animation=not verbose)

    msg.divider("Load Model")
    with msg.loading(f"Loading model {model}"):
        nlp = spacy.load(model)
        msg.good("Done.")

    if output_dir is not None:
        output_dir = Path(output_dir / new_model_name)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    entities = list(srsly.read_jsonl(kb_dir / "entities.jsonl"))
    aliases = list(srsly.read_jsonl(kb_dir / "aliases.jsonl"))
    kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=INPUT_DIM)

    # set up the data
    entity_ids = []
    descriptions = []
    ent_label_map = {}
    freqs = []
    for e in entities:
        entity_ids.append(e["id"])
        descriptions.append(e.get("description", ""))
        if e.get("label"):
            ent_label_map[e["id"]]=e["label"]
        freqs.append(100)

    # msg.divider("Train EntityEncoder")

    # with msg.loading("Starting training EntityEncoder"):
    #     # training entity description encodings
    #     # this part can easily be replaced with a custom entity encoder
    #     encoder = EntityEncoder(nlp=nlp, input_dim=INPUT_DIM, desc_width=DESC_WIDTH, epochs=n_iter)
    #     encoder.train(description_list=descriptions, to_print=True)
    #     msg.good("Done Training")

    msg.divider("Apply EntityEncoder")

    with msg.loading("Applying EntityEncoder to descriptions"):
        # get the pretrained entity vectors
        embeddings = [nlp.make_doc(desc).vector for desc in descriptions]
        msg.good("Finished, embeddings created")

    with msg.loading("Setting kb entities and aliases"):
        # set the entities, can also be done by calling `kb.add_entity` for each entity
        for i in range(len(entity_ids)):
            entity = entity_ids[i]
            if not kb.contains_entity(entity):
                kb.add_entity(entity, freqs[i], embeddings[i])

        for a in aliases:
            ents = [e for e in a["entities"] if kb.contains_entity(e)]
            n_ents = len(ents)
            if n_ents > 0:
                prior_prob = [1.0 / n_ents] * n_ents
                kb.add_alias(alias=a["alias"], entities=ents, probabilities=prior_prob)

        msg.good("Done adding entities and aliases to kb")

    msg.divider("Create ANN Index")

    cg = CandidateGenerator().fit(kb.get_alias_strings(), verbose=True)

    ann_linker = nlp.add_pipe("ann_linker", last=True)
    ann_linker.set_kb(kb)
    ann_linker.set_cg(cg)
    ann_linker.set_entity_lables(ent_label_map)
    if nlp.has_pipe("ner"):
        nlp.disable_pipe("ner")
    nlp.meta["name"] = new_model_name
    nlp.to_disk(output_dir)
    nlp.from_disk(output_dir)


if __name__ == "__main__":
    typer.run(create_index)
