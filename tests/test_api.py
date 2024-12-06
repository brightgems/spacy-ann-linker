# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import srsly
from spacy_ann.api.app import app
from starlette.requests import Request
from starlette.testclient import TestClient
from fastapi import FastAPI


def create_test_app(trained_linker):
    """创建测试用的应用实例"""
    from spacy_ann.api.app import app as base_app

    # 创建新的应用实例
    app = FastAPI()

    # 复制原始应用的路由
    app.router = base_app.router

    # 添加中间件
    @app.middleware("http")
    async def add_nlp_to_state(request: Request, call_next):
        request.state.nlp = trained_linker
        response = await call_next(request)
        return response
        
    return app

def test_docs_redirect():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.url.path.endswith("docs")


def test_link(trained_linker):
    app = create_test_app(trained_linker)
    client = TestClient(app)

    example_request = srsly.read_json(
        Path(__file__).parent.parent / "spacy_ann/api/example_request.json"
    )

    res = client.post("/link", json=example_request)
    assert res.status_code == 200

    data = res.json()

    for doc in data["documents"]:
        for span in doc["spans"]:
            assert "id" in span
