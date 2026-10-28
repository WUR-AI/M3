"""SURF request options belong in ChatOpenAI.extra_body."""

import warnings

from src import config


def test_surf_extra_body_is_explicit_and_merges_caller_options(monkeypatch):
    monkeypatch.setattr(config, "SURF_API_BASE", "https://example.invalid/v1")
    monkeypatch.setattr(config, "SURF_API_KEY", "test-key")
    monkeypatch.setattr(config, "SURF_ENABLE_THINKING", False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = config.create_llm(
            model_name="default-text-large", provider="surf",
            model_kwargs={"extra_body": {"chat_template_kwargs": {"foo": 1}}},
            extra_body={"chat_template_kwargs": {"enable_thinking": True}, "bar": 2},
        )

    assert not any("extra_body" in str(w.message) for w in caught)
    assert "extra_body" not in model.model_kwargs
    assert model.extra_body == {
        "chat_template_kwargs": {"enable_thinking": True, "foo": 1},
        "bar": 2,
    }


def test_surf_mistral_has_no_template_options(monkeypatch):
    monkeypatch.setattr(config, "SURF_API_BASE", "https://example.invalid/v1")
    monkeypatch.setattr(config, "SURF_API_KEY", "test-key")

    model = config.create_llm(
        model_name="mistralai/Devstral-Small-2-24B-Instruct-2512",
        provider="surf",
    )

    assert model.extra_body is None
