# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""A failed Bedrock model listing must raise, not report zero models.

OpenAIMixin.list_models() re-raises when list_provider_model_ids() fails, and
ModelsRoutingTable.refresh() keeps the registered models when list_models()
raises. An empty list instead means "the provider has no models", so with
refresh_models enabled one failed listing unregisters every Bedrock model.
"""

from unittest.mock import MagicMock, patch

import pytest

from ogx.core.routing_tables.models import ModelsRoutingTable
from ogx.providers.registry.inference import available_providers
from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
from ogx.providers.remote.inference.bedrock.config import BedrockConfig

BEDROCK_SPEC = next(p for p in available_providers() if p.provider_type == "remote::bedrock")


def _sigv4_adapter(client: MagicMock) -> BedrockInferenceAdapter:
    adapter = BedrockInferenceAdapter(config=BedrockConfig(region_name="us-east-1", refresh_models=True))
    adapter._bedrock_client = client
    adapter.__provider_spec__ = BEDROCK_SPEC
    adapter.__provider_id__ = "bedrock"
    return adapter


def _listing(*model_ids: str) -> dict:
    return {"modelSummaries": [{"modelId": m, "modelLifecycleStatus": "ACTIVE"} for m in model_ids]}


async def test_sigv4_listing_failure_raises():
    client = MagicMock()
    client.list_foundation_models.side_effect = ConnectionError("Could not connect to the endpoint URL")
    adapter = _sigv4_adapter(client)

    with patch.object(adapter, "_should_use_sigv4", return_value=True):
        with pytest.raises(ConnectionError):
            await adapter.list_provider_model_ids()


async def test_sigv4_listing_returns_active_models():
    client = MagicMock()
    client.list_foundation_models.return_value = _listing("anthropic.claude-3-haiku", "amazon.titan-embed")
    adapter = _sigv4_adapter(client)

    with patch.object(adapter, "_should_use_sigv4", return_value=True):
        assert list(await adapter.list_provider_model_ids()) == ["anthropic.claude-3-haiku", "amazon.titan-embed"]


class _UnreachablePage:
    """Like AsyncOpenAI's models.list(): the request fails when the page is iterated."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise ConnectionError("mantle endpoint unreachable")


async def test_bearer_listing_failure_raises():
    adapter = BedrockInferenceAdapter(config=BedrockConfig(region_name="us-east-1", api_key="token"))
    failing = MagicMock()
    failing.models.list.return_value = _UnreachablePage()

    with (
        patch.object(adapter, "_should_use_sigv4", return_value=False),
        patch("ogx.providers.remote.inference.bedrock.bedrock.AsyncOpenAI", return_value=failing),
    ):
        with pytest.raises(ConnectionError):
            await adapter.list_provider_model_ids()


async def test_failed_refresh_keeps_registered_bedrock_models(cached_disk_dist_registry):
    client = MagicMock()
    client.list_foundation_models.return_value = _listing("anthropic.claude-3-haiku")
    adapter = _sigv4_adapter(client)
    table = ModelsRoutingTable({"bedrock": adapter}, cached_disk_dist_registry, {})
    await table.initialize()

    with patch.object(adapter, "_should_use_sigv4", return_value=True):
        await table.refresh()
        assert await table.has_model("bedrock/anthropic.claude-3-haiku")

        # The next periodic refresh cannot reach Bedrock.
        client.list_foundation_models.side_effect = ConnectionError("Could not connect to the endpoint URL")
        await table.refresh()

    assert await table.has_model("bedrock/anthropic.claude-3-haiku")
