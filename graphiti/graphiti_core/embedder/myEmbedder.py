"""
Copyright 2025, Your Company, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, List, Union, Iterable
import aiohttp
from .client import EmbedderClient, EmbedderConfig


class LocalEmbedderConfig(EmbedderConfig):
    """
    Configuration for the local embedding model service.
    """
    base_url: str = "http://11.251.225.96:49502"
    endpoint: str = "/embed"
    embedding_dim: int = 384  # 默认维度，可配置


class LocalEmbedder(EmbedderClient):
    """
    Client for calling locally deployed embedding models via HTTP.

    This client sends requests to a local HTTP server that hosts the embedding model.
    """

    def __init__(self, config: LocalEmbedderConfig | None = None):
        if config is None:
            config = LocalEmbedderConfig()
        self.config = config
        self.base_url = config.base_url
        self.endpoint = config.endpoint
        self.embedding_dim = config.embedding_dim

    async def create(
        self, input_data: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]
    ) -> List[float]:
        """
        Create an embedding for a single input item.
        """
        if isinstance(input_data, list) and all(isinstance(i, str) for i in input_data):
            return await self.create_batch(input_data)[0]

        if isinstance(input_data, str):
            payload = {"input": input_data}
        else:
            raise ValueError("Unsupported input type for embedding")

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}{self.endpoint}", json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to get embedding: {await response.text()}")
                result = await response.json()

        return result["embedding"][:self.embedding_dim]

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of input strings.
        """
        payload = {"input": input_data_list}

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}{self.endpoint}", json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to get embeddings: {await response.text()}")
                result = await response.json()

        return [emb[:self.embedding_dim] for emb in result["embedding"]]