import json
import logging
import os
import sys
import uuid
from typing import List, Union
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import asyncio

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
)
from src.core.node_extractor.prompts.docx_extractor_prompt import (
    prompt as prompt_template,
)
from src.core.node_extractor.prompts.docx_summarization_prompts import SUMMARIZATION_PROMPT
from src.core.llm_client import LLMClientFactory
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class DescriptionNode(BaseModel):
    """Mô tả thông tin chi tiết của một entity hoặc relationship."""
    type: str
    text: str
    embedding: str

    model_config = ConfigDict(extra="forbid")  # Không cho phép field lạ


class Entity(BaseModel):
    """Định nghĩa một thực thể (entity)."""

    id: str
    name: str
    type: str
    description_node: Union[DescriptionNode, str]

    model_config = ConfigDict(extra="forbid")


class Relationship(BaseModel):
    """Định nghĩa mối quan hệ giữa các entity."""

    id: str
    source: str
    target: str
    type: str
    description_node: DescriptionNode

    model_config = ConfigDict(extra="forbid")


class Metadata(BaseModel):
    """
    Metadata cho chunk.
    Cho phép key bất kỳ nếu muốn lưu trữ thông tin linh hoạt.
    """

    model_config = ConfigDict(extra="allow")


class ChunkResult(BaseModel):
    """Kết quả trích xuất cho một chunk."""

    chunk_id: str
    chunk_text: str
    metadata: Metadata
    embedding: str
    entities: List[Entity]
    relationships: List[Relationship]

    model_config = ConfigDict(extra="forbid")


class ExtractionResult(BaseModel):
    """Kết quả trích xuất tổng thể."""

    entities: List[Entity]
    relationships: List[Relationship]

    model_config = ConfigDict(extra="forbid")


class DocxNodeExtractor:
    """Bộ trích xuất entity và relationship từ chunk văn bản, summary và embedding"""
    def __init__(self, model_name: str = "gpt-4.1", embedding_model: str = "embedding"):
        try:
            factory = LLMClientFactory()
            self.llm_client = factory.get_client(model_name)

            # Initialize embeddings client
            self.embeddings = factory.get_client(embedding_model)
            self.prompt_template = prompt_template
        except Exception as e:
            logging.error(f"Failed to initialize DocxNodeExtractor: {e}")
            raise

    async def get_chunk_summary(self, chunk_text: str, max_length: int = 100) -> str:
        try:
            summary_prompt = SUMMARIZATION_PROMPT.format(
                text=chunk_text,
                max_length=max_length
            )
            response = await asyncio.to_thread(
                self.llm_client["client"].chat.completions.create,
                model=self.llm_client["model"],
                messages=[
                    {"role": "system", "content": "You are an expert summarizer."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=max_length + 50
            )
            
            summary = response.choices[0].message.content
            return summary.strip()

        except Exception as e:
            logging.warning(f"Failed to generate summary: {e}")
            return chunk_text

    async def get_text_embedding(self, text: str) -> List[float]:
        try:
            response = await asyncio.to_thread(
                self.embeddings["client"].embeddings.create,
                model=self.embeddings["model"],
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.warning(f"Failed to generate embedding: {e}")
            return []

    async def _call_llm(self, text):
        """Gọi LLM để trích xuất entity/relationship."""
        response = await asyncio.to_thread(
            self.llm_client["client"].responses.parse,
            model=self.llm_client["model"],
            input=[
                {"role": "system", "content": "Extract entities and relationships from the provided markdown."},
                {"role": "user", "content": self.prompt_template + "\n\n" + text},
            ],
            text_format=ExtractionResult,
        )
        return response.output_parsed

    @staticmethod
    def _generate_entity_id():
        """Tạo ID duy nhất cho entity."""
        return f"entity_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _generate_rel_id():
        """Tạo ID duy nhất cho relationship."""
        return f"rel_{uuid.uuid4().hex[:8]}"

    async def extract(self, chunks):
        """Trích xuất thông tin từ danh sách chunks."""
        results = []

        for idx, chunk in enumerate(chunks, start=1):
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk["chunk_text"]
            metadata = chunk.get("metadata", {})

            logging.info(
                f"[{idx}/{len(chunks)}] Processing chunk_id={chunk_id}"
            )

            # Tạo summary cho chunk
            logging.info(f"Generating summary for chunk {chunk_id}")
            chunk_summary = await self.get_chunk_summary(chunk_text)
            
            # Tạo embedding cho chunk_text
            logging.info(f"Generating embedding for chunk {chunk_id}")
            chunk_embedding = await self.get_text_embedding(chunk_text)

            # Call LLM (trả về Pydantic object)
            parsed_obj = await self._call_llm(chunk_text)
            logging.debug(f"Raw parsed object: {parsed_obj}")

            # Chuyển sang dict để xử lý
            if hasattr(parsed_obj, "model_dump"):
                parsed = parsed_obj.model_dump()  
            elif hasattr(parsed_obj, "dict"):
                parsed = parsed_obj.dict()
            else:
                logging.warning(
                    f"Unexpected parsed type: {type(parsed_obj)}"
                )
                parsed = {"entities": [], "relationships": []}

            # Gán ID cho entities và tạo embedding cho description_node
            entity_id_map = {}
            for ent in parsed.get("entities", []):
                eid = self._generate_entity_id()
                entity_id_map[ent["name"]] = eid
                ent["id"] = eid
                
                # Tạo embedding cho entity description_node
                if "description_node" in ent and isinstance(ent["description_node"], dict):
                    desc_text = ent["description_node"].get("text", "")
                    if desc_text:
                        logging.debug(f"Generating embedding for entity {ent['id']} description")
                        ent["description_node"]["embedding"] = await self.get_text_embedding(desc_text)
                    else:
                        ent["description_node"]["embedding"] = []

            # Gán ID cho relationships và loại bỏ những cái bị lỗi
            valid_relationships = []
            for rel in parsed.get("relationships", []):
                source = rel.get("source")
                target = rel.get("target")

                # Bỏ qua nếu source hoặc target không có trong entity_id_map
                if source not in entity_id_map or target not in entity_id_map:
                    logging.warning(
                        f"❌ Bỏ qua relationship lỗi: {source} -> {target}"
                    )
                    continue

                rel["id"] = self._generate_rel_id()
                rel["source"] = entity_id_map[source]
                rel["target"] = entity_id_map[target]
                
                if "description_node" in rel:
                    rel["description_node"]["embedding"] = await self.get_text_embedding(rel["description_node"].get("text", ""))

                valid_relationships.append(rel)
            parsed["relationships"] = valid_relationships
            
            # Tạo kết quả chunk
            chunk_result = {
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "summary": chunk_summary,  
                "metadata": metadata,
                "embedding": chunk_embedding, 
                "entities": parsed.get("entities", []),
                "relationships": parsed.get("relationships", []),
            }

            logging.info(
                f"✅ Found {len(chunk_result['entities'])} entities, "
                f"{len(chunk_result['relationships'])} relationships "
                f"in chunk_id={chunk_id}"
            )
            results.append(chunk_result)
        return results


async def main():
    extractor = DocxNodeExtractor()
    input_file = "data/output/chunks/elpasoamiandmdmsrfpv19_10012019-final.json"

    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    output_data = await extractor.extract(chunks)

    output_file = "data/output/nodes/elpasoamiandmdmsrfpv19_10012019-final.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    logging.info(f"✅ Extraction completed. Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())