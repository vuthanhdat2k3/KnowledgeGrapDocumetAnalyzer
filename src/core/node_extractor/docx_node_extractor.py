import json
import logging
import os
import sys
import uuid
from typing import List, Union
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
import asyncio
import re

# Conditional imports
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("langchain_openai not available. Some features might not work.")

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
)
from src.core.node_extractor.prompts.docx_extractor_prompt import (
    prompt as prompt_template,
)
from src.core.node_extractor.prompts.docx_summarization_prompts import SUMMARIZATION_PROMPT
from src.core.llm_client import LLMClientFactory

load_dotenv()

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ===================== MODELS =====================

class DescriptionNode(BaseModel):
    type: str
    text: str
    embedding: str

    model_config = ConfigDict(extra="forbid")


class Entity(BaseModel):
    id: str
    name: str
    type: str
    description_node: Union[DescriptionNode, str]

    model_config = ConfigDict(extra="forbid")


class Relationship(BaseModel):
    id: str
    source: str
    target: str
    type: str
    description_node: DescriptionNode

    model_config = ConfigDict(extra="forbid")


class Metadata(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChunkResult(BaseModel):
    chunk_id: str
    chunk_text: str
    metadata: Metadata
    embedding: str
    entities: List[Entity]
    relationships: List[Relationship]

    model_config = ConfigDict(extra="forbid")


class ExtractionResult(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

    model_config = ConfigDict(extra="forbid")


# ===================== UTILS =====================

def extract_json(text: str):
    """Tìm và trích xuất JSON từ text, ưu tiên block có cả 'entities' và 'relationships'."""
    if not text:
        return None
    # Loại bỏ code block nếu có
    text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)

    # Thử tìm block JSON có cả entities và relationships
    json_pattern = r'\{[^{}]*"entities"[^{}]*:[^{}]*\[[^\]]*\][^{}]*"relationships"[^{}]*:[^{}]*\[[^\]]*\][^{}]*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group(0)

    # Fallback: lấy block {...} đầu tiên
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


# ===================== MAIN EXTRACTOR =====================

class DocxNodeExtractor:
    def __init__(self, model_name: str = "gemini-2.5-flash", embedding_model: str = "embedding"):
        try:
            logging.info(f"Initializing DocxNodeExtractor with model={model_name}, embedding_model={embedding_model}")
            factory = LLMClientFactory()
            self.llm_client = factory.get_client(model_name)
            self.embeddings = factory.get_client(embedding_model)
            logging.info(f"Embedding model: {self.embeddings['model']} (type: {self.embeddings['type']})")
            self.prompt_template = prompt_template
        except Exception as e:
            logging.error(f"Failed to initialize DocxNodeExtractor: {e}")
            raise

    async def get_chunk_summary(self, chunk_text: str, max_length: int = 100) -> str:
        """Sinh summary ngắn gọn cho 1 chunk."""
        try:
            summary_prompt = SUMMARIZATION_PROMPT.format(
                text=chunk_text,
                max_length=max_length
            )
            response = await asyncio.to_thread(
                self.llm_client["client"].models.generate_content,
                model=self.llm_client["model"],
                contents=[
                    {"role": "user", "parts": [{"text": "You are an expert summarizer."}]},
                    {"role": "user", "parts": [{"text": summary_prompt}]}
                ],
                config={"temperature": 0.3, "max_output_tokens": max_length + 50}
            )

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if candidate.content.parts:
                    return candidate.content.parts[0].text.strip()

            return chunk_text
        except Exception as e:
            logging.warning(f"Failed to generate summary: {e}")
            return chunk_text

    async def get_text_embedding(self, text: str) -> List[float]:
        """Sinh embedding cho text."""
        try:
            if self.embeddings["type"] == "embedding_sentence_transformer":
                embeddings = await asyncio.to_thread(
                    self.embeddings["client"].encode,
                    [text]
                )
                return embeddings[0].tolist() if len(embeddings) > 0 else []
            else:
                response = await asyncio.to_thread(
                    self.embeddings["client"].embeddings.create,
                    model=self.embeddings["model"],
                    input=text
                )
                return response.data[0].embedding
        except Exception as e:
            logging.warning(f"Failed to generate embedding: {e}", exc_info=True)
            return []

    async def _call_llm(self, text: str, max_retries: int = 3):
        """Gọi Gemini để trích xuất entity/relationship, ép trả JSON thủ công với retry."""
        prompt = (
            "INSTRUCTION: You are a specialized entity extractor that ALWAYS returns valid JSON.\n\n"
            "TASK: Analyze the provided markdown text and extract all entities and relationships.\n\n"
            "VERY IMPORTANT: Return ONLY valid JSON that follows this EXACT schema - nothing else:\n"
            f"{ExtractionResult.model_json_schema()}\n\n"
            "If you can't find any entities or relationships, return:\n"
            '{"entities": [], "relationships": []}\n\n'
            "Do NOT include explanation, markdown, or code blocks. Return ONLY the JSON.\n\n"
            f"MARKDOWN CONTENT:\n{text}"
        )

        for attempt in range(1, max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self.llm_client["client"].models.generate_content,
                    model=self.llm_client["model"],
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config={"temperature": 0.0}
                )

                raw_text = None
                if hasattr(response, "text") and response.text:
                    raw_text = response.text
                elif hasattr(response, "candidates") and response.candidates:
                    cand = response.candidates[0]
                    if cand.content.parts:
                        raw_text = cand.content.parts[0].text

                if raw_text:
                    logging.debug(f"🔍 Raw Gemini response: {raw_text[:500]}...")
                    json_str = extract_json(raw_text)
                    if json_str:
                        try:
                            parsed_json = json.loads(json_str)
                            return ExtractionResult(**parsed_json)
                        except Exception as e:
                            logging.error(f"❌ JSON parse error: {e}")
                            return ExtractionResult(entities=[], relationships=[])

                logging.warning("⚠️ Gemini response empty or invalid format")
                return ExtractionResult(entities=[], relationships=[])

            except Exception as e:
                logging.error(f"❌ Failed to call Gemini (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # backoff
                else:
                    return ExtractionResult(entities=[], relationships=[])

    @staticmethod
    def _generate_entity_id():
        return f"entity_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _generate_rel_id():
        return f"rel_{uuid.uuid4().hex[:8]}"

    async def extract(self, chunks):
        """Trích xuất thông tin từ danh sách chunks."""
        results = []

        for idx, chunk in enumerate(chunks, start=1):
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk["chunk_text"]
            metadata = chunk.get("metadata", {})

            logging.info(f"[{idx}/{len(chunks)}] Processing chunk_id={chunk_id}")

            logging.info(f"Generating summary for chunk {chunk_id}")
            chunk_summary = await self.get_chunk_summary(chunk_text)

            logging.info(f"Generating embedding for chunk {chunk_id}")
            chunk_embedding = await self.get_text_embedding(chunk_text)

            parsed_obj = await self._call_llm(chunk_text)

            if hasattr(parsed_obj, "model_dump"):
                parsed = parsed_obj.model_dump()
            elif hasattr(parsed_obj, "dict"):
                parsed = parsed_obj.dict()
            else:
                logging.warning(f"Unexpected parsed type: {type(parsed_obj)}")
                parsed = {"entities": [], "relationships": []}

            entity_id_map = {}
            for ent in parsed.get("entities", []):
                if not ent.get("id"):  # giữ id nếu LLM đã trả
                    ent["id"] = self._generate_entity_id()
                entity_id_map[ent["name"]] = ent["id"]

                if "description_node" in ent and isinstance(ent["description_node"], dict):
                    desc_text = ent["description_node"].get("text", "")
                    if desc_text:
                        ent["description_node"]["embedding"] = await self.get_text_embedding(desc_text)
                    else:
                        ent["description_node"]["embedding"] = []

            valid_relationships = []
            for rel in parsed.get("relationships", []):
                if not rel.get("id"):
                    rel["id"] = self._generate_rel_id()

                source = rel.get("source")
                target = rel.get("target")

                # Nếu source/target là name thì map sang id
                if source in entity_id_map:
                    rel["source"] = entity_id_map[source]
                if target in entity_id_map:
                    rel["target"] = entity_id_map[target]

                if not rel["source"] or not rel["target"]:
                    logging.warning(f"❌ Bỏ qua relationship lỗi: {source} -> {target}")
                    continue

                if "description_node" in rel:
                    rel["description_node"]["embedding"] = await self.get_text_embedding(
                        rel["description_node"].get("text", "")
                    )

                valid_relationships.append(rel)
            parsed["relationships"] = valid_relationships

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


# ===================== ENTRYPOINT =====================

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
