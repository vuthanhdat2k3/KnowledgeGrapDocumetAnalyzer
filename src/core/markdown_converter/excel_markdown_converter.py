import os
import re
import logging
import time
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path

import pandas as pd
from google.genai.errors import ServerError  # ✅ để bắt lỗi 503

from src.core.markdown_converter.base_markdown_converter import BaseMarkdownConverter
from src.core.llm_client import LLMClientFactory
from src.core.markdown_converter.prompts.excel_prompts import EXCEL_MARKDOWN_ENHANCER_USER_PROMPT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@dataclass
class ContextBuffer:
    """Buffer to maintain last conversation context for LLM."""
    currently_in_table: bool = False
    current_table_header: Optional[List[str]] = None
    last_three_rows: List[List[str]] = field(default_factory=list)

    def update(self, currently_in_table: bool, header: Optional[List[str]], new_rows: List[List[str]]):
        logging.debug(f"Updating context: currently_in_table={currently_in_table}, header={header}, new_rows(last 3)={new_rows[-3:]}")
        self.currently_in_table = currently_in_table
        self.current_table_header = header
        self.last_three_rows = (self.last_three_rows + new_rows)[-3:]

    def reset(self):
        logging.info("Resetting context buffer for new sheet.")
        self.currently_in_table = False
        self.current_table_header = None
        self.last_three_rows = []

    def to_context_string(self) -> str:
        context_parts = [
            f"Currently processing: {'TABLE' if self.currently_in_table else 'NON-TABLE content'}"
        ]
        if self.current_table_header:
            context_parts.append(f"Table headers: {' | '.join(self.current_table_header)}")
        if self.last_three_rows:
            context_parts.append("Last 3 rows:")
            for i, row in enumerate(self.last_three_rows, 1):
                context_parts.append(f"  Row {i}: {' | '.join(str(cell) for cell in row)}")
        context_str = "\n".join(context_parts)
        logging.debug(f"Context string built:\n{context_str}")
        return context_str

class ExcelMarkdownConverter(BaseMarkdownConverter):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        factory = LLMClientFactory()
        llm_info = factory.get_client(model_name)
        self.model = llm_info["model"]
        self.client = llm_info["client"]
        self.context_buffer = ContextBuffer()

    def _normalize_filename(self, filename: str) -> str:
        filename = re.sub(r"[^a-zA-Z0-9\s\-\(\)\[\]]", "-", filename)
        filename = re.sub(r"\s+", " ", filename)
        return filename.strip()

    def _build_user_prompt(self, context: str, content: str) -> str:
        return EXCEL_MARKDOWN_ENHANCER_USER_PROMPT.format(context=context, content=content)

    def _format_cell_content(self, value):
        if pd.isna(value):
            return ""
        value = str(value).strip()
        value = value.replace('\n', '<br>')
        value = value.replace('\r', '')
        value = value.replace('|', '&#124;')
        value = value.replace('*', '\\*').replace('_', '\\_')
        return value

    def df_to_markdown_rows(self, df: pd.DataFrame) -> List[str]:
        """Convert DataFrame to list of markdown table rows (including header and separator)."""
        rows = []
        header = [self._format_cell_content(col) for col in df.columns]
        rows.append("| " + " | ".join(header) + " |")
        rows.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
        for _, row in df.iterrows():
            row_md = "| " + " | ".join(self._format_cell_content(v) for v in row) + " |"
            rows.append(row_md)
        logging.info(f"Converted DataFrame to {len(rows)} markdown rows.")
        return rows

    def chunk_rows(self, rows: List[str], max_length: int = 12000) -> List[List[str]]:
        """Chunk rows so that each chunk's total character length does not exceed max_length."""
        chunks = []
        current_chunk = []
        current_length = 0
        for row in rows:
            row_len = len(row) + 1  # +1 for newline
            if current_length + row_len > max_length and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_length = 0
            current_chunk.append(row)
            current_length += row_len
        if current_chunk:
            chunks.append(current_chunk)
        logging.info(f"Chunked rows into {len(chunks)} chunks (max_length={max_length}).")
        return chunks

    def process_chunk_with_llm(self, context: str, chunk: str) -> Dict[str, Any]:
        """Gọi Gemini với retry và fallback khi quá tải hoặc JSON lỗi."""
        prompt = self._build_user_prompt(context, chunk)
        logging.debug(f"Sending prompt to LLM. Context length: {len(context)}, Chunk length: {len(chunk)}")

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config={"temperature": 0.0, "max_output_tokens": 16000},
                )
                response_content = response.text.strip().strip("```json").strip()
                logging.debug(f"LLM response (first 300 chars): {response_content[:300]}...")

                try:
                    result = json.loads(response_content)
                    logging.info("LLM response successfully parsed as JSON.")
                    return result
                except Exception:
                    logging.warning("Failed to parse LLM response as JSON, returning raw text.")
                    return {"content": response_content}

            except ServerError as e:
                if e.status_code == 503:
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    logging.warning(f"Gemini overloaded (503). Retry {attempt}/{max_retries} after {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
            except Exception as e:
                logging.error(f"Unexpected LLM error: {e}")
                return {"content": f"[LLM error: {e}]"}

        logging.error("Gemini failed after retries, fallback to raw chunk")
        return {"content": chunk}

    def convert_to_markdown(self, file_path: str, *args, **kwargs) -> str:
        logging.info(f"Reading Excel file: {file_path}")
        sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
        md_content = ""
        for sheet_name, df in sheets.items():
            logging.info(f"Processing sheet: {sheet_name}")
            md_content += f"# {sheet_name}\n\n"
            self.context_buffer.reset()
            rows = self.df_to_markdown_rows(df)
            chunks = self.chunk_rows(rows)
            for i, chunk_rows in enumerate(chunks):
                chunk_str = "\n".join(chunk_rows)
                context_str = self.context_buffer.to_context_string()
                logging.info(f"Processing chunk {i+1}/{len(chunks)} for sheet '{sheet_name}'")
                llm_result = self.process_chunk_with_llm(context_str, chunk_str)
                if isinstance(llm_result, dict):
                    md_content += llm_result.get("content", "") + "\n"
                    currently_in_table = llm_result.get("currently_in_table", False)
                    current_table_header = llm_result.get("current_table_header", None)
                    last_three_rows = llm_result.get("last_three_rows", [])[-3:]
                    self.context_buffer.update(currently_in_table, current_table_header, last_three_rows)
                else:
                    md_content += str(llm_result) + "\n"
        logging.info("Markdown conversion complete.")
        return md_content

if __name__ == "__main__":
    converter = ExcelMarkdownConverter()
    markdown = converter.convert_to_markdown(
        "data/sample_documents/iiPay-Global-Payroll-Request-for-Proposal-Template-1_described.xlsx"
    )
    with open(
        "data/sample_documents/iiPay-Global-Payroll-Request-for-Proposal-Template-1_described.md",
        "w", encoding="utf-8"
    ) as f:
        f.write(markdown)
