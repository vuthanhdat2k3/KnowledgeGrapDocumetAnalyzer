#!/usr/bin/env python3
"""
All Prompts Templates

Chứa tất cả các prompt templates được sử dụng trong hệ thống QA
"""

# Language Detection
LANGUAGE_DETECTION_PROMPT = """
Analyze the following text samples and determine the primary language.
Return ONLY the language name in English (e.g., "English", "Vietnamese", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Chinese", "Japanese", "Korean", "Arabic", "Hindi", "Dutch", "Polish", "Turkish", "Thai", "Swedish", etc.).

If multiple languages are present, return the dominant one.
If the language cannot be determined, return "Unknown".

Text samples:
{combined_text}

Language:
"""

# Translation
TRANSLATION_PROMPT = """
Translate the following question to {target_language}. 
Keep the meaning and intent exactly the same.
Return only the translated question, nothing else.
If the question is already in {target_language}, return it unchanged.

Question: {question}

Translation:
"""

# Keyword Extraction
KEYWORD_EXTRACTION_PROMPT = """
Extract 5-10 most important keywords/key phrases from the following text.
Return them as a comma-separated list.
Focus on nouns, proper nouns, and important concepts.
Preserve original language - do not translate.

Text: {text}

Keywords:
"""

# Answer Generation
ANSWER_GENERATION_PROMPT = """Based on the provided knowledge graph information, please answer the following question:

Question: {question}

Information from Knowledge Graph:
{context}

Requirements:
1. Analyze if the provided context contains sufficient information to answer the question
2. Respond in the exact format below
3. If there is sufficient information: provide a detailed answer in Vietnamese and set Answerable to True
4. If there is insufficient information: set Answerable to False and leave Answer empty
5. Include relevant context excerpt that was used for the answer (if any)

Format your response EXACTLY as follows:

Answer: [Your detailed answer in Vietnamese, or leave empty if no sufficient information]
========
Answerable: [True or False]
========
Related Context: [Relevant excerpt from the context that supports your answer, or leave empty if no information]
"""
