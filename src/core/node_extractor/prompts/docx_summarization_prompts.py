"""
Prompts for text summarization tasks.
"""

SUMMARIZATION_PROMPT = """
## ROLE
You are an expert summarizer.

## TASK
Your task:
- Read the text carefully and summarize it in the **same language** as the original text.
- Provide a summary of the text, strictly limited to {max_length} words.
- Focus on the **main points, key facts, and essential context**.
- The summary should be **concise but complete in meaning** — avoid unnecessary details, but do not omit important information.
- Keep sentences clear and coherent.
- Do not add interpretations, opinions, or information not present in the text.

## CONTEXT
Text to summarize:
{text}

Summary:
""" 