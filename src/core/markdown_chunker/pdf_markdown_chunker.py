from src.core.markdown_chunker.base_markdown_chunker import BaseMarkdownChunker

class PdfMarkdownChunker(BaseMarkdownChunker):
    def chunk_from_file(self, file_path):
        """
        Chunk a PDF file into smaller parts, which have the meaning of paragraphs.
        Returns a list of text chunks, or save them to a json file.
        """
        pass