from abc import ABC, abstractmethod

class BaseMarkdownChunker(ABC):
    def __init__(self, max_tokens=700):
        self.max_tokens = max_tokens

    @abstractmethod
    def chunk_from_file(self, file_path, *args, **kwargs):
        pass
