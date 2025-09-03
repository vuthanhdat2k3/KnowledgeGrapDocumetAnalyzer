from abc import ABC, abstractmethod

class BaseMarkdownConverter(ABC):
    @abstractmethod
    def convert_to_markdown(self, file_path: str, *args, **kwargs) -> str:
        """
        Convert given file to markdown string.
        """
        pass
