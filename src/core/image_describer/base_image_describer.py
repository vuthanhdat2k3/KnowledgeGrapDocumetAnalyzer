from abc import ABC, abstractmethod

class BaseImageDescriber(ABC):
    def __init__(self, llm_client):
        self.llm = llm_client

    @abstractmethod
    def extract_images_and_contexts(self, *args, **kwargs):
        """
        Trích xuất các (image, context) từ file.
        """
        pass

    @abstractmethod
    def generate_descriptions(self, image_context_pairs, *args, **kwargs):
        """
        Nhận pair (image, context) → trả về description cho image đó
        """
        pass

    @abstractmethod
    def replace_images_with_description(self, *args, **kwargs):
        """
        Replace description của ảnh vào trong file.
        """
        pass

    @abstractmethod
    def run(self, file_path, *args, **kwargs):
        """
        Chạy quy trình xử lý ảnh từ file:
        1. Mở file
        2. Trích xuất ảnh và ngữ cảnh
        3. Sinh mô tả cho ảnh
        4. Replace mô tả vào file
        5. Return file mới
        """
        pass
