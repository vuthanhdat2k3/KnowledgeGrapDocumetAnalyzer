import fitz
import base64
import io
import json
import logging
import os
from PIL import Image
from typing import Dict, List, Tuple, Any
from src.core.llm_client import LLMClientFactory
from src.core.image_describer.base_image_describer import BaseImageDescriber
from src.core.image_describer.prompts.pdf_image_describer_prompt import PDF_IMAGE_ANALYSIS_PROMPT

class PdfImageDescriber(BaseImageDescriber):
    def __init__(self):
        # Initialize LLM client
        factory = LLMClientFactory()
        claude_info = factory.get_client("gpt-4.1-nano")
        self.model = claude_info["model"]
        self.client = claude_info["client"]
        
        # Call parent constructor (note: parent expects llm_client param)
        super().__init__(self.client)

    def extract_images_and_contexts(self, pdf_path: str, context_window: int = 200) -> List[Tuple[Dict, str, str]]:
        """
        Extract images and their surrounding context from PDF
        
        Returns:
            List of tuples containing (image_data_dict, context_before, context_after)
        """
        doc = fitz.open(pdf_path)
        image_context_pairs = []
        global_index = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Lấy text blocks
            blocks = page.get_text("blocks")
            
            # Lấy thông tin và ảnh thật
            images_info = []
            
            # Method 1: Standard get_images()
            for img_index, img in enumerate(page.get_images()):
                try:
                    # Lấy ảnh thật
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Xử lý alpha channel và colorspace
                    if pix.alpha:
                        # Nếu có alpha, loại bỏ alpha channel
                        pix = fitz.Pixmap(pix, 0)  # Remove alpha
                    
                    if pix.colorspace and pix.colorspace.n > 3:
                        # Nếu là CMYK, convert sang RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    # Convert sang bytes để tạo PIL Image
                    img_data = pix.tobytes("png")
                    img_pil = Image.open(io.BytesIO(img_data))
                    
                    # Resize image nếu quá lớn để tránh 413 error
                    original_size = img_pil.size
                    max_size = 1024  # Max width/height
                    if max(img_pil.size) > max_size:
                        img_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                        logging.info(f"Resized image from {original_size} to {img_pil.size}")
                    
                    # Convert sang base64 với chất lượng tối ưu
                    buffered = io.BytesIO()
                    if img_pil.mode == 'RGBA':
                        img_pil = img_pil.convert('RGB')
                    img_pil.save(buffered, format="JPEG", quality=85, optimize=True)
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    # Lấy bbox
                    img_rects = page.get_image_rects(xref)
                    for rect in img_rects:
                        images_info.append({
                            'bbox': rect,
                            'y_center': (rect[1] + rect[3]) / 2,
                            'page': page_num,
                            'image_base64': img_base64,
                            'image_size': img_pil.size,
                            'source': 'get_images'
                        })
                    
                    pix = None  # Cleanup
                    
                except Exception as e:
                    logging.warning(f"Could not extract image {img_index} on page {page_num}: {e}")
                    continue
            
            # Method 2: Extract from text blocks (for embedded/replaced images)
            text_dict = page.get_text("dict")
            image_block_count = 0
            
            for block in text_dict.get("blocks", []):
                if block.get("type") == 1:  # Image block
                    image_block_count += 1
                    try:
                        bbox = block.get("bbox")
                        if bbox and "image" in block:
                            # Try direct image data access
                            try:
                                image_data = block.get("image")
                                if image_data:
                                    img_pil = Image.open(io.BytesIO(image_data))
                                    
                                    # Skip if already found via get_images() (same size and similar position)
                                    is_duplicate = False
                                    for existing in images_info:
                                        if (abs(existing['image_size'][0] - img_pil.size[0]) < 10 and 
                                            abs(existing['image_size'][1] - img_pil.size[1]) < 10 and
                                            abs(existing['y_center'] - (bbox[1] + bbox[3]) / 2) < 20):
                                            is_duplicate = True
                                            break
                                    
                                    if not is_duplicate and img_pil.size[0] > 20 and img_pil.size[1] > 20:
                                        # Resize image nếu quá lớn
                                        original_size = img_pil.size
                                        max_size = 1024
                                        if max(img_pil.size) > max_size:
                                            img_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                                            logging.info(f"Resized text block image from {original_size} to {img_pil.size}")
                                        
                                        # Convert sang base64
                                        buffered = io.BytesIO()
                                        if img_pil.mode == 'RGBA':
                                            img_pil = img_pil.convert('RGB')
                                        img_pil.save(buffered, format="JPEG", quality=85, optimize=True)
                                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                        
                                        rect = fitz.Rect(bbox)
                                        images_info.append({
                                            'bbox': rect,
                                            'y_center': (rect[1] + rect[3]) / 2,
                                            'page': page_num,
                                            'image_base64': img_base64,
                                            'image_size': img_pil.size,
                                            'source': 'text_block'
                                        })
                                        
                                        logging.info(f"Extracted image from text block {image_block_count} on page {page_num}: {img_pil.size}")
                                        
                            except Exception as e:
                                logging.warning(f"Could not extract image from text block {image_block_count} on page {page_num}: {e}")
                    
                    except Exception as e:
                        logging.warning(f"Error processing text block {image_block_count} on page {page_num}: {e}")
                        continue
            
            # Tạo danh sách tất cả elements
            all_elements = []
            
            # Thêm text blocks
            for block in blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                if block_type == 0:  # Text block
                    all_elements.append({
                        'type': 'text',
                        'y': y0,
                        'content': text.strip()
                    })
            
            # Thêm image elements
            for img_info in images_info:
                all_elements.append({
                    'type': 'image',
                    'y': img_info['y_center'],
                    'index': global_index,
                    'page': page_num,
                    'image_base64': img_info['image_base64'],
                    'image_size': img_info['image_size'],
                    'source': img_info.get('source', 'unknown')
                })
                global_index += 1
            
            # Sắp xếp theo vị trí Y
            all_elements.sort(key=lambda x: x['y'])
            
            # Tìm context cho mỗi image
            for i, element in enumerate(all_elements):
                if element['type'] == 'image':
                    # Lấy context trước và sau image
                    context_before = []
                    context_after = []
                    
                    # Context trước image
                    j = i - 1
                    chars_before = 0
                    while j >= 0 and chars_before < context_window:
                        if all_elements[j]['type'] == 'text':
                            text = all_elements[j]['content']
                            if chars_before + len(text) <= context_window:
                                context_before.insert(0, text)
                                chars_before += len(text)
                            else:
                                remaining = context_window - chars_before
                                context_before.insert(0, text[-remaining:])
                                break
                        j -= 1
                    
                    # Context sau image
                    j = i + 1
                    chars_after = 0
                    while j < len(all_elements) and chars_after < context_window:
                        if all_elements[j]['type'] == 'text':
                            text = all_elements[j]['content']
                            if chars_after + len(text) <= context_window:
                                context_after.append(text)
                                chars_after += len(text)
                            else:
                                remaining = context_window - chars_after
                                context_after.append(text[:remaining])
                                break
                        j += 1
                    
                    # Add to image_context_pairs list
                    image_data_dict = {
                        'index': element['index'],
                        'page': element['page'],
                        'image_base64': element['image_base64'],
                        'image_size': element['image_size'],
                        'y_position': element['y'],
                        'source': element.get('source', 'unknown')
                    }
                    
                    image_context_pairs.append((
                        image_data_dict,
                        " ".join(context_before),
                        " ".join(context_after)
                    ))
        
        doc.close()
        return image_context_pairs

    def generate_descriptions(self, image_context_pairs: List[Tuple[Dict, str, str]]) -> Dict[str, Dict]:
        """
        Generate descriptions for images using context and vision API
        
        Args:
            image_context_pairs: List of (image_data_dict, context_before, context_after) tuples
            
        Returns:
            Dictionary mapping image indices to description data
        """
        descriptions = {}
        
        for image_data_dict, context_before, context_after in image_context_pairs:
            image_index = image_data_dict['index']
            logging.info(f"Describing image {image_index} on page {image_data_dict['page']}...")
            
            description = self.get_image_description_with_vision(
                context_before,
                context_after, 
                image_data_dict['image_base64'],
                image_index
            )
            
            descriptions[str(image_index)] = {
                'page': image_data_dict['page'],
                'context_before': context_before,
                'context_after': context_after,
                'description': description,
                'y_position': image_data_dict['y_position'],
                'image_size': image_data_dict['image_size'],
                'source': image_data_dict.get('source', 'unknown')
            }
        
        return descriptions

    def get_image_description_with_vision(self, context_before: str, context_after: str, 
                                        image_base64: str, image_index: int) -> str:
        prompt = PDF_IMAGE_ANALYSIS_PROMPT.format(
            context_before=context_before,
            context_after=context_after
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.2,
            )
            
            description = response.choices[0].message.content.strip()
            word_count = len(description.split())
            logging.info(f"Generated vision description for image {image_index}: {word_count} words - {description[:50]}...")
            return description
            
        except Exception as e:
            logging.error(f"Error generating vision description for image {image_index}: {e}")
            # Simple fallback without additional LLM call
            return f"Image content at position {image_index} (vision API unavailable)"

    def replace_images_with_description(self):
        """pass"""
        return
    
    def run(self, pdf_path: str, context_window: int = 200) -> Dict[str, Dict]:
        """
        Main execution method that orchestrates the full workflow
        
        Args:
            pdf_path: Path to the PDF file
            context_window: Number of characters for context extraction
            
        Returns:
            Dictionary of generated descriptions
        """
        logging.info(f"Extracting images from PDF: {pdf_path}")
        
        # Step 1: Extract images and contexts
        image_context_pairs = self.extract_images_and_contexts(pdf_path, context_window)
        
        # Step 2: Generate descriptions
        descriptions = self.generate_descriptions(image_context_pairs)
        
        # Step 3: Save descriptions to file
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = "src/core/image_describer/output/pdf_describer"
        os.makedirs(output_dir, exist_ok=True)
        
        json_file = os.path.join(output_dir, f"{base_name}_image_descriptions_vision.json")
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(descriptions, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Image descriptions saved to: {json_file}")
        
        return pdf_path

