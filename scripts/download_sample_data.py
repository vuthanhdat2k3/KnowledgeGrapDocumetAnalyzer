#!/usr/bin/env python3
"""
Script download các file mẫu từ URLs được cung cấp
"""
import os
import requests
import logging
from urllib.parse import urlparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, local_path: str) -> bool:
    """
    Download file từ URL về local path
    
    Args:
        url: URL của file cần download
        local_path: Đường dẫn local để lưu file
        
    Returns:
        bool: True nếu download thành công
    """
    try:
        logger.info(f"📥 Đang download: {url}")
        
        # Tạo directory nếu chưa có
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download file với stream để handle large files
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Lưu file
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(local_path)
        logger.info(f"✅ Download thành công: {local_path} ({file_size:,} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi download {url}: {e}")
        return False


def download_sample_documents():
    """Download tất cả sample documents"""
    
    # URLs của sample documents từ anh Thắng
    sample_urls = [
        {
            "url": "https://self-help.org/docs/default-source/pdfs/rfps/software-development-services-rfp-1---revised-20240809.pdf",
            "filename": "software-development-rfp.pdf",
            "description": "Software Development Services RFP"
        },
        {
            "url": "https://www.mhlw.go.jp/content/10808000/seishu_doc003.pdf",
            "filename": "mhlw-document.pdf", 
            "description": "MHLW Document (Japanese)"
        }
    ]
    
    # Thư mục lưu sample documents
    sample_dir = "data/sample_documents"
    
    logger.info("🚀 Bắt đầu download sample documents...")
    
    success_count = 0
    total_count = len(sample_urls)
    
    for doc in sample_urls:
        local_path = os.path.join(sample_dir, doc["filename"])
        
        # Kiểm tra file đã tồn tại chưa
        if os.path.exists(local_path):
            logger.info(f"⏭️ File đã tồn tại: {local_path}")
            success_count += 1
            continue
        
        # Download file
        if download_file(doc["url"], local_path):
            success_count += 1
            
            # Tạo metadata file
            metadata_path = local_path + ".meta"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"URL: {doc['url']}\n")
                f.write(f"Description: {doc['description']}\n")
                f.write(f"Downloaded: {os.path.getctime(local_path)}\n")
    
    logger.info(f"📊 Kết quả: {success_count}/{total_count} files download thành công")
    return success_count == total_count


def create_sample_text_files():
    """Tạo một số file text mẫu cho testing"""
    
    sample_texts = [
        {
            "filename": "mobile-app-requirements.txt",
            "content": """
Mobile Banking Application Requirements

Overview:
We need to develop a mobile banking application for iOS and Android platforms.
The application should provide core banking features including account management,
money transfer, bill payment, and investment tracking.

Key Features:
1. User Authentication - Biometric login, 2FA
2. Account Dashboard - View balances, transaction history
3. Money Transfer - Internal and external transfers
4. Bill Payment - Utility bills, credit cards
5. Investment Portal - Stock trading, portfolio management
6. Customer Support - In-app chat, FAQ section

Technical Requirements:
- React Native or Flutter framework
- RESTful API integration
- SQLite local database
- Push notifications
- Offline mode capabilities
- End-to-end encryption

Security Requirements:
- Multi-factor authentication
- Encrypted data storage
- Session management
- Fraud detection
- PCI DSS compliance

Performance Requirements:
- Load time < 3 seconds
- Support 100k+ concurrent users
- 99.9% uptime SLA
- Real-time transaction processing
            """
        },
        {
            "filename": "ecommerce-platform-spec.txt",
            "content": """
E-commerce Platform Specification

Project Overview:
Development of a comprehensive e-commerce platform supporting multiple vendors,
product catalog management, order processing, and payment integration.

Core Modules:

1. Vendor Management
   - Vendor registration and verification
   - Store customization
   - Commission management
   - Analytics dashboard

2. Product Catalog
   - Product listing and categorization
   - Inventory management
   - Pricing and promotions
   - Product reviews and ratings

3. Order Management
   - Shopping cart functionality
   - Checkout process
   - Order tracking
   - Return and refund handling

4. Payment Gateway
   - Multiple payment methods
   - Secure payment processing
   - Refund management
   - Payment analytics

5. User Management
   - Customer registration
   - Profile management
   - Wishlist functionality
   - Order history

Technology Stack:
- Frontend: React.js with Redux
- Backend: Node.js with Express
- Database: MongoDB
- Cache: Redis
- Search: Elasticsearch
- Payment: Stripe, PayPal integration

Deployment:
- AWS cloud infrastructure
- Docker containerization
- CI/CD pipeline
- Load balancing
- Auto-scaling
            """
        }
    ]
    
    sample_dir = "data/sample_documents"
    os.makedirs(sample_dir, exist_ok=True)
    
    logger.info("📝 Đang tạo sample text files...")
    
    for sample in sample_texts:
        file_path = os.path.join(sample_dir, sample["filename"])
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample["content"].strip())
        
        logger.info(f"✅ Tạo file: {file_path}")


def verify_downloads():
    """Kiểm tra và verify các files đã download"""
    
    sample_dir = "data/sample_documents"
    
    if not os.path.exists(sample_dir):
        logger.warning(f"⚠️ Thư mục {sample_dir} không tồn tại")
        return False
    
    files = os.listdir(sample_dir)
    
    if not files:
        logger.warning(f"⚠️ Không có file nào trong {sample_dir}")
        return False
    
    logger.info(f"📂 Tìm thấy {len(files)} files trong {sample_dir}:")
    
    total_size = 0
    for file in files:
        file_path = os.path.join(sample_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            logger.info(f"  📄 {file} ({size:,} bytes)")
    
    logger.info(f"📊 Tổng dung lượng: {total_size:,} bytes")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download sample documents")
    parser.add_argument("--download", action="store_true", default=True, 
                       help="Download files từ URLs")
    parser.add_argument("--create-samples", action="store_true",
                       help="Tạo sample text files")
    parser.add_argument("--verify", action="store_true",
                       help="Verify downloads")
    
    args = parser.parse_args()
    
    try:
        if args.download:
            download_sample_documents()
        
        if args.create_samples:
            create_sample_text_files()
        
        if args.verify:
            verify_downloads()
        
        logger.info("🎉 Hoàn tất download sample data!")
        
    except KeyboardInterrupt:
        logger.info("👋 Đã hủy bỏ download")
    except Exception as e:
        logger.error(f"💥 Lỗi không mong đợi: {e}") 