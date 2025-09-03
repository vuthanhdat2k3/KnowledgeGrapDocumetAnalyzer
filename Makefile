# Makefile for Knowledge Graph Document Analyzer
# Personal Project

.PHONY: help setup docker-build docker-up docker-down docker-logs clean test lint format install-dev

# Default target
help:
	@echo "🕸️  Knowledge Graph Document Analyzer - Makefile Commands"
	@echo "=========================================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make setup          - Complete project setup"
	@echo "  make docker-up       - Start all services with Docker"
	@echo "  make docker-down     - Stop all services"
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  make docker-build    - Build Docker images"
	@echo "  make docker-rebuild  - Rebuild images from scratch"
	@echo "  make docker-logs     - View container logs"
	@echo "  make docker-shell    - Shell into app container"
	@echo ""
	@echo "📄 Document Processing:"
	@echo "  make test-pandoc     - Test pandoc installation"
	@echo "  make demo-pandoc     - Run pandoc conversion demo"
	@echo "  make process-docs    - Process sample documents"
	@echo ""
	@echo "🔧 Development:"
	@echo "  make install-dev     - Install development dependencies"
	@echo "  make test            - Run tests"
	@echo "  make lint            - Run linting"
	@echo "  make format          - Format code"
	@echo "  make clean           - Clean up generated files"
	@echo ""
	@echo "📊 Database:"
	@echo "  make neo4j-shell     - Neo4j shell access"
	@echo "  make neo4j-status    - Check Neo4j status"
	@echo "  make reset-db        - Reset database (CAUTION!)"

# Team setup (FIRST COMMAND TO RUN)
team-setup:
	@echo "🎯 TEAM SETUP - Setting up everything for teams..."
	python scripts/team_setup.py

# Cài đặt dependencies
install:
	@echo "📦 Cài đặt dependencies..."
	pip install -r requirements.txt
	@echo "✅ Cài đặt hoàn tất!"

# Setup project
setup:
	@echo "🔧 Setup project..."
	python run_app.py --setup
	@echo "✅ Setup hoàn tất!"

# Chạy ứng dụng
run:
	@echo "🚀 Chạy ứng dụng..."
	python run_app.py

# Development mode
dev:
	@echo "🔧 Chạy development mode..."
	python run_app.py --dev

# Chạy tests (TODO: Implement by QA team)
test:
	@echo "🧪 Running tests..."
	@echo "⚠️ TODO: Implement test suite"
	@echo "Expected: pytest tests/ -v"

# Test coverage (TODO: Implement)
test-coverage:
	@echo "📊 Test coverage..."
	@echo "⚠️ TODO: Implement test coverage"
	@echo "Expected: pytest --cov=src tests/ --cov-report=html"

# Hiển thị trạng thái
status:
	@echo "📊 Trạng thái hệ thống..."
	python run_app.py --status

# Format code
format:
	@echo "🎨 Format code..."
	black src/ config/ scripts/ --line-length=100
	@echo "✅ Code đã được format!"

# Lint code
lint:
	@echo "🔍 Lint code..."
	flake8 src/ config/ scripts/ --max-line-length=100 --ignore=E203,W503
	@echo "✅ Lint hoàn tất!"

# Type check
type-check:
	@echo "🔍 Type checking..."
	mypy src/ --ignore-missing-imports
	@echo "✅ Type check hoàn tất!"

# Docker commands
docker-up:
	@echo "🐳 Starting Docker containers..."
	docker-compose up -d
	@echo "✅ Docker containers started!"
	@echo "🌐 Neo4j Browser: http://localhost:7474"
	@echo "🌐 Streamlit App: http://localhost:8501"

docker-down:
	@echo "🐳 Stopping Docker containers..."
	docker-compose down
	@echo "✅ Docker containers stopped!"

docker-rebuild:
	@echo "🐳 Rebuilding Docker containers..."
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Docker containers rebuilt!"

docker-logs:
	@echo "📋 Docker container logs..."
	docker-compose logs -f

# Database commands
db-setup:
	@echo "🗄️ Setup Neo4j database..."
	python scripts/setup_neo4j.py --setup
	@echo "✅ Database setup hoàn tất!"

db-clean:
	@echo "🗑️ Clean Neo4j database..."
	python scripts/setup_neo4j.py --cleanup
	@echo "✅ Database cleaned!"

# Clean up
clean:
	@echo "🧹 Dọn dẹp files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	@echo "✅ Dọn dẹp hoàn tất!"

# Development environment
venv:
	@echo "🐍 Tạo virtual environment..."
	python -m venv venv
	@echo "✅ Virtual environment created!"
	@echo "Activate with: source venv/bin/activate"

install-dev:
	@echo "📦 Cài đặt development dependencies..."
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy
	@echo "✅ Development dependencies installed!"

# Quick start
quick-start: install setup run

# Full development setup
dev-setup: venv install-dev setup format lint
	@echo "🎉 Development environment ready!"
	@echo "⚠️ TODO: Implement modules as per team assignments"

# CI/CD commands
ci-test:
	@echo "🔄 Running CI tests..."
	@echo "⚠️ TODO: Implement CI test pipeline"

# Monitoring
logs:
	@echo "📋 Application logs..."
	@echo "⚠️ TODO: Implement logging system"

monitor:
	@echo "📊 System monitoring..."
	@echo "Neo4j status:"
	curl -s http://localhost:7474/db/data/ | jq '.neo4j_version' || echo "Neo4j not responding"
	@echo "Streamlit status:"
	curl -s http://localhost:8501/_stcore/health || echo "Streamlit not responding"

# TODO commands (for team reference)
todo-backend:
	@echo "👥 Backend Team TODOs:"
	@echo "- src/core/document_processor/ modules"
	@echo "- src/core/utils/ utilities"
	@echo "- Integration with KG engine"

todo-ai:
	@echo "🤖 AI Team TODOs:"
	@echo "- src/core/knowledge_graph/graphiti_wrapper.py implementation"
	@echo "- src/team1_product_analysis/ modules"
	@echo "- src/team2_clarification/ modules"

todo-frontend:
	@echo "🎨 Frontend Team TODOs:"
	@echo "- Enhanced Streamlit UI components"
	@echo "- KG visualization"
	@echo "- Real-time analysis dashboard"

todo-qa:
	@echo "🧪 QA Team TODOs:"
	@echo "- tests/ directory structure"
	@echo "- Unit and integration tests"
	@echo "- Performance testing" 

# Knowledge Graph Building
build-kg:
	@echo "🔨 Building Knowledge Graph automatically..."
	docker-compose exec app python scripts/auto_build_kg.py

build-kg-local:
	@echo "🔨 Building KG locally (without Docker)..."
	python scripts/auto_build_kg.py 

# Document Processing Commands
test-pandoc:
	@echo "🔍 Testing Pandoc installation..."
	@docker-compose exec app python -c "from src.core.utils.pandoc_converter import pandoc_converter; print(f'Pandoc available: {pandoc_converter.is_available()}'); print(f'Supported formats: {pandoc_converter.get_supported_formats()}')"

demo-pandoc:
	@echo "🔄 Running Pandoc conversion demo..."
	@docker-compose exec app python scripts/example_pandoc_usage.py

process-docs:
	@echo "📄 Processing sample documents..."
	@docker-compose exec app python -c "from scripts.process_sample_docs import main; main()" 