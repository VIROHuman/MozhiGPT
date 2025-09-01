# MozhiGPT Makefile
# Usage: make <target>

.PHONY: help install train serve test clean docker-build docker-run

# Default target
help:
	@echo "🇹🇦 MozhiGPT - Tamil-First ChatGPT"
	@echo ""
	@echo "Available commands:"
	@echo "  install       Install dependencies"
	@echo "  download      Download datasets"
	@echo "  train         Train the Tamil model"
	@echo "  serve         Start the API server"
	@echo "  test          Test the API endpoints"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run with Docker"
	@echo "  clean         Clean up generated files"
	@echo "  dev           Start development server"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# Download datasets
download:
	@echo "📥 Downloading datasets..."
	python scripts/download_dataset.py

# Train the model
train:
	@echo "🚂 Training Tamil GPT model..."
	python train.py

# Start the server
serve:
	@echo "🚀 Starting MozhiGPT server..."
	python scripts/start_server.py

# Start development server
dev:
	@echo "🛠️ Starting development server with reload..."
	python scripts/start_server.py --reload

# Test the API
test:
	@echo "🧪 Testing MozhiGPT API..."
	python scripts/test_api.py

# Test specific endpoint
test-health:
	python scripts/test_api.py --test health

test-chat:
	python scripts/test_api.py --test chat

test-stream:
	python scripts/test_api.py --test stream

test-ws:
	python scripts/test_api.py --test ws

# Docker commands
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t mozhigpt .

docker-run:
	@echo "🐳 Running with Docker..."
	docker run -p 8000:8000 mozhigpt

docker-dev:
	@echo "🐳 Starting Docker development environment..."
	docker-compose --profile dev up

docker-prod:
	@echo "🐳 Starting Docker production environment..."
	docker-compose up -d mozhigpt-api

# Utility commands
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Setup for first time users
setup: install download
	@echo "✅ Setup complete! You can now run 'make train' to train the model."

# Quick start (install, download, train, serve)
quickstart: setup train serve

# Lint code
lint:
	@echo "🔍 Linting code..."
	black --check .
	flake8 .
	isort --check-only .

# Format code
format:
	@echo "✨ Formatting code..."
	black .
	isort .

# Run all checks
check: lint test
	@echo "✅ All checks passed!"
