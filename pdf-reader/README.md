# 📄 PDF Reader Module

Intelligent PDF processing and question-answering system.

## 🎯 Features

- Document parsing and text extraction
- Smart chunking and segmentation
- Semantic search across documents
- Question-answering on PDF content
- Citation and source tracking
- Batch processing capabilities

## 🚀 Quick Start

### Basic Usage

```python
from pdf_reader import PDFReader

# Initialize reader
reader = PDFReader(api_key="your-openai-key")

# Load PDF
reader.load("document.pdf")

# Ask questions
response = reader.query("What is the main topic?")
print(response)
```

### Command Line

```bash
python pdf_reader.py --file document.pdf --query "Your question here"
```

## 📂 File Structure

```
pdf-reader/
├── pdf_reader.py      # Main module
├── config.py          # Configuration
├── utils.py           # Utility functions
├── README.md          # This file
└── requirements.txt   # Dependencies
```

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

## 💡 Advanced Usage

### Custom Chunking

```python
reader = PDFReader(
    chunk_size=1000,
    chunk_overlap=100,
    api_key="your-key"
)
```

### Batch Processing

```python
for pdf_file in pdf_files:
    reader.load(pdf_file)
    results = reader.query_batch(queries)
    save_results(results)
```

## 🔧 Configuration

Edit `config.py` to customize:
- Chunk size and overlap
- Vector store type (Chroma, FAISS, etc.)
- Embedding model
- LLM model and parameters

## 📝 Notes

- Supports PDF files up to 100MB
- Uses OpenAI embeddings by default
- Implements caching for performance
- Handles multi-column PDFs

## 🐛 Troubleshooting

See parent README for common issues.
