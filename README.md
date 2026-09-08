# 🧠 GenAI Projects

A comprehensive collection of **Generative AI and agentic system projects** demonstrating practical applications of LLMs, RAG, and AI agent patterns.

---

## 📚 Projects Overview

| Project | Description | Tech Stack | Status |
|---------|-------------|------------|--------|
| **🤖 Supervisor Agent** | Multi-step agent orchestration with task planning | LangChain, LLM, Python | ✅ Working |
| **🕷️ Web Data Agent** | Web scraping + data extraction with AI reasoning | BeautifulSoup, Langchain, OpenAI | ✅ Working |
| **🌐 Web Agent** | Autonomous web navigation and information gathering | Langchain, ChromeDriver, LLM | ✅ Working |
| **📋 TCS Agent** | Enterprise workflow automation agent | Langchain, Python | ✅ Working |
| **🩺 Medical SQL Data Bot** | Context-aware SQL query generation from natural language | SQLAlchemy, Langchain, OpenAI | ✅ Complete |
| **💬 History-Aware Bot** | Conversational AI with conversation memory | Langchain, Vector Store | ✅ Complete |
| **📖 RAG Concepts** | Retrieval-Augmented Generation deep dive | Langchain, Chroma, Vector Embeddings | ✅ Complete |
| **📄 PDF Reader** | Intelligent PDF document processing and Q&A | PyPDF, Langchain, Embeddings | ✅ Complete |
| **✈️ Travel Planner** | AI-powered travel planning agent | Langchain, APIs | 🔄 In Development |

---

## 🚀 Quick Start

### Prerequisites

```bash
python 3.9+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Lava888/GenAI-Projects.git
   cd GenAI-Projects
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Setup Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_api_key_here
GOOGLE_API_KEY=your_google_api_key
# Add other API keys as needed
```

⚠️ **Never commit `.env` files or API keys!**

---

## 📁 Directory Structure

```
GenAI-Projects/
├── working-bots/                    # Production-ready agents
│   ├── supervisor_agent.ipynb
│   ├── web_data_agent.ipynb
│   └── README.md
├── scripts/                         # Python scripts
│   ├── med_sql_data_bot.py
│   ├── history_aware_bot.py
│   └── README.md
├── notebooks/                       # Jupyter notebooks
│   ├── rag_concepts_2.ipynb
│   ├── tcs_agent.ipynb
│   ├── web_agent.ipynb
│   └── README.md
├── pdf-reader/                      # PDF processing module
│   ├── pdf_reader.py
│   └── README.md
├── travel-planner/                  # Travel planning agent
│   ├── travel_planner.py
│   └── README.md
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── CONTRIBUTING.md                  # Contribution guidelines
└── Summary.md                       # Project patterns & concepts
```

---

## 🧰 Core Dependencies

### LLM & Agent Frameworks
```
langchain>=0.1.0
langchain-openai>=0.1.0
langchain-community>=0.0.1
```

### Vector & RAG
```
langchain-chroma>=0.1.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
```

### Data Processing
```
pandas>=2.0.0
pypdf>=3.17.0
bs4>=4.12.0
```

### UI & APIs
```
streamlit>=1.28.0
httpx>=0.24.0
requests>=2.31.0
```

---

## 📖 Project Details

### 🤖 Supervisor Agent
**Location:** `working-bots/supervisor_agent.ipynb`

Demonstrates multi-step task orchestration where agents break down complex problems into subtasks.

**Key Features:**
- Task decomposition and planning
- Agent coordination and execution
- Error handling and fallbacks
- Result aggregation

**Use Cases:**
- Complex data processing pipelines
- Multi-step business workflows
- Research and analysis tasks

### 📄 PDF Reader
**Location:** `pdf-reader/`

Intelligent PDF processing system with semantic search and Q&A capabilities.

**Features:**
- Document parsing and chunking
- Semantic search across documents
- Question-answering on PDF content
- Citation and source tracking

### 🩺 Medical SQL Data Bot
**Location:** `scripts/med_sql_data_bot.py`

Converts natural language queries into SQL statements for medical databases.

**Features:**
- Context-aware query generation
- Schema understanding
- Result formatting and validation
- Error handling for complex queries

### 💬 History-Aware Bot
**Location:** `scripts/history_aware_bot.py`

Conversational AI with persistent conversation memory.

**Features:**
- Conversation history management
- Context-aware responses
- Multi-turn conversations
- Memory persistence

### 📖 RAG Concepts
**Location:** `notebooks/rag_concepts_2.ipynb`

Deep dive into Retrieval-Augmented Generation patterns.

**Topics Covered:**
- Vector embeddings and similarity search
- Document chunking strategies
- Retrieval optimization
- RAG pipeline implementation

---

## 🔧 Usage Examples

### Running a Jupyter Notebook

```bash
jupyter notebook working-bots/supervisor_agent.ipynb
```

### Running a Python Script

```bash
python scripts/med_sql_data_bot.py
```

### Using in Your Code

```python
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(api_key="your-key")
agent = initialize_agent(...)
result = agent.run("Your query here")
print(result)
```

---

## 🎓 Learning Path

1. **Start Here:** `Summary.md` - Understand GenAI patterns
2. **Basics:** `notebooks/rag_concepts_2.ipynb` - Learn RAG fundamentals
3. **Simple Projects:** `scripts/history_aware_bot.py` - Build conversational AI
4. **Advanced:** `working-bots/supervisor_agent.ipynb` - Multi-agent systems
5. **Production:** Adapt patterns to your use cases

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Adding New Projects

1. Create a new folder with a descriptive name
2. Include a `README.md` with project description
3. Add your code/notebooks
4. Update main README with project overview
5. Submit a pull request

---

## 📚 Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [RAG Patterns](https://docs.langchain.com/docs/modules/chains/popular/)
- [LLM Agents Guide](https://python.langchain.com/docs/modules/agents/)
- [Retrieval-Augmented Generation Papers](https://arxiv.org/abs/2005.11401)

---

## ⚠️ Important Notes

### Security
- **Never commit `.env` files** or credentials
- Store API keys in environment variables
- Review code before sharing
- Check for sensitive data in notebooks

### Performance
- Large files (PDFs, models) can be memory-intensive
- Monitor API usage and costs
- Implement rate limiting for API calls
- Test with small datasets first

### Best Practices
- Use virtual environments
- Keep dependencies updated
- Document your code
- Test thoroughly before production use
- Use `.gitignore` to exclude sensitive files

---

## 🐛 Troubleshooting

### Common Issues

**ImportError: No module named 'langchain'**
```bash
pip install langchain langchain-openai
```

**OpenAI API Key not found**
```bash
export OPENAI_API_KEY='your-key-here'
# Or create .env file with: OPENAI_API_KEY=your-key-here
```

**FAISS installation issues**
```bash
pip install faiss-cpu  # CPU version
pip install faiss-gpu  # GPU version (requires CUDA)
```

---

## 📊 Project Statistics

- **Total Projects:** 9
- **Notebooks:** 4
- **Python Scripts:** 3
- **Modules:** 2
- **Main Concepts:** RAG, Agents, LLMs

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) file for details.

---

## 📧 Contact & Support

For questions or issues:
- Open a [GitHub Issue](https://github.com/Lava888/GenAI-Projects/issues)
- Check [documentation](./Summary.md)
- Visit [GitHub Profile](https://github.com/Lava888)

---

**Last Updated:** September 2026

**Maintained by:** [@Lava888](https://github.com/Lava888)
