# 👶 Working Bots - Production Ready Agents

Production-ready, tested LLM agents ready for deployment.

## 📂 Contents

### Supervisor Agent (`supervisor_agent.ipynb`)
Multi-agent orchestration for complex task workflows.

**Features:**
- Task planning and decomposition
- Agent coordination
- Error handling and retries
- Result aggregation

**Use Cases:**
- Complex data pipelines
- Multi-step workflows
- Research automation

**To Run:**
```bash
jupyter notebook supervisor_agent.ipynb
```

### Web Data Agent (`web_data_agent.ipynb`)
Automated web scraping and data extraction with AI reasoning.

**Features:**
- Web navigation and crawling
- Data extraction with NLP
- Structured data generation
- Error recovery

**Use Cases:**
- Market research automation
- Content aggregation
- Data collection pipelines

**To Run:**
```bash
jupyter notebook web_data_agent.ipynb
```

## ⚙️ Installation

See parent directory `requirements.txt` for all dependencies.

```bash
pip install -r ../requirements.txt
```

## 🚀 Quick Start

1. Install dependencies
2. Set up `.env` with API keys
3. Open notebook in Jupyter
4. Run cells in order
5. Modify for your use case

## 📝 Notes

- These are production-tested agents
- Customize prompts for your domain
- Monitor API usage and costs
- Implement proper error handling before deploying
