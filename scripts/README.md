# 🐍 Python Scripts - Ready to Use

Production-ready Python scripts for specific AI tasks.

## 📂 Contents

### Medical SQL Data Bot (`med_sql_data_bot.py`)
Natural language to SQL converter for medical databases.

**Features:**
- NLP-based query generation
- Schema understanding
- Query validation
- Result formatting

**Usage:**
```bash
python med_sql_data_bot.py
```

**Example:**
```python
from med_sql_data_bot import MedicalQueryBot

bot = MedicalQueryBot(api_key="your-key")
result = bot.query("Show me patient records from January")
print(result)
```

### History-Aware Bot (`history_aware_bot.py`)
Conversational AI with conversation memory.

**Features:**
- Multi-turn conversations
- Conversation history tracking
- Context-aware responses
- Memory persistence

**Usage:**
```bash
python history_aware_bot.py
```

**Example:**
```python
from history_aware_bot import ConversationBot

bot = ConversationBot(api_key="your-key")
response = bot.chat("Hello, what can you do?")
print(response)
```

## 🚀 Setup

1. **Install dependencies**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Set up environment**
   ```bash
   export OPENAI_API_KEY='your-key'
   # or create .env file
   ```

3. **Run scripts**
   ```bash
   python script_name.py
   ```

## 🔧 Customization

Each script is designed to be easily customizable:

1. Modify prompts for your domain
2. Adjust temperature and parameters
3. Change model (GPT-4, Claude, etc.)
4. Add custom preprocessing/postprocessing

## 📝 Notes

- Scripts are production-ready
- Implement error handling for your use case
- Monitor API costs
- Test thoroughly before deployment
- Use in automation workflows or APIs
