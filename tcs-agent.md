## **Supervisor-Agent Architecture Diagram**

```
                 ┌──────────────────┐
                 │      User        │
                 │  (query input)   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Supervisor Agent │
                 │  (Routing logic) │
                 └───────┬──────────┘
        Route query     │
        based on type   │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌──────────────────┐           ┌──────────────────┐
│     TCS Agent    │           │     WEB Agent    │
│ (Domain: TCS RAG)│          │ (Domain: Web RAG)│
└─────────┬────────┘           └─────────┬────────┘
          │                                │
          │ Calls Tools / RAG DBs          │ Calls Tools / RAG DBs
          ▼                                ▼
   ┌───────────────┐                 ┌───────────────┐
   │ search_tcs_rag│                 │ search_web_rag│
   │  (vector DB)  │                 │  (vector DB)  │
   └───────────────┘                 └───────────────┘
          │                                │
          └──────────────┬─────────────────┘
                         ▼
                 ┌──────────────────┐
                 │   LLM Reasoning  │
                 │ Generates Answer │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   User Response  │
                 └──────────────────┘
```

---

### **Flow Explanation**

1. **User sends a query** → Supervisor receives it.
2. **Supervisor analyzes query** → decides which sub-agent should handle it:

   * **TCS Agent** → specialized in internal domain knowledge (TCS database)
   * **WEB Agent** → handles general or web-based queries
3. **Agent calls its tools** → queries RAG vector database for top matching documents.
4. **LLM interprets retrieved documents** → generates an answer in natural language.
5. **Response is returned to the Supervisor** → then delivered to the **User**.
6. **Error handling** ensures fallback if a tool fails or no answer is found.

---

### **Why this Diagram Helps in Hackathons**

* Clearly shows **hierarchical delegation** (Supervisor → Sub-agent → Tool).
* Highlights **RAG integration + LLM reasoning**.
* Demonstrates **modularity** — easy to add more agents or knowledge sources.
* Judges immediately see the **complexity and sophistication** of your multi-agent system.

---

If you want, I can also **make a visually polished diagram with boxes and arrows** (like a slide-ready image) that you can directly use in hackathon presentations.

Do you want me to create that?
