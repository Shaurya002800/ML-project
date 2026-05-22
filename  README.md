# Vivayu

**Vivayu** is an AI-powered AgTech platform engineered to optimize crop health and maximize agricultural productivity through multi-modal predictive diagnostics and automated expert assistance. The system combines resource-constrained IoT edge sensors with advanced machine learning models to detect crop diseases early, protecting yields before outbreaks spread.

---

### **Core Architecture**
* **Predictive Diagnostics:** Utilizes highly optimized **XGBoost** classification models trained on real-time Internet of Things (IoT) Volatile Organic Compound (VOC) sensor data to identify specific plant disease signatures from localized gas emissions.
* **Intelligent Assistant (AgriGPT):** A voice-first, localized Retrieval-Augmented Generation (**RAG**) assistant built using **LangChain**, **FAISS** vector databases, and **Groq** LLM endpoints. It delivers immediate, context-aware agricultural advisory, market data, and treatment prescriptions.
* **Automated Reporting:** Generates downloadable, dynamic pesticide prescriptions and compliance-ready crop health summary reports directly through the application interface.
* **Web Delivery:** Implements a fast, interactive user interface built via a live-deployed **Streamlit** cloud application.

---

### **Key Features**
* **Early-Stage VOC Detection:** Categorizes metabolic shifts in crops using environmental and gas sensor inputs, bypassing standard optical delays associated with visual camera-based disease identification.
* **Voice-First Accessibility:** Features natural language processing tailored for hands-free operations, allowing farmers to speak queries and receive audio-guided agricultural insights.
* **Low-Latency RAG:** Queries large localized agricultural vector stores instantly through highly optimized Groq language models and semantic similarity search.
* **Actionable Prescriptions:** Provides automated, data-driven treatment and chemical management guides structured as formal, downloadable PDFs.

---

### **Tech Stack**
* **AI/ML & Core NLP:** Python, XGBoost, LangChain, FAISS Vector Store, Groq API
* **Frontend & Presentation:** Streamlit, Tailwind CSS components
* **Data Pipelines & Storage:** NumPy, Pandas, Scikit-learn
* **Hardware Integration:** IoT Gas and Environmental Sensor Arrays (VOC, MQ-series sensors)

---

### **System Workflow**
1. **Data Acquisition:** Low-power sensor nodes capture local environmental metadata and gas chemical densities from fields.
2. **ML Classification:** The pre-trained XGBoost engine evaluates the sensor array matrix to predict the probability of specific fungal, bacterial, or environmental crop stress.
3. **RAG Injection:** If anomalies are detected, the system extracts precise chemical treatment documentation from the vector database using FAISS semantic search.
4. **Context Synthesis:** LangChain compiles the extracted documents and current sensor diagnostics, dispatching them to Groq to generate a localized, natural-sounding mitigation checklist.
5. **UI Update:** The live Streamlit application displays the status updates and exposes a secure option to download the pesticide prescription report.

---

### **Project Status**
Developed as a high-fidelity functional prototype demonstrating accurate sensor-driven predictive diagnostics paired with generative LLM orchestration for smart agricultural ecosystems.