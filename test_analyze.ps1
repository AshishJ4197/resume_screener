# test_analyze.ps1
# Usage:
#   1) Edit the paths/text below
#   2) Run:  powershell -ExecutionPolicy Bypass -File .\test_analyze.ps1

$API = "http://127.0.0.1:8000/api/v1/analyze"

# ---- EDIT THESE ----
$ResumePath = "JALADI ASHISH RESUME_oracle.pdf"
$UseJDFile  = $false                      # $true to send a JD file instead of JD text
$JDFilePath = "C:\path\to\your\jd.pdf"
$JDText = @"
🧠 Job Title: Associate / Junior Generative AI Engineer (Fresher)
Location: [Hybrid / Onsite / Remote — e.g., Bangalore, Pune, Hyderabad]
Experience: 0–1 Years
Employment Type: Full-time
Department: AI / Machine Learning / Product Engineering
About the Role

We are looking for an enthusiastic and innovative Generative AI Engineer (Fresher) to join our AI team.
In this role, you’ll work on designing, fine-tuning, and deploying large language model (LLM)-based applications using frameworks like LangChain, LlamaIndex, and OpenAI APIs. You will collaborate with data scientists, backend engineers, and product teams to build intelligent systems that understand, reason, and generate human-like text and content.

This is an exciting opportunity for someone passionate about AI, NLP, and prompt engineering, eager to learn real-world GenAI workflows and contribute to production-ready AI systems.

Key Responsibilities

Design and implement LLM-powered applications using frameworks such as LangChain, LlamaIndex, or Haystack.

Develop prompt engineering pipelines and retrieval-augmented generation (RAG) workflows.

Work with vector databases (FAISS, Chroma, Pinecone, Weaviate, Milvus) for semantic search and document retrieval.

Fine-tune or adapt open-source LLMs (LLaMA, Mistral, Falcon, Gemma, etc.) for domain-specific tasks.

Integrate AI APIs (OpenAI, Anthropic, Google Gemini, etc.) into web or backend systems.

Collaborate with backend teams to deploy AI models using FastAPI, Flask, or Streamlit.

Analyze and evaluate model performance (precision, recall, BLEU, ROUGE, perplexity, etc.).

Write efficient, production-quality Python code for data preprocessing, model integration, and automation.

Stay up-to-date with advancements in Generative AI, Transformer architectures, and MLOps practices.

Technical Skills (Required)
🧩 Programming & Tools:

Proficiency in Python and experience with libraries like transformers, torch, langchain, sentence-transformers, chromadb.

Familiarity with REST APIs and JSON for integrating AI services.

Experience with Jupyter Notebooks or VS Code for experimentation.

Understanding of data preprocessing, text embeddings, and tokenization.

🤖 AI/ML Concepts:

Basic understanding of:

Neural networks and transformers (BERT, GPT, etc.)

RAG (Retrieval Augmented Generation)

Embeddings & vector search

Fine-tuning and model evaluation

Awareness of Hugging Face ecosystem and OpenAI API usage.

🗄️ Databases & Backend:

Familiarity with NoSQL / vector databases (Chroma, Pinecone, FAISS, Milvus).

Knowledge of FastAPI or Flask for backend integration.

Basic understanding of cloud services (AWS, GCP, or Azure).

Soft Skills

Strong problem-solving and analytical abilities.

Curiosity and a learning mindset toward cutting-edge AI technologies.

Ability to work collaboratively in a fast-paced environment.

Good written and verbal communication skills for technical documentation.

Educational Qualifications

B.E. / B.Tech / M.Tech / MCA in Computer Science, Artificial Intelligence, Data Science, or related fields.

Good understanding of Mathematics, Statistics, and Machine Learning fundamentals.

Academic or personal projects related to AI / NLP / LLMs are a plus.

Bonus / Nice-to-Have Skills

Experience with LangGraph, CrewAI, or OpenDevin-style agent frameworks.

Exposure to MLOps pipelines (Docker, MLflow, Hugging Face Hub).

Knowledge of document loaders, PDF parsing, or knowledge base QA systems.

Understanding of prompt optimization, few-shot learning, and tool use in LLMs.
"@
# --------------------

if (!(Test-Path $ResumePath)) {
  Write-Error "Resume file not found: $ResumePath"
  exit 1
}

# Build curl args robustly for PowerShell
$Args = @("--silent","-X","POST","--form","resume_file=@$ResumePath")

if ($UseJDFile) {
  if (!(Test-Path $JDFilePath)) {
    Write-Error "JD file not found: $JDFilePath"
    exit 1
  }
  $Args += @("--form","jd_file=@$JDFilePath")
} else {
  # NOTE: Keep the JD text modest in size; the endpoint accepts large text, but shell quoting gets messy.
  $Args += @("--form","jd_text=$JDText")
}

Write-Host "Calling $API ..."
$response = & curl.exe @Args $API

# Save full response for inspection
"$(Get-Date -Format o)`n$response" | Out-File -Encoding utf8 response.json
Write-Host "Saved response → $(Resolve-Path .\response.json)"

# Try to summarize
try {
  $json = $response | ConvertFrom-Json
  Write-Host ("Score: {0} | Eligible: {1} | Run ID: {2}" -f $json.score, $json.eligible, $json.run_id)
  if ($json.contacts) {
    Write-Host ("Name: {0} | Email: {1} | Phone: {2} | LinkedIn: {3}" -f `
      $json.contacts.name, $json.contacts.email, $json.contacts.phone, $json.contacts.linkedin)
  }
} catch {
  Write-Warning "Could not parse JSON. Raw response saved to response.json"
}
