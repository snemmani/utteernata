# Shankar Nag Nemmani
**Lead AI Systems & Post-Training Engineer**  
Hyderabad, India · +91 95025 62524 · shankar.nemmani@yahoo.com · [linkedin.com/in/shankar-nag-nemmani](https://linkedin.com/in/shankar-nag-nemmani) · [github.com/snemmani](https://github.com/snemmani)

---

### SUMMARY
AI Systems and Post-Training Engineer with 13 years of backend, distributed systems, and applied machine learning experience. Specializes in building autonomous agentic execution loops, parameter-efficient fine-tuning (LoRA/QLoRA), alignment (DPO), and adversarial evaluation harnesses. First-named inventor on US Patent Application 2026/0010616 A1 for automated schema-driven test generation. Shipped production agent workflows and deep learning classifiers with rigorous ground-truth evaluation across 20+ enterprise SaaS product families.

---

### SELECTED OPEN SOURCE & RESEARCH SYSTEMS

- **AEGIS: Adversarially-Hardened Autonomous Code Agent & Eval Harness**  
  *Open-source post-trained model & evaluation benchmark for secure tool invocation*
  - Curated a synthetic dataset of 5k+ security triage, patch reasoning, and multi-turn prompt injection vectors using teacher models with strict rubric validation.
  - Fine-tuned Qwen 2.5-Coder 7B via SFT + DPO using LoRA/Unsloth; achieved an 18% reduction in tool-poisoning vulnerability while maintaining 94% code-repair accuracy.
  - Served with vLLM under continuous batching and PagedAttention, achieving <80ms time-to-first-token (TTFT) and streaming structured tool invocations.
  - Released a reproducible evaluation harness benchmarked against SWE-bench and CyberSecEval with deterministic test sandboxes.

- **US Patent Application 2026/0010616 A1: Schema-Driven Test Generation Engine**  
  *First-named inventor · Assigned to Oracle International Corporation · Published Jan 2026*
  - Invented an engine that parses complex API schemas to synthesize context-specific test payloads and orchestrates layer 4–7 evaluations with automated false-positive reduction algorithms.

- **JARVIS: Multi-Agent Orchestration & Adversarial Testbed**  
  *LangGraph, Python, FastAPI, Docker*
  - Architected a multi-agent orchestration runtime with state persistence, tool calling, and network isolation.
  - Implemented an adversarial testbed evaluating confused-deputy attacks, tool poisoning, and prompt injection vectors mapped to MITRE ATLAS and OWASP GenAI Top 10.

---

### EXPERIENCE

**Oracle** — Hyderabad, India  
**Lead AI Systems & Security Engineer** *(Functional Manager, SaaS Security)* | *Mar 2021 – Present*
- Shipped an autonomous LLM defect-triage agent leveraging Codex/OpenAI models with custom skill harnesses; triages incoming scan reports and files structured defects automatically at **90% true-positive rate**, cutting triage turnaround from days to minutes.
- Designed and trained a deep neural network on ~20,000 historically triaged findings across 20 Oracle SaaS product families; achieved **92% precision and 96% recall**, reducing manual review volume by **70%**.
- Architected internal AI services using FastAPI, Redis caching, and Docker, establishing standardized schema verification and guardrail layers for internal agent tooling.
- Led black-box adversarial ML evaluations (HopSkipJump, ZOO) against classifiers, quantifying query-budget vs. evasion success boundaries to establish robust deployment thresholds.
- Spearheaded the AI systems review practice for Oracle SaaS: defined threat models, deterministic evaluation criteria, and guardrails for model-backed product features.

**Pramati Technologies** — Hyderabad, India  
**Principal Data Engineer** | *Oct 2020 – Mar 2021*
- Architected high-throughput data processing pipelines, Django services, and REST APIs for enterprise data analytics products.

**JP Morgan Chase & Co** — Hyderabad, India  
**Associate (Software & ML Engineering)** | *Jul 2018 – Oct 2020*
- Developed and deployed production NLP text-classification models using Python, scikit-learn, and vector embeddings, establishing data validation harnesses and baseline benchmark comparisons.
- Architected high-volume distributed data ingestion and streaming services on Apache Kafka and GraphQL serving downstream analytical consumers across the firm.
- Led engineering teams building test-driven backend systems with rigorous automated test suites and continuous deployment pipelines.

---

### TECHNICAL SKILLS

- **Model Post-Training & DL**: PyTorch, Hugging Face (Transformers, TRL, PEFT), SFT, LoRA/QLoRA, DPO, GRPO, Unsloth, Tokenizers (BPE), Deep Neural Networks
- **LLM Systems & Inference**: vLLM, SGLang, PagedAttention, Continuous Batching, Quantization (AWQ, GPTQ), LangGraph, Streaming Tool Invocation
- **Evaluation & AI Safety**: Deterministic Benchmark Design, LLM-as-a-Judge Calibration, Adversarial ML (HopSkipJump, ZOO), Red-Teaming (Garak, LLM Guard), MITRE ATLAS
- **Core Engineering**: Python, Java, C++ (basics), SQL, Linux, Docker, FastAPI, Apache Kafka, Git, CI/CD Automation

---

### EDUCATION

**B.E. in Instrumentation and Control Engineering**  
Manipal Institute of Technology, Manipal University (2013)
