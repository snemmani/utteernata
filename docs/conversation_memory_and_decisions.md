# Conversation Memory & Strategic Decisions Log

**User:** Shankar Nag Nemmani  
**Location:** Hyderabad, India  
**Date Initiated:** November 2, 2026  
**Project:** Utteernata (Career Pivot to Tier-1 Applied AI & Post-Training Builder)  
**Primary Goal:** Transition from an enterprise security management role into a top-tier Applied AI / Post-Training Research & Builder role at frontier AI labs or tier-1 R&D centers, avoiding offshore/vendor delegated maintenance loops.

---

## 1. Baseline Audit & Key Diagnoses

### Initial State
* **Experience:** 13 years across software, data, and security engineering; currently Manager, SaaS Security at Oracle India.
* **Key Achievements:** Shipped LLM defect-triage agent (90% TPR), trained deep neural network on 20k static-analysis findings (92% precision, 96% recall), first-named inventor on US Patent Application 2026/0010616 A1 (schema-driven testing).
* **Pain Point:** Not getting interview calls; felt rusty; worried about clearing top AI/ML research company technical screens.
* **Deep Learning Baseline:** User acknowledged knowing a little bit of TensorFlow/Keras, but **zero prior PyTorch experience**.

### Root Cause of Why Calls Were Blocked
1. **Title & Categorization Trap:** Recruiters and ATS parsed the resume as "Enterprise AppSec / SecOps Manager" due to recent titles and legacy security tool keywords (BurpSuite, Fortify, CEH, ITIL).
2. **Dated Stack Signals:** Mentions of Gensim, NLTK, Keras, CEH, and TCS Perl migrations signaled a 2018 enterprise profile rather than modern LLM systems engineering.
3. **Offshore Perception:** Standard applications from India GCCs without high-visibility public artifacts often get routed to regional IT vendor maintenance tracks.
4. **The Unfair Advantage (Untapped):** Deep intersection of **Adversarial ML + LLM Red-Teaming + Systems Testing + Patent**—one of the highest-demand niches in AI safety and reasoning if framed properly.

---

## 2. Strategic Decisions Recorded (/grill-me Alignment)

| Decision Point | Chosen Option | Rationale |
| :--- | :--- | :--- |
| **Target Role Archetype** | **Applied AI Builder + Post-Training Research Engineer** | Combines agentic harness engineering (vLLM, tools, evals) with post-training alignment (SFT, LoRA, DPO, GRPO). |
| **Technical Rustiness Baseline** | **All 5 Core Areas** | Acknowledged need to rebuild PyTorch internals, post-training pipelines, serving systems, DSA/LeetCode, and eval benchmarks. |
| **Time Commitment** | **30+ Hours per Week** | Full immersion mode: early mornings, evenings, and full weekend engineering blocks (~400 hours over 13 weeks). |
| **Flagship Project Strategy** | **AEGIS: Post-Trained Reasoning Agent + Adversarial Robustness & Evals** | High-signal, reproducible open-source artifact combining fine-tuning (Qwen 2.5-Coder), vLLM serving, and a defensive eval benchmark against prompt injection. |
| **Hardware Environment** | **Local NVIDIA RTX 5070 (12GB VRAM) + 64GB RAM** + On-demand Cloud credits | Prototyping, inference, and QLoRA 4-bit fine-tuning (via Unsloth) locally; cloud A100s on RunPod for heavy DPO/multi-GPU runs. |
| **Company Targeting** | **Dual-Track (Global Remote + Tier-1 R&D in India/Relocation)** | Targeting Anthropic, Scale AI, Mistral, Databricks, Uber AI, Google DeepMind India, Microsoft GenAI India, Protect AI, Lakera. |
| **Interview Preparation** | **Split Focus (1 hr DSA daily + PyTorch live-coding)** | Daily LeetCode patterns combined with live-coding Transformer components from scratch. |
| **Resume & Branding Overhaul** | **Aggressive Modernization** | Retitle role to *Lead AI Systems & Security Engineer*, prune CEH, ITIL, TCS Perl, and SAST analyst tooling; foreground PyTorch, vLLM, and post-training. |

---

## 3. Milestones & Checkpoints

- **Month 1 (Weeks 1–4):** PyTorch Foundations (from zero), Transformer architecture from scratch, KV Cache, RoPE, daily LeetCode patterns, and finalized 1-page modernized resume.
- **Month 2 (Weeks 5–8):** Parameter-efficient fine-tuning (LoRA/QLoRA with Unsloth on RTX 5070), DPO/GRPO alignment, vLLM high-throughput serving, and Flagship Project Phase 1 (dataset + training).
- **Month 3 (Weeks 9–12):** Flagship Project Phase 2 & 3 (agent harness, deterministic eval benchmark, technical paper/blog), ML systems design, mock interviews, and warm outbound referral engine.

---

## 4. Key Files in this Repository

* `README.md` - High-level repo summary and getting started guide.
* `docs/90_day_ai_research_builder_masterplan.md` - Full 12-week syllabus, schedule, and study materials.
* `docs/revamped_resume_shankar_nemmani.md` - The ATS-optimized 1-page modern resume.
* `docs/pytorch_zero_to_hero_guide.md` - Dedicated TensorFlow-to-PyTorch primer and coding guide.
* `src/day01_tensors_autograd.py` - Day 1 executable PyTorch script testing GPU, autograd, and training loops.
