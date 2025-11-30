🤖 Chatbot + Recommendation System — Architecture, Training, and Enterprise Practices

A comprehensive study note summarizing key concepts in Chatbot × Recsys × RAG × LLM × Offline RL.
Author: Yingzuo Liu
Last Updated: 2025-11

📘 Overview

This repository summarizes my learning and understanding of Chatbot × Recommendation System design, especially in e-commerce / customer service settings.

Covered topics include:

Retrieval-first Chatbot architecture (widely used in customer service)

Generation-first Chatbot architecture (LLM Agent style)

Multi-intent & multi-action scoring

RAG hallucination control

Dual-encoder retrieval, FAISS, MLP reranking

Multi-turn dialogue & intent drift

IPS / DR / SNIPS for unbiased offline evaluation

Latency optimization

Training methods: SFT / DPO / PPO / GRPO

Why dual-encoder instead of cross-encoder

How Chatbot integrates with Recsys design

This is a structured and interview-ready summary.

🏛 1. Enterprise Chatbot Architectures

Two common enterprise designs:

A. Retrieval-first

B. Generation-first

Each serves a different purpose and is widely used in large-scale production systems.

🅰 Retrieval-first Architecture（检索优先）

📌 90% of customer service systems use this.
Ideal for: FAQ, refund, logistics, policy QA, e-commerce customer support, hybrid recsys.

Core Idea

Retrieve first → optional generation.
LLM is not the knowledge source. It only rewrites surfaces.

Pipeline
User Query
  ↓
NLU (intent classifier) + Encoder (text embedding)
  ↓
FAISS / Milvus vector retrieval (Top-K)
  ↓
Reranker (MLP or lightweight transformer)
  ↓
LLM converts retrieved knowledge to natural language (optional)

Advantages

High safety

Deterministic and controllable

No hallucination

Fast latency (ms-level)

Knowledge easily updated (update vector DB only)

Perfect fit for customer support & enterprise QA

Enterprise Characteristics

LLM is post-processing, not the decision-maker

MLP reranker is often enough (cheap & fast)

Strict filters for safety

Multi-intent & multi-action scoring for business logic

When to use

Refund/return policy

Logistics tracking

After-sale Q&A

Recsys recall + reranking

Corporate knowledge base search

Large-scale, low-latency production

🅱 Generation-first Architecture（生成优先）

📌 Used for: LLM Agents, workflow automation, complex multi-turn reasoning.

Core Idea

LLM first plans → then decides whether to retrieve, call tools, or act.

LLM acts as:

Planner

Reasoner

Tool orchestrator

State machine

Pipeline
User Query
  ↓
LLM performs planning + intent reasoning
  ↓
LLM decides:
    - retrieve or not?
    - call tools/APIs?
    - clarify missing info?
    - return final answer?
  ↓
Execute sub-steps (RAG / Tools)
  ↓
LLM integrates results and responds

Enterprise Characteristics

Designed for complex multi-turn interactions

Tool calling is core

Strong need for rule-engine + schema validation

Works well with automation workflows

More flexible, but less stable

When to use

Intelligent CS assistant (automated refund, auto-generate forms)

Multi-step tasks (address change + delivery change)

Complex business rules with conditional logic

Enterprise internal agent (Jira/SAP/Confluence integration)

🆚 Retrieval-first vs Generation-first
Comparison	Retrieval-first	Generation-first
Philosophy	Select the correct answer	Plan and execute
Reliability	Very high	Medium (requires constraints)
Latency	Low	High
Hallucination	Near-zero	Possible
Multi-turn reasoning	Limited	Strong
Tool calling	Optional	Core
Recsys integration	Excellent	Decent
Knowledge updates	Fast	Requires retraining or prompts
Use Case	Customer support, FAQ, Recsys	Agent automation, multi-step tasks
Practical rule:

Need correctness → Retrieval-first
Need reasoning/automation → Generation-first

🧠 2. Multi-intent & Multi-action Scoring

Customer queries often contain multiple intents:

“Refund + logistics check”

“Address change + compensation”

“Return + stock availability”

Your approach:

Multi-label scoring
score(intent_i | query, history)
score(action_j | context, selected_intents)


Keep top intents & actions → final decision uses business rules + confidence.

🔁 3. Multi-turn Dialogue & Intent Drift

Intent changes over turns.

Inspired by DIN/DIEN idea:

Attention selects relevant past actions

Gating captures interest drift

Transformer > RNN for stability & long context

🔍 4. Vector Retrieval (FAISS)
Why not cross-encoder?

Requires concatenation and transformer for each candidate → slow

Cannot pre-build index

High latency

Why dual-encoder?

Build vector DB offline

Dot-product retrieval (1–3ms)

Scale to millions of items

Suitable for multilingual alignment

🎯 5. Ranking (Why MLP is Enough?)

Embedding already encodes semantics.

MLP advantages:

microsecond inference

simple & robust

avoids heavy transformer reranker cost

common in YouTube, TikTok recommender stacks

📚 6. RAG Hallucination Control

Three-layer safety mechanisms:

① Semantic Confidence

Embedding margin too small → uncertain.

② Rule-based validation

Amount must be positive

Dates must be valid

Refund policy must match DB structure

③ Fallback to Human

Low score → escalate to agent.

🧮 7. IPS / DR / SNIPS（偏差校正 & Offline RL）

Interactions are biased: user only clicks what old system shows.

IPS

Importance sampling adjusts for old-policy bias.

DR

IPS + model estimation → lower variance, more stable.

SNIPS

Normalized IPS to avoid huge weights.

Usage

Estimate new agent's reward safely

Adjust intent/action model learning

Reduce bias in recommendation-reranker training

⚙️ 8. Latency Optimization

Practical tips you applied:

INT8/FP16 model quantization

FAISS IVF-PQ / HNSW

Redis warm vector cache

MLP reranker only

LLM called after retrieval

Context window control

🧪 9. Training Strategies
SFT

Align tone, format, persona, safety.

DPO / ORPO

Align model to preference pairs (politeness, safety, correctness).

PPO / GRPO

Reinforcement learning:

reward = task completion + safety + customer satisfaction

GRPO minimizes KL drift + natural gradient updates

🧩 10. Combined System Architecture

Unified design combining both LLM and Recsys:

            ┌──────────────────────────────┐
            │        User Query            │
            └──────────────────────────────┘
                          ↓
        NLU + Encoder + Context Tracking
                          ↓
        Multi-intent & Multi-action Scoring
                          ↓
      ┌──────────── Retrieval-first ────────────┐
      │  Vector DB → FAISS → MLP Rerank → LLM    │
      └──────────────────────────────────────────┘
                          ↓
      ┌──────────── Generation-first ───────────┐
      │ LLM Planning → Tool Calling → RAG       │
      └─────────────────────────────────────────┘
                          ↓
                     Final Answer
                          ↓
                   Safety Validation
