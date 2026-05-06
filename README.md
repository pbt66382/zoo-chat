# zoo-chat

Design notes for a **universal AI customer service** layer that can support a broad hardware and software portfolio—similar in spirit to how a company like Zoom might serve **video phones, meeting software, headsets, mice, meeting room displays, and calling services** with one coherent assistant experience.

---

## 1. Goal

Build one AI front door that:

- Understands **what the customer is trying to do** (intent), not only which SKU they mention.
- Routes to the **right product context** (meetings vs. devices vs. telephony) without forcing the user to pick a category first.
- Answers from **trusted knowledge** (docs, policies, firmware notes) and **live systems** (order status, subscription, device registration) where APIs exist.
- **Escalates cleanly** to human agents with a structured handoff when confidence is low or the case is sensitive (billing disputes, security, legal).

---

## 2. Why “universal” is hard

Products differ in **symptoms** (e.g., “no audio in a meeting” vs. “headset won’t pair”), **data** (account vs. serial number), and **resolution paths** (app settings vs. driver vs. RMA). A single model prompt is not enough; you need **shared orchestration** plus **per-domain knowledge and tools**.

---

## 3. Recommended approach (high level)

### A. Shared conversation layer

- **Intent detection and slot filling**: map utterances to intents (setup, connectivity, billing, feature how-to, incident) and slots (product line, OS, error text).
- **Product router**: classify or retrieve the most likely product line(s); allow **clarifying questions** only when needed.
- **Policy and tone layer**: brand voice, legal disclaimers, and “do not guess” rules for hardware safety or regulated claims.

### B. Knowledge per domain (not one blob)

- **Structured product catalog**: attributes, compatibility matrices, known issues by firmware version.
- **Retrieval (RAG)** over manuals, FAQs, and internal runbooks; **separate indexes or metadata filters** per product family to reduce cross-product confusion.
- **Version-aware answers**: meetings software changes quickly; pin answers to **app version** or **release channel** when possible.

### C. Tools and integrations

- Read-only tools first: **order / entitlement lookup**, **device registration**, **ticket creation**, **link to downloads**.
- **Human handoff** tool: pass transcript, intent, slots, and suggested next steps to the CRM or agent desktop.

### D. Quality and safety

- **Grounding**: cite or internally trace answers to retrieved chunks or tool outputs; refuse when evidence is missing.
- **PII handling**: minimize collection; redact logs; regional retention rules.
- **Evaluation**: golden questions per product line, regression sets after each knowledge or model change.

---

## 4. Phased rollout

1. **MVP**: one channel (e.g., web chat), top intents, FAQ + doc RAG, ticket creation, human escalation.
2. **Expand**: add tools per product (headset pairing flows, room kit diagnostics scripts where safe).
3. **Optimize**: routing quality, proactive diagnostics (with user consent), multilingual support, voice channel if needed.

---

## 5. Success metrics

- **Containment** with quality (not raw deflection): resolved without human, verified by sampling or CSAT.
- **First-contact resolution** and **average handle time** for escalated cases (handoff package should shorten agent time).
- **Wrong-product rate**: how often the assistant answers as if the issue were for another line—keep this near zero.

---

This repository is a starting point for product and engineering discussion; implementation stacks (e.g., LangGraph, CRM APIs, vector DB) can be chosen to match your existing ecosystem.
