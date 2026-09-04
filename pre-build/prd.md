# PRD — NLP: Fine-Tune a Transformer for Sentiment / Sarcasm Detection

> Template based on `homebase/guideline.md` §1. Fill in before writing code. If any point is not clear, ASK the user.

## Project Goal / Vision
- What problem is being solved? Why does this project exist?
- *(Fill: classify text as positive/negative or sarcastic/not using a fine-tuned transformer)*

## Target Audience
- Who will use this? Skill level, platform, needs.
- *(Fill: e.g., social media analysts; demo users via web app)*

## Features / Scope
- **Must-have (MVP)**
  - Dataset loading & label distribution analysis
  - Text preprocessing + tokenization (HuggingFace tokenizers)
  - Fine-tune pretrained transformer (DistilBERT / RoBERTa)
  - Training loop with GPU support + mixed precision
  - Evaluation: accuracy, precision, recall, F1, confusion matrix
  - Prediction via API or web app
- **Should-have (v2)**
  - Class imbalance handling (weights / oversampling)
  - Explainability (LIME / attention visualization)
  - Compare 2+ model sizes
- **Won't-have**
  - *(Fill: explicitly out of scope)*

## User Stories
- As a user, I can paste a text and receive a label + confidence.
- As a user, I can see which words drove the prediction.

## Acceptance Criteria
- Achieve target F1 (e.g., ≥ 0.85 on held-out test set).
- Predicts a single text in ≤ *(fill)* seconds on local hardware.
- Web/API output includes label + probability.

## Non-Functional Requirements
- **Performance**: inference latency ≤ *(fill)*; reasonable training time on available hardware.
- **Security**: input length limits; no file path exposure.
- **Accessibility**: *(fill)*
- **Scalability**: single-user at MVP.
- **Platform**: runs locally on Windows; GPU optional, CPU fallback works.

## Success Metrics
- *(Fill: F1 / accuracy on test set, qualitative error analysis)*

## Assumptions & Constraints
- Pretrained weights downloadable from HuggingFace Hub.
- Dataset from Kaggle (Sentiment140 / Sarcasm News Headlines).
- Long training avoided on CPU; use small model if no GPU.

## Deadline / Milestones
- *(Fill: expected delivery dates per milestone)*

---
> Rule: If the PRD is not clear, ASK the user instead of guessing.