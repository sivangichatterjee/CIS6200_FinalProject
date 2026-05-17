# Error Distribution Shift in Knowledge-Distilled RAG Systems

Investigating whether knowledge distillation in RAG pipelines causes student models to inherit teacher retrieval blind spots or develop structurally different failure modes.

## Research Question

Does knowledge distillation in a RAG pipeline cause the student model to inherit the teacher's retrieval blind spots, or does it develop structurally different failure modes?

## Key Findings

- Teacher (Llama 3.3 70B) achieves **79.9% accuracy** on 338 validation claims
- Student (Llama 3.2 1B, LoRA fine-tuned) achieves **65.7% accuracy**, a 14.2 percentage point gap
- The student does not simply inherit the teacher's failure modes. It develops a qualitatively new one: hallucination (FP7) increases **30x** from 0.3% to 8.9%
- Both models fail primarily by ignoring retrieved evidence (FP4: 17.5% teacher vs. 22.5% student), confirming the generation side is the bottleneck, not retrieval
- The student amplifies the teacher's existing CONTRADICT under-prediction bias, predicting SUPPORT 78.4% of the time vs. 63.9% ground truth frequency

## Dataset

[SciFact](https://huggingface.co/datasets/allenai/scifact) -- 5,183 biomedical paper abstracts and 1,261 annotated scientific claims. Each claim is labeled SUPPORT or CONTRADICT relative to a specific evidence document in the corpus. 957 training claims and 338 validation claims have ground-truth evidence annotations.

## Project Structure

```
RAG_KD/
  rag_data_loading.ipynb        -- Load and inspect SciFact dataset, save corpus.json and claims files
  rag_chunking.ipynb            -- Six chunking strategies: document, sentence, fixed 128, recursive 128, recursive 256, semantic
  rag_indexing.ipynb            -- Build FAISS indexes for all six strategies using bge-small-en-v1.5
  rag_retrieval_eval_fixed.ipynb -- Hit Rate and MRR benchmarking across all strategies, select winner
  day6_teacher.ipynb            -- Teacher pipeline: retrieve evidence, call Llama 3.3 70B via Together AI, log responses
  student_finetuning.ipynb      -- LoRA fine-tuning of Llama 3.2 1B on teacher logs, student inference on val claims
  day11_failure_analysis.ipynb  -- LLM-as-judge evaluation, Seven Failure Points categorization, comparison chart

  chunks/                       -- Chunk JSON files for each strategy
  indexes/                      -- FAISS index files and chunk metadata
  logs/
    teacher_logs.json           -- 957 teacher-generated responses on train claims
    teacher_val_logs.json       -- 338 teacher responses on validation claims
    student_logs.json           -- 338 student responses on validation claims
    teacher_judgments.json      -- Judge categorizations for teacher failures
    student_judgments.json      -- Judge categorizations for student failures
    training_data.json          -- Formatted training pairs for LoRA fine-tuning
    failure_comparison.png      -- 2x7 failure distribution chart (key result figure)
```

## Retrieval Benchmark Results

| Strategy | Chunks | Hit@5 | MRR@5 |
|---|---|---|---|
| Document-level | 5,183 | 0.893 | 0.800 |
| Sentence-level | 45,952 | 0.929 | 0.807 |
| Fixed 128 | 17,400 | 0.891 | 0.796 |
| **Recursive 128 (winner)** | **18,638** | **0.923** | **0.813** |
| Recursive 256 | 9,905 | 0.914 | 0.806 |
| Semantic | 10,594 | 0.899 | 0.799 |

Recursive 128-token chunking was selected for all downstream experiments.

## Tech Stack

- **Embedding model**: BAAI/bge-small-en-v1.5 (384 dimensions, cosine similarity)
- **Vector store**: FAISS IndexFlatIP with L2-normalized embeddings
- **Chunking**: LangChain RecursiveCharacterTextSplitter and SemanticChunker
- **Teacher model**: Llama 3.3 70B Instruct Turbo via Together AI API
- **Student model**: Llama 3.2 1B Instruct, fine-tuned with Unsloth + LoRA (r=16, 3 epochs)
- **Judge model**: Llama 3.3 70B via Together AI
- **Training hardware**: NVIDIA RTX PRO 6000 Blackwell (102GB VRAM) on Google Colab Pro

## Setup

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set API keys before running teacher or judge notebooks:

```bash
export TOGETHER_API_KEY="your_key_here"
```

## Reproduction Order

Run notebooks in this order to reproduce all results:

1. `rag_data_loading.ipynb` -- produces `corpus.json`, `claims_train.json`, `claims_val.json`
2. `rag_chunking.ipynb` -- produces `chunks/*.json`
3. `rag_indexing.ipynb` -- produces `indexes/`
4. `rag_retrieval_eval_fixed.ipynb` -- produces `eval/retrieval_metrics.csv`
5. `day6_teacher.ipynb` -- produces `logs/teacher_logs.json`, `logs/teacher_val_logs.json`, `logs/training_data.json`
6. `student_finetuning.ipynb` -- produces `logs/student_logs.json` (requires Colab GPU)
7. `day11_failure_analysis.ipynb` -- produces `logs/*_judgments.json`, `logs/failure_comparison.png`

## Authors

Sivangi Chatterjee, Amanda Huie, Sanskriti Sarkar
University of Pennsylvania, CIS 6200 Advanced Machine Learning, 2025
