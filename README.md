

# SciFact RAG Research Project

Investigating whether knowledge distillation in RAG pipelines causes 
student models to inherit teacher retrieval blind spots or develop 
new failure modes.

## Research Question
Does teacher distillation on RAG outputs cause the student model to 
inherit the teacher's retrieval blind spots, or does it develop 
structurally different failure modes?

## Dataset
[SciFact](https://huggingface.co/datasets/allenai/scifact) — 1,261 
scientific claims paired with 5,183 biomedical paper abstracts.

## Project Structure
- `rag_data_loading.ipynb` — Load and inspect SciFact dataset
- `day2_chunking.ipynb` — 4 chunking strategies
- `day3_indexing.ipynb` — Build vector indexes
- `day4_evaluation.ipynb` — Hit Rate and MRR benchmarking

## Setup
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
