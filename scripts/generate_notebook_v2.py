import json
import os

# Define the notebook structure as a Python dictionary
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "id": "intro",
   "metadata": {},
   "source": [
    "# 🧪 Evaluation & Benchmarking\n",
    "\n",
    "A professional RAG system isn't just about getting answers—it's about measuring quality.\n",
    "This notebook demonstrates how to benchmark CUBO using **Information Retrieval (IR) metrics** (Recall, MRR) and **Generative Metrics** (Faithfulness, Relevance).\n",
    "\n",
    "## What You'll Learn\n",
    "1. Create a synthetic \"Ground Truth\" dataset.\n",
    "2. Run retrieval evaluation (Recall@K).\n",
    "3. Run generative evaluation (RAGAS).\n",
    "4. Analyze the results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "setup",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "# Add CUBO to path\n",
    "cubo_root = Path(\".\").resolve().parent\n",
    "if str(cubo_root) not in sys.path:\n",
    "    sys.path.insert(0, str(cubo_root))\n",
    "\n",
    "from cubo.core import CuboCore\n",
    "from cubo.evaluation.metrics import IRMetricsEvaluator\n",
    "import pandas as pd\n",
    "\n",
    "# Initialize core engine (Laptop Mode auto-detected)\n",
    "core = CuboCore()\n",
    "core.initialize_components()\n",
    "print(\"✅ CUBO Core Initialized\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dataset",
   "metadata": {},
   "source": [
    "## 1️⃣ Create Ground Truth Dataset\n",
    "\n",
    "For evaluation, we need `(Query, Relevant_Doc_ID)` pairs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "create-data",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create dummy corpus\n",
    "documents = [
",
    "    {\"text\": \"The Eiffel Tower is located in Paris, France.\", \"id\": \"doc_paris\", \"file_path\": \"paris.txt\"},\n",
    "    {\"text\": \"The Colosseum is an ancient amphitheater in Rome, Italy.\", \"id\": \"doc_rome\", \"file_path\": \"rome.txt\"},\n",
    "    {\"text\": \"Sushi is a traditional dish from Japan.\", \"id\": \"doc_japan\", \"file_path\": \"japan.txt\"},\n",
    "    {\"text\": \"Pizza originated in Naples, Italy.\", \"id\": \"doc_naples\", \"file_path\": \"naples.txt\"}
",
    " ]\n",
    "\n",
    "# Index documents\n",
    "core.add_documents(documents)\n",
    "print(f\"✅ Indexed {len(documents)} documents\")\n",
    "\n",
    "# Define Ground Truth (Query -> List of Relevant Doc IDs)\n",
    "ground_truth = {
",
    "    \"Where is the Eiffel Tower?\": [\"doc_paris\"]},\n",
    "    \"Tell me about Italian food\": [\"doc_rome\", \"doc_naples\"]},\n",
    "    \"Famous buildings in Italy\": [\"doc_rome\"]},\n",
    "    \"Japanese cuisine\": [\"doc_japan\"]
",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "retrieval-eval",
   "metadata": {},
   "source": [
    "## 2️⃣ Evaluate Retrieval (IR Metrics)\n",
    "\n",
    "We calculate **Recall@K** (did we find it?) and **MRR** (was it the first result?)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "run-retrieval",
   "metadata": {},
   "outputs": [],
   "source": [
    "results = []\n",
    "\n",
    "for query, relevant_ids in ground_truth.items():\n",
    "    # Retrieve top 3\n",
    "    retrieved = core.query_retrieve(query, top_k=3)\n",
    "    retrieved_ids = [doc['metadata'].get('id') for doc in retrieved]\n",
    "    \n",
    "    # Calculate metrics\n",
    "    metrics = IRMetricsEvaluator.evaluate_retrieval(\n",
    "        question_id=query,\n",
    "        retrieved_ids=retrieved_ids,\n",
    "        ground_truth={query: relevant_ids},\n",
    "        k_values=[1, 3]\n",
    "    )\n",
    "    \n",
    "    results.append({\n",
    "        \"query\": query,\n",
    "        \"hits\": retrieved_ids,\n",
    "        \"recall@1\": metrics[\"recall_at_k\"][1],\n",
    "        \"recall@3\": metrics[\"recall_at_k\"][3],\n",
    "        \"mrr\": metrics[\"mrr\"]\n",
    "    })\n",
    "\n",
    "# Display Results\n",
    "df = pd.DataFrame(results)\n",
    "print(f\"Average Recall@3: {df['recall@3'].mean():.2f}\")\n",
    "print(f\"Average MRR: {df['mrr'].mean():.2f}\")\n",
    "display(df)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ragas-eval",
   "metadata": {},
   "source": [
    "## 3️⃣ Evaluate Generation (RAGAS)\n",
    "\n",
    "We use the Local LLM as a judge to verify if the answer is faithful to the context.\n",
    "\n",
    "> **Note:** This requires `ragas` installed. If running on a laptop CPU, this step might be slow."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "run-ragas",
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    from cubo.evaluation.ragas_evaluator import run_ragas_evaluation\n",
    "    \n",
    "    # Prepare data for RAGAS\n",
    "    questions = list(ground_truth.keys())\n",
    "    # Note: Flattened list of strings for ground truths if possible, but RAGAS accepts lists\n",
    "    ground_truths_list = [[\"Paris, France\"], [\"Pizza and Colosseum\"], [\"Colosseum\"], [\"Sushi\"]]
",
    "    \n",
    "    # Generate real answers\n",
    "    contexts = []\n",
    "    answers = []\n",
    "    \n",
    "    print(\"Generating answers for RAGAS...\")\n",
    "    for q in questions:\n",
    "        res = core.query_and_generate(q, top_k=2)\n",
    "        answers.append(res['answer'])\n",
    "        contexts.append([d['document'] for d in res['sources']])\n",
    "\n",
    "    # Run Evaluation\n",
    "    scores = run_ragas_evaluation(\n",
    "        questions=questions,\n",
    "        contexts=contexts,\n",
    "        ground_truths=ground_truths_list,\n",
    "        answers=answers\n",
    "    )\n",
    "    \n",
    "    print(\"\n\")\n",
    "    print(\"📊 RAGAS Scores:\")\n",
    "    print(scores)\n",
    "except ImportError:\n",
    "    print(\"ℹ️ RAGAS not installed. Skipping generation eval.\")\n",
    "except Exception as e:\n",
    "    print(f\"⚠️ RAGAS Eval skipped (requires heavier compute): {e}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "conclusion",
   "metadata": {},
   "source": [
    "## 🎯 Conclusion\n",
    "\n",
    "This workflow allows you to scientifically measure the impact of changes (like switching embedding models or enabling reranking) on your specific data."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

os.makedirs("examples", exist_ok=True)
with open("examples/04_evaluation_benchmarking.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Notebook generated successfully.")
