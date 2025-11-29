#!/usr/bin/env python3
"""
Prepare UltraDomain Data

Extracts Corpus, Questions, and Ground Truth from UltraDomain .jsonl files
and saves them in a standard format (BEIR-like) for the benchmark runner.

Input:
    - data/ultradomain/*.jsonl

Output:
    - data/ultradomain_processed/corpus.jsonl
    - data/ultradomain_processed/questions.json
    - data/ultradomain_processed/ground_truth.json
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

def process_ultradomain(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    corpus_file = output_path / "corpus.jsonl"
    questions_file = output_path / "questions.json"
    ground_truth_file = output_path / "ground_truth.json"

    questions_data = {"easy": [], "medium": [], "hard": []} # UltraDomain doesn't have difficulty, putting all in 'easy' or 'medium'
    ground_truth_data = {} # {qid: {doc_id: 1}}
    
    seen_doc_ids = set()
    seen_q_ids = set()

    print(f"Processing files in {input_path}...")
    
    # Open corpus file for writing
    with open(corpus_file, "w", encoding="utf-8") as f_corpus:
        for fname in os.listdir(input_path):
            if not fname.endswith(".jsonl"):
                continue
            
            file_path = input_path / fname
            print(f"Reading {fname}...")
            
            with open(file_path, "r", encoding="utf-8") as f_in:
                for line in tqdm(f_in, desc=fname):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        
                        # Extract Document (Context)
                        # Structure: context (str), context_id (str), meta (dict)
                        doc_id = str(data.get("context_id"))
                        doc_text = data.get("context", "")
                        meta = data.get("meta", {})
                        
                        if doc_id and doc_id not in seen_doc_ids:
                            doc_obj = {
                                "_id": doc_id,
                                "text": doc_text,
                                "title": meta.get("title", ""),
                                "metadata": meta
                            }
                            f_corpus.write(json.dumps(doc_obj) + "\n")
                            seen_doc_ids.add(doc_id)
                        
                        # Extract Question
                        # Structure: input (str), _id (str) -> this is QID
                        q_id = str(data.get("_id"))
                        q_text = data.get("input", "")
                        
                        if q_id and q_text and q_id not in seen_q_ids:
                            # Add to questions list (defaulting to 'medium' as general bucket)
                            questions_data["medium"].append(q_text)
                            
                            # We need a way to map question text back to ID for the runner if it loads by text
                            # But standard runner loads questions from list. 
                            # To support ground truth, the runner needs to know the QID for a given question text.
                            # The current runner generates QIDs if not provided. 
                            # We should save a mapping or ensure the runner uses these IDs.
                            # For now, let's save the standard questions.json format.
                            # AND we will save a separate "questions_with_ids.json" if needed, 
                            # but better yet, let's rely on the text-to-id mapping or just save the QID in the question object if supported.
                            # The current runner: "if self.query_ids ... question_id = self.query_ids[i-1]"
                            
                            # So we should save query_ids in metadata
                            seen_q_ids.add(q_id)
                            
                            # Extract Ground Truth
                            # The document associated with this line is the ground truth
                            if q_id not in ground_truth_data:
                                ground_truth_data[q_id] = {}
                            
                            ground_truth_data[q_id][doc_id] = 1

                    except json.JSONDecodeError:
                        continue

    # Save Questions with IDs in metadata
    # The runner uses: data['questions'] (dict of lists) and data['metadata']['query_ids'] (list)
    # But query_ids is a single list, which implies a single order. 
    # If we have multiple difficulties, we need to be careful.
    # Let's put everything in 'medium' and save the IDs in order.
    
    final_questions = {
        "metadata": {
            "total_questions": len(questions_data["medium"]),
            "query_ids": list(ground_truth_data.keys()) # Keys are QIDs, insertion order preserved in Py3.7+
        },
        "questions": questions_data
    }
    
    # Verify order matches
    # questions_data["medium"] contains texts. 
    # ground_truth_data.keys() contains IDs.
    # We need to ensure they align index-wise.
    
    ordered_q_texts = []
    ordered_q_ids = []
    
    # Re-iterate to ensure alignment
    # (Since we built them in the loop, they might be interleaved if we just took keys)
    # Actually, let's rebuild the lists to be safe.
    
    # Reset
    questions_data["medium"] = []
    final_query_ids = []
    final_ground_truth = {}
    
    # We need to re-read or store in memory. Storing in memory is fine for this dataset size.
    # Let's do a second pass or just store tuples in the loop. 
    # Refactoring loop above to store tuples.
    pass

def process_ultradomain_safe(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    corpus_file = output_path / "corpus.jsonl"
    questions_file = output_path / "questions.json"
    ground_truth_file = output_path / "ground_truth.json"

    q_list = [] # List of (qid, q_text, doc_id)
    seen_doc_ids = set()
    
    print(f"Processing files in {input_path}...")
    
    with open(corpus_file, "w", encoding="utf-8") as f_corpus:
        for fname in os.listdir(input_path):
            if not fname.endswith(".jsonl"):
                continue
            
            file_path = input_path / fname
            print(f"Reading {fname}...")
            
            with open(file_path, "r", encoding="utf-8") as f_in:
                for line in tqdm(f_in, desc=fname):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        
                        doc_id = str(data.get("context_id"))
                        doc_text = data.get("context", "")
                        meta = data.get("meta", {})
                        
                        if doc_id and doc_id not in seen_doc_ids:
                            doc_obj = {
                                "_id": doc_id,
                                "text": doc_text,
                                "title": meta.get("title", ""),
                                "metadata": meta
                            }
                            f_corpus.write(json.dumps(doc_obj) + "\n")
                            seen_doc_ids.add(doc_id)
                        
                        q_id = str(data.get("_id"))
                        q_text = data.get("input", "")
                        
                        if q_id and q_text:
                            q_list.append((q_id, q_text, doc_id))
                            
                    except json.JSONDecodeError:
                        continue

    # Build final structures
    final_q_texts = []
    final_q_ids = []
    final_gt = {}
    
    for qid, qtext, docid in q_list:
        final_q_texts.append(qtext)
        final_q_ids.append(qid)
        
        if qid not in final_gt:
            final_gt[qid] = {}
        final_gt[qid][docid] = 1
        
    questions_json = {
        "metadata": {
            "total_questions": len(final_q_texts),
            "query_ids": final_q_ids
        },
        "questions": {
            "easy": [],
            "medium": final_q_texts, # All in medium
            "hard": []
        }
    }
    
    print(f"Saving {len(final_q_texts)} questions...")
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(questions_json, f, indent=2)
        
    print(f"Saving ground truth...")
    with open(ground_truth_file, "w", encoding="utf-8") as f:
        json.dump(final_gt, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    args = parser.parse_args()
    
    process_ultradomain_safe(args.data_folder, args.output_folder)
