"""Unit tests for experiment metrics, statistics, and validation logic."""

import pytest
import json
from pathlib import Path
import evaluate_workflows as eval_funcs
import analyze_results


def test_lexical_metrics_known_inputs():
    """Verify NLG metric calculations with simple known inputs."""
    ref = "the quick brown fox jumps over the lazy dog"
    cand = "the quick brown fox jumps over the lazy dog"
    
    # Exact match
    bleus = eval_funcs.calculate_bleu(ref, cand)
    assert bleus[0] == pytest.approx(1.0)
    assert bleus[3] == pytest.approx(1.0)
    
    _, _, r1 = eval_funcs.calculate_rouge_n(ref, cand, 1)
    assert r1 == pytest.approx(1.0)
    
    _, _, rl = eval_funcs.calculate_rouge_l(ref, cand)
    assert rl == pytest.approx(1.0)
    
    assert eval_funcs.word_jaccard(ref, cand) == pytest.approx(1.0)
    assert eval_funcs.cosine_similarity_tf(ref, cand) == pytest.approx(1.0)
    assert eval_funcs.levenshtein_similarity(ref, cand) == pytest.approx(1.0)

    # Partial match
    cand_partial = "the quick brown cat jumps over the lazy dog"
    assert eval_funcs.word_jaccard(ref, cand_partial) < 1.0
    assert eval_funcs.cosine_similarity_tf(ref, cand_partial) < 1.0
    assert eval_funcs.levenshtein_similarity(ref, cand_partial) < 1.0


def test_retrieval_metrics():
    """Verify precision@5, recall@5, and MRR calculations."""
    gold_ids = ["doc1", "doc2", "doc3"]
    
    # Perfect retrieval in top 3
    retrieved_1 = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    p_5, r_5, rr = analyze_results.compute_retrieval_metrics(retrieved_1, gold_ids)
    assert p_5 == pytest.approx(3 / 5)
    assert r_5 == pytest.approx(1.0)
    assert rr == pytest.approx(1.0)  # first is doc1 (rank 1)
    
    # Delayed retrieval
    retrieved_2 = ["doc4", "doc5", "doc1", "doc6", "doc7"]
    p_5, r_5, rr = analyze_results.compute_retrieval_metrics(retrieved_2, gold_ids)
    assert p_5 == pytest.approx(1 / 5)
    assert r_5 == pytest.approx(1 / 3)
    assert rr == pytest.approx(1 / 3)  # first is doc1 (rank 3)

    # Empty retrieved
    p_5, r_5, rr = analyze_results.compute_retrieval_metrics([], gold_ids)
    assert p_5 == pytest.approx(0.0)
    assert r_5 == pytest.approx(0.0)
    assert rr == pytest.approx(0.0)

    # Empty gold
    p_5, r_5, rr = analyze_results.compute_retrieval_metrics(retrieved_1, [])
    assert p_5 is None
    assert r_5 is None
    assert rr is None


def test_cliffs_delta():
    """Verify Cliff's Delta effect size calculation."""
    # Identical groups -> delta = 0
    x1 = [1, 2, 3, 4, 5]
    y1 = [1, 2, 3, 4, 5]
    assert analyze_results.cliffs_delta(x1, y1) == pytest.approx(0.0)
    
    # x strictly greater than y -> delta = 1
    x2 = [6, 7, 8]
    y2 = [1, 2, 3]
    assert analyze_results.cliffs_delta(x2, y2) == pytest.approx(1.0)
    
    # x strictly less than y -> delta = -1
    x3 = [1, 2]
    y3 = [9, 10]
    assert analyze_results.cliffs_delta(x3, y3) == pytest.approx(-1.0)

    # Mixed groups
    x4 = [1, 5, 5, 8]
    y4 = [2, 2, 4, 6]
    # pairs (x, y):
    # (1, 2) x<y, (1, 2) x<y, (1, 4) x<y, (1, 6) x<y  => 4 less
    # (5, 2) x>y, (5, 2) x>y, (5, 4) x>y, (5, 6) x<y  => 3 greater, 1 less
    # (5, 2) x>y, (5, 2) x>y, (5, 4) x>y, (5, 6) x<y  => 3 greater, 1 less
    # (8, 2) x>y, (8, 2) x>y, (8, 4) x>y, (8, 6) x>y  => 4 greater
    # Total greater: 4 + 3 + 3 = 10; Total less: 4 + 1 + 1 = 6
    # delta = (10 - 6) / 16 = 4/16 = 0.25
    assert analyze_results.cliffs_delta(x4, y4) == pytest.approx(0.25)


def test_dataset_validation(tmp_path):
    """Verify schema validator accepts good dataset and rejects malformed ones."""
    good_dataset = [
        {
            "id": "q01",
            "type": "simple",
            "query": "What is the capital of France?",
            "gold": "Paris",
            "gold_doc_ids": ["france_cap"]
        }
    ]
    # Should pass without error
    analyze_results.validate_dataset_schema(good_dataset)

    # Missing field
    bad_dataset_1 = [
        {
            "id": "q01",
            "type": "simple",
            "query": "What is the capital of France?"
            # missing gold and gold_doc_ids
        }
    ]
    with pytest.raises(ValueError):
        analyze_results.validate_dataset_schema(bad_dataset_1)

    # Invalid type
    bad_dataset_2 = [
        {
            "id": "q01",
            "type": "invalid_type",
            "query": "What is the capital of France?",
            "gold": "Paris",
            "gold_doc_ids": ["france_cap"]
        }
    ]
    with pytest.raises(ValueError):
        analyze_results.validate_dataset_schema(bad_dataset_2)

    # gold_doc_ids not a list
    bad_dataset_3 = [
        {
            "id": "q01",
            "type": "simple",
            "query": "What is the capital of France?",
            "gold": "Paris",
            "gold_doc_ids": "france_cap"
        }
    ]
    with pytest.raises(ValueError):
        analyze_results.validate_dataset_schema(bad_dataset_3)


def test_jsonl_round_trip(tmp_path):
    """Verify that writing and reading run rows preserves data types and content."""
    row = {
        "query_id": "q01",
        "query_type": "simple",
        "mode": "all",
        "repeat": 0,
        "gold": "Paris",
        "gold_doc_ids": ["france_cap"],
        "response": "Paris is capital",
        "elapsed_s": 1.25,
        "backend_ms": 1150.0,
        "route": "general",
        "timings": {"planner": 10.0, "general_qa": 1140.0},
        "steps": 2,
        "retrieved_ids": ["france_cap", "france_other"],
        "prompt_tokens": 120,
        "completion_tokens": 80,
        "success": True
    }
    
    file_path = tmp_path / "test_runs.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    assert len(lines) == 1
    read_row = json.loads(lines[0])
    
    assert read_row["query_id"] == "q01"
    assert read_row["elapsed_s"] == 1.25
    assert read_row["success"] is True
    assert read_row["gold_doc_ids"] == ["france_cap"]
    assert read_row["timings"]["planner"] == 10.0
