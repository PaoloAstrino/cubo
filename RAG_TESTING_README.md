# CUBO RAG Testing Framework

This directory contains a comprehensive testing framework for evaluating your RAG (Retrieval-Augmented Generation) system's performance.

## Files

- `test_questions.json` - 150 test questions across 3 difficulty levels
- `run_rag_tests.py` - Python script to run systematic tests
- `test_results.log` - Log file for test execution details

## Question Categories

### Easy Questions (50)
- Basic factual recall
- Simple "what/who/where" questions
- Direct information retrieval

### Medium Questions (50)
- Require inference and analysis
- Understanding relationships between concepts
- Moderate synthesis of information

### Hard Questions (50)
- Deep analysis and critical thinking
- Thematic understanding
- Complex synthesis across multiple concepts

## Usage

### Quick Test Run
```bash
python run_rag_tests.py
```

### Limited Testing
```bash
# Test only 10 questions from each difficulty
python run_rag_tests.py --easy-limit 10 --medium-limit 10 --hard-limit 10
```

### Custom Output
```bash
python run_rag_tests.py --output my_test_results.json
```

## Test Results

Results are saved to `test_results.json` with:
- Success rates by difficulty level
- Processing times
- Individual question results
- Overall performance metrics

## Integration

To integrate with your actual RAG system, modify the `run_single_test` method in `run_rag_tests.py` to call your RAG pipeline instead of the mock response.

## Question Sources

Questions are based on the animal stories in the `data/` directory:
- Cat story (Whiskers' adventures)
- Dog story (Buddy's loyalty)
- Frog story (Hopper's heroism)
- Other animal stories

## Performance Metrics

The framework tracks:
- Question success rate
- Processing time per question
- Performance by difficulty level
- Overall system reliability