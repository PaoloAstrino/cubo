# CUBO Evaluation System

A comprehensive evaluation and analytics system for RAG (Retrieval-Augmented Generation) applications.

## Overview

The CUBO Evaluation System provides detailed metrics and analytics to monitor and improve your RAG application's performance. It goes beyond basic evaluation to provide actionable insights and comprehensive data storage.

## Features

### 📊 Core Metrics (RAG Triad)

- **Answer Relevance**: How well the answer addresses the question
- **Context Relevance**: How relevant retrieved contexts are to the query
- **Groundedness**: How well the answer is supported by the contexts

### 🔍 Advanced Metrics

- **Answer Quality**: Length, readability, structure analysis
- **Context Utilization**: Diversity, coverage, efficiency metrics
- **Response Efficiency**: Speed, throughput, resource usage
- **Information Completeness**: Coverage and alignment analysis

### 📈 Analytics & Insights

- **Performance Trends**: Track improvements over time
- **Error Analysis**: Identify common failure patterns
- **Quality Insights**: Automated recommendations for improvement
- **Comparative Analysis**: Benchmark against historical performance

### 💾 Data Storage

- **SQLite Database**: Persistent storage of all evaluation data
- **Comprehensive Schema**: Store questions, answers, contexts, metadata
- **Export Capabilities**: CSV/JSON export for external analysis
- **Session Tracking**: Group evaluations by user sessions

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   QueryWidget   │───▶│ Evaluation      │───▶│   Database      │
│                 │    │ Integration     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Advanced Metrics │    │   Dashboard     │
                       │ & Analytics      │    │   GUI           │
                       └──────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Automatic Evaluation

Every query in CUBO is automatically evaluated:

```python
# Queries are automatically evaluated when processed
# Results stored in evaluation/evaluation.db
```

### 2. View Dashboard

```python
from evaluation.dashboard import show_evaluation_dashboard
show_evaluation_dashboard()
```

### 3. Manual Evaluation

```python
from evaluation.integration import evaluate_query_sync

result = evaluate_query_sync(
    question="What is AI?",
    answer="AI is artificial intelligence...",
    contexts=["AI definition...", "ML explanation..."],
    response_time=1.5
)
```

## Database Schema

The evaluation database stores comprehensive data:

```sql
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    session_id TEXT,
    question TEXT,
    answer TEXT,
    response_time REAL,
    contexts TEXT,           -- JSON array
    context_metadata TEXT,   -- JSON array
    model_used TEXT,
    embedding_model TEXT,
    retrieval_method TEXT,
    chunking_method TEXT,

    -- RAG Triad Scores
    answer_relevance_score REAL,
    context_relevance_score REAL,
    groundedness_score REAL,

    -- Advanced Metrics
    answer_length INTEGER,
    context_count INTEGER,
    total_context_length INTEGER,
    average_context_similarity REAL,
    answer_confidence REAL,

    -- Quality Flags
    has_answer BOOLEAN,
    is_fallback_response BOOLEAN,
    error_occurred BOOLEAN,
    error_message TEXT,

    -- User Feedback
    user_rating INTEGER,
    user_feedback TEXT
);
```

## API Reference

### EvaluationDatabase

```python
from evaluation.database import EvaluationDatabase

db = EvaluationDatabase()

# Store evaluation
db.store_evaluation(evaluation)

# Get recent evaluations
recent = db.get_recent_evaluations(limit=50)

# Get metrics summary
metrics = db.get_metrics_summary(days=30)

# Export data
db.export_to_csv("evaluation_data.csv", days=30)
```

### AdvancedEvaluator

```python
from evaluation.metrics import AdvancedEvaluator

evaluator = AdvancedEvaluator()

# Comprehensive evaluation
metrics = await evaluator.evaluate_comprehensive(
    question, answer, contexts, response_time
)

# Individual metrics
quality = evaluator.evaluate_answer_quality(answer)
utilization = evaluator.evaluate_context_utilization(question, contexts)
```

### EvaluationIntegrator

```python
from evaluation.integration import get_evaluation_integrator

integrator = get_evaluation_integrator(generator, retriever)

# Evaluate query
result = await integrator.evaluate_query(
    question, answer, contexts, response_time
)

# Get summary
metrics = integrator.get_metrics_summary(days=30)
```

## Dashboard Features

### 📈 Key Metrics Overview

- Total queries and success rates
- Average RAG triad scores
- Response time statistics
- Error rate tracking

### 📋 Recent Evaluations Table

- Chronological list of recent queries
- Color-coded performance scores
- Response time and status indicators
- Searchable and filterable

### 📊 Performance Trends

- Daily performance metrics
- Trend analysis with slope calculations
- Best/worst performing days
- Error rate trends

### 💡 Insights & Recommendations

- Automated insights from data patterns
- Actionable recommendations for improvement
- Performance bottleneck identification
- Quality improvement suggestions

### 📤 Data Export

- CSV and JSON export formats
- Configurable date ranges
- Comprehensive data inclusion
- Ready for external analysis

## Configuration

Evaluation settings can be configured in `config.json`:

```json
{
  "evaluation": {
    "enabled": true,
    "auto_evaluate": true,
    "store_contexts": true,
    "max_contexts_stored": 10,
    "export_format": "json"
  }
}
```

## Integration with CUBO

The evaluation system integrates seamlessly with CUBO:

1. **Automatic Evaluation**: Every query is evaluated in the background
2. **GUI Integration**: Evaluation menu in main window
3. **Service Manager**: Uses existing error recovery and async execution
4. **Configuration**: Leverages existing config system

## Performance Considerations

- **Async Evaluation**: Doesn't block query responses
- **Efficient Storage**: Optimized database schema with indexes
- **Batch Processing**: Handles multiple evaluations efficiently
- **Memory Management**: Limits stored context size

## Troubleshooting

### Common Issues

1. **Database Locked**: Close other instances accessing the database
2. **Missing Dependencies**: Install pandas: `pip install pandas`
3. **Large Database**: Export old data and create new database

### Logs

Evaluation logs are stored in the main CUBO log file with the `evaluation` prefix.

## Future Enhancements

- **User Feedback Integration**: Allow users to rate responses
- **A/B Testing**: Compare different model configurations
- **Real-time Alerts**: Notify when performance drops
- **Advanced Visualizations**: Charts and graphs for trends
- **Model Comparison**: Evaluate different models side-by-side

## Contributing

The evaluation system is designed to be extensible. Add new metrics by:

1. Extending `AdvancedEvaluator` with new methods
2. Adding columns to the database schema
3. Updating the dashboard to display new metrics
4. Adding export support for new data

---

**For questions or issues, check the main CUBO documentation or create an issue in the repository.**
