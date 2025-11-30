from cubo.config import config
from cubo.main import CUBOApp
from cubo.security.security import security_manager
from cubo.utils.logger import logger_instance


def test_query_scrubbed_in_logs(tmp_path):
    log_file = tmp_path / "test_log.jsonl"
    query = "this is my secret query"

    # Test when scrub_queries = True
    config.set("logging.log_file", str(log_file))
    config.set("logging.format", "json")
    config.set("logging.scrub_queries", True)
    config.set("logging.enable_queue", False)
    logger_instance.shutdown()
    logger_instance._setup_logging()
    # Ensure module-level loggers referencing the previous logger are updated
    from cubo import main as cubomain

    cubomain.logger = logger_instance.get_logger()

    app = CUBOApp()
    # Use the command line display helper to log the query
    app._display_command_line_results(query, top_docs=["doc1"], response="ok")

    with open(log_file, encoding="utf-8") as f:
        lines = f.readlines()

    assert any("Query:" in l for l in lines)
    # Find the last Query line
    qlines = [l for l in lines if "Query:" in l]
    qline = qlines[-1]
    # When scrub enabled, ensure full query is not in logs
    assert query not in qline
    # And hash representation is present
    assert security_manager.hash_sensitive_data(query) in qline

    # Test when scrub_queries = False
    config.set("logging.scrub_queries", False)
    logger_instance.shutdown()
    logger_instance._setup_logging()
    # Re-update module logger pointer after re-init
    cubomain.logger = logger_instance.get_logger()

    app._display_command_line_results(query, top_docs=["doc1"], response="ok")

    with open(log_file, encoding="utf-8") as f:
        lines = f.readlines()

    qlines = [l for l in lines if "Query:" in l]
    qline = qlines[-1]
    assert query in qline
