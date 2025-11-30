# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.2.x   | :white_check_mark: |
| 1.1.x   | :white_check_mark: |
| < 1.1   | :x:                |

## Reporting a Vulnerability

We take the security of CUBO seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Where to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@example.com**

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

### What to Include

Please include the following information in your report:

- Type of issue (e.g., path traversal, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- We will acknowledge your email within 48 hours
- We will send a more detailed response within 7 days indicating the next steps
- We will keep you informed about the progress towards a fix and announcement
- We may ask for additional information or guidance

## Security Best Practices

When using CUBO in production:

### 1. **Environment Variables**
Never hardcode sensitive information. Use environment variables:
```bash
export CUBO_ENCRYPTION_KEY="your-32-byte-key-here"
export CUBO_MODEL_PATH="/secure/path/to/model"
```

### 2. **File Upload Security**
- Set appropriate `max_file_size_mb` in `config.json`
- Validate file types before processing
- Use path sanitization (already implemented in `Utils.sanitize_path`)

### 3. **API Security**
- Deploy behind a reverse proxy (nginx, Caddy)
- Enable HTTPS/TLS in production
- Implement rate limiting at the proxy level
- Use API keys for authentication (if exposing externally)

### 4. **Data Privacy**
- CUBO processes data locally - no data is sent to external services
- For GDPR compliance, document your data retention policy
- Implement data deletion endpoints for user requests

### 5. **Network Isolation**
- Run CUBO in a private network if possible
- Do not expose the API directly to the internet without authentication
- Use firewall rules to restrict access

### 6. **Dependency Security**
- Regularly update dependencies: `pip install --upgrade -r requirements.txt`
- Run security audits: `pip-audit` or `safety check`
- Monitor for CVEs in dependencies

### 7. **Secret Management**
- Never commit secrets to git
- Use `.env.example` as a template (secrets should be in `.env` which is gitignored)
- Rotate encryption keys periodically

## Known Security Considerations

### Local LLM Execution
CUBO uses Ollama for local LLM inference. Ensure Ollama is properly configured and not exposed to untrusted networks.

### Document Processing
When processing user-uploaded documents:
- PDFs may contain malicious scripts (use sandboxing if processing untrusted files)
- Excel/CSV files may contain formulas (already disabled in our implementation)

### Embeddings Storage
- Embedding vectors are stored in FAISS indices
- SQLite metadata stores may contain sensitive document metadata
- Ensure proper file permissions on data directories

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find similar problems
3. Prepare fixes for all supported versions
4. Release patches as soon as possible

We will credit security researchers in our release notes (unless anonymity is requested).

## Comments on This Policy

If you have suggestions on how this process could be improved, please submit a pull request or open an issue in GitHub.
