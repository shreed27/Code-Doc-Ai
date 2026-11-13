"""
Security utilities for code-to-docs GitHub Action

This module provides security functions to prevent:
- Credential leakage in error messages
- Path traversal attacks
- Unauthorized file access
"""

import os
import re
import subprocess
from pathlib import Path


def sanitize_output(text, sensitive_tokens=None):
    """
    Remove sensitive tokens from text output.

    Args:
        text: Text to sanitize
        sensitive_tokens: Additional sensitive tokens to remove

    Returns:
        str: Sanitized text with tokens replaced by ***TOKEN***
    """
    if not text:
        return text

    if sensitive_tokens is None:
        sensitive_tokens = []

    # Add all known sensitive env vars to the list
    for env_var in [
        "GH_TOKEN",
        "MODEL_API_KEY",
        "JIRA_API_TOKEN",
        "GOOGLE_SA_KEY",
    ]:
        val = os.environ.get(env_var, "")
        if val:
            sensitive_tokens.append(val)

    # Dynamically scan os.environ for secret-like keys
    sensitive_name_keywords = (
        "KEY",
        "SECRET",
        "PASSWORD",
        "TOKEN",
        "AUTH",
        "CREDENTIAL",
        "PASS",
    )
    for k, v in os.environ.items():
        if any(kw in k.upper() for kw in sensitive_name_keywords):
            if v and len(v.strip()) >= 5 and v not in sensitive_tokens:
                sensitive_tokens.append(v)

    # Replace all sensitive tokens, longest first to avoid partial replacements
    sanitized = text
    for token in sorted(sensitive_tokens, key=len, reverse=True):
        if token and len(token) > 0:
            sanitized = sanitized.replace(token, "***TOKEN***")

    return sanitized


def redact_secrets(text: str) -> str:
    """
    Redact secrets including AWS keys, GitHub PATs, JWTs, OpenAI keys, Slack tokens,
    Stripe keys, database credentials, authorization headers, private key blocks,
    and prefixed secret variable assignments.

    Preserves code structure, variable names, and formatting for comprehension while
    scrubbing the actual secret values.
    """
    if not text:
        return text

    # 1. AWS access key ID (e.g. AKIAIOSFODNN7EXAMPLE)
    text = re.sub(r"\bAKIA[0-9A-Z]{16}\b", "[REDACTED_AWS_KEY]", text)

    # 2. AWS secret access key in assignment
    text = re.sub(
        r"(?i)\b((?:aws_secret_access_key|aws_secret_key)\s*[:=]\s*['\"]?)[A-Za-z0-9/+=]{40}(['\"]?)",
        r"\1[REDACTED_AWS_SECRET_KEY]\2",
        text,
    )

    # 3. GitHub tokens (ghp_, github_pat_, gho_, ghu_, ghs_, ghr_)
    text = re.sub(r"\bgh[pousr]_[0-9a-zA-Z]{36,255}\b", "[REDACTED_GITHUB_TOKEN]", text)
    text = re.sub(r"\bgithub_pat_[0-9a-zA-Z_]{20,255}\b", "[REDACTED_GITHUB_TOKEN]", text)

    # 4. JSON Web Tokens (JWT)
    text = re.sub(
        r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\b",
        "[REDACTED_JWT]",
        text,
    )

    # 5. OpenAI-style API keys (sk-...)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{20,}\b", "[REDACTED_OPENAI_KEY]", text)
    text = re.sub(r"\bsk-proj-[A-Za-z0-9_-]{20,}\b", "[REDACTED_OPENAI_KEY]", text)

    # 6. Slack tokens (xoxb-, xoxp-, xoxa-, xoxr-)
    text = re.sub(r"\bxox[baprs]-[0-9a-zA-Z-]{10,}\b", "[REDACTED_SLACK_TOKEN]", text)

    # 7. Stripe keys (sk_live_, rk_live_, sk_test_, rk_test_)
    text = re.sub(r"\b[sr]k_(?:live|test)_[0-9a-zA-Z]{24,}\b", "[REDACTED_STRIPE_KEY]", text)

    # 8. Database connection strings containing credentials
    text = re.sub(
        r"\b((?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp|mssql):\/\/[^\s:@/]+:)[^\s:@/]+(@[^\s\"']+)\b",
        r"\1[REDACTED_PASSWORD]\2",
        text,
    )

    # 9. Authorization headers and dictionary fields
    text = re.sub(
        r"(?i)(['\"]?Authorization['\"]?\s*:\s*['\"]?(?:Bearer|Basic)\s+)[A-Za-z0-9._~+/-]{10,}(['\"]?)",
        r"\1[REDACTED_TOKEN]\2",
        text,
    )
    text = re.sub(
        r"(?i)(['\"]?Authorization['\"]?\s*:\s*['\"])Basic\s+[A-Za-z0-9+/=]{6,}(['\"])",
        r"\1Basic [REDACTED_TOKEN]\2",
        text,
    )
    text = re.sub(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/-]{16,}\b", r"\1[REDACTED_TOKEN]", text)

    # 10. Private key blocks
    text = re.sub(
        r"-----BEGIN [A-Z0-9_ -]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9_ -]*PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        text,
    )

    # 11. Context-aware secret assignments (YAML, JSON, Python, Shell, .env, prose, headers)
    # Matches keys like client_secret, DB_PASSWORD, password, api_key, X-API-Key, etc.
    secret_key_re = re.compile(
        r"(?i)(?:^|[\s,;{(])((?:export\s+)?['\"]?([A-Za-z0-9_.-]*(?:API[_-]?KEY|SECRET|PASSWORD|ACCESS[_-]?KEY|PRIVATE[_-]?KEY|AUTH[_-]?TOKEN|BEARER[_-]?TOKEN|CLIENT[_-]?SECRET)[A-Za-z0-9_.-]*)['\"]?\s*[:=]\s*)(?:(['\"])([^'\"\n\r]+)(['\"])|([^\s,;'\"}{\][#\n\r]+))"
    )

    def _replace_secret_assignment(match: re.Match) -> str:
        prefix = match.group(1)
        # If quoted
        if match.group(3) and match.group(5):
            q1 = match.group(3)
            val = match.group(4)
            q2 = match.group(5)
            if val.startswith("[REDACTED"):
                return match.group(0)
            leading = match.group(0)[: match.start(1) - match.start(0)]
            return f"{leading}{prefix}{q1}[REDACTED]{q2}"
        # If unquoted
        val = match.group(6)
        if not val or val.startswith("[REDACTED"):
            return match.group(0)
        # Avoid redacting boolean/null types or common non-secret code tokens
        if val.lower() in ("null", "none", "true", "false", "str", "int", "bool"):
            return match.group(0)
        leading = match.group(0)[: match.start(1) - match.start(0)]
        return f"{leading}{prefix}[REDACTED]"

    text = secret_key_re.sub(_replace_secret_assignment, text)

    # 12. Sanitize active environment variables
    return sanitize_output(text)


def run_command_safe(cmd, check=False, capture_output=True, **kwargs):
    """
    Run subprocess command with sanitized output.

    Args:
        cmd: Command to run as list
        check: Raise exception on non-zero return code
        capture_output: Capture stdout and stderr
        **kwargs: Additional subprocess.run arguments

    Returns:
        CompletedProcess: Result with sanitized output
    """
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, **kwargs)

        if check and result.returncode != 0:
            # Sanitize error output before raising
            sanitized_stderr = sanitize_output(result.stderr)
            sanitized_stdout = sanitize_output(result.stdout)
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=sanitized_stdout, stderr=sanitized_stderr
            )

        return result
    except subprocess.CalledProcessError as e:
        # Re-raise with sanitized output
        e.stderr = sanitize_output(str(e.stderr)) if e.stderr else None
        e.stdout = sanitize_output(str(e.stdout)) if e.stdout else None
        raise
    except Exception as e:
        # Sanitize any other exceptions that might contain tokens
        e.args = tuple(sanitize_output(str(arg)) for arg in e.args)
        raise


def validate_file_path(file_path, base_dir=None):
    """
    Validate file path is within allowed directory.

    Args:
        file_path: Path to validate
        base_dir: Base directory (defaults to current directory)

    Returns:
        bool: True if safe, False otherwise
    """
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    try:
        file_path = Path(file_path)
        # Resolve to absolute path
        resolved = (base_dir / file_path).resolve()

        # Check if it's within base directory
        resolved.relative_to(base_dir.resolve())

        return True
    except (ValueError, OSError):
        return False


def setup_git_credentials(token, repo_url):
    """
    Setup git credential helper for secure authentication.

    Args:
        token: GitHub token
        repo_url: Repository URL

    Returns:
        bool: True if successful, False otherwise
    """
    # Use git config to set credential helper
    try:
        helper_cmd = f"!f() {{ echo 'username=x-access-token'; echo 'password={token}'; }}; f"

        run_command_safe(["git", "config", "credential.helper", helper_cmd], check=False)
        return True
    except subprocess.CalledProcessError as e:
        # Don't print the command (contains token) - only print exit code
        print(f"Warning: Git credential setup failed with exit code {e.returncode}")
        return False
    except Exception as e:
        # Generic exception - sanitize just in case
        error_msg = sanitize_output(str(e))
        print(f"Warning: Could not setup git credentials: {error_msg}")
        return False


def validate_docs_file_extension(file_path):
    """
    Validate file has allowed documentation extension.

    Args:
        file_path: Path to check

    Returns:
        bool: True if .adoc, .md, or .rst, False otherwise
    """
    return file_path.endswith(".adoc") or file_path.endswith(".md") or file_path.endswith(".rst")


def validate_docs_subfolder(subfolder_path):
    """
    Validate docs subfolder path is safe.

    Args:
        subfolder_path: Subfolder path to validate

    Returns:
        bool: True if safe, False otherwise
    """
    if not subfolder_path:
        return True

    # Security: ensure no path traversal
    if ".." in subfolder_path or subfolder_path.startswith("/"):
        return False

    return True
