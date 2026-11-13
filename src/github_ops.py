"""
GitHub and git operations for code-to-docs.

Handles PR diffs, commit info extraction, docs environment setup,
and pushing/creating PRs in the documentation repo.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from config import get_branch_name, get_docs_repo_url, get_pr_title_prefix
from security_utils import (
    run_command_safe,
    sanitize_output,
    setup_git_credentials,
    validate_docs_subfolder,
)


@dataclass
class SourceFileChange:
    """Represents a source file change extracted from git."""

    path: str
    old_path: str
    status: str  # "added", "modified", "deleted", "renamed"
    old_code: str
    new_code: str
    error: str | None = None


def get_changed_source_files(
    repo_root: str | Path,
    pr_base: str = "origin/main",
) -> dict[str, SourceFileChange]:
    """
    Extract changed source files using git diff -M --raw anchored at repo_root.

    Never relies on current working directory. Correctly tracks additions,
    modifications, deletions, and renames between the merge-base and HEAD.
    Returns structured errors rather than empty strings when files cannot be read.
    """
    root = Path(repo_root).resolve()
    changes: dict[str, SourceFileChange] = {}

    # 1. Resolve merge-base
    merge_base_result = run_command_safe(
        ["git", "merge-base", pr_base, "HEAD"], check=False, cwd=str(root)
    )
    if merge_base_result.returncode == 0 and merge_base_result.stdout.strip():
        base_rev = merge_base_result.stdout.strip()
    else:
        base_rev = pr_base

    # 2. Get raw diff with rename detection
    raw_res = run_command_safe(
        ["git", "diff", "-M", "--raw", f"{base_rev}...HEAD"], check=False, cwd=str(root)
    )
    if raw_res.returncode != 0:
        raw_res = run_command_safe(
            ["git", "diff", "-M", "--raw", base_rev, "HEAD"], check=False, cwd=str(root)
        )

    if raw_res.returncode != 0 or not raw_res.stdout.strip():
        return changes

    for line in raw_res.stdout.strip().splitlines():
        if not line or not line.startswith(":"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        meta_parts = parts[0].split()
        status_info = meta_parts[-1]
        status_char = status_info[0]

        if status_char == "R":
            old_p = parts[1].strip()
            new_p = parts[2].strip() if len(parts) > 2 else old_p
            status = "renamed"
        elif status_char == "A":
            old_p = parts[1].strip()
            new_p = parts[1].strip()
            status = "added"
        elif status_char == "D":
            old_p = parts[1].strip()
            new_p = parts[1].strip()
            status = "deleted"
        else:
            old_p = parts[1].strip()
            new_p = parts[1].strip()
            status = "modified"

        # Focus on python files
        if not new_p.endswith(".py") and not old_p.endswith(".py"):
            continue

        old_code = ""
        new_code = ""
        error = None

        # Read old version from base_rev
        if status != "added":
            old_git = run_command_safe(
                ["git", "show", f"{base_rev}:{old_p}"], check=False, cwd=str(root)
            )
            if old_git.returncode == 0:
                old_code = old_git.stdout
            else:
                error = f"Failed to retrieve revision {base_rev[:7]}:{old_p}: {old_git.stderr or 'git error'}"

        # Read new version from working tree / HEAD
        if status != "deleted" and error is None:
            file_on_disk = root / new_p
            if file_on_disk.is_file():
                try:
                    with open(file_on_disk, encoding="utf-8") as f:
                        new_code = f.read()
                except Exception as read_err:
                    error = f"Failed to read file on disk at {file_on_disk}: {read_err}"
            else:
                new_git = run_command_safe(
                    ["git", "show", f"HEAD:{new_p}"], check=False, cwd=str(root)
                )
                if new_git.returncode == 0:
                    new_code = new_git.stdout
                else:
                    error = f"Failed to retrieve HEAD:{new_p}: {new_git.stderr or 'file not found'}"

        changes[new_p] = SourceFileChange(
            path=new_p,
            old_path=old_p,
            status=status,
            old_code=old_code,
            new_code=new_code,
            error=error,
        )

    return changes


def get_diff():
    """
    Get the full diff for the entire PR, not just the latest commit.
    Uses safe subprocess execution.
    """
    pr_base = os.environ.get("PR_BASE", "origin/main")
    pr_number = os.environ.get("PR_NUMBER", "unknown")

    print(f"Getting diff for PR #{pr_number} against base: {pr_base}")

    try:
        merge_base_result = run_command_safe(["git", "merge-base", pr_base, "HEAD"], check=False)

        if merge_base_result.returncode == 0:
            merge_base = merge_base_result.stdout.strip()
            print(f"Using merge-base: {merge_base[:7]}...{merge_base[-7:]}")

            files_result = run_command_safe(
                ["git", "diff", "--name-only", f"{merge_base}...HEAD"], check=False
            )
            if files_result.returncode == 0:
                changed_files = files_result.stdout.strip().split("\n")
                changed_files = [f for f in changed_files if f.strip()]
                print(f"Files changed in entire PR: {changed_files}")

            result = run_command_safe(["git", "diff", f"{merge_base}...HEAD"], check=False)
            diff_method = f"merge-base ({merge_base[:7]}...HEAD)"
        else:
            print("Warning: Could not find merge-base, using fallback diff method")
            result = run_command_safe(["git", "diff", f"{pr_base}...HEAD"], check=False)
            diff_method = f"direct ({pr_base}...HEAD)"

        diff_content = result.stdout.strip() if result.stdout else ""
        print(f"Diff method: {diff_method}")
        print(f"Diff size: {len(diff_content)} characters")

        return diff_content
    except Exception as e:
        print(f"❌ Error getting diff: {sanitize_output(str(e))}")
        return ""


def get_commit_info():
    """
    Get PR information for the documentation PR reference.
    Uses safe subprocess execution.
    """
    try:
        pr_number = os.environ.get("PR_NUMBER")

        current_commit_result = run_command_safe(["git", "rev-parse", "HEAD"], check=False)
        if current_commit_result.returncode != 0:
            return None
        commit_hash = current_commit_result.stdout.strip()

        remote_url = run_command_safe(["git", "config", "--get", "remote.origin.url"], check=False)
        if remote_url.returncode != 0:
            return None

        # Convert SSH URL to HTTPS if needed
        repo_url = remote_url.stdout.strip()
        if repo_url.startswith("git@github.com:"):
            repo_url = repo_url.replace("git@github.com:", "https://github.com/").replace(
                ".git", ""
            )
        elif repo_url.endswith(".git"):
            repo_url = repo_url.replace(".git", "")

        short_hash = commit_hash[:7]

        result = {"repo_url": repo_url, "current_commit": commit_hash, "short_hash": short_hash}

        if pr_number and pr_number.strip() and pr_number != "unknown":
            result["pr_number"] = pr_number
            result["pr_url"] = f"{repo_url}/pull/{pr_number}"

        return result

    except Exception as e:
        print(f"Warning: Could not get commit info: {sanitize_output(str(e))}")
        return None


def setup_docs_environment():
    """
    Set up docs environment — either local subfolder or clone separate repo.
    Uses secure git operations.
    """
    docs_subfolder = os.environ.get("DOCS_SUBFOLDER")
    branch_name = get_branch_name(os.environ.get("PR_NUMBER"))

    if docs_subfolder:
        # Use local subfolder (same repo)
        if not validate_docs_subfolder(docs_subfolder):
            print(f"❌ Security: Invalid docs subfolder path: {docs_subfolder}")
            return False

        if not os.path.exists(docs_subfolder):
            print(f"ERROR: Docs subfolder '{docs_subfolder}' not found")
            return False

        os.chdir(docs_subfolder)
        return True
    else:
        # Clone separate repository with secure credentials
        try:
            print("Cloning separate docs repository")
            docs_repo_url = get_docs_repo_url()

            gh_token = os.environ.get("GH_TOKEN")
            if gh_token:
                setup_git_credentials(gh_token, docs_repo_url)

            run_command_safe(["git", "clone", docs_repo_url, "docs_repo"], check=True)
            os.chdir("docs_repo")

            result = run_command_safe(
                ["git", "ls-remote", "--heads", "origin", branch_name], check=False
            )

            if result.stdout and result.stdout.strip():
                print(f"Reusing existing branch: {branch_name}")
                run_command_safe(["git", "fetch", "origin", branch_name], check=True)
                run_command_safe(["git", "checkout", branch_name], check=True)
                run_command_safe(["git", "pull", "origin", branch_name], check=True)
            else:
                print(f"Creating new branch: {branch_name}")
                run_command_safe(["git", "checkout", "-b", branch_name], check=True)

            return True
        except Exception as e:
            print(f"❌ Failed to setup docs environment: {sanitize_output(str(e))}")
            return False


def push_and_open_pr(modified_files, commit_info=None):
    """
    Push changes and create PR in docs repository.
    Uses secure credential helper to prevent token leakage.
    """
    branch_name = get_branch_name(os.environ.get("PR_NUMBER"))

    try:
        run_command_safe(["git", "add"] + modified_files, check=True)

        commit_msg = f"{get_pr_title_prefix()}Auto-generated doc updates from code changes"
        if commit_info:
            if "pr_number" in commit_info:
                commit_msg += f"\n\nPR Link: {commit_info['pr_url']}"
                commit_msg += f"\nLatest commit: {commit_info['short_hash']}"
            else:
                commit_url = f"{commit_info['repo_url']}/commit/{commit_info['current_commit']}"
                commit_msg += f"\n\nCommit Link: {commit_url}"
                commit_msg += f"\nLatest commit: {commit_info['short_hash']}"
        commit_msg += "\n\nAssisted-by: code-to-docs AI"

        run_command_safe(["git", "commit", "-m", commit_msg], check=True)

        gh_token = os.environ.get("GH_TOKEN")
        if not gh_token:
            raise ValueError("GH_TOKEN environment variable not set")

        # Clear GitHub Actions default authentication that interferes with GH_TOKEN
        run_command_safe(
            ["git", "config", "--unset-all", "http.https://github.com/.extraheader"],
            check=False,
            capture_output=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        docs_repo_url = get_docs_repo_url()
        setup_git_credentials(gh_token, docs_repo_url)
        run_command_safe(["git", "remote", "set-url", "origin", docs_repo_url], check=True)

        print(f"Pushing to branch {branch_name}...")
        run_command_safe(
            ["git", "push", "--set-upstream", "origin", branch_name, "--force-with-lease"],
            check=True,
        )
        print("✅ Successfully pushed changes")

        # Build PR body
        pr_body = "This PR updates the following documentation files based on code changes:\n\n"
        pr_body += "\n".join([f"- `{f}`" for f in modified_files])
        if commit_info:
            pr_body += "\n\n---\n**Source:**\n"
            if "pr_number" in commit_info:
                pr_body += f"- PR: {commit_info['pr_url']}\n"
            pr_body += f"- Commit: `{commit_info['short_hash']}`"
        pr_body += "\n\n*Assisted by code-to-docs AI*"

        # Check if PR already exists for this branch
        print("Checking for existing pull request...")
        check_pr = run_command_safe(
            ["gh", "pr", "list", "--head", branch_name, "--state", "open", "--json", "number"],
            check=False,
            env={**os.environ, "GH_TOKEN": gh_token},
        )

        existing_pr = check_pr.stdout.strip() if check_pr.returncode == 0 else "[]"

        if existing_pr and existing_pr != "[]":
            print("✅ Existing PR found — branch updated with new changes")
        else:
            print("Creating pull request...")
            run_command_safe(
                [
                    "gh",
                    "pr",
                    "create",
                    "--title",
                    f"{get_pr_title_prefix()}Auto-Generated Doc Updates from Code PR",
                    "--body",
                    pr_body,
                    "--base",
                    os.environ.get("DOCS_BASE_BRANCH") or "main",
                    "--head",
                    branch_name,
                ],
                check=True,
                env={**os.environ, "GH_TOKEN": gh_token},
            )
            print("✅ Successfully created PR")

    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {sanitize_output(str(e))}")
        if e.stderr:
            print(f"Error details: {sanitize_output(e.stderr)}")
        raise
    except Exception as e:
        print(f"❌ Error in push_and_open_pr: {sanitize_output(str(e))}")
        raise
