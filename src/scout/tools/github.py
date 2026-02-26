from __future__ import annotations
"""
GitHub API Tools for Scout MCP Server.

This module contains extracted GitHub API tools:
- scout_pr_info: Fetch PR metadata
- scout_pr: Create GitHub PRs
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from scout.tool_output import ToolOutput
from scout.tools import log_tool_invocation, simple_cache

# Shared configuration
VENV_PYTHON = "/Users/vivariumenv1/Vivarium/.venv/bin/python"
REPO_ROOT = Path("/Users/vivariumenv1/Vivarium")


def _run_command(
    cmd: list[str],
    timeout: int = 30,
    cwd: Path = None,
) -> subprocess.CompletedProcess:
    """Shared subprocess.run wrapper."""
    import os

    if cwd is None:
        cwd = REPO_ROOT
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )


def _get_github_token() -> str:
    """Get GitHub token from environment."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable is not set")
    return token


def _get_repo() -> str:
    """Get repository from environment or derive from git."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return repo
    try:
        result = _run_command(["git", "remote", "get-url", "origin"], timeout=10)
        if result.returncode == 0:
            url = result.stdout.strip()
            if url.startswith("git@github.com:"):
                path = url.replace("git@github.com:", "")
                return path.replace(".git", "")
            elif "github.com" in url:
                parts = url.split("github.com")
                if len(parts) > 1:
                    path = parts[1].strip("/")
                    return path.replace(".git", "")
    except Exception:
        pass
    raise RuntimeError(
        "Could not determine GitHub repository. Set GITHUB_REPOSITORY env var."
    )


def _github_api_get(path: str) -> dict | list:
    """Make a GET request to GitHub API."""
    import urllib.error
    import urllib.request

    token = _get_github_token()
    url = f"https://api.github.com{path}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API error {e.code}: {body}")


def _fetch_pull_request(pr_number: int) -> dict:
    """Fetch pull request data from GitHub API."""
    repo = _get_repo()
    return _github_api_get(f"/repos/{repo}/pulls/{pr_number}")


# =============================================================================
# PR Info Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@log_tool_invocation
def scout_pr_info(pr_number: int, ctx: Any = None) -> ToolOutput:
    """
    Fetch pull request metadata from GitHub.

    Use this tool to retrieve information about a specific pull request including
    title, description/body, author, status, and other metadata.

    Example: `scout_pr_info(pr_number=42)`

    Args:
        pr_number: The pull request number (e.g., 42 for PR #42).

    Returns:
        ToolOutput with tool_name="pr_info", content=json_string, cost_usd=0.0
    """
    try:
        pr_data = _fetch_pull_request(pr_number)

        result = {
            "number": pr_data.get("number"),
            "title": pr_data.get("title"),
            "body": pr_data.get("body"),
            "state": pr_data.get("state"),
            "html_url": pr_data.get("html_url"),
            "diff_url": pr_data.get("diff_url"),
            "patch_url": pr_data.get("patch_url"),
            "user": (
                {
                    "login": pr_data.get("user", {}).get("login"),
                    "html_url": pr_data.get("user", {}).get("html_url"),
                }
                if pr_data.get("user")
                else None
            ),
            "head": (
                {
                    "ref": pr_data.get("head", {}).get("ref"),
                    "sha": pr_data.get("head", {}).get("sha"),
                }
                if pr_data.get("head")
                else None
            ),
            "base": (
                {
                    "ref": pr_data.get("base", {}).get("ref"),
                    "sha": pr_data.get("base", {}).get("sha"),
                }
                if pr_data.get("base")
                else None
            ),
            "merged": pr_data.get("merged"),
            "mergeable": pr_data.get("mergeable"),
            "mergeable_state": pr_data.get("mergeable_state"),
            "created_at": pr_data.get("created_at"),
            "updated_at": pr_data.get("updated_at"),
            "closed_at": pr_data.get("closed_at"),
            "merged_at": pr_data.get("merged_at"),
            "comments": pr_data.get("comments"),
            "review_comments": pr_data.get("review_comments"),
            "commits": pr_data.get("commits"),
            "additions": pr_data.get("additions"),
            "deletions": pr_data.get("deletions"),
            "changed_files": pr_data.get("changed_files"),
            "draft": pr_data.get("draft"),
            "merge_date": pr_data.get("merged_at"),
        }

        content = json.dumps(result, indent=2)
        return ToolOutput.from_content(
            tool_name="pr_info",
            content=content,
            cost_usd=0.0,
            metadata={"pr_number": pr_number},
        )

    except RuntimeError as e:
        error_msg = str(e)
        if "404" in error_msg:
            content = json.dumps(
                {
                    "status": "error",
                    "message": f"Pull request #{pr_number} not found",
                    "hint": "Check that the PR number is correct and exists in the repository",
                },
                indent=2,
            )
        else:
            content = json.dumps(
                {
                    "status": "error",
                    "message": "Failed to fetch pull request",
                    "details": error_msg,
                },
                indent=2,
            )
        return ToolOutput.from_content(
            tool_name="pr_info",
            content=content,
            cost_usd=0.0,
            metadata={"pr_number": pr_number, "error": True},
        )


# =============================================================================
# PR Create Tool
# =============================================================================


@log_tool_invocation
async def scout_pr(
    title: str,
    body: str = "",
    base_branch: str = "main",
    draft: bool = False,
    dry_run: bool = False,
    use_dry_run_cache: bool = False,
    json_output: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Create a GitHub Pull Request with smart analysis and LLM-generated body.

    This tool provides a comprehensive PR creation workflow:
    1. Analyzes current git changes efficiently
    2. Generates a comprehensive PR body using LLM
    3. Creates the PR via GitHub CLI

    Args:
        title: PR title (required)
        body: Optional manual body. If empty, auto-generates using LLM
        base_branch: Target branch (default: main)
        draft: Create as draft PR (default: False)
        dry_run: Generate PR body without creating PR (default: False)
        use_dry_run_cache: Use cached body from previous dry run (default: False)
        json_output: Return structured JSON

    Returns:
        ToolOutput with tool_name="pr", content=result_string, cost_usd=0.0

    Example: `scout_pr(title="feat: add dark mode", draft=True)`
    """
    import asyncio

    # Normalize branch name
    if base_branch == "master":
        check_cmd = ["git", "branch", "--list", "master"]
        if _run_command(check_cmd).returncode != 0:
            base_branch = "main"

    # Load cached body if requested
    if use_dry_run_cache:
        cached_body = _load_pr_cache(title, base_branch)
        if cached_body:
            body = cached_body
        elif not body:
            error_msg = "No dry run cache found. Run with dry_run=True first."
            if json_output:
                return ToolOutput.from_content(
                    tool_name="pr",
                    content=_success_response(False, error=error_msg),
                    cost_usd=0.0,
                    metadata={"error": True, "json_output": True},
                )
            return ToolOutput.from_content(
                tool_name="pr",
                content=f"Error: {error_msg}",
                cost_usd=0.0,
                metadata={"error": True},
            )

    # Gather git context and generate body
    if not body:
        git_context = _gather_pr_context_deep(base_branch)

        from vivarium.scout.llm.minimax import call_minimax_async_detailed

        body_prompt = f"""Generate a PR description for:

Title: {title}

Changes:
{git_context}

Requirements:
- No generic templates ("This PR adds...")
- Explain WHY each file changed, WHAT it does
- Include how to verify changes work"""

        try:
            result = await call_minimax_async_detailed(
                prompt=body_prompt,
                system="Senior engineer creating PR descriptions. Be concise.",
                max_tokens=1500,
                model="MiniMax-M2.1",
            )
            if hasattr(result, "response_text"):
                body = result.response_text.strip()
            else:
                body = f"## TLDR\n{title}\n\n## Changes\nSee diff."
        except Exception as e:
            body = f"## TLDR\n{title}\n\n(Error generating body: {e})"

    # Handle dry run
    if dry_run:
        cache_file = _save_pr_cache(title, base_branch, body)
        if json_output:
            return ToolOutput.from_content(
                tool_name="pr",
                content=json.dumps(
                    {
                        "success": True,
                        "dry_run": True,
                        "title": title,
                        "base_branch": base_branch,
                        "body": body,
                        "cache_file": cache_file,
                    }
                ),
                cost_usd=0.0,
                metadata={"dry_run": True, "json_output": True},
            )
        dry_run_content = f"""## DRY RUN - PR Body Preview
========================================
Title: {title}
Base: {base_branch}
Cache saved to: {cache_file}

---

{body}

========================================
To create PR, run without dry_run
"""
        return ToolOutput.from_content(
            tool_name="pr",
            content=dry_run_content,
            cost_usd=0.0,
            metadata={"dry_run": True},
        )

    # Create PR
    body_file = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    body_file.write(body)
    body_file.close()

    cmd = [
        "gh",
        "pr",
        "create",
        "--title",
        title,
        "--body-file",
        body_file.name,
        "--base",
        base_branch,
    ]
    if draft:
        cmd.append("--draft")

    try:
        result = _run_command(cmd, timeout=60)
        os.unlink(body_file.name)

        if result.returncode == 0:
            pr_url = result.stdout.strip()
            if json_output:
                return ToolOutput.from_content(
                    tool_name="pr",
                    content=json.dumps(
                        {
                            "success": True,
                            "url": pr_url,
                            "title": title,
                            "base": base_branch,
                        }
                    ),
                    cost_usd=0.0,
                    metadata={"json_output": True},
                )
            return ToolOutput.from_content(
                tool_name="pr",
                content=f"PR created: {pr_url}",
                cost_usd=0.0,
                metadata={"url": pr_url},
            )
        else:
            error_msg = result.stderr or result.stdout
            if json_output:
                return ToolOutput.from_content(
                    tool_name="pr",
                    content=_success_response(False, error=error_msg),
                    cost_usd=0.0,
                    metadata={"error": True, "json_output": True},
                )
            return ToolOutput.from_content(
                tool_name="pr",
                content=f"Error creating PR: {error_msg}",
                cost_usd=0.0,
                metadata={"error": True},
            )
    except subprocess.TimeoutExpired:
        os.unlink(body_file.name)
        if json_output:
            return ToolOutput.from_content(
                tool_name="pr",
                content=_success_response(False, error="Timeout creating PR"),
                cost_usd=0.0,
                metadata={"error": True, "json_output": True},
            )
        return ToolOutput.from_content(
            tool_name="pr",
            content="Error: Timeout creating PR",
            cost_usd=0.0,
            metadata={"error": True},
        )
    except Exception as e:
        try:
            os.unlink(body_file.name)
        except:
            pass
        if json_output:
            return ToolOutput.from_content(
                tool_name="pr",
                content=_success_response(False, error=str(e)),
                cost_usd=0.0,
                metadata={"error": True, "json_output": True},
            )
        return ToolOutput.from_content(
            tool_name="pr",
            content=f"Error creating PR: {e}",
            cost_usd=0.0,
            metadata={"error": True},
        )


# =============================================================================
# Helper Functions (from main module)
# =============================================================================


def _success_response(success: bool = True, **data: Any) -> str:
    """Standardized success response format."""
    data["success"] = success
    return json.dumps(data, indent=2)


def _gather_pr_context_deep(base_branch: str, max_files: int = 50) -> str:
    """Gather git context for PR body generation."""
    # Find the branch point
    cmd = ["git", "merge-base", "HEAD", base_branch]
    result = _run_command(cmd)
    if result.returncode != 0:
        return "Could not determine branch point."

    base_commit = result.stdout.strip()

    # Get changed files
    cmd = ["git", "diff", "--name-only", base_commit, "HEAD"]
    result = _run_command(cmd)
    if result.returncode != 0:
        return "Could not determine changed files."

    files = result.stdout.strip().split("\n")[:max_files]
    if not files:
        return "No changes detected."

    # Get diff summary
    cmd = ["git", "diff", "--stat", base_commit, "HEAD"]
    result = _run_command(cmd)

    return f"""Branch: {base_branch} -> HEAD
Base: {base_commit}
Changed files ({len(files)}):
{result.stdout}"""


def _load_pr_cache(title: str, base_branch: str) -> str | None:
    """Load cached PR body."""
    import hashlib

    cache_key = hashlib.sha256(f"{title}:{base_branch}".encode()).hexdigest()[:16]
    cache_dir = REPO_ROOT / ".scout_logs" / "pr_cache"
    cache_file = cache_dir / f"{cache_key}.md"

    if cache_file.exists():
        return cache_file.read_text()
    return None


def _save_pr_cache(title: str, base_branch: str, body: str) -> str:
    """Save PR body to cache."""
    import hashlib

    cache_key = hashlib.sha256(f"{title}:{base_branch}".encode()).hexdigest()[:16]
    cache_dir = REPO_ROOT / ".scout_logs" / "pr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.md"

    cache_file.write_text(body)
    return str(cache_file)
