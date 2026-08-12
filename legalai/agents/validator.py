"""Validation Agent - Validates the quality of responses before sending to user."""

import re
from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from agents.base import BaseAgent
import config


class ValidationAgent(BaseAgent):
    """Validation Agent checks response quality and completeness."""

    def __init__(self):
        """Initialize the Validation Agent."""
        super().__init__(temperature=0.0)
        self.prompt = ChatPromptTemplate.from_template(config.VALIDATION_PROMPT)

    def _parse_validation(self, response: str) -> Dict[str, Any]:
        """Parse the validation response into structured format.

        Args:
            response: Raw LLM response.

        Returns:
            Dictionary with 'pass' (bool), 'issues' (str), 'source_relevant' (bool), and 'retry_fetch' (bool).
        """
        result = {
            "pass": True,
            "issues": "",
            "source_relevant": True,
            "retry_fetch": False,
            "parsed": True,
        }

        # Try to parse PASS: true/false. An unparseable validator response used to
        # default to PASS, so a malformed judgement silently became approval and the
        # validation stage measured nothing.
        pass_match = re.search(r"PASS:\s*(true|false|yes|no)", response, re.IGNORECASE)
        if pass_match:
            pass_val = pass_match.group(1).lower()
            result["pass"] = pass_val in ["true", "yes"]
        else:
            result["pass"] = False
            result["parsed"] = False
            result["issues"] = (
                "Validator output could not be parsed (no PASS field); treated as FAIL. "
                f"Raw output: {response.strip()[:300]}"
            )

        # Try to parse ISSUES:
        issues_match = re.search(r"ISSUES:\s*(.+?)(?=\n\n|$)", response, re.DOTALL | re.IGNORECASE)
        if issues_match:
            result["issues"] = issues_match.group(1).strip()
        else:
            # Try alternative format
            issues_match = re.search(r"ISSUES:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
            if issues_match:
                result["issues"] = issues_match.group(1).strip()

        # Try to parse SOURCE_RELEVANT: true/false
        source_match = re.search(r"SOURCE_RELEVANT:\s*(true|false|yes|no)", response, re.IGNORECASE)
        if source_match:
            source_val = source_match.group(1).lower()
            result["source_relevant"] = source_val in ["true", "yes"]

        # Try to parse RETRY_FETCH: true/false
        retry_match = re.search(r"RETRY_FETCH:\s*(true|false|yes|no)", response, re.IGNORECASE)
        if retry_match:
            retry_val = retry_match.group(1).lower()
            result["retry_fetch"] = retry_val in ["true", "yes"]

        return result

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the draft response.

        Args:
            state: The current state with 'query' and 'draft_response'.

        Returns:
            Updated state with 'validation_result' populated.
        """
        query = state.get("query", "")
        draft_response = state.get("draft_response", "")

        self.log(f"Validating response for: {query[:50]}...")

        if not draft_response:
            self.log("Warning: Empty draft response to validate")
            state["validation_result"] = {
                "pass": False,
                "issues": "Empty response generated",
                "source_relevant": False,
                "retry_fetch": False
            }
            return state

        # Get sources from retrieved docs
        retrieved_docs = state.get("retrieved_docs", [])
        sources_text = ""
        if retrieved_docs:
            sources = []
            for doc in retrieved_docs[:5]:  # Limit to first 5 sources
                # Ingestion writes the document label under 'name'; reading only
                # 'source' made every source render as "Unknown", so the validator
                # was judging source relevance against a list of blanks.
                if hasattr(doc, 'metadata') and doc.metadata:
                    metadata = doc.metadata or {}
                    source = metadata.get('name') or metadata.get('source') or 'Unknown'
                elif isinstance(doc, dict):
                    metadata = doc.get('metadata', {}) or {}
                    source = metadata.get('name') or metadata.get('source') or 'Unknown'
                else:
                    source = 'Unknown'
                sources.append(f"- {source}")
            sources_text = "\n".join(sources) if sources else "No sources available"
        else:
            sources_text = "No sources retrieved"

        # Create the chain and invoke
        chain = self.prompt | self.llm

        response = chain.invoke({
            "query": query,
            "response": draft_response,
            "sources": sources_text,
        })

        # Parse the validation result
        validation_result = self._parse_validation(response.content)
        state["validation_result"] = validation_result

        # Record tokens
        self._record_tokens(state, "validator", response)

        if not validation_result.get("parsed", True):
            self.log("WARNING validator output unparseable; recorded as FAIL, not PASS")
            state["validator_parse_failures"] = state.get("validator_parse_failures", 0) + 1

        status = "PASS" if validation_result["pass"] else "FAIL"
        self.log(f"Validation result: {status}")

        if not validation_result["pass"]:
            self.log(f"Issues: {validation_result['issues'][:100]}...")

        if not validation_result.get("source_relevant", True):
            self.log("Sources deemed irrelevant - may need to re-fetch")

        if validation_result.get("retry_fetch", False):
            self.log("Retry fetch requested by validator")

        return state

    def should_retry(self, state: Dict[str, Any]) -> bool:
        """Check if the workflow should retry based on validation.

        Args:
            state: The current state.

        Returns:
            True if validation failed and we haven't exceeded max iterations.
        """
        validation = state.get("validation_result", {})
        iteration = state.get("iteration_count", 0)

        # Don't retry if validation passed
        if validation.get("pass", True):
            return False

        # Don't retry if we've hit the max iterations
        if iteration >= config.MAX_ITERATIONS:
            self.log(f"Max iterations ({config.MAX_ITERATIONS}) reached, skipping retry")
            return False

        return True

    def should_retry_fetch(self, state: Dict[str, Any]) -> bool:
        """Check if we should re-fetch sources based on validation.

        Args:
            state: The current state.

        Returns:
            True if sources need to be re-fetched.
        """
        validation = state.get("validation_result", {})

        # Check if validator specifically requested a re-fetch
        if validation.get("retry_fetch", False):
            return True

        # Also re-fetch if sources are marked as not relevant
        if not validation.get("source_relevant", True):
            return True

        return False
