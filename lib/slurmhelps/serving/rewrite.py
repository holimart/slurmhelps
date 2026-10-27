"""Rewrite denied strings in OpenAI-compatible requests and responses.

Used by the gateway for Qwen3.8's memorised Alibaba OSS proxy URL: once that
URL is in the conversation the model copies it (the "spiral"), so the gateway

* scrubs every string in the request history (``messages`` / ``prompt``)
  before the model sees it, and
* rewrites the model's output (JSON or SSE stream), so the client never
  stores the URL in the first place.

Rules are regexes applied case-insensitively, in order. A match is replaced by
the rule's replacement text. Matches are whole "tokens": runs of characters
that can appear in a URL or host name (no whitespace, quotes, brackets or
backslashes), which keeps tool-call JSON arguments valid.

Streaming: a URL arrives split over many deltas. For each output field the
rewriter holds back the trailing token (it may still grow into a match) and
releases text once a token boundary arrives, the choice finishes, or the held
token exceeds ``max_hold`` characters.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Callable

TOKEN_CHARS = r"[^\s\"'<>()\[\]{}\\`]"
_TRAILING_TOKEN = re.compile(TOKEN_CHARS + r"+\Z")

# Output text fields in OpenAI-compatible chat/completion payloads.
TEXT_FIELDS = ("content", "reasoning", "reasoning_content", "text")


class Rule:
    def __init__(self, pattern: str, replacement: str, name: str = "") -> None:
        self.name = name or pattern
        self.replacement = replacement
        self.regex = re.compile(pattern, re.IGNORECASE)


def default_rules() -> list[Rule]:
    """Built-in rules for the Qwen3.8 routify habit (no 'alibaba' in replacements)."""
    words = r"(?:alibaba|aliyun|alicdn|dashscope|routi[fz]|ossaccesskeyid)"
    return [
        Rule(r"https?://" + TOKEN_CHARS + r"*?" + words + TOKEN_CHARS + r"*",
             "https://blocked.invalid/removed-url", "url"),
        Rule(TOKEN_CHARS + r"*?" + words + TOKEN_CHARS + r"*", "removed-host", "host"),
        Rule(r"LTAI[0-9A-Za-z]{12,}", "REMOVED_KEY", "key"),
    ]


def load_rules(path: str) -> list[Rule]:
    """Read ``{"rules": [{"pattern", "replacement", "name"?}, ...]}`` (or a bare list)."""
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, dict):
        data = data.get("rules", [])
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty list of rules")
    rules = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("pattern"), str) \
                or not isinstance(item.get("replacement"), str):
            raise ValueError(f"{path}: each rule needs string 'pattern' and 'replacement'")
        rules.append(Rule(item["pattern"], item["replacement"], str(item.get("name", ""))))
    return rules


class Rewriter:
    """Applies rules to text; counts and optionally logs every replacement."""

    def __init__(
        self,
        rules: list[Rule] | None = None,
        *,
        max_hold: int = 4096,
        log_path: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.rules = rules if rules is not None else default_rules()
        self.max_hold = max_hold
        self.log_path = log_path or None
        self.clock = clock
        self._lock = threading.Lock()
        self.counts = {"request": 0, "response": 0}

    # -- text --------------------------------------------------------------
    def rewrite_text(self, text: str, *, where: str = "response", request_id: str = "") -> str:
        if not text:
            return text
        events: list[dict[str, str]] = []
        for rule in self.rules:
            def repl(match: re.Match[str], rule: Rule = rule) -> str:
                if match.group(0) != rule.replacement:
                    events.append({"rule": rule.name, "original": match.group(0)})
                return rule.replacement
            text = rule.regex.sub(repl, text)
        if events:
            self._record(where, request_id, events)
        return text

    def _record(self, where: str, request_id: str, events: list[dict[str, str]]) -> None:
        with self._lock:
            self.counts[where] = self.counts.get(where, 0) + len(events)
            if not self.log_path:
                return
            try:
                with open(self.log_path, "a", encoding="utf-8") as stream:
                    for event in events:
                        stream.write(json.dumps({"ts": round(self.clock(), 3), "where": where,
                                                 "request_id": request_id, **event}) + "\n")
            except OSError:
                pass

    def _walk(self, value: Any, where: str, request_id: str) -> Any:
        if isinstance(value, str):
            return self.rewrite_text(value, where=where, request_id=request_id)
        if isinstance(value, list):
            return [self._walk(item, where, request_id) for item in value]
        if isinstance(value, dict):
            return {key: self._walk(item, where, request_id) for key, item in value.items()}
        return value

    # -- requests ----------------------------------------------------------
    def scrub_request(self, payload: dict[str, Any], request_id: str = "") -> dict[str, Any]:
        """Rewrite the conversation history (not tool schemas or sampling params)."""
        for key in ("messages", "prompt", "input"):
            if key in payload:
                payload[key] = self._walk(payload[key], "request", request_id)
        return payload

    # -- non-streamed responses -------------------------------------------
    def rewrite_response(self, payload: dict[str, Any], request_id: str = "") -> dict[str, Any]:
        if isinstance(payload.get("choices"), list):
            payload["choices"] = self._walk(payload["choices"], "response", request_id)
        return payload

    def stream(self, request_id: str = "") -> "StreamRewriter":
        return StreamRewriter(self, request_id)


class StreamRewriter:
    """Per-response state for rewriting SSE chunks with a hold-back buffer."""

    def __init__(self, rewriter: Rewriter, request_id: str = "") -> None:
        self.rewriter = rewriter
        self.request_id = request_id
        # (choice index, field, tool index or -1) -> held text
        self.held: dict[tuple[int, str, int], str] = {}
        self.template: dict[str, Any] = {}

    def _push(self, key: tuple[int, str, int], piece: str, final: bool) -> str:
        buffer = self.held.pop(key, "") + (piece or "")
        if not final:
            match = _TRAILING_TOKEN.search(buffer)
            if match and len(match.group(0)) <= self.rewriter.max_hold:
                head, tail = buffer[: match.start()], match.group(0)
                if tail:
                    self.held[key] = tail
                buffer = head
        return self.rewriter.rewrite_text(buffer, where="response", request_id=self.request_id)

    def chunk(self, obj: dict[str, Any]) -> dict[str, Any]:
        """Rewrite one parsed SSE ``data:`` object in place and return it."""
        if not self.template:
            self.template = {k: obj[k] for k in ("id", "object", "created", "model") if k in obj}
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            index = int(choice.get("index", 0) or 0)
            final = choice.get("finish_reason") is not None
            target = choice.get("delta") if isinstance(choice.get("delta"), dict) else choice
            seen: set[tuple[int, str, int]] = set()
            for field in TEXT_FIELDS:
                if isinstance(target.get(field), str):
                    key = (index, field, -1)
                    seen.add(key)
                    target[field] = self._push(key, target[field], final)
            for call in target.get("tool_calls") or []:
                fn = call.get("function") if isinstance(call, dict) else None
                if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                    key = (index, "arguments", int(call.get("index", 0) or 0))
                    seen.add(key)
                    fn["arguments"] = self._push(key, fn["arguments"], final)
            if final:
                self._flush_into(index, target, skip=seen)
        return obj

    def _flush_into(self, index: int, target: dict[str, Any], skip: set) -> None:
        for key in sorted(k for k in self.held if k[0] == index and k not in skip):
            text = self._push(key, "", True)
            if not text:
                continue
            _, field, tool_index = key
            if field == "arguments":
                calls = target.setdefault("tool_calls", [])
                calls.append({"index": tool_index, "function": {"arguments": text}})
            else:
                target[field] = (target.get(field) or "") + text

    def drain(self) -> list[dict[str, Any]]:
        """Chunks carrying any text still held when the stream ends."""
        out = []
        for index in sorted({k[0] for k in self.held}):
            delta: dict[str, Any] = {}
            self._flush_into(index, delta, skip=set())
            if delta:
                kind = self.template.get("object", "")
                choice: dict[str, Any] = {"index": index, "finish_reason": None}
                if kind == "text_completion":
                    choice["text"] = delta.get("text", "")
                else:
                    choice["delta"] = delta
                out.append({**self.template, "choices": [choice]})
        return out
