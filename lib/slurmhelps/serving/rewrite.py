"""Rewrite denied strings in OpenAI-compatible requests and responses.

Used by the gateway for Qwen3.8's memorised Alibaba OSS proxy URL: once that
URL is in the conversation the model copies it (the "spiral"), so the gateway

* repairs the request history before the model sees it: each tool call to a
  denied URL is pointed at the URL the model meant (recovered from its own
  reasoning) or, when that cannot be recovered, removed together with its
  result (``repair_history``); remaining denied strings in the history become
  plain words such as "(link removed)", not a URL the model could copy, and
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

# What the gateway puts in place of a denied URL in model output. It comes back
# in the next request's history, where repair_history treats it as denied too.
BLOCKED_URL = "https://blocked.invalid/removed-url"


class Rule:
    """``replacement`` is used in model output; ``history`` (default: the same)
    in the request history."""

    def __init__(self, pattern: str, replacement: str, name: str = "", history: str | None = None) -> None:
        self.name = name or pattern
        self.replacement = replacement
        self.history = replacement if history is None else history
        self.regex = re.compile(pattern, re.IGNORECASE)


def default_rules() -> list[Rule]:
    """Built-in rules for the Qwen3.8 routify habit (no 'alibaba' in replacements)."""
    words = r"(?:alibaba|aliyun|alicdn|dashscope|routi[fz]|ossaccesskeyid)"
    return [
        Rule(r"https?://" + TOKEN_CHARS + r"*?" + words + TOKEN_CHARS + r"*",
             BLOCKED_URL, "url", history="(link removed)"),
        Rule(TOKEN_CHARS + r"*?" + words + TOKEN_CHARS + r"*", "removed-host", "host", history="(host removed)"),
        Rule(r"LTAI[0-9A-Za-z]{12,}", "REMOVED_KEY", "key", history="(key removed)"),
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
        history = item.get("history")
        if history is not None and not isinstance(history, str):
            raise ValueError(f"{path}: 'history' must be a string")
        rules.append(Rule(item["pattern"], item["replacement"], str(item.get("name", "")), history))
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
            new = rule.history if where == "request" else rule.replacement

            def repl(match: re.Match[str], rule: Rule = rule, new: str = new) -> str:
                if match.group(0) not in (rule.replacement, rule.history):
                    events.append({"rule": rule.name, "original": match.group(0)})
                return new
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
        if isinstance(payload.get("messages"), list):
            payload["messages"] = self.repair_history(payload["messages"], request_id)
        for key in ("messages", "prompt", "input"):
            if key in payload:
                payload[key] = self._walk(payload[key], "request", request_id)
        return payload

    def is_denied_url(self, url: str) -> bool:
        return url.startswith(BLOCKED_URL) or any(r.regex.search(url) for r in self.rules)

    def repair_history(self, messages: list[Any], request_id: str = "") -> list[Any]:
        """Point denied tool-call URLs at the intended URL, or drop the call.

        For each assistant message, every tool call whose arguments hold a
        denied URL is handled with ``recover_target``. A recovered call keeps
        its id and result with the URL swapped (in the call, and wherever else
        that exact URL appears in the history), so the model sees that it tried
        the right page. An unrecoverable call is removed with its tool result;
        an assistant message left with no text and no calls is removed too.
        Returns a new list; inputs are not mutated.
        """
        swaps: dict[str, str] = {}
        dropped: set[str] = set()
        out: list[Any] = []
        events: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                out.append(message)
                continue
            if message.get("role") == "tool" and message.get("tool_call_id") in dropped:
                continue
            calls = message.get("tool_calls")
            if message.get("role") != "assistant" or not isinstance(calls, list) or not calls:
                out.append(message)
                continue
            prose = "\n".join(str(message.get(k) or "") for k in ("reasoning", "reasoning_content", "content"))
            used = {u for c in calls for u in _call_urls(c) if not self.is_denied_url(u)}
            kept, changed = [], False
            for call in calls:
                bad = [u for u in _call_urls(call) if self.is_denied_url(u)]
                if not bad:
                    kept.append(call)
                    continue
                changed = True
                targets = {}
                for u in bad:
                    # The placeholder is shared by many calls: never reuse its swap.
                    known = None if u.startswith(BLOCKED_URL) else swaps.get(u)
                    targets[u] = known or recover_target(u, prose, used, self.is_denied_url)
                if all(targets.values()):
                    for u, t in targets.items():
                        if not u.startswith(BLOCKED_URL):
                            swaps[u] = t
                        used.add(t)
                        events.append({"rule": "swap", "original": u, "target": t})
                    order = sorted(targets, key=len, reverse=True)
                    kept.append(_replace_strings(call, lambda s, o=order, t=targets: _swap_all(s, o, t)))
                else:
                    if isinstance(call, dict) and call.get("id"):
                        dropped.add(call["id"])
                    events.append({"rule": "drop", "original": bad[0]})
            if not changed:
                out.append(message)
                continue
            message = {**message, "tool_calls": kept}
            if not kept:
                del message["tool_calls"]
                if not str(message.get("content") or "").strip():
                    continue  # a bare-reasoning assistant turn would only confuse the template
            out.append(message)
        if events:
            self._record("request", request_id, events)
        if swaps:
            order = sorted(swaps, key=len, reverse=True)  # longest first: no partial swaps
            out = _replace_strings(out, lambda s: _swap_all(s, order, swaps))
        # Placeholders left in prose (from earlier rewritten output) become plain words.
        return _replace_strings(out, lambda s: s.replace(BLOCKED_URL, "(link removed)") if BLOCKED_URL in s else s)

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


# -- history repair helpers -------------------------------------------------

_URL_RE = re.compile(r"https?://" + TOKEN_CHARS + r"+")
_EMBEDDED_RE = re.compile(r"https?://" + TOKEN_CHARS + r"+?/\d+/(https?://" + TOKEN_CHARS + r"+)")
_ARXIV_SUFFIX_RE = re.compile(r"\.(\d{4,5})(?:v\d+)?(?:[?#]|$)")
_ARXIV_ID_RE = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])")
_TRAILING_PUNCT = ".,;:!?"


def _call_urls(call: Any) -> list[str]:
    fn = call.get("function") if isinstance(call, dict) else None
    args = fn.get("arguments") if isinstance(fn, dict) else None
    if not isinstance(args, str):
        return []
    return [u.rstrip(_TRAILING_PUNCT) for u in _URL_RE.findall(args.replace("\\/", "/"))]


def recover_target(url: str, prose: str, used: set[str], denied: Callable[[str], bool]) -> str | None:
    """The URL the model meant when it wrote the denied ``url``, or None.

    1. A clean URL embedded in it (``routified.com/0/https://arxiv.org/...``).
    2. Its file suffix is the tail of an arXiv id the reasoning names
       (``.../e5f7....08946?Expires=...`` with "2408.08946" in the reasoning).
    3. The last clean URL the reasoning names that no other call in the same
       message already uses.
    """
    embedded = _EMBEDDED_RE.match(url)
    if embedded and not denied(embedded.group(1)):
        return embedded.group(1).rstrip(_TRAILING_PUNCT)
    path = url.split("?", 1)[0]
    suffix = _ARXIV_SUFFIX_RE.search(path + ("?" if "?" in url else ""))
    if suffix:
        ids = [m for m in _ARXIV_ID_RE.findall(prose) if m.endswith("." + suffix.group(1))]
        if ids:
            return f"https://arxiv.org/abs/{ids[-1]}"
    for candidate in reversed(_URL_RE.findall(prose)):
        candidate = candidate.rstrip(_TRAILING_PUNCT)
        if not denied(candidate) and candidate not in used:
            return candidate
    return None


def _swap_all(text: str, order: list[str], swaps: dict[str, str]) -> str:
    for old in order:
        if old in text:
            text = text.replace(old, swaps[old])
        escaped = json.dumps(old)[1:-1]
        if escaped != old and escaped in text:
            text = text.replace(escaped, json.dumps(swaps[old])[1:-1])
    return text


def _replace_strings(value: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_replace_strings(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _replace_strings(v, fn) for k, v in value.items()}
    return value
