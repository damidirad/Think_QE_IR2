"""LiteLLM proxy backend (added for the API extension, e.g. --generation_model litellm/Qwen3.6).

Same interface as VLLMCompletion (completion_chat, completion_chat_batch, last_prompts, last_meta,
verbose), so thinkqe.py is unchanged apart from choosing the class. Defaults mirror the local vLLM runs:
  - thinking on (the model's chat-template default); --no_thinking sends enable_thinking=False instead
    of the local "Okay, I think I have finished thinking.\n</think>" prefill (not possible over a chat API)
  - prompt + output limited to the same 32768-token window as vLLM's max_model_len
  - one request per sample (exact per-sample token counts and finish reasons for --log_trajectory),
    sent in parallel; sample i uses seed*1000+i so different --seed runs never share a request seed
  - the reasoning returned in `reasoning_content` is put back in front of the answer as
    "<reasoning>\n</think>\n\n<answer>", so --think_split "</think>\n" and the trajectory logs work as
    for local runs; a trace cut off while reasoning has no "</think>", like a truncated local generation

API key: $LITELLM_API_KEY, else `api_token` (or LITELLM_API_KEY) in the repo's .env (gitignored).
The key is never put in argv, run_config.json or the logs.

Robustness (the shared proxy answers 429 "maximum allowed number of requests", and its gateway answers 504
to any non-streamed request that runs longer than ~90 s):
  - requests are streamed (stream=True, usage in the last chunk), so long reasoning traces are not cut off.
  - every request is retried on 408/409/429/5xx, timeouts and connection errors with exponential backoff
    (10 s doubling, capped at 5 min, with jitter) until THINKQE_API_MAX_WAIT seconds of waiting; then the
    job stops visibly. Other errors (e.g. 400 context too long) stop at once.
  - successful responses are cached on disk, keyed by a hash of (model, messages, sampling params, seed).
    A resubmitted job replays finished requests instead of sending them again. Each process appends to its
    own file in the cache folder (no cross-node append races); all files are read at start.

Environment options (all optional):
  LITELLM_API_KEY           key (overrides .env)
  THINKQE_API_CONTEXT       token window for prompt + output, default 32768 (vLLM max_model_len)
  THINKQE_API_CONCURRENCY   parallel requests, default 8
  THINKQE_API_TIMEOUT       seconds per request, default 3600 (32k-token traces are slow)
  THINKQE_API_MAX_WAIT      total seconds of backoff per request before giving up, default 2700 (45 min)
  THINKQE_API_CACHE         cache folder, default $WORK/api_cache (or <repo>/api_cache); "off" disables it
  THINKQE_CHAT_KWARGS       chat-template options, e.g. '{"enable_thinking": false}', sent as chat_template_kwargs
"""
import glob
import hashlib
import json
import math
import os
import random
import re
import socket
import threading
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import httpx
import openai
from openai import OpenAI
from tqdm import tqdm

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

DEFAULT_API_BASE = "https://llmproxy.uva.nl/v1"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Sample = namedtuple("Sample", "text finish_reason prompt_tokens completion_tokens reasoning content")


def _load_api_key():
    key = os.environ.get("LITELLM_API_KEY")
    if key:
        return key
    path = os.path.join(_REPO, ".env")
    if os.path.exists(path):
        values = {}
        with open(path) as fp:
            for line in fp:
                m = re.match(r'\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*)', line)
                if m:
                    values[m.group(1)] = m.group(2).strip().strip('"\'')
        for name in ("LITELLM_API_KEY", "api_token"):
            if values.get(name):
                return values[name]
    raise RuntimeError("No LiteLLM API key: set LITELLM_API_KEY or put api_token=... in " + path)


class LiteLLMCompletion(object):
    def __init__(self, model_name="Qwen3.6", api_base=DEFAULT_API_BASE, control_thinking=False, seed=0):
        self.model_name = model_name[len("litellm/"):] if model_name.startswith("litellm/") else model_name
        self.seed = seed
        self.context = int(os.environ.get("THINKQE_API_CONTEXT", "32768"))
        self.concurrency = int(os.environ.get("THINKQE_API_CONCURRENCY", "8"))
        self.max_wait = float(os.environ.get("THINKQE_API_MAX_WAIT", "2700"))
        # retries are done by _create (long backoff), not by the SDK (a few seconds at most)
        self.client = OpenAI(base_url=api_base, api_key=_load_api_key(),
                             timeout=float(os.environ.get("THINKQE_API_TIMEOUT", "3600")), max_retries=0)
        self.chat_kwargs = json.loads(os.environ.get("THINKQE_CHAT_KWARGS", "{}"))
        if control_thinking:
            self.chat_kwargs["enable_thinking"] = False
        self._lock = threading.Lock()
        self.usage = {"requests": 0, "cached": 0, "retries": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._init_cache()
        print(f"LiteLLM backend: model={self.model_name} api_base={api_base} context={self.context} "
              f"concurrency={self.concurrency} max_wait={self.max_wait:.0f}s chat_template_kwargs={self.chat_kwargs} "
              f"cache={self.cache_dir or 'off'} ({len(self.cache)} cached responses)")
        self.last_prompts = []
        self.last_meta = []
        self.verbose = True

    # ---------- disk cache ----------
    def _init_cache(self):
        self.cache = {}
        setting = os.environ.get("THINKQE_API_CACHE", "")
        if setting.lower() == "off":
            self.cache_dir, self._cache_file = None, None
            return
        root = setting or os.path.join(os.environ.get("WORK", _REPO), "api_cache")
        self.cache_dir = os.path.join(root, re.sub(r"[^A-Za-z0-9._-]", "_", self.model_name))
        os.makedirs(self.cache_dir, exist_ok=True)
        # Files are named <host>_<pid>_<start time>; read oldest first and keep the FIRST response per key, so
        # processes that ran at the same time and both sent a request resolve to the same answer every time.
        paths = sorted(glob.glob(os.path.join(self.cache_dir, "*.jsonl")),
                       key=lambda p: (int(re.search(r"_(\d+)\.jsonl$", p).group(1)) if re.search(r"_(\d+)\.jsonl$", p)
                                      else 0, p))
        for path in paths:
            with open(path, encoding="utf-8") as fp:
                for line in fp:
                    try:
                        rec = json.loads(line)
                        self.cache.setdefault(rec["key"], rec["response"])
                    except (ValueError, KeyError):
                        pass   # a line cut off by a killed job
        self._cache_file = os.path.join(self.cache_dir, f"{socket.gethostname()}_{os.getpid()}_{int(time.time())}.jsonl")

    def _cache_put(self, key, response):
        if not self._cache_file:
            return
        line = json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n"
        with self._lock:
            self.cache[key] = response
            with open(self._cache_file, "a", encoding="utf-8") as fp:
                fp.write(line)

    # ---------- request (streamed) with long backoff ----------
    def _stream_once(self, params):
        """One streamed request. The proxy's gateway answers 504 to non-streamed requests that run longer than
        ~90 s (long reasoning traces); a stream keeps the connection active, so those finish."""
        stream = self.client.chat.completions.create(**params, stream=True, stream_options={"include_usage": True})
        reasoning, content, finish, usage = [], [], None, None
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            for choice in chunk.choices:
                d = choice.delta
                extra = d.model_extra or {}
                r = getattr(d, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning")
                if r:
                    reasoning.append(r)
                if d.content:
                    content.append(d.content)
                finish = choice.finish_reason or finish
        if usage is None or finish is None:
            raise openai.APIConnectionError(message="stream ended without usage/finish_reason", request=None)
        return {"reasoning": "".join(reasoning) or None, "content": "".join(content), "finish_reason": finish,
                "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}

    def _create(self, **kwargs):
        waited, attempt = 0.0, 0
        while True:
            try:
                return self._stream_once(kwargs)
            except (openai.APITimeoutError, openai.APIConnectionError, openai.APIStatusError, httpx.HTTPError) as e:
                status = getattr(e, "status_code", None)
                msg = str(e)
                retryable = status is None or status in RETRY_STATUS
                if not retryable or "budget" in msg.lower() or waited >= self.max_wait:
                    raise
                wait = min(300.0, 10.0 * 2 ** attempt) * random.uniform(0.5, 1.0)
                wait = min(wait, self.max_wait - waited + 1)
                attempt += 1
                with self._lock:
                    self.usage["retries"] += 1
                print(f"[LiteLLM retry] {type(e).__name__} {status or ''}: attempt {attempt}, waiting {wait:.0f}s "
                      f"(waited {waited:.0f}s so far) | {msg[:120]}", flush=True)
                time.sleep(wait)
                waited += wait

    # ---------- helpers ----------
    @staticmethod
    def _render(messages):
        """Readable prompt for the logs (the chat template itself is applied by the server)."""
        return "\n\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)

    def _max_tokens(self, messages, max_tokens):
        # Local vLLM stops at prompt + output = max_model_len. The server's tokenizer is not available here, so
        # estimate the prompt generously (3 characters per token + template overhead) and leave the rest to output.
        est_prompt = math.ceil(sum(len(m["content"]) for m in messages) / 3) + 64
        return max(1, min(max_tokens, self.context - est_prompt))

    def _sample(self, messages, max_tokens, temperature, top_p, seed):
        params = dict(model=self.model_name, messages=messages, n=1,
                      max_tokens=self._max_tokens(messages, max_tokens), temperature=temperature, top_p=top_p,
                      seed=seed, extra_body={"chat_template_kwargs": self.chat_kwargs} if self.chat_kwargs else None)
        key = hashlib.sha256(json.dumps(params, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        resp = self.cache.get(key)
        if resp is not None:
            with self._lock:
                self.usage["cached"] += 1
        else:
            resp = self._create(**params)
            with self._lock:
                self.usage["requests"] += 1
                self.usage["prompt_tokens"] += resp["prompt_tokens"]
                self.usage["completion_tokens"] += resp["completion_tokens"]
            self._cache_put(key, resp)
        reasoning, content = resp["reasoning"], resp["content"]
        if reasoning:
            text = reasoning.rstrip("\n")
            if content or resp["finish_reason"] != "length":   # reasoning finished -> answer follows
                text += "\n</think>\n\n" + content.lstrip("\n")
        else:                                                   # thinking off (or think tags already inline)
            text = content
        return Sample(text, resp["finish_reason"], resp["prompt_tokens"], resp["completion_tokens"],
                      reasoning or "", content)

    @staticmethod
    def _meta(samples):
        return {
            "prompt_tokens": samples[0].prompt_tokens,
            "finish_reasons": [s.finish_reason for s in samples],   # "stop" or "length"
            "completion_tokens": [s.completion_tokens for s in samples],
            # separate fields for callers that need them (rag_eval.py); thinkqe.py's logs ignore these keys
            "reasoning": [s.reasoning for s in samples],
            "content": [s.content for s in samples],
        }

    def _run(self, messages_list, n, max_tokens, temperature, top_p, progress):
        jobs = [(pi, i) for pi in range(len(messages_list)) for i in range(n)]

        def one(job):
            pi, i = job
            return self._sample(messages_list[pi], max_tokens, temperature, top_p, self.seed * 1000 + i)

        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(jobs))) as pool:
            it = pool.map(one, jobs)
            if progress:
                it = tqdm(it, total=len(jobs), desc="LiteLLM requests", mininterval=30)
            flat = list(it)   # input order; the first error (after the client's retries) stops the job visibly
        grouped = [flat[pi * n:(pi + 1) * n] for pi in range(len(messages_list))]
        self.last_prompts = [self._render(m) for m in messages_list]
        self.last_meta = [self._meta(g) for g in grouped]
        return [[s.text for s in g] for g in grouped]

    def _usage_line(self):
        u = self.usage
        return (f"LiteLLM usage so far: {u['requests']} requests, {u['prompt_tokens']} prompt + "
                f"{u['completion_tokens']} completion tokens | {u['cached']} from cache, {u['retries']} retries")

    # ---------- interface used by thinkqe.py ----------
    def completion_chat(self, messages, top_passages=None, max_tokens=32768, temperature=1, top_p=1, n=2):
        if self.verbose:
            print(f"args: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, n={n}")
            print(f"INPUT TO the model: {self._render(messages)}")
        responses = self._run([messages], n, max_tokens, temperature, top_p, progress=False)[0]
        if self.verbose:
            print(self._usage_line())
        return responses

    def completion_chat_batch(self, messages_list, max_tokens=32768, temperature=1, top_p=1, n=2):
        print(f"batch args: prompts={len(messages_list)}, max_tokens={max_tokens}, temperature={temperature}, "
              f"top_p={top_p}, n={n}")
        responses = self._run(messages_list, n, max_tokens, temperature, top_p, progress=True)
        print(self._usage_line())
        return responses
