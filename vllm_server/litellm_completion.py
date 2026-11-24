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

Environment options (all optional):
  LITELLM_API_KEY           key (overrides .env)
  THINKQE_API_CONTEXT       token window for prompt + output, default 32768 (vLLM max_model_len)
  THINKQE_API_CONCURRENCY   parallel requests, default 16
  THINKQE_API_TIMEOUT       seconds per request, default 3600 (32k-token traces are slow)
  THINKQE_API_MAX_RETRIES   retries on connection errors / 408 / 409 / 429 / 5xx, default 8
  THINKQE_CHAT_KWARGS       chat-template options, e.g. '{"enable_thinking": false}', sent as chat_template_kwargs
"""
import json
import math
import os
import re
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from tqdm import tqdm

DEFAULT_API_BASE = "https://llmproxy.uva.nl/v1"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Sample = namedtuple("Sample", "text finish_reason prompt_tokens completion_tokens")


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
        self.concurrency = int(os.environ.get("THINKQE_API_CONCURRENCY", "16"))
        self.client = OpenAI(base_url=api_base, api_key=_load_api_key(),
                             timeout=float(os.environ.get("THINKQE_API_TIMEOUT", "3600")),
                             max_retries=int(os.environ.get("THINKQE_API_MAX_RETRIES", "8")))
        self.chat_kwargs = json.loads(os.environ.get("THINKQE_CHAT_KWARGS", "{}"))
        if control_thinking:
            self.chat_kwargs["enable_thinking"] = False
        print(f"LiteLLM backend: model={self.model_name} api_base={api_base} context={self.context} "
              f"concurrency={self.concurrency} chat_template_kwargs={self.chat_kwargs}")
        self.last_prompts = []
        self.last_meta = []
        self.verbose = True
        self._lock = threading.Lock()
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}

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
        r = self.client.chat.completions.create(
            model=self.model_name, messages=messages, n=1,
            max_tokens=self._max_tokens(messages, max_tokens), temperature=temperature, top_p=top_p, seed=seed,
            extra_body={"chat_template_kwargs": self.chat_kwargs} if self.chat_kwargs else None)
        choice = r.choices[0]
        extra = choice.message.model_extra or {}
        reasoning = getattr(choice.message, "reasoning_content", None) or extra.get("reasoning_content") \
            or extra.get("reasoning")
        content = choice.message.content or ""
        if reasoning:
            text = reasoning.rstrip("\n")
            if content or choice.finish_reason != "length":   # reasoning finished -> answer follows
                text += "\n</think>\n\n" + content.lstrip("\n")
        else:                                                 # thinking off (or think tags already inline)
            text = content
        with self._lock:
            self.usage["requests"] += 1
            self.usage["prompt_tokens"] += r.usage.prompt_tokens
            self.usage["completion_tokens"] += r.usage.completion_tokens
        return Sample(text, choice.finish_reason, r.usage.prompt_tokens, r.usage.completion_tokens)

    @staticmethod
    def _meta(samples):
        return {
            "prompt_tokens": samples[0].prompt_tokens,
            "finish_reasons": [s.finish_reason for s in samples],   # "stop" or "length"
            "completion_tokens": [s.completion_tokens for s in samples],
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
                f"{u['completion_tokens']} completion tokens")

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
