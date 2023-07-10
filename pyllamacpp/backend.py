#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified backend for llama.cpp and ggllm.cpp
"""
import _pyllamacpp as llama
import _pyggllmcpp as falcon

from enum import Enum


class BackendType(Enum):
    llamacpp = 0
    ggllmcpp = 1


class UnifiedBackend:

    def __init__(self, backend: BackendType):
        self.backend = backend

    def context_default_params(self):
        if self.backend == BackendType.llamacpp:
            return llama.llama_context_default_params()
        elif self.backend == BackendType.ggllmcpp:
            return falcon.falcon_context_default_params()
        else:
            # maybe other backends in the future
            pass

    def init_from_file(self, model_path, params):
        if self.backend == BackendType.llamacpp:
            return llama.llama_init_from_file(model_path, params)
        elif self.backend == BackendType.ggllmcpp:
            falcon.falcon_init_backend()
            return falcon.falcon_init_from_file(model_path, params)
        else:
            pass

    def gpt_params(self):
        return llama.gpt_params()

    def n_ctx(self, ctx):
        if self.backend == BackendType.llamacpp:
            return llama.llama_n_ctx(ctx)
        elif self.backend == BackendType.ggllmcpp:
            return falcon.falcon_n_ctx(ctx)
        else:
            pass

    def tokenize(self, ctx, text, add_bos):
        if self.backend == BackendType.llamacpp:
            return llama.llama_tokenize(ctx, text, add_bos)
        elif self.backend == BackendType.ggllmcpp:
            return falcon.falcon_tokenize(ctx, text, add_bos)
        else:
            pass

    def set_rng_seed(self, ctx, seed):
        if self.backend == BackendType.llamacpp:
            return llama.llama_set_rng_seed(ctx, seed)
        elif self.backend == BackendType.ggllmcpp:
            return falcon.falcon_set_rng_seed(ctx, seed)
        else:
            pass

    def eval(self, ctx, tokens, n_eval, n_past, n_threads, debug_timings=0):
        if self.backend == BackendType.llamacpp:
            return llama.llama_eval(ctx, tokens, n_eval, n_past, n_threads)
        elif self.backend == BackendType.ggllmcpp:
            falcon.falcon_prepare_buffers(ctx, n_eval, len(tokens))
            return falcon.falcon_eval(ctx, tokens, n_eval, n_past, n_threads, debug_timings)
        else:
            pass

    def sample_next_token(self, ctx, gpt_params, las_n_tokens):
        if self.backend == BackendType.llamacpp:
            return llama.llama_sample_next_token(ctx, gpt_params, las_n_tokens)
        elif self.backend == BackendType.ggllmcpp:
            return falcon.falcon_sample_next_token(ctx, gpt_params, las_n_tokens)
        else:
            pass

    def token_to_str(self, ctx, token):
        if self.backend == BackendType.llamacpp:
            return llama.llama_token_to_str(ctx, token)
        elif self.backend == BackendType.ggllmcpp:
            return falcon.falcon_token_to_str(ctx, token)
        else:
            pass

    def llama_generate(self, ctx, gpt_params, callback):
        return llama.llama_generate(ctx, gpt_params, callback)

    def free(self, ctx):
        if self.backend == BackendType.llamacpp:
            return llama.llama_free(ctx)
        elif self.backend == BackendType.ggllmcpp:
            return falcon.falcon_free(ctx)
        else:
            pass

    def token_eos(self):
        if self.backend == BackendType.llamacpp:
            return llama.llama_token_eos()
        elif self.backend == BackendType.ggllmcpp:
            return falcon.falcon_token_eos()
        else:
            pass

