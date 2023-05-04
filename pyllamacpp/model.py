#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a simple Python API around [llama.cpp](https://github.com/ggerganov/llama.cpp)
"""

import logging
from pathlib import Path
from typing import Callable, Tuple, Union, List, Generator
import pyllamacpp.constants as constants
from pyllamacpp._logger import set_log_level

__author__ = "abdeladim-s"
__github__ = "https://github.com/abdeladim-s/pyllamacpp"
__copyright__ = "Copyright 2023, "
__license__ = "MIT"

import logging
import _pyllamacpp as pp


class Model:
    """
    A simple Python class on top of llama.cpp

    Example usage
    ```python
    from pyllamacpp.model import Model

    model = Model(ggml_model='path/to/ggml/model')
    for token in model.generate("Tell me a joke ?"):
        print(token, end='', flush=True)
    ```
    """
    _new_text_callback = None

    def __init__(self,
                 model_path: str,
                 prompt_context: str = '',
                 prompt_prefix: str = '',
                 prompt_suffix: str = '',
                 log_level: int = logging.ERROR,
                 n_ctx: int = 512,
                 seed: int = 0,
                 n_parts: int = -1,
                 f16_kv: bool = False,
                 logits_all: bool = False,
                 vocab_only: bool = False,
                 use_mlock: bool = False,
                 embedding: bool = False):
        """
        :param model_path: the path to the ggml model
        :param prompt_context: the global context of the interaction
        :param prompt_prefix: the prompt prefix
        :param prompt_suffix: the prompt suffix
        :param log_level: logging level, set to INFO by default
        :param n_ctx: LLaMA context
        :param seed: random seed
        :param n_parts: LLaMA n_parts
        :param f16_kv: use fp16 for KV cache
        :param logits_all: the llama_eval() call computes all logits, not just the last one
        :param vocab_only: only load the vocabulary, no weights
        :param use_mlock: force system to keep model in RAM
        :param embedding: embedding mode only
        """

        # set logging level
        set_log_level(log_level)
        self._ctx = None

        if not Path(model_path).is_file():
            raise Exception(f"File {model_path} not found!")

        self.llama_params = pp.llama_context_default_params()
        # update llama_params
        self.llama_params.n_ctx = n_ctx
        self.llama_params.seed = seed
        self.llama_params.n_parts = n_parts
        self.llama_params.f16_kv = f16_kv
        self.llama_params.logits_all = logits_all
        self.llama_params.vocab_only = vocab_only
        self.llama_params.use_mlock = use_mlock
        self.llama_params.embedding = embedding

        self._ctx = pp.llama_init_from_file(model_path, self.llama_params)

        # gpt params
        self.gpt_params = pp.gpt_params()

        self.res = ""

        self._n_ctx = pp.llama_n_ctx(self._ctx)
        self._last_n_tokens = [0] * self._n_ctx  # n_ctx elements
        self._n_past = 0
        self.prompt_cntext = prompt_context
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix

        self._prompt_context_tokens = []
        self._prompt_prefix_tokens = []
        self._prompt_suffix_tokens = []

        self.reset()

    def reset(self) -> None:
        """Resets the context"""
        self._prompt_context_tokens = pp.llama_tokenize(self._ctx, self.prompt_cntext, True)
        self._prompt_prefix_tokens = pp.llama_tokenize(self._ctx, self.prompt_prefix, True)
        self._prompt_suffix_tokens = pp.llama_tokenize(self._ctx, self.prompt_suffix, True)
        self._last_n_tokens = [0] * self._n_ctx  # n_ctx elements
        self._n_past = 0

    def generate(self,
                 prompt: str,
                 n_predict: Union[None, int] = None,
                 antiprompt: str = None,
                 infinite_generation: bool = False,
                 n_threads: int = 4,
                 repeat_last_n: int = 64,
                 top_k: int = 40,
                 top_p: float = 0.95,
                 temp: float = 0.8,
                 repeat_penalty: float = 1.10) -> Generator:
        """
        Runs llama.cpp inference and yields new predicted tokens from the prompt provided as input

        :param prompt: The prompt :)
        :param n_predict: if n_predict is not None, the inference will stop if it reaches `n_predict` tokens, otherwise
                          it will continue until `EOS`
        :param antiprompt: aka the stop word, the generation will stop if this word is predicted,
                           keep it None to handle it in your own way
        :param infinite_generation: set it to `True` to make the generation go infinitely
        :param n_threads: The number of CPU threads
        :param repeat_last_n: last n tokens to penalize
        :param top_k: top K sampling parameter
        :param top_p: top P sampling parameter
        :param temp: temperature
        :param repeat_penalty: repeat penalty sampling parameter
        :return: Tokens generator
        """
        input_tokens = self._prompt_prefix_tokens + pp.llama_tokenize(self._ctx, prompt,
                                                                      True) + self._prompt_suffix_tokens
        if len(input_tokens) > self._n_ctx - 4:
            raise Exception('Prompt too long!')
        predicted_tokens = []
        predicted_token = 0

        # add global context for the first time
        if self._n_past == 0:
            for tok in self._prompt_context_tokens:
                predicted_tokens.append(tok)
                self._last_n_tokens.pop(0)
                self._last_n_tokens.append(tok)

        # consume input tokens
        for tok in input_tokens:
            predicted_tokens.append(tok)
            self._last_n_tokens.pop(0)
            self._last_n_tokens.append(tok)

        n_remain = 0
        if antiprompt is not None:
            sequence_queue = []
            stop_word = antiprompt.strip()

        while infinite_generation or predicted_token != pp.llama_token_eos():
            if len(predicted_tokens) > 0:
                if (pp.llama_eval(self._ctx,
                                  predicted_tokens,
                                  len(predicted_tokens),
                                  self._n_past,
                                  n_threads)):
                    raise Exception("failed to eval the model!")
                self._n_past += len(predicted_tokens)
                predicted_tokens.clear()

            predicted_token = pp.llama_sample_top_p_top_k(self._ctx,
                                                          self._last_n_tokens[self._n_ctx - repeat_last_n:],
                                                          repeat_last_n,
                                                          top_k,
                                                          top_p,
                                                          temp,
                                                          repeat_penalty)

            predicted_tokens.append(predicted_token)
            # tokens come as raw undecoded bytes,
            # and we decode them, replacing those that can't be decoded.
            # i decoded here for fear of breaking the stopword logic, 
            token_str = pp.llama_token_to_str(self._ctx, predicted_token).decode('utf-8', "replace")
            if antiprompt is not None:
                if token_str == '\n':
                    sequence_queue.append(token_str)
                    continue
                if len(sequence_queue) != 0:
                    if stop_word.startswith(''.join(sequence_queue).strip()):
                        sequence_queue.append(token_str)
                        if ''.join(sequence_queue).strip() == stop_word:
                            break
                        else:
                            continue
                    else:
                        # consume sequence queue tokens
                        while len(sequence_queue) != 0:
                            yield sequence_queue.pop(0)
                        sequence_queue = []
            self._last_n_tokens.pop(0)
            self._last_n_tokens.append(predicted_token)
            yield token_str
            if n_predict is not None:
                if n_remain == n_predict:
                    break
                else:
                    n_remain += 1

    @staticmethod
    def _set_params(params, kwargs: dict) -> None:
        """
        Private method to set the kwargs params to the `Params` class
        :param kwargs: dict like object for the different params
        :return: None
        """
        for param in kwargs:
            setattr(params, param, kwargs[param])

    def _call_new_text_callback(self, text) -> None:
        """
        Internal new_segment_callback, it just calls the user's callback with the `Segment` object
        :return: None
        """
        if Model._new_text_callback is not None:
            Model._new_text_callback(text)
        # save res
        self.res += text.decode('utf-8', 'replace')

    def cpp_generate(self, prompt: str,
                     n_predict: int = 128,
                     new_text_callback: Callable[[bytes], None] = None,
                     n_threads: int = 4,
                     repeat_last_n: int = 64,
                     top_k: int = 40,
                     top_p: float = 0.95,
                     temp: float = 0.8,
                     repeat_penalty: float = 1.10,
                     n_batch: int = 8,
                     n_keep: int = 0,
                     interactive: bool = False,
                     antiprompt: List = [],
                     ignore_eos: bool = False,
                     instruct: bool = False,
                     verbose_prompt: bool = False,
                     ) -> str:
        """
        The generate function from `llama.cpp`

        :param prompt: the prompt
        :param n_predict: number of tokens to generate
        :param new_text_callback: a callback function called when new text is generated, default `None`
        :param n_threads: The number of CPU threads
        :param repeat_last_n: last n tokens to penalize
        :param top_k: top K sampling parameter
        :param top_p: top P sampling parameter
        :param temp: temperature
        :param repeat_penalty: repeat penalty sampling parameter
        :param n_batch: GPT params n_batch
        :param n_keep: GPT params n_keep
        :param interactive: interactive communication
        :param antiprompt: list of anti prompts
        :param ignore_eos: Ignore LLaMA EOS
        :param instruct: Activate instruct mode
        :param verbose_prompt: verbose prompt
        :return: the new generated text
        """
        self.gpt_params.prompt = prompt
        self.gpt_params.n_predict = n_predict
        # update other params if any
        self.gpt_params.n_threads = n_threads
        self.gpt_params.repeat_last_n = repeat_last_n
        self.gpt_params.top_k = top_k
        self.gpt_params.top_p = top_p
        self.gpt_params.temp = temp
        self.gpt_params.repeat_penalty = repeat_penalty
        self.gpt_params.n_batch = n_batch
        self.gpt_params.n_keep = n_keep
        self.gpt_params.interactive = interactive
        self.gpt_params.antiprompt = antiprompt
        self.gpt_params.ignore_eos = ignore_eos
        self.gpt_params.instruct = instruct
        self.gpt_params.verbose_prompt = verbose_prompt

        # assign new_text_callback
        self.res = ""
        Model._new_text_callback = new_text_callback

        # run the prediction
        pp.llama_generate(self._ctx, self.gpt_params, self._call_new_text_callback)
        return self.res

    @staticmethod
    def get_params(params) -> dict:
        """
        Returns a `dict` representation of the params
        :return: params dict
        """
        res = {}
        for param in dir(params):
            if param.startswith('__'):
                continue
            res[param] = getattr(params, param)
        return res

    def llama_print_timings(self):
        pp.llama_print_timings(self._ctx)

    @staticmethod
    def llama_print_system_info():
        pp.llama_print_system_info()

    def __del__(self):
        if self._ctx:
            pp.llama_free(self._ctx)
