#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a simple Python API around [llama.cpp](https://github.com/ggerganov/llama.cpp)
"""

import logging
from pathlib import Path
from typing import Callable, Tuple, Union
import pyllamacpp.constants as constants
from pyllamacpp._logger import set_log_level

__author__ = "abdeladim-s"
__github__ = "https://github.com/abdeladim-s/pyllamacpp"
__copyright__ = "Copyright 2023, "
__license__ = "MIT"

import logging
import sys
import _pyllamacpp as pp


class Model:
    """
    A simple Python class on top of llama.cpp

    Example usage
    ```python
    from pyllamacpp.model import Model
    import sys

    model = Model(ggml_model='./models/ggml-model-f16-q4_0.bin')
    for token in model.generate("Tell me a joke ?"):
        print(token, end='', flush=True)
    ```
    """
    _new_text_callback = None

    def __init__(self,
                 ggml_model: str,
                 prompt_context=constants.PROMPT_CONTEXT,
                 prompt_prefix=constants.PROMPT_PREFIX,
                 prompt_suffix=constants.PROMPT_SUFFIX,
                 anti_prompts=[],
                 log_level: int = logging.ERROR,
                 **llama_params):
        """
        :param ggml_model: the path to the ggml model
        :param prompt_context: the global context of the interaction, default to [PROMPT_CONTEXT](/pyllamacpp/#pyllamacpp.constants.PROMPT_CONTEXT)
        :param prompt_prefix: the prompt prefix, default to [PROMPT_PREFIX](/pyllamacpp/#pyllamacpp.constants.PROMPT_PREFIX)
        :param prompt_suffix: the prompt suffix, default to [PROMPT_SUFFIX](/pyllamacpp/#pyllamacpp.constants.PROMPT_SUFFIX)
        :param anti_prompts: The inference will stop if an anti_prompt is detected, it will always contain the `prompt_prefix`
        :param log_level: logging level, set to INFO by default
        :param llama_params: keyword arguments for different whisper.cpp parameters,
                        see [PARAMS_SCHEMA](/pyllamacpp/#pyllamacpp.constants.LLAMA_CONTEXT_PARAMS_SCHEMA)
        """
        # set logging level
        set_log_level(log_level)
        self._ctx = None

        if not Path(ggml_model).is_file():
            raise Exception(f"File {ggml_model} not found!")

        self.llama_params = pp.llama_context_default_params()
        # update llama_params
        self._set_params(self.llama_params, llama_params)

        self._ctx = pp.llama_init_from_file(ggml_model, self.llama_params)

        # gpt params
        self.gpt_params = pp.gpt_params()

        self.res = ""

        self._n_ctx = pp.llama_n_ctx(self._ctx)
        self._last_n_tokens = [0] * self._n_ctx  # n_ctx elements
        self._n_past = 0
        self.prompt_cntext = prompt_context
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.anti_prompts = anti_prompts

        self._prompt_context_tokens = []
        self._prompt_prefix_tokens = []

        self.reset()

    def reset(self):
        self._prompt_context_tokens = pp.llama_tokenize(self._ctx, self.prompt_cntext, True)
        self._prompt_prefix_tokens = pp.llama_tokenize(self._ctx, self.prompt_prefix, True)
        self.anti_prompts.append(self.prompt_prefix)
        self._last_n_tokens = [0] * self._n_ctx  # n_ctx elements
        self._n_past = 0

    def _is_anti_prompt(self, predicted_word: str) -> Tuple[bool, Union[None, str]]:
        """
        Returns True if an anti_prompt is detected
        :param predicted_word: the predicted word
        :return: Tuple[bool, Union[str, None]]
        """
        if predicted_word == '':
            return False, None
        for word in self.anti_prompts:
            if word.startswith(predicted_word):
                if word == predicted_word:
                    return True, None
                else:
                    return True, predicted_word

        return False, word

    def generate(self,
                 prompt: str,
                 n_predict: Union[None, int] = None,
                 infinite_generation: bool = False,
                 n_threads: int = 4,
                 repeat_last_n: int = 128,
                 top_k: int = 40,
                 top_p: float = 0.95,
                 temp: float = 0.8,
                 repeat_penalty: float = 1.10,
                 verbose: bool = True):
        """
        Runs llama.cpp inference and yields new predicted tokens from the prompt provided as input

        :param prompt: The prompt :)
        :param n_predict: if n_predict is not None, the inference will stop if it reaches `n_predict` tokens, otherwise
                          it will continue until `EOS`
        :param infinite_generation: set it to `True` to make the generation go infinitely
        :param n_threads: The number of CPU threads
        :param repeat_last_n: last n tokens to penalize
        :param top_k: top K sampling parameter
        :param top_p: top P sampling parameter
        :param temp: temperature
        :param repeat_penalty: repeat penalty sampling parameter
        :param verbose: if `True`, `llama.cpp` stuff will be printed
        :return: Tokens generator
        """
        prompt = f' {self.prompt_prefix}{prompt}{self.prompt_suffix}'
        input_tokens = pp.llama_tokenize(self._ctx, prompt, True)
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

        predicted_word = ""
        n_remain = 0

        tokens_queue = []

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
            token_str = pp.llama_token_to_str(self._ctx, predicted_token)
            predicted_word = predicted_word + token_str
            anti_prompt_flag, previous_word = self._is_anti_prompt(predicted_word)
            if anti_prompt_flag and previous_word is None:
                logging.info(f'Anti prompt {predicted_word} detected'.strip())
                break
            elif anti_prompt_flag and previous_word is not None:
                predicted_word = previous_word
                tokens_queue.append(token_str)
                continue
            else:
                self._last_n_tokens.pop(0)
                self._last_n_tokens.append(predicted_token)
                predicted_word = token_str
                # consume tokens_queue first
                while len(tokens_queue) != 0:
                    yield tokens_queue.pop(0)

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
        self.res += text

    def _generate(self, prompt: str,
                 n_predict: int = 128,
                 new_text_callback: Callable[[str], None] = None,
                 verbose: bool = False,
                 **gpt_params) -> str:
        """
        Runs llama.cpp inference to generate new text content from the prompt provided as input

        :param prompt: the prompt
        :param n_predict: number of tokens to generate
        :param new_text_callback: a callback function called when new text is generated, default `None`
        :param verbose: print some info about the inference
        :param gpt_params: any other llama.cpp params see [PARAMS_SCHEMA](/pyllamacpp/#pyllamacpp.constants.GPT_PARAMS_SCHEMA)
        :return: the new generated text
        """
        self.gpt_params.prompt = prompt
        self.gpt_params.n_predict = n_predict
        # update other params if any
        self._set_params(self.gpt_params, gpt_params)

        # assign new_text_callback
        self.res = ""
        Model._new_text_callback = new_text_callback

        # run the prediction
        pp.llama_generate(self._ctx, self.gpt_params, self._call_new_text_callback, verbose)
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

    @staticmethod
    def get_params_schema() -> dict:
        """
        A simple link to [PARAMS_SCHEMA](/pyllamacpp/#pyllamacpp.constants.PARAMS_SCHEMA)
        :return: dict of params schema
        """
        return constants.GPT_PARAMS_SCHEMA

    def llama_print_timings(self):
        pp.llama_print_timings(self._ctx)

    @staticmethod
    def llama_print_system_info():
        pp.llama_print_system_info()

    def __del__(self):
        if self._ctx:
            pp.llama_free(self._ctx)
