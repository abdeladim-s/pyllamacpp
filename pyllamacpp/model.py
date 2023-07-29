#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a simple Python API around [llama.cpp](https://github.com/ggerganov/llama.cpp)
"""

import logging
import time
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
                 n_gpu_layers: int = 0,
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
        :param n_gpu_layers: number of layers to store in VRAM
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
        self.llama_params.n_gpu_layers = n_gpu_layers
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
        self._prompt_suffix_tokens = pp.llama_tokenize(self._ctx, self.prompt_suffix, False)
        self._last_n_tokens = [0] * self._n_ctx  # n_ctx elements
        self._n_past = 0

    def tokenize(self, text: str):
        """
        Returns a list of tokens for the text
        :param text: text to be tokenized
        :return: List of tokens
        """
        return pp.llama_tokenize(self._ctx, text, True)

    def detokenize(self, tokens: list):
        """
        Returns a list of tokens for the text
        :param text: text to be tokenized
        :return: A string representing the text extracted from the tokens
        """
        return pp.llama_tokens_to_str(self._ctx, tokens)

    def generate(self,
                 prompt: str,
                 n_predict: Union[None, int] = None,
                 n_threads: int = 4,
                 seed: Union[None, int] = None,
                 antiprompt: str = None,
                 n_batch: int = 512,
                 n_keep: int = 0,
                 top_k: int = 40,
                 top_p: float = 0.95,
                 tfs_z: float = 1.00,
                 typical_p: float = 1.00,
                 temp: float = 0.8,
                 repeat_penalty: float = 1.10,
                 repeat_last_n: int = 64,
                 frequency_penalty: float = 0.00,
                 presence_penalty: float = 0.00,
                 mirostat: int = 0,
                 mirostat_tau: int = 5.00,
                 mirostat_eta: int = 0.1,
                 infinite_generation: bool = False) -> Generator:
        """
        Runs llama.cpp inference and yields new predicted tokens from the prompt provided as input

        :param prompt: The prompt :)
        :param n_predict: if n_predict is not None, the inference will stop if it reaches `n_predict` tokens, otherwise
                          it will continue until `EOS`
        :param n_threads: The number of CPU threads
        :param seed: Set rng seed, leave it None for random
        :param antiprompt: aka the stop word, the generation will stop if this word is predicted,
                           keep it None to handle it in your own way
        :param n_batch: batch size for prompt processing (must be >=32 to use BLAS)
        :param n_keep: number of tokens to keep from initial prompt
        :param top_k: top K sampling parameter, <= 0 to use vocab size
        :param top_p: top P sampling parameter, 1.0 = disabled
        :param tfs_z: tfs_z sampling parameter, 1.0 = disabled
        :param typical_p: typical_p sampling parameter, 1.0 = disabled
        :param temp: Temperature, 1.0 = disabled
        :param repeat_penalty: repeat penalty sampling parameter, 1.0 = disabled
        :param repeat_last_n: last n tokens to penalize (0 = disable penalty, -1 = context size)
        :param frequency_penalty: 0.0 = disabled
        :param presence_penalty: 0.0 = disabled
        :param mirostat: 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        :param mirostat_tau: target entropy
        :param mirostat_eta: learning rate
        :param infinite_generation: set it to `True` to make the generation go infinitely

        :return: Tokens generator
        """
        # update params
        self.gpt_params.n_batch = n_batch
        self.gpt_params.n_keep = n_keep
        self.gpt_params.top_k = top_k
        self.gpt_params.top_p = top_p
        self.gpt_params.tfs_z = tfs_z
        self.gpt_params.typical_p = typical_p
        self.gpt_params.temp = temp
        self.gpt_params.repeat_penalty = repeat_penalty
        self.gpt_params.repeat_last_n = repeat_last_n
        self.gpt_params.frequency_penalty = frequency_penalty
        self.gpt_params.presence_penalty = presence_penalty
        self.gpt_params.mirostat = mirostat
        self.gpt_params.mirostat_tau = mirostat_tau
        self.gpt_params.mirostat_eta = mirostat_eta

        if seed is not None:
            pp.llama_set_rng_seed(self._ctx, seed)
        else:
            seed = int(time.time())
            pp.llama_set_rng_seed(self._ctx, seed)

        input_tokens = self._prompt_prefix_tokens + \
                       pp.llama_tokenize(self._ctx, prompt, len(self._prompt_prefix_tokens) == 0) + \
                       self._prompt_suffix_tokens

        # input_tokens = pp.llama_tokenize(self._ctx, prompt, True)

        if len(input_tokens) > self._n_ctx - 4:
            raise Exception('Prompt too long!')
        predicted_tokens = []
        predicted_token = 0

        # add global context if no past yet
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

        n_ctx = pp.llama_n_ctx(self._ctx)

        while infinite_generation or predicted_token != pp.llama_token_eos():
            if len(predicted_tokens) > 0:
                # infinite text generation via context swapping
                if (self._n_past + len(predicted_tokens)) > n_ctx:
                    n_left = self._n_past - self.gpt_params.n_keep
                    self._n_past = max(1, self.gpt_params.n_keep)
                    predicted_tokens[:0] = self._last_n_tokens[
                                           n_ctx - n_left // 2 - len(predicted_tokens):len(self._last_n_tokens) - len(
                                               predicted_tokens)]

                for i in range(0, len(predicted_tokens), self.gpt_params.n_batch):
                    n_eval = len(predicted_tokens) - i
                    if n_eval > self.gpt_params.n_batch:
                        n_eval = self.gpt_params.n_batch

                    pp.llama_eval(self._ctx,
                                  predicted_tokens[i:],
                                  n_eval,
                                  self._n_past,
                                  n_threads)
                    self._n_past += n_eval

            predicted_tokens.clear()

            # sampling
            predicted_token = pp.sample_next_token(self._ctx, self.gpt_params, self._last_n_tokens)

            predicted_tokens.append(predicted_token)
            # tokens come as raw undecoded bytes,
            # and we decode them, replacing those that can't be decoded.
            # I decoded here for fear of breaking the stopword logic,
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
            if n_predict is not None:
                if n_remain == n_predict:
                    break
                else:
                    n_remain += 1
            yield token_str

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
        decoded_text = text.decode('utf-8', 'replace')

        if Model._new_text_callback is not None:
            Model._new_text_callback(decoded_text)
        # save res
        self.res += decoded_text

    def cpp_generate(self, prompt: str,
                     n_predict: int = 128,
                     new_text_callback: Callable[[bytes], None] = None,
                     n_threads: int = 4,
                     top_k: int = 40,
                     top_p: float = 0.95,
                     tfs_z: float = 1.00,
                     typical_p: float = 1.00,
                     temp: float = 0.8,
                     repeat_penalty: float = 1.10,
                     repeat_last_n: int = 64,
                     frequency_penalty: float = 0.00,
                     presence_penalty: float = 0.00,
                     mirostat: int = 0,
                     mirostat_tau: int = 5.00,
                     mirostat_eta: int = 0.1,
                     n_batch: int = 8,
                     n_keep: int = 0,
                     interactive: bool = False,
                     antiprompt: List = [],
                     instruct: bool = False,
                     verbose_prompt: bool = False,
                     ) -> str:
        """
        The generate function from `llama.cpp`

        :param prompt: the prompt
        :param n_predict: number of tokens to generate
        :param new_text_callback: a callback function called when new text is generated, default `None`
        :param n_threads: The number of CPU threads
        :param top_k: top K sampling parameter, <= 0 to use vocab size
        :param top_p: top P sampling parameter, 1.0 = disabled
        :param tfs_z: tfs_z sampling parameter, 1.0 = disabled
        :param typical_p: typical_p sampling parameter, 1.0 = disabled
        :param temp: Temperature, 1.0 = disabled
        :param repeat_penalty: repeat penalty sampling parameter, 1.0 = disabled
        :param repeat_last_n: last n tokens to penalize (0 = disable penalty, -1 = context size)
        :param frequency_penalty: 0.0 = disabled
        :param presence_penalty: 0.0 = disabled
        :param mirostat: 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        :param mirostat_tau: target entropy
        :param mirostat_eta: learning rate
        :param n_batch: GPT params n_batch
        :param n_keep: GPT params n_keep
        :param interactive: interactive communication
        :param anti_prompt: list of anti prompts
        :param instruct: Activate instruct mode
        :param verbose_prompt: verbose prompt
        :return: the new generated text
        """
        self.gpt_params.prompt = prompt
        self.gpt_params.n_predict = n_predict
        self.gpt_params.n_threads = n_threads
        self.gpt_params.top_k = top_k
        self.gpt_params.top_p = top_p
        self.gpt_params.tfs_z = tfs_z
        self.gpt_params.typical_p = typical_p
        self.gpt_params.temp = temp
        self.gpt_params.repeat_penalty = repeat_penalty
        self.gpt_params.repeat_last_n = repeat_last_n
        self.gpt_params.frequency_penalty = frequency_penalty
        self.gpt_params.presence_penalty = presence_penalty
        self.gpt_params.mirostat = mirostat
        self.gpt_params.mirostat_tau = mirostat_tau
        self.gpt_params.mirostat_eta = mirostat_eta
        self.gpt_params.n_batch = n_batch
        self.gpt_params.n_keep = n_keep
        self.gpt_params.interactive = interactive
        self.gpt_params.antiprompt = antiprompt
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

    def get_embeddings(self) -> List[float]:
        """
        Get the embeddings for the input

        :return the last embeddings vector from the context (shape: [n_embd] (1-dimensional))
        """
        assert self.llama_params.embedding, "The model should be instanciated with embedding=True to get the embeddings"
        return pp.llama_get_embeddings(self._ctx)

    def get_prompt_embeddings(self,
                              prompt: str,
                              n_threads: int = 4,
                              n_batch: int = 512
                              ) -> List[float]:
        """
        Get the embeddings of a specific prompt

        :warning: this will reset the context

        :param prompt: the prompt :)
        :param n_threads: The number of CPU threads
        :param n_batch: batch size for prompt processing (must be >=32 to use BLAS)
        :return The embeddings vector
        """
        assert self.llama_params.embedding, "The model should be instanced with embedding=True to get the embeddings"
        self.reset()
        tokens = self.tokenize(prompt)
        for i in range(0, len(tokens), n_batch):
            n_eval = len(tokens) - i
            if n_eval > n_batch:
                n_eval = n_batch

            pp.llama_eval(self._ctx,
                          tokens[i:],
                          n_eval,
                          0,
                          n_threads)
        embeddings = self.get_embeddings()
        self.reset()
        return embeddings

    def __del__(self):
        if self._ctx:
            pp.llama_free(self._ctx)
