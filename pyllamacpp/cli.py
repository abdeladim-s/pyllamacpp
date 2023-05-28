#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple command line interface
"""

import argparse
import importlib.metadata
import logging

__version__ = importlib.metadata.version('pyllamacpp')

__header__ = f"""

██████╗ ██╗   ██╗██╗     ██╗      █████╗ ███╗   ███╗ █████╗  ██████╗██████╗ ██████╗ 
██╔══██╗╚██╗ ██╔╝██║     ██║     ██╔══██╗████╗ ████║██╔══██╗██╔════╝██╔══██╗██╔══██╗
██████╔╝ ╚████╔╝ ██║     ██║     ███████║██╔████╔██║███████║██║     ██████╔╝██████╔╝
██╔═══╝   ╚██╔╝  ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║██║     ██╔═══╝ ██╔═══╝ 
██║        ██║   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║╚██████╗██║     ██║     
╚═╝        ╚═╝   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝     ╚═╝     
                                                                                    

PyLLaMACpp
A simple Command Line Interface to test the package
Version: {__version__} 

         
=========================================================================================
"""

from pyllamacpp.model import Model

LLAMA_CONTEXT_PARAMS_SCHEMA = {
    'n_ctx': {
        'type': int,
        'description': "text context",
        'options': None,
        'default': 512
    },
    'seed': {
        'type': int,
        'description': "RNG seed",
        'options': None,
        'default': -1
    },
    'f16_kv': {
        'type': bool,
        'description': "use fp16 for KV cache",
        'options': None,
        'default': False
    },
    'logits_all': {
        'type': bool,
        'description': "the llama_eval() call computes all logits, not just the last one",
        'options': None,
        'default': False
    },
    'vocab_only': {
        'type': bool,
        'description': "only load the vocabulary, no weights",
        'options': None,
        'default': False
    },
    'use_mlock': {
        'type': bool,
        'description': "force system to keep model in RAM",
        'options': None,
        'default': False
    },
    'embedding': {
        'type': bool,
        'description': "embedding mode only",
        'options': None,
        'default': False
    }
}

GPT_PARAMS_SCHEMA = {
    'n_predict': {
            'type': int,
            'description': "Number of tokens to predict",
            'options': None,
            'default': 256
    },
    'n_threads': {
            'type': int,
            'description': "Number of threads",
            'options': None,
            'default': 4
    },
    'repeat_last_n': {
            'type': int,
            'description': "Last n tokens to penalize",
            'options': None,
            'default': 64
    },
    # sampling params
    'n_keep': {
        'type': int,
        'description': "n_keep",
        'options': None,
        'default': 48
    },
    'top_k': {
            'type': int,
            'description': "top_k",
            'options': None,
            'default': 40
    },
    'top_p': {
            'type': float,
            'description': "top_p",
            'options': None,
            'default': 0.95
    },
    'temp': {
            'type': float,
            'description': "temp",
            'options': None,
            'default': 0.8
    },
    'repeat_penalty': {
            'type': float,
            'description': "repeat_penalty",
            'options': None,
            'default': 1.0
    },
    'n_batch': {
            'type': int,
            'description': "batch size for prompt processing",
            'options': None,
            'default': 1024
    }
}


def _get_llama_context_params(args) -> dict:
    """
    Helper function to get params from argparse as a `dict`
    """
    params = {}
    for arg in args.__dict__:
        if arg in LLAMA_CONTEXT_PARAMS_SCHEMA.keys() and getattr(args, arg) is not None:
            if LLAMA_CONTEXT_PARAMS_SCHEMA[arg]['type'] is bool:
                if getattr(args, arg).lower() == 'false':
                    params[arg] = False
                else:
                    params[arg] = True
            else:
                params[arg] = LLAMA_CONTEXT_PARAMS_SCHEMA[arg]['type'](getattr(args, arg))
    return params


def _get_gpt_params(args) -> dict:
    """
    Helper function to get params from argparse as a `dict`
    """
    params = {}
    for arg in args.__dict__:
        if arg in GPT_PARAMS_SCHEMA.keys() and getattr(args, arg) is not None:
            if GPT_PARAMS_SCHEMA[arg]['type'] is bool:
                if getattr(args, arg).lower() == 'false':
                    params[arg] = False
                else:
                    params[arg] = True
            else:
                params[arg] = GPT_PARAMS_SCHEMA[arg]['type'](getattr(args, arg))
    return params


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


PROMPT_CONTEXT = "Below is an instruction that describes a task. Write a response that appropriately completes the " \
                 "request\n"
PROMPT_PREFIX = "\n\n##Instruction:\n"
PROMPT_SUFFIX = "\n\n##Response:\n"


def run(args):
    print(f"[+] Running model `{args.model}`")
    llama_params = _get_llama_context_params(args)
    print(f"[+] LLaMA context params: `{llama_params}`")
    gpt_params = _get_gpt_params(args)
    print(f"[+] GPT params: `{gpt_params}`")
    model = Model(model_path=args.model,
                  prompt_context=PROMPT_CONTEXT,
                  prompt_prefix=PROMPT_PREFIX,
                  prompt_suffix=PROMPT_SUFFIX,
                  **llama_params)
    print("...")
    print("[+] Press Ctrl+C to Stop ... ")
    print("...")

    while True:
        try:
            prompt = input("You: ")
            if prompt == '':
                continue
            print(f"{bcolors.OKBLUE}AI: {bcolors.ENDC}", end='', flush=True)
            for token in model.generate(prompt, antiprompt=PROMPT_PREFIX.strip(), **gpt_params):
                print(f"{bcolors.OKCYAN}{token}{bcolors.ENDC}", end='', flush=True)
            print()
        except KeyboardInterrupt:
            break


def main():
    print(__header__)

    parser = argparse.ArgumentParser(description="This is like a chatbot, You can start the conversation with `Hi, "
                                                 "can you help me ?`\nPay attention though that it may hallucinate!",
                                     allow_abbrev=True)
    # Positional args
    parser.add_argument('model', type=str, help="The path of the model file")

    # add params from LLAMA_CONTEXT_PARAMS_SCHEMA
    for param in LLAMA_CONTEXT_PARAMS_SCHEMA:
        param_fields = LLAMA_CONTEXT_PARAMS_SCHEMA[param]
        parser.add_argument(f'--{param}',
                            help=f'{param_fields["description"]}')

    for param in GPT_PARAMS_SCHEMA:
        param_fields = GPT_PARAMS_SCHEMA[param]
        parser.add_argument(f'--{param}',
                            help=f'{param_fields["description"]}')

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
