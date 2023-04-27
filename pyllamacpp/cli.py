#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple command line interface
"""

import argparse
import importlib.metadata
import logging

import pyllamacpp.constants as constants

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


def _get_llama_context_params(args) -> dict:
    """
    Helper function to get params from argparse as a `dict`
    """
    params = {}
    for arg in args.__dict__:
        if arg in constants.LLAMA_CONTEXT_PARAMS_SCHEMA.keys() and getattr(args, arg) is not None:
            if constants.LLAMA_CONTEXT_PARAMS_SCHEMA[arg]['type'] is bool:
                if getattr(args, arg).lower() == 'false':
                    params[arg] = False
                else:
                    params[arg] = True
            else:
                params[arg] = constants.LLAMA_CONTEXT_PARAMS_SCHEMA[arg]['type'](getattr(args, arg))
    return params


def _get_gpt_params(args) -> dict:
    """
    Helper function to get params from argparse as a `dict`
    """
    params = {}
    for arg in args.__dict__:
        if arg in constants.GPT_PARAMS_SCHEMA.keys() and getattr(args, arg) is not None:
            if constants.GPT_PARAMS_SCHEMA[arg]['type'] is bool:
                if getattr(args, arg).lower() == 'false':
                    params[arg] = False
                else:
                    params[arg] = True
            else:
                params[arg] = constants.GPT_PARAMS_SCHEMA[arg]['type'](getattr(args, arg))
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


def run(args):
    print(f"[+] Running model `{args.model}`")
    llama_params = _get_llama_context_params(args)
    print(f"[+] LLaMA context params: `{llama_params}`")
    gpt_params = _get_gpt_params(args)
    print(f"[+] GPT params: `{gpt_params}`")
    model = Model(ggml_model=args.model, **llama_params)
    print("...")
    print("[+] Press Ctrl+C to Stop ... ")
    print("...")
    while True:
        try:
            prompt = input("You: ")
            if prompt == '':
                continue
            print(f"{bcolors.OKCYAN}AI: {bcolors.ENDC}", end='', flush=True)
            for tok in model.generate(prompt, **gpt_params):
                print(f"{bcolors.OKCYAN}{tok}{bcolors.ENDC}", end='', flush=True)
            print()
        except KeyboardInterrupt:
            break


def main():
    print(__header__)

    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Positional args
    parser.add_argument('model', type=str, help="The path of the model file")

    # add params from LLAMA_CONTEXT_PARAMS_SCHEMA
    for param in constants.LLAMA_CONTEXT_PARAMS_SCHEMA:
        param_fields = constants.LLAMA_CONTEXT_PARAMS_SCHEMA[param]
        parser.add_argument(f'--{param}',
                            help=f'{param_fields["description"]}')

    for param in constants.GPT_PARAMS_SCHEMA:
        param_fields = constants.GPT_PARAMS_SCHEMA[param]
        parser.add_argument(f'--{param}',
                            help=f'{param_fields["description"]}')

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
