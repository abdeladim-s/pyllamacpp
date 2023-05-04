# PyLLaMACpp
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPi version](https://badgen.net/pypi/v/pyllamacpp)](https://pypi.org/project/pyllamacpp/)

Python bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp)


<p align="center">
  <img src="./docs/demo.gif">
</p>


For those who don't know, `llama.cpp` is a port of Facebook's LLaMA model in pure C/C++:

<blockquote>

- Without dependencies
- Apple silicon first-class citizen - optimized via ARM NEON
- AVX2 support for x86 architectures
- Mixed F16 / F32 precision
- 4-bit quantization support
- Runs on the CPU

</blockquote>

# Table of contents
<!-- TOC -->
* [Installation](#installation)
* [CLI](#cli)
* [Tutorial](#tutorial)
    * [Quick start](#quick-start)
    * [Interactive Dialogue](#interactive-dialogue)
    * [Attribute a persona to the language model](#attribute-a-persona-to-the-language-model)
* [API reference](#api-reference)
* [Supported models](#supported-models)
* [Discussions and contributions](#discussions-and-contributions)
* [License](#license)
<!-- TOC -->

# Installation
1. The easy way is to install the prebuilt wheels
```bash
pip install pyllamacpp
```

However, the compilation process of `llama.cpp` is taking into account the architecture of the target `CPU`, 
so you might need to build it from source:

```shell
pip install git+https://github.com/abdeladim-s/pyllamacpp.git
```

# CLI 

You can run the following simple command line interface to test the package once it is installed:

```shell
pyllamacpp path/to/ggml/model
```

```shell
pyllamacpp -h

usage: pyllamacpp [-h] [--n_ctx N_CTX] [--n_parts N_PARTS] [--seed SEED] [--f16_kv F16_KV] [--logits_all LOGITS_ALL]
                  [--vocab_only VOCAB_ONLY] [--use_mlock USE_MLOCK] [--embedding EMBEDDING] [--n_predict N_PREDICT] [--n_threads N_THREADS]
                  [--repeat_last_n REPEAT_LAST_N] [--top_k TOP_K] [--top_p TOP_P] [--temp TEMP] [--repeat_penalty REPEAT_PENALTY]
                  [--n_batch N_BATCH]
                  model

This is like a chatbot, You can start the conversation with `Hi, can you help me ?` Pay attention though that it may hallucinate!

positional arguments:
  model                 The path of the model file

options:
  -h, --help            show this help message and exit
  --n_ctx N_CTX         text context
  --n_parts N_PARTS
  --seed SEED           RNG seed
  --f16_kv F16_KV       use fp16 for KV cache
  --logits_all LOGITS_ALL
                        the llama_eval() call computes all logits, not just the last one
  --vocab_only VOCAB_ONLY
                        only load the vocabulary, no weights
  --use_mlock USE_MLOCK
                        force system to keep model in RAM
  --embedding EMBEDDING
                        embedding mode only
  --n_predict N_PREDICT
                        Number of tokens to predict
  --n_threads N_THREADS
                        Number of threads
  --repeat_last_n REPEAT_LAST_N
                        Last n tokens to penalize
  --top_k TOP_K         top_k
  --top_p TOP_P         top_p
  --temp TEMP           temp
  --repeat_penalty REPEAT_PENALTY
                        repeat_penalty
  --n_batch N_BATCH     batch size for prompt processing
```

# Tutorial

### Quick start
A simple `Pythonic` API is built on top of `llama.cpp` C/C++ functions. You can call it from Python as follows:

```python
from pyllamacpp.model import Model

model = Model(model_path='./models/gpt4all-model.bin')
for token in model.generate("Tell me a joke ?"):
    print(token, end='', flush=True)
```

### Interactive Dialogue
You can set up an interactive dialogue by simply keeping the `model` variable alive:

```python
from pyllamacpp.model import Model

model = Model(model_path='/path/to/ggml/model')
while True:
    try:
        prompt = input("You: ", flush=True)
        if prompt == '':
            continue
        print(f"AI:", end='')
        for token in model.generate(prompt):
            print(f"{token}", end='', flush=True)
        print()
    except KeyboardInterrupt:
        break
```
### Attribute a persona to the language model

The following is an example showing how to _"attribute a persona to the language model"_ :

```python
from pyllamacpp.model import Model

prompt_context = """Act as Bob. Bob is helpful, kind, honest,
and never fails to answer the User's requests immediately and with precision. 

User: Nice to meet you Bob!
Bob: Welcome! I'm here to assist you with anything you need. What can I do for you today?
"""

prompt_prefix = "\nUser:"
prompt_suffix = "\nBob:"

model = Model(model_path='/path/to/ggml/model',
              prompt_context=prompt_context,
              prompt_prefix=prompt_prefix,
              prompt_suffix=prompt_suffix)

while True:
  try:
    prompt = input("User: ")
    if prompt == '':
      continue
    print(f"Bob: ", end='')
    for token in model.generate(prompt, antiprompt='User:'):
      print(f"{token}", end='', flush=True)
      print()
  except KeyboardInterrupt:
    break
```

# Supported models

Fully tested with [GPT4All](https://github.com/nomic-ai/gpt4all) model, see [PyGPT4All](https://github.com/nomic-ai/pygpt4all).

But all models supported by `llama.cpp` should be supported as well:

<blockquote>

**Supported models:**

- [X] LLaMA 🦙
- [X] [Alpaca](https://github.com/ggerganov/llama.cpp#instruction-mode-with-alpaca)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [Vicuna](https://github.com/ggerganov/llama.cpp/discussions/643#discussioncomment-5533894)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)

</blockquote>

# Advanced usage
For advanced users, you can access the [llama.cpp C-API](https://github.com/ggerganov/llama.cpp/blob/master/llama.h) functions directly to make your own logic.
All functions from `llama.h` are exposed with the binding module [`_pyllamacpp`](https://abdeladim-s.github.io/pyllamacpp/#_pyllamacpp).

# API reference
You can check the [API reference documentation](https://abdeladim-s.github.io/pyllamacpp/) for more details.

# FAQs
* [How to build pyllamacpp without AVX2 or FMA.](https://github.com/nomic-ai/pygpt4all/issues/71)
* [pyllamacpp not support M1 chips MacBook](https://github.com/nomic-ai/pygpt4all/issues/57#issuecomment-1519197837)
* [ImportError: DLL failed while importing _pyllamacpp](https://github.com/nomic-ai/pygpt4all/issues/53#issuecomment-1529772010)

# Discussions and contributions
If you find any bug, please open an [issue](https://github.com/abdeladim-s/pyllamacpp/issues).

If you have any feedback, or you want to share how you are using this project, feel free to use the [Discussions](https://github.com/abdeladim-s/pyllamacpp/discussions) and open a new topic.

# License

This project is licensed under the same license as [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE) (MIT  [License](./LICENSE)).

