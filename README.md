# PyLLaMACpp - ggllm

* Python bindings for [ggllm.cpp](https://github.com/cmp-nct/ggllm.cpp)
* `ggllm.cpp` is a fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) to run quantized Falcon Models.
* You can run Falcon as well as LlaMA models using the same `pyllamacpp` API.


# Table of contents
<!-- TOC -->
* [Installation](#installation)
* [CLI](#cli)
* [Tutorial](#tutorial)
    * [Quick start](#quick-start)
    * [Interactive Dialogue](#interactive-dialogue)
    * [Attribute a persona to the language model](#attribute-a-persona-to-the-language-model)
* [Supported models](#supported-models)
* [Discussions and contributions](#discussions-and-contributions)
* [License](#license)
<!-- TOC -->

# Installation

```shell
pip install git+https://github.com/abdeladim-s/pyllamacpp/tree/ggllm.cpp
```

# CLI 

You can run the following simple command line interface to test the package once it is installed:

```shell
pyllamacpp path/to/model.bin
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

```python
from pyllamacpp.model import Model

model = Model(model_path='/path/to/model.bin')
for token in model.generate("Tell me a joke ?\n"):
    print(token, end='', flush=True)
```

### Interactive Dialogue
You can set up an interactive dialogue by simply keeping the `model` variable alive:

```python
from pyllamacpp.model import Model

model = Model(model_path='/path/to/model.bin')
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

model = Model(model_path='/path/to/model.bin',
              n_ctx=512,
              prompt_context=prompt_context,
              prompt_prefix=prompt_prefix,
              prompt_suffix=prompt_suffix)

while True:
  try:
    prompt = input("User: ")
    if prompt == '':
      continue
    print(f"Bob: ", end='')
    for token in model.generate(prompt,
                                antiprompt='User:',
                                n_threads=6,
                                n_batch=1024,
                                n_predict=256,
                                n_keep=48,
                                repeat_penalty=1.0, ):
      print(f"{token}", end='', flush=True)
    print()
  except KeyboardInterrupt:
    break
```

# Supported models
All models supported by `ggllm.cpp` (The new [GGCC format](https://huggingface.co/TheBloke/falcon-7b-instruct-GGML)) and `llama.cpp` should be supported.

# Discussions and contributions
If you find any bug, please open an [issue](https://github.com/abdeladim-s/pyllamacpp/issues).

If you have any feedback, or you want to share how you are using this project, feel free to use the [Discussions](https://github.com/abdeladim-s/pyllamacpp/discussions) and open a new topic.

# License

MIT  [License](./LICENSE).

