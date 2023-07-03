/**
 ********************************************************************************
 * @file    main.h
 * @author  [abdeladim-s](https://github.com/abdeladim-s)
 * @date    2023
 * @brief   Python bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) using Pybind11
 * @brief   llama.cpp is licensed under MIT Copyright (c) 2023 Georgi Gerganov,
            please see [llama.cpp License](./llama.cpp_LICENSE)
 * @par
 * COPYRIGHT NOTICE: (c) 2023.
 ********************************************************************************
 */

#include <string>
#include <vector>
#include <random>
#include <thread>


struct gpt_params {
    int32_t seed          = -1;  // RNG seed
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_predict     = -1;  // new tokens to predict
    int32_t n_ctx         = 512; // context size
    int32_t n_batch       = 512; // batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep        = 0;   // number of tokens to keep from initial prompt
    int32_t n_gpu_layers  = 0;   // number of layers to store in VRAM

    // sampling parameters
    std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens
    int32_t top_k             = 40;    // <= 0 to use vocab size
    float   top_p             = 0.95f; // 1.0 = disabled
    float   tfs_z             = 1.00f; // 1.0 = disabled
    float   typical_p         = 1.00f; // 1.0 = disabled
    float   temp              = 0.80f; // 1.0 = disabled
    float   repeat_penalty    = 1.10f; // 1.0 = disabled
    int32_t repeat_last_n     = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   frequency_penalty = 0.00f; // 0.0 = disabled
    float   presence_penalty  = 0.00f; // 0.0 = disabled
    int     mirostat          = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau      = 5.00f; // target entropy
    float   mirostat_eta      = 0.10f; // learning rate

    std::string model             = "models/7B/ggml-model.bin"; // model path
    std::string prompt            = "";
    std::string path_prompt_cache = "";  // path to file for saving/loading prompt eval state
    std::string input_prefix      = "";  // string to prefix user inputs with
    std::string input_suffix      = "";  // string to suffix user inputs with
    std::vector<std::string> antiprompt; // string upon seeing which more user input is prompted

    std::string lora_adapter = "";  // lora adapter path
    std::string lora_base    = "";  // base model path for the lora adapter

    bool memory_f16        = true;  // use f16 instead of f32 for memory kv
    bool random_prompt     = false; // do not randomize prompt if none provided
    bool use_color         = false; // use color to distinguish generations and inputs
    bool interactive       = false; // interactive mode
    bool prompt_cache_all  = false; // save user input and generations to prompt cache

    bool embedding         = false; // get only sentence embedding
    bool interactive_first = false; // wait for user input immediately
    bool multiline_input   = false; // reverse the usage of `\`

    bool instruct          = false; // instruction mode (used for Alpaca models)
    bool penalize_nl       = true;  // consider newlines as a repeatable token
    bool perplexity        = false; // compute perplexity over the prompt
    bool use_mmap          = true;  // use mmap for faster loads
    bool use_mlock         = false; // use mlock to keep model in memory
    bool mem_test          = false; // compute maximum memory usage
    bool verbose_prompt    = false; // print prompt tokens before generation
};


#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"
