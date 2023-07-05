/**
 ********************************************************************************
 * @file    main.cpp
 * @author  [Abdeladim Sadiki](https://github.com/abdeladim-s)
 * @date    2023
 * @brief   Python bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) using Pybind11
 * @brief   llama.cpp is licensed under MIT Copyright (c) 2023 Georgi Gerganov,
            please see [llama.cpp License](./llama.cpp_LICENSE)
 * @par
 * COPYRIGHT NOTICE: (c) 2023.
 ********************************************************************************
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "../llama.cpp/llama.h"
#include "main.h"



#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

static bool is_interacting = false;


py::function py_llama_progress_callback;

struct llama_context_wrapper {
    llama_context* ptr;
};

struct llama_context_wrapper llama_init_from_file_wrapper(const char * path_model, struct llama_context_params  params){
    struct llama_context * ctx = llama_init_from_file(path_model, params);
    struct llama_context_wrapper ctw_w;
    ctw_w.ptr = ctx;
    return ctw_w;
}


void llama_free_wrapper(struct llama_context_wrapper * ctx_w){
    llama_free(ctx_w->ptr);
}

void llama_apply_lora_from_file_wrapper(struct llama_context_wrapper * ctx_w,
                                        const char * path_lora,
                                        const char * path_base_model,
                                        int   n_threads){
    struct llama_context * ctx = ctx_w->ptr;
    llama_apply_lora_from_file(ctx, path_lora, path_base_model, n_threads);
}

int llama_get_kv_cache_token_count_wrapper(struct llama_context_wrapper * ctx_w){
    return llama_get_kv_cache_token_count(ctx_w->ptr);
}

void llama_set_rng_seed_wrapper(struct llama_context_wrapper * ctx_w, int seed){
    llama_set_rng_seed(ctx_w->ptr, seed);
}

size_t llama_get_state_size_wrapper(struct llama_context_wrapper * ctx_w){
    return llama_get_state_size(ctx_w->ptr);
}

bool llama_load_session_file_wrapper(struct llama_context_wrapper * ctx_w,
                                     const char * path_session,
                                     py::array_t<llama_token> tokens,
                                     size_t n_token_capacity,
                                     size_t * n_token_count_out){
    struct llama_context * ctx = ctx_w->ptr;
    py::buffer_info buf = tokens.request();
    llama_token *tokens_ptr = static_cast<llama_token *>(buf.ptr);
    return llama_load_session_file(ctx, path_session, tokens_ptr, n_token_capacity, n_token_count_out);

}

bool llama_save_session_file_wrapper(struct llama_context_wrapper * ctx_w,
                                    const char * path_session,
                                   py::array_t<llama_token> tokens,
                                    size_t n_token_count){
    struct llama_context * ctx = ctx_w->ptr;
    py::buffer_info buf = tokens.request();
    llama_token *tokens_ptr = static_cast<llama_token *>(buf.ptr);
    return llama_save_session_file(ctx, path_session, tokens_ptr, n_token_count);

}

int llama_eval_wrapper(struct llama_context_wrapper * ctx_w,
               py::array_t<llama_token> tokens,
               int   n_tokens,
               int   n_past,
               int   n_threads){
   struct llama_context * ctx = ctx_w->ptr;
   py::buffer_info buf = tokens.request();
   llama_token *tokens_ptr = static_cast<llama_token *>(buf.ptr);
   return llama_eval(ctx, tokens_ptr, n_tokens, n_past, n_threads);
}

std::vector<llama_token> llama_tokenize_wrapper(
        struct llama_context_wrapper * ctx_w,
        const std::string & text,
        bool   add_bos){

    struct llama_context * ctx = ctx_w->ptr;
    std::vector<llama_token> tokens((text.size() + (int)add_bos));
    int new_size = llama_tokenize(ctx, text.c_str(), tokens.data(), tokens.size(), add_bos);
    assert(new_size >= 0);
    tokens.resize(new_size);
    return tokens;
}


std::vector<llama_token> llama_tokenize_wrapper_2(
        struct llama_context * ctx,
        const std::string & text,
        bool   add_bos){

        std::vector<llama_token> tokens((text.size() + (int)add_bos));
        int new_size = llama_tokenize(ctx, text.c_str(), tokens.data(), tokens.size(), add_bos);
        assert(new_size >= 0);
        tokens.resize(new_size);
        return tokens;
}


std::string llama_tokens_to_str_wrapper(struct llama_context_wrapper* ctx_w, py::array_t<llama_token> tokens_array) {
    std::string result;
    struct llama_context * ctx = ctx_w->ptr;
    bool all_tokens_valid = true;

    for (int i = 0; i < tokens_array.size(); i++) {
        llama_token token = tokens_array.at(i);
        if (token >= llama_n_vocab(ctx)) {
            all_tokens_valid = false;
            break;
        }

        result += llama_token_to_str(ctx, token);
    }

    if (all_tokens_valid) {
        return result;
    } else {
        return "";
    }
}

int llama_n_vocab_wrapper(struct llama_context_wrapper * ctx_w){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_n_vocab(ctx);
}
int llama_n_ctx_wrapper(struct llama_context_wrapper * ctx_w){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_n_ctx(ctx);
}
int llama_n_embd_wrapper(struct llama_context_wrapper * ctx_w){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_n_embd(ctx);
}

pybind11::array wrap_array_ptr(float *v) {
  auto capsule = py::capsule(
      &v, [](void *v) { delete reinterpret_cast<std::vector<float> *>(v); });
  return py::array(static_cast<pybind11::ssize_t>(sizeof(v)), v,
                   capsule);
}

//py::array llama_get_logits_wrapper(struct llama_context_wrapper * ctx_w){
//   struct llama_context * ctx = ctx_w->ptr;
//   auto logits = llama_get_logits(ctx);
////    std::vector<float> logits_vect;
////    logits_vect.assign(std::begin(logits), std::end(logits));
//   return wrap_array_ptr(logits);
////    return py::buffer_info(
////            m.data(),                               /* Pointer to buffer */
////            sizeof(float),                          /* Size of one scalar */
////            py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
////            2,                                      /* Number of dimensions */
////            { m.rows(), m.cols() },                 /* Buffer dimensions */
////            { sizeof(float) * m.cols(),             /* Strides (in bytes) for each index */
////              sizeof(float) }
////        );
//
//}

float * llama_get_logits_wrapper(struct llama_context_wrapper * ctx_w){
   struct llama_context * ctx = ctx_w->ptr;
  return llama_get_logits(ctx);

}

std::vector<float> llama_get_embeddings_wrapper(struct llama_context_wrapper * ctx_w){
    struct llama_context * ctx = ctx_w->ptr;
    int n_embed = llama_n_embd(ctx);
    float * embed_ptr = llama_get_embeddings(ctx);
    std::vector<float> embeddings(embed_ptr, embed_ptr + n_embed);
    return embeddings;
}

const char * llama_token_to_str_wrapper(struct llama_context_wrapper * ctx_w, llama_token token){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_token_to_str(ctx, token);
}

// sampling fcts

void llama_sample_repetition_penalty_wrapper(struct llama_context_wrapper * ctx_w,
                                     llama_token_data_array * candidates,
                                     py::array_t<llama_token> last_tokens,
                                     size_t last_tokens_size,
                                     float penalty){
    struct llama_context * ctx = ctx_w->ptr;
    py::buffer_info buf = last_tokens.request();
    llama_token *last_tokens_ptr = static_cast<llama_token *>(buf.ptr);
    llama_sample_repetition_penalty(ctx, candidates, last_tokens_ptr, last_tokens_size, penalty);
}

void llama_sample_frequency_and_presence_penalties_wrapper(struct llama_context_wrapper * ctx_w,
                                             llama_token_data_array * candidates,
                                             py::array_t<llama_token> last_tokens,
                                             size_t last_tokens_size,
                                             float alpha_frequency,
                                             float alpha_presence){
    struct llama_context * ctx = ctx_w->ptr;
    py::buffer_info buf = last_tokens.request();
    llama_token *last_tokens_ptr = static_cast<llama_token *>(buf.ptr);
    llama_sample_frequency_and_presence_penalties(ctx, candidates, last_tokens_ptr, last_tokens_size, alpha_frequency, alpha_presence);
}

void llama_sample_softmax_wrapper(struct llama_context_wrapper * ctx_w,
                                             llama_token_data_array * candidates){
    struct llama_context * ctx = ctx_w->ptr;
    llama_sample_softmax(ctx, candidates);
}

void llama_sample_top_k_wrapper(struct llama_context_wrapper * ctx_w,
                                llama_token_data_array * candidates,
                                int k,
                                size_t min_keep){
    struct llama_context * ctx = ctx_w->ptr;
    llama_sample_top_k(ctx, candidates, k, min_keep);
}

void llama_sample_top_p_wrapper(struct llama_context_wrapper * ctx_w,
                                llama_token_data_array * candidates,
                                float p,
                                size_t min_keep){
    struct llama_context * ctx = ctx_w->ptr;
    llama_sample_top_p(ctx, candidates, p, min_keep);
}

void llama_sample_tail_free_wrapper(struct llama_context_wrapper * ctx_w,
                                llama_token_data_array * candidates,
                                float z,
                                size_t min_keep){
    struct llama_context * ctx = ctx_w->ptr;
    llama_sample_tail_free(ctx, candidates, z, min_keep);
}

void llama_sample_typical_wrapper(struct llama_context_wrapper * ctx_w,
                                llama_token_data_array * candidates,
                                float p,
                                size_t min_keep){
    struct llama_context * ctx = ctx_w->ptr;
    llama_sample_typical(ctx, candidates, p, min_keep);
}

void llama_sample_temperature_wrapper(struct llama_context_wrapper * ctx_w,
                                llama_token_data_array * candidates,
                                float temp){
    struct llama_context * ctx = ctx_w->ptr;
    llama_sample_temperature(ctx, candidates, temp);
}

llama_token llama_sample_token_mirostat_wrapper(struct llama_context_wrapper * ctx_w,
                                llama_token_data_array * candidates,
                                float tau,
                                float eta,
                                int m,
                                float * mu){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_sample_token_mirostat(ctx, candidates, tau, eta, m, mu);
}

llama_token llama_sample_token_mirostat_v2_wrapper(struct llama_context_wrapper * ctx_w,
                                llama_token_data_array * candidates,
                                float tau,
                                float eta,
                                float * mu){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_sample_token_mirostat_v2(ctx, candidates, tau, eta, mu);
}

llama_token llama_sample_token_greedy_wrapper(struct llama_context_wrapper * ctx_w,
                                llama_token_data_array * candidates){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_sample_token_greedy(ctx, candidates);
}

llama_token llama_sample_token_wrapper(struct llama_context_wrapper * ctx_w,
                                llama_token_data_array * candidates){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_sample_token(ctx, candidates);
}


void llama_print_timings_wrapper(struct llama_context_wrapper * ctx_w){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_print_timings(ctx);
}

void llama_reset_timings_wrapper(struct llama_context_wrapper * ctx_w){
    struct llama_context * ctx = ctx_w->ptr;
    return llama_reset_timings(ctx);
}

llama_token sample_next_token(struct llama_context_wrapper * ctx_w, gpt_params params, std::vector<llama_token> & last_n_tokens){
    // helper function to sample next token (based on the main example)
    struct llama_context * ctx = ctx_w->ptr;
    const int n_ctx = llama_n_ctx(ctx);
    llama_token id = 0;

    const float   temp            = params.temp;
    const int32_t top_k           = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
    const float   top_p           = params.top_p;
    const float   tfs_z           = params.tfs_z;
    const float   typical_p       = params.typical_p;
    const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
    const float   repeat_penalty  = params.repeat_penalty;
    const float   alpha_presence  = params.presence_penalty;
    const float   alpha_frequency = params.frequency_penalty;
    const int     mirostat        = params.mirostat;
    const float   mirostat_tau    = params.mirostat_tau;
    const float   mirostat_eta    = params.mirostat_eta;
    const bool    penalize_nl     = params.penalize_nl;

    {
        auto logits  = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(ctx);

        // Apply params.logit_bias map
        for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
            logits[it->first] += it->second;
        }

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

        // Apply penalties
        float nl_logit = logits[llama_token_nl()];
        auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
        llama_sample_repetition_penalty(ctx, &candidates_p,
                                        last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                        last_n_repeat, repeat_penalty);
        llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                      last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                      last_n_repeat, alpha_frequency, alpha_presence);
        if (!penalize_nl) {
            logits[llama_token_nl()] = nl_logit;
        }

        if (temp <= 0) {
            // Greedy sampling
            id = llama_sample_token_greedy(ctx, &candidates_p);
        } else {
            if (mirostat == 1) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                const int mirostat_m = 100;
                llama_sample_temperature(ctx, &candidates_p, temp);
                id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
            } else if (mirostat == 2) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                llama_sample_temperature(ctx, &candidates_p, temp);
                id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
            } else {
                // Temperature sampling
                llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                llama_sample_temperature(ctx, &candidates_p, temp);
                id = llama_sample_token(ctx, &candidates_p);
            }
        }
        // printf("`%d`", candidates_p.size);

//        last_n_tokens.erase(last_n_tokens.begin());
//        last_n_tokens.push_back(id);
    }

    return id;
}



std::string gpt_random_prompt(std::mt19937 & rng) {
    const int r = rng() % 10;
    switch (r) {
        case 0: return "So";
        case 1: return "Once upon a time";
        case 2: return "When";
        case 3: return "The";
        case 4: return "After";
        case 5: return "If";
        case 6: return "import";
        case 7: return "He";
        case 8: return "She";
        case 9: return "They";
        default: return "To";
    }

    return "The";
}
static llama_context ** g_ctx;

int llama_generate(struct llama_context_wrapper * ctx_w, gpt_params params, py::function new_text_callback){
    // main example from llama.cpp
    struct llama_context * ctx = ctx_w->ptr;
    g_ctx = &ctx;

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                        "expect poor results\n", __func__, params.n_ctx);
    }
    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed  = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    // determine the maximum memory usage needed to do inference for the given n_batch and n_predict parameters
    // uncomment the "used_mem" line in llama.cpp to see the results
    if (params.mem_test) {
        {
            const std::vector<llama_token> tmp(params.n_batch, 0);
            llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        }

        {
            const std::vector<llama_token> tmp = { 0, };
            llama_eval(ctx, tmp.data(), tmp.size(), params.n_predict - 1, params.n_threads);
        }

        llama_print_timings(ctx);
        llama_free(ctx);

        return 0;
    }

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        fprintf(stderr, "%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());

        // fopen to check for existing session
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);

            session_tokens.resize(params.n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_load_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);

            fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
            fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
        }
    }

    // tokenize the prompt
    auto embd_inp = llama_tokenize_wrapper(ctx_w, params.prompt, true);

    const int n_ctx = llama_n_ctx(ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (n_matching_session_tokens >= embd_inp.size()) {
            fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct) {
        params.n_keep = (int)embd_inp.size();
    }

    // prefix & suffix for instruct mode
    const auto inp_pfx = llama_tokenize_wrapper(ctx_w, "\n\n### Instruction:\n\n", true);
    const auto inp_sfx = llama_tokenize_wrapper(ctx_w, "\n\n### Response:\n\n", false);

    // in instruct mode, we inject a prefix and a suffix to each input by the user
    if (params.instruct) {
        params.interactive_first = true;
        params.antiprompt.push_back("### Instruction:\n\n");
    }

    // enable interactive mode if reverse prompt or interactive start is specified
    if (params.antiprompt.size() != 0 || params.interactive_first) {
        params.interactive = true;
    }

    // determine newline token
    auto llama_token_newline = llama_tokenize_wrapper(ctx_w, "\n", false);

    if (params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
        }
        if (params.n_keep > 0) {
            fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                fprintf(stderr, "%s", llama_token_to_str(ctx, embd_inp[i]));
            }
            fprintf(stderr, "'\n");
        }
        fprintf(stderr, "\n");
    }

    if (params.interactive) {

        fprintf(stderr, "%s: interactive mode on.\n", __func__);

        if (params.antiprompt.size()) {
            for (auto antiprompt : params.antiprompt) {
                fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str());
            }
        }

        if (!params.input_prefix.empty()) {
            fprintf(stderr, "Input prefix: '%s'\n", params.input_prefix.c_str());
        }

        if (!params.input_suffix.empty()) {
            fprintf(stderr, "Input suffix: '%s'\n", params.input_suffix.c_str());
        }
    }

    fprintf(stderr, "sampling: repeat_last_n = %d, repeat_penalty = %f, presence_penalty = %f, frequency_penalty = %f, top_k = %d, tfs_z = %f, top_p = %f, typical_p = %f, temp = %f, mirostat = %d, mirostat_lr = %f, mirostat_ent = %f\n",
            params.repeat_last_n, params.repeat_penalty, params.presence_penalty, params.frequency_penalty, params.top_k, params.tfs_z, params.top_p, params.typical_p, params.temp, params.mirostat, params.mirostat_eta, params.mirostat_tau);
    fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    fprintf(stderr, "\n\n");

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    if (params.interactive) {
        is_interacting = params.interactive_first;
    }

    bool is_antiprompt = false;
    bool input_echo    = true;

    // HACK - because session saving incurs a non-negligible delay, for now skip re-saving session
    // if we loaded a session with at least 75% similarity. It's currently just used to speed up the
    // initial prompt so it doesn't need to be an exact match.
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < (embd_inp.size() * 3 / 4);

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<llama_token> embd;

    while (n_remain != 0 || params.interactive) {
        // predict
        if (embd.size() > 0) {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - params.n_keep;

                // always keep the first token - BOS
                n_past = std::max(1, params.n_keep);

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());

                // stop saving session if we run out of context
                path_session = "";

                //printf("\n---\n");
                //printf("resetting: '");
                //for (int i = 0; i < (int) embd.size(); i++) {
                //    printf("%s", llama_token_to_str(ctx, embd[i]));
                //}
                //printf("'\n");
                //printf("\n---\n");
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }
                if (llama_eval(ctx, &embd[i], n_eval, n_past, params.n_threads)) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    return 1;
                }
                n_past += n_eval;
            }

            if (embd.size() > 0 && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // out of user input, sample next token
            const float   temp            = params.temp;
            const int32_t top_k           = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
            const float   top_p           = params.top_p;
            const float   tfs_z           = params.tfs_z;
            const float   typical_p       = params.typical_p;
            const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float   repeat_penalty  = params.repeat_penalty;
            const float   alpha_presence  = params.presence_penalty;
            const float   alpha_frequency = params.frequency_penalty;
            const int     mirostat        = params.mirostat;
            const float   mirostat_tau    = params.mirostat_tau;
            const float   mirostat_eta    = params.mirostat_eta;
            const bool    penalize_nl     = params.penalize_nl;

            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session) {
                need_to_save_session = false;
                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
            }

            llama_token id = 0;

            {
                auto logits  = llama_get_logits(ctx);
                auto n_vocab = llama_n_vocab(ctx);

                // Apply params.logit_bias map
                for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                    logits[it->first] += it->second;
                }

                std::vector<llama_token_data> candidates;
                candidates.reserve(n_vocab);
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                }

                llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                // Apply penalties
                float nl_logit = logits[llama_token_nl()];
                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                llama_sample_repetition_penalty(ctx, &candidates_p,
                                                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                last_n_repeat, repeat_penalty);
                llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                              last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                              last_n_repeat, alpha_frequency, alpha_presence);
                if (!penalize_nl) {
                    logits[llama_token_nl()] = nl_logit;
                }

                if (temp <= 0) {
                    // Greedy sampling
                    id = llama_sample_token_greedy(ctx, &candidates_p);
                } else {
                    if (mirostat == 1) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        const int mirostat_m = 100;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                    } else if (mirostat == 2) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                    } else {
                        // Temperature sampling
                        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                        llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                        llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token(ctx, &candidates_p);
                    }
                }
                // printf("`%d`", candidates_p.size);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // replace end of text token with newline token when in interactive mode
            if (id == llama_token_eos() && params.interactive && !params.instruct) {
                id = llama_token_newline.front();
                if (params.antiprompt.size() != 0) {
                    // tokenize and inject first reverse prompt
                    const auto first_antiprompt = llama_tokenize_wrapper(ctx_w, params.antiprompt.front(), false);
                    embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                }
            }

            // add it to the context
            embd.push_back(id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo) {
            for (auto id : embd) {
                // @NOTE: model.cpp_generate invokes llama_generate() with a python callback function.
                // @NOTE: we need to make sure that when the python callback function is called, it gets callback with raw bytes
                // @NOTE: of generated token, else pybind with implicitly try to decode it to Unicode, and cause UnicodeDecodeError
                std::string tok_str = llama_token_to_str(ctx, id);
                new_text_callback(py::bytes(tok_str));
            }
            fflush(stdout);
        }
        // reset color to default if we there is no pending user input
//        if (input_echo && (int)embd_inp.size() == n_consumed) {
//            console_set_color(con_st, CONSOLE_COLOR_DEFAULT);
//        }

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos()) {
            if (params.instruct) {
                is_interacting = true;
            } else {
                fprintf(stderr, " [end of text]\n");
                break;
            }
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    llama_print_timings(ctx);
    return 0;
}

// @NOTE: Experimental
class LLaMAModel {

public:
    struct llama_context * ctx;
//    gpt_params params;
    std::string path_session;
    std::vector<llama_token> session_tokens;
    std::vector<llama_token> embd_inp;
    std::vector<llama_token> last_n_tokens;
    int n_ctx;
    int n_past             = 0;
    int n_remain = 25;
    int n_consumed         = 0;
    int n_session_consumed = 0;
    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool need_to_save_session = false;
    size_t n_matching_session_tokens = 0;
    bool is_interacting = false;

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_out;

    // Constructor
    LLaMAModel(struct llama_context_wrapper * ctx_w, gpt_params g_params, int n): last_n_tokens(n) {
        ctx = ctx_w->ptr;
        g_ctx = &ctx;
        n_ctx = n;
        last_n_tokens.reserve(n_ctx);
        std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    }

    int setup(gpt_params & params){
        path_session = params.path_prompt_cache;
        if (!path_session.empty()) {
            fprintf(stderr, "%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());

            // fopen to check for existing session
            FILE * fp = std::fopen(path_session.c_str(), "rb");
            if (fp != NULL) {
                std::fclose(fp);

                session_tokens.resize(params.n_ctx);
                size_t n_token_count_out = 0;
                if (!llama_load_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                    fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                    return 1;
                }
                session_tokens.resize(n_token_count_out);
                llama_set_rng_seed(ctx, params.seed);

                fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
            } else {
                fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
            }
        }


        // in instruct mode, we inject a prefix and a suffix to each input by the user
        if (params.instruct) {
            params.interactive_first = true;
            params.antiprompt.push_back("### Instruction:\n\n");
        }

        // enable interactive mode if interactive start is specified
        if (params.interactive_first) {
            params.interactive = true;
        }


        if (params.verbose_prompt) {
            fprintf(stderr, "\n");
            fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
            fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
            for (int i = 0; i < (int) embd_inp.size(); i++) {
                fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
            }
            if (params.n_keep > 0) {
                fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
                for (int i = 0; i < params.n_keep; i++) {
                    fprintf(stderr, "%s", llama_token_to_str(ctx, embd_inp[i]));
                }
                fprintf(stderr, "'\n");
            }
            fprintf(stderr, "\n");
        }

        fprintf(stderr, "sampling: repeat_last_n = %d, repeat_penalty = %f, presence_penalty = %f, frequency_penalty = %f, top_k = %d, tfs_z = %f, top_p = %f, typical_p = %f, temp = %f, mirostat = %d, mirostat_lr = %f, mirostat_ent = %f\n",
                params.repeat_last_n, params.repeat_penalty, params.presence_penalty, params.frequency_penalty, params.top_k, params.tfs_z, params.top_p, params.typical_p, params.temp, params.mirostat, params.mirostat_eta, params.mirostat_tau);
        fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
        fprintf(stderr, "\n\n");

        need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();
        n_remain = params.n_predict;
        // tokenize the prompt
        if (params.interactive_first || params.instruct || !params.prompt.empty() || session_tokens.empty()) {
            // Add a space in front of the first character to match OG llama tokenizer behavior
            params.prompt.insert(0, 1, ' ');

            embd_inp = llama_tokenize_wrapper_2(ctx, params.prompt, true);
        } else {
            embd_inp = session_tokens;
        }

        if ((int) embd_inp.size() > n_ctx - 4) {
            fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
            return 1;
        }
        // debug message about similarity of saved session, if applicable
        size_t n_matching_session_tokens = 0;
        if (session_tokens.size()) {
            for (llama_token id : session_tokens) {
                if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                    break;
                }
                n_matching_session_tokens++;
            }
            if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
                fprintf(stderr, "%s: using full prompt from session file\n", __func__);
            } else if (n_matching_session_tokens >= embd_inp.size()) {
                fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
            } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
                fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                        __func__, n_matching_session_tokens, embd_inp.size());
            } else {
                fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
                        __func__, n_matching_session_tokens, embd_inp.size());
            }
        }

        if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct) {
            params.n_keep = (int)embd_inp.size();
        }
        if (params.interactive) {
            is_interacting = params.interactive_first;
        }

        return 0;
    }

    void update_prompt(gpt_params & params, std::string buffer){
        // prefix & suffix for instruct mode
        const auto inp_pfx = llama_tokenize_wrapper_2(ctx, "\n\n### Instruction:\n\n", true);
        const auto inp_sfx = llama_tokenize_wrapper_2(ctx, "\n\n### Response:\n\n", false);


        if (!params.input_prefix.empty()) {
            buffer += params.input_prefix;
            printf("%s", buffer.c_str());
        }

        // Add tokens to embd only if the input buffer is non-empty
        // Entering a empty line lets the user pass control back
        if (buffer.length() > 1) {
            // append input suffix if any
            if (!params.input_suffix.empty()) {
                buffer += params.input_suffix;
                printf("%s", params.input_suffix.c_str());
            }

            // instruct mode: insert instruction prefix
            if (params.instruct && !is_antiprompt) {
                n_consumed = embd_inp.size();
                embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
            }

            auto line_inp = llama_tokenize_wrapper_2(ctx, buffer, false);
            embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

            // instruct mode: insert response suffix
            if (params.instruct) {
                embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
            }

            n_remain -= line_inp.size();
        }
//        input_echo = false;

    }

    int generate(gpt_params & params){
        // determine newline token
        auto llama_token_newline = llama_tokenize_wrapper_2(ctx, "\n", false);


        // predict
        if (embd.size() > 0) {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - params.n_keep;

                // always keep the first token - BOS
                n_past = std::max(1, params.n_keep);

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());

                // stop saving session if we run out of context
                path_session.clear();

                //printf("\n---\n");
                //printf("resetting: '");
                //for (int i = 0; i < (int) embd.size(); i++) {
                //    printf("%s", llama_token_to_str(ctx, embd[i]));
                //}
                //printf("'\n");
                //printf("\n---\n");
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }
                if (llama_eval(ctx, &embd[i], n_eval, n_past, params.n_threads)) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    return 1;
                }
                n_past += n_eval;
            }

            if (embd.size() > 0 && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();
        embd_out.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // out of user input, sample next token
            const float   temp            = params.temp;
            const int32_t top_k           = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
            const float   top_p           = params.top_p;
            const float   tfs_z           = params.tfs_z;
            const float   typical_p       = params.typical_p;
            const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float   repeat_penalty  = params.repeat_penalty;
            const float   alpha_presence  = params.presence_penalty;
            const float   alpha_frequency = params.frequency_penalty;
            const int     mirostat        = params.mirostat;
            const float   mirostat_tau    = params.mirostat_tau;
            const float   mirostat_eta    = params.mirostat_eta;
            const bool    penalize_nl     = params.penalize_nl;

            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session) {
                need_to_save_session = false;
                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
            }

            llama_token id = 0;

            {
                auto logits  = llama_get_logits(ctx);
                auto n_vocab = llama_n_vocab(ctx);

                // Apply params.logit_bias map
                for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                    logits[it->first] += it->second;
                }

                std::vector<llama_token_data> candidates;
                candidates.reserve(n_vocab);
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                }

                llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                // Apply penalties
                float nl_logit = logits[llama_token_nl()];
                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                llama_sample_repetition_penalty(ctx, &candidates_p,
                                                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                last_n_repeat, repeat_penalty);
                llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                              last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                              last_n_repeat, alpha_frequency, alpha_presence);
                if (!penalize_nl) {
                    logits[llama_token_nl()] = nl_logit;
                }

                if (temp <= 0) {
                    // Greedy sampling
                    id = llama_sample_token_greedy(ctx, &candidates_p);
                } else {
                    if (mirostat == 1) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        const int mirostat_m = 100;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                    } else if (mirostat == 2) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                    } else {
                        // Temperature sampling
                        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                        llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                        llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token(ctx, &candidates_p);
                    }
                }
                // printf("`%d`", candidates_p.size);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // replace end of text token with newline token when in interactive mode
            if (id == llama_token_eos() && params.interactive && !params.instruct) {
                id = llama_token_newline.front();
                if (params.antiprompt.size() != 0) {
                    // tokenize and inject first reverse prompt
                    const auto first_antiprompt = llama_tokenize_wrapper_2(ctx, params.antiprompt.front(), false);
                    embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                }
            }

            // add it to the context
            embd.push_back(id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }


//        // display text
//        if (input_echo) {
//            for (auto id : embd) {
//                embd_out.push_back(id);
//            }
//        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {

            // check for reverse prompt
            if (params.antiprompt.size()) {
                std::string last_output;
                for (auto id : last_n_tokens) {
                    last_output += llama_token_to_str(ctx, id);
                }

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                                              ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                                              : 0;

                    if (last_output.find(antiprompt.c_str(), search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        return 2;  // break
                    }
                }
            }

            if (n_past > 0) {
                is_interacting = false;
            }
        }

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos()) {
            if (params.instruct) {
                is_interacting = true;
            } else {
                fprintf(stderr, " [end of text]\n");
                return 2;
            }
        }

        if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
            n_remain = params.n_predict;
            is_interacting = true;
        }

        return 0;
    }

};
///////////////////


PYBIND11_MODULE(_pyllamacpp, m) {
    m.doc() = R"pbdoc(
        PyLLaMACpp: Python binding for llama.cpp
        -----------------------

        .. currentmodule:: _pyllamacpp

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    py::class_<gpt_params>(m,"gpt_params" /*,py::dynamic_attr()*/)
        .def(py::init<>())
        .def_readwrite("seed", &gpt_params::seed)
        .def_readwrite("n_threads", &gpt_params::n_threads)
        .def_readwrite("n_predict", &gpt_params::n_predict)
        .def_readwrite("n_ctx", &gpt_params::n_ctx)
        .def_readwrite("n_batch", &gpt_params::n_batch)
        .def_readwrite("n_keep", &gpt_params::n_keep)

        .def_readwrite("logit_bias", &gpt_params::logit_bias)
        .def_readwrite("top_k", &gpt_params::top_k)
        .def_readwrite("top_p", &gpt_params::top_p)
        .def_readwrite("tfs_z", &gpt_params::tfs_z)
        .def_readwrite("typical_p", &gpt_params::typical_p)
        .def_readwrite("temp", &gpt_params::temp)
        .def_readwrite("repeat_penalty", &gpt_params::repeat_penalty)
        .def_readwrite("repeat_last_n", &gpt_params::repeat_last_n)
        .def_readwrite("frequency_penalty", &gpt_params::frequency_penalty)
        .def_readwrite("presence_penalty", &gpt_params::presence_penalty)
        .def_readwrite("mirostat", &gpt_params::mirostat)
        .def_readwrite("mirostat_tau", &gpt_params::mirostat_tau)
        .def_readwrite("mirostat_eta", &gpt_params::mirostat_eta)

        .def_readwrite("model", &gpt_params::model)
        .def_readwrite("prompt", &gpt_params::prompt)
        .def_readwrite("path_prompt_cache", &gpt_params::path_prompt_cache)
        .def_readwrite("input_prefix", &gpt_params::input_prefix)
        .def_readwrite("input_suffix", &gpt_params::input_suffix)
        .def_readwrite("antiprompt", &gpt_params::antiprompt)

        .def_readwrite("lora_adapter", &gpt_params::lora_adapter)
        .def_readwrite("lora_base", &gpt_params::lora_base)

        .def_readwrite("memory_f16", &gpt_params::memory_f16)
        .def_readwrite("random_prompt", &gpt_params::random_prompt)
        .def_readwrite("use_color", &gpt_params::use_color)
        .def_readwrite("interactive", &gpt_params::interactive)

        .def_readwrite("embedding", &gpt_params::embedding)
        .def_readwrite("interactive_first", &gpt_params::interactive_first)
        .def_readwrite("multiline_input", &gpt_params::multiline_input)

        .def_readwrite("instruct", &gpt_params::instruct)
        .def_readwrite("penalize_nl", &gpt_params::penalize_nl)
        .def_readwrite("perplexity", &gpt_params::perplexity)
        .def_readwrite("use_mmap", &gpt_params::use_mmap)
        .def_readwrite("use_mlock", &gpt_params::use_mlock)
        .def_readwrite("mem_test", &gpt_params::mem_test)
        .def_readwrite("verbose_prompt", &gpt_params::verbose_prompt)
        ;

    py::class_<llama_context_wrapper>(m,"llama_context");

    py::class_<llama_token_data>(m,"llama_token_data")
        .def(py::init<>())
        .def_readwrite("id", &llama_token_data::id)
        .def_readwrite("p", &llama_token_data::p)
        .def_readwrite("logit", &llama_token_data::logit);

    PYBIND11_NUMPY_DTYPE(llama_token_data, id, p, logit); // to use llama_tokens in numpy
    py::class_<llama_token_data_array>(m,"llama_token_data_array")
        .def(py::init<>())
        .def_readwrite("size", &llama_token_data_array::size)
        .def_readwrite("sorted", &llama_token_data_array::sorted)
        .def_property("data", [](llama_token_data_array &self) {
            std::vector<llama_token_data> data(self.data, self.data + self.size);
            for(int i=0; i<self.size; i++){
                data[i] = self.data[i];
            }
            return data;
        },
        [](llama_token_data_array &self, py::array_t<llama_token_data> tokens) {
//            py::buffer_info buf = tokens.request();
//            llama_token_data *tokens_ptr = static_cast<llama_token_data *>(buf.ptr);
//            self.data = tokens_ptr;
        });

    py::class_<llama_context_params>(m,"llama_context_params")
        .def(py::init<>())
        .def_readwrite("n_ctx", &llama_context_params::n_ctx)
        .def_readwrite("n_gpu_layers", &llama_context_params::n_gpu_layers)
        .def_readwrite("seed", &llama_context_params::seed)
        .def_readwrite("f16_kv", &llama_context_params::f16_kv)
        .def_readwrite("logits_all", &llama_context_params::logits_all)
        .def_readwrite("vocab_only", &llama_context_params::vocab_only)
        .def_readwrite("use_mlock", &llama_context_params::use_mlock)
        .def_readwrite("embedding", &llama_context_params::embedding)
        .def_property("progress_callback", [](llama_context_params &self) {},
            [](llama_context_params &self, py::function callback) {
            py_llama_progress_callback = callback;
            self.progress_callback = [](float progress, void *ctx) {
                py_llama_progress_callback(progress, ctx);
                };
        })

        .def_readwrite("progress_callback_user_data", &llama_context_params::progress_callback_user_data);

    py::enum_<llama_ftype>(m, "llama_ftype")
    .value("LLAMA_FTYPE_ALL_F32", llama_ftype::LLAMA_FTYPE_ALL_F32)
    .value("LLAMA_FTYPE_MOSTLY_F16", llama_ftype::LLAMA_FTYPE_MOSTLY_F16)
    .value("LLAMA_FTYPE_MOSTLY_Q4_0", llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_0)
    .value("LLAMA_FTYPE_MOSTLY_Q4_1", llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_1)
    .value("LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16", llama_ftype::LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16)
    .value("LLAMA_FTYPE_MOSTLY_Q8_0", llama_ftype::LLAMA_FTYPE_MOSTLY_Q8_0)
    .value("LLAMA_FTYPE_MOSTLY_Q5_0", llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_0)
    .value("LLAMA_FTYPE_MOSTLY_Q5_1", llama_ftype::LLAMA_FTYPE_MOSTLY_Q5_1)
    .export_values();

    m.def("llama_context_default_params", &llama_context_default_params);

    m.def("llama_mmap_supported", &llama_mmap_supported);
    m.def("llama_mlock_supported", &llama_mlock_supported);


    m.def("llama_init_from_file", &llama_init_from_file_wrapper);

    m.def("llama_free", &llama_free_wrapper);
    m.def("llama_model_quantize", &llama_model_quantize);
    m.def("llama_apply_lora_from_file", &llama_apply_lora_from_file_wrapper);
    m.def("llama_get_kv_cache_token_count", &llama_get_kv_cache_token_count_wrapper);
    m.def("llama_set_rng_seed", &llama_set_rng_seed_wrapper);
    m.def("llama_get_state_size", &llama_get_state_size_wrapper);
    m.def("llama_load_session_file", &llama_load_session_file_wrapper);
    m.def("llama_save_session_file", &llama_save_session_file_wrapper);

    m.def("llama_model_quantize", &llama_model_quantize);
    m.def("llama_eval", &llama_eval_wrapper);
    m.def("llama_tokenize", &llama_tokenize_wrapper);
    m.def("llama_n_vocab", &llama_n_vocab_wrapper);
    m.def("llama_n_ctx", &llama_n_ctx_wrapper);
    m.def("llama_n_embd", &llama_n_embd_wrapper);
    m.def("llama_get_logits", &llama_get_logits_wrapper);
    m.def("llama_get_embeddings", &llama_get_embeddings_wrapper);
    m.def("llama_token_to_str", [](struct llama_context_wrapper * ctx_w, llama_token token){
        //@NOTE: model.generate() calls pp.llama_token_to_str -> llama_token_to_str_wrapper()
        //@NOTE: we need to make sure that llama_token_to_str_wrapper() returns raw bytes
        //@NOTE: to prevent implicit conversion of const char* to unicode on python side, leading to UnicodeDecodeError
        return py::bytes(llama_token_to_str_wrapper(ctx_w, token));
    });
    m.def("llama_tokens_to_str", &llama_tokens_to_str_wrapper);


    m.def("llama_token_bos", &llama_token_bos);
    m.def("llama_token_eos", &llama_token_eos);
    m.def("llama_token_nl", &llama_token_nl);

    // sampling fcts
    m.def("llama_sample_repetition_penalty", &llama_sample_repetition_penalty_wrapper);
    m.def("llama_sample_frequency_and_presence_penalties", &llama_sample_frequency_and_presence_penalties_wrapper);
    m.def("llama_sample_softmax", &llama_sample_softmax_wrapper);
    m.def("llama_sample_top_k", &llama_sample_top_k_wrapper);
    m.def("llama_sample_top_p", &llama_sample_top_p_wrapper);
    m.def("llama_sample_tail_free", &llama_sample_tail_free_wrapper);
    m.def("llama_sample_typical", &llama_sample_typical_wrapper);
    m.def("llama_sample_temperature", &llama_sample_temperature_wrapper);
    m.def("llama_sample_token_mirostat", &llama_sample_token_mirostat_wrapper);
    m.def("llama_sample_token_mirostat_v2", &llama_sample_token_mirostat_v2_wrapper);
    m.def("llama_sample_token_greedy", &llama_sample_token_greedy_wrapper);
    m.def("llama_sample_token", &llama_sample_token_wrapper);

    m.def("sample_next_token", &sample_next_token); // helper function / not part of the llama.h API

    m.def("llama_print_timings", &llama_print_timings_wrapper);
    m.def("llama_reset_timings", &llama_reset_timings_wrapper);

    m.def("llama_print_system_info", &llama_print_system_info);

    m.def("llama_generate", &llama_generate);

    py::class_<LLaMAModel>(m,"LLaMAModel")
        .def(py::init<llama_context_wrapper *, gpt_params, int>())
        .def_readwrite("embd", &LLaMAModel::embd)
        .def_readwrite("n_remain", &LLaMAModel::n_remain)
        .def_readwrite("is_antiprompt", &LLaMAModel::is_antiprompt)
        .def_readwrite("n_consumed", &LLaMAModel::n_consumed)
        .def_readwrite("last_n_tokens", &LLaMAModel::last_n_tokens)
        .def_readwrite("input_echo", &LLaMAModel::input_echo)
        .def_readwrite("embd_out", &LLaMAModel::embd_out)

        .def("setup", &LLaMAModel::setup)
        .def("generate", &LLaMAModel::generate)
        .def("update_prompt", &LLaMAModel::update_prompt)

       ;


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
