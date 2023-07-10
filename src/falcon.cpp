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
#include <string>

#include "main.h"
#include "falcon.h"
#include "../ggllm.cpp/cmpnct_unicode.h"



#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

struct falcon_context_wrapper {
    falcon_context* ptr;
};

struct falcon_context_wrapper falcon_init_from_file_wrapper(const char * path_model, struct falcon_context_params  params){
    struct falcon_context * ctx = falcon_init_from_file(path_model, params);
    struct falcon_context_wrapper ctw_w;
    ctw_w.ptr = ctx;
    return ctw_w;
}

void falcon_prepare_buffers_wrapper(falcon_context_wrapper *ctx_w, int n_batch, int n_ctx){
    struct falcon_context * ctx = ctx_w->ptr;
    return falcon_prepare_buffers(ctx, n_batch, n_ctx);
}


int falcon_eval_wrapper(
        struct falcon_context_wrapper * ctx_w,
        py::array_t<falcon_token> tokens,
        int   n_tokens,
        int   n_past,
        int   n_threads,
        int debug_timings){
    struct falcon_context * ctx = ctx_w->ptr;
    py::buffer_info buf = tokens.request();
    falcon_token *tokens_ptr = static_cast<falcon_token *>(buf.ptr);
    return falcon_eval(ctx, tokens_ptr, n_tokens, n_past, n_threads, debug_timings);
}

int falcon_get_vocab_wrapper(
        const struct falcon_context_wrapper * ctx_w,
        const char * * strings,
        float * scores,
        int   capacity){
    struct falcon_context * ctx = ctx_w->ptr;
    return falcon_get_vocab(ctx, strings, scores, capacity);
}


std::vector<falcon_token> falcon_tokenize_wrapper(
        struct falcon_context_wrapper * ctx_w,
        const std::string & text,
        bool   add_bos){

    struct falcon_context * ctx = ctx_w->ptr;
    std::vector<falcon_token> tokens((text.size() + (int)add_bos));
    int new_size = falcon_tokenize(ctx, text.c_str(), tokens.data(), static_cast<int>(tokens.size()), add_bos);
    assert(new_size >= 0);
    tokens.resize(new_size);
    return tokens;
}

falcon_token falcon_sample_next_token(struct falcon_context_wrapper * ctx_w, gpt_params params, std::vector<falcon_token> & last_n_tokens){
    // helper function to sample next token (based on the main example)
    struct falcon_context * ctx = ctx_w->ptr;
    const int n_ctx = falcon_n_ctx(ctx);
    falcon_token id = 0;

    const float   temp            = params.temp;
    const int32_t top_k           = params.top_k <= 0 ? falcon_n_vocab(ctx) : params.top_k;
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
        auto logits  = falcon_get_logits(ctx);
        auto n_vocab = falcon_n_vocab(ctx);

        // Apply params.logit_bias map
        for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
            logits[it->first] += it->second;
        }

        std::vector<falcon_token_data> candidates;
        candidates.reserve(n_vocab);
        for (falcon_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(falcon_token_data{token_id, logits[token_id], 0.0f});
        }

        falcon_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

        // Apply penalties
        float nl_logit = logits[falcon_token_nl()];
        auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
        llama_sample_repetition_penalty(ctx, &candidates_p,
                                        last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                        last_n_repeat, repeat_penalty);
        llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                      last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                      last_n_repeat, alpha_frequency, alpha_presence);
        if (!penalize_nl) {
            logits[falcon_token_nl()] = nl_logit;
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
    }

    return id;
}




///////////////////


//void falcon_module(py::module_ &m) {

PYBIND11_MODULE(_pyggllmcpp, m) {

     m.attr("FALCON_FILE_MAGIC_GGCC") = py::int_(FALCON_FILE_MAGIC_GGCC);

     py::class_<falcon_context_wrapper>(m,"falcon_context");
//     py::class_<falcon_tokenizer>(m,"falcon_tokenizer");

    py::class_<falcon_token_data>(m,"falcon_token_data")
            .def(py::init<>())
            .def_readwrite("id", &falcon_token_data::id)
            .def_readwrite("p", &falcon_token_data::p)
            .def_readwrite("logit", &falcon_token_data::logit);

    py::class_<falcon_token_data_array>(m,"falcon_token_data_array")
        .def(py::init<>())
        .def_readwrite("data", &falcon_token_data_array::data)
        .def_readwrite("size", &falcon_token_data_array::size)
        .def_readwrite("sorted", &falcon_token_data_array::sorted);

    py::class_<falcon_context_params>(m,"falcon_context_params")
            .def(py::init<>())
            .def_readwrite("n_ctx", &falcon_context_params::n_ctx)
            .def_readwrite("n_batch", &falcon_context_params::n_batch)
            .def_readwrite("n_gpu_layers", &falcon_context_params::n_gpu_layers)
            .def_readwrite("i_gpu_start", &falcon_context_params::i_gpu_start)
            .def_readwrite("i_gpu_last", &falcon_context_params::i_gpu_last)
            .def_readwrite("main_gpu", &falcon_context_params::main_gpu)
//            .def_readwrite("tensor_split", &falcon_context_params::tensor_split)
            .def_readwrite("seed", &falcon_context_params::seed)

            .def_readwrite("f16_kv", &falcon_context_params::f16_kv)
            .def_readwrite("logits_all", &falcon_context_params::logits_all)
            .def_readwrite("vocab_only", &falcon_context_params::vocab_only)
            .def_readwrite("use_mmap", &falcon_context_params::use_mmap)
            .def_readwrite("use_mlock", &falcon_context_params::use_mlock)
            .def_readwrite("embedding", &falcon_context_params::embedding);


//            .def_readwrite("progress_callback_user_data", &falcon_context_params::progress_callback_user_data);

    m.def("falcon_context_default_params", &falcon_context_default_params);

    m.def("falcon_init_backend", &falcon_init_backend);
    m.def("falcon_init_from_file", &falcon_init_from_file_wrapper);
    m.def("falcon_prepare_buffers", &falcon_prepare_buffers_wrapper);
    m.def("falcon_eval", &falcon_eval_wrapper);

    m.def("falcon_tokenize", &falcon_tokenize_wrapper);

    m.def("falcon_n_vocab", [](struct falcon_context_wrapper * ctx_w) {
        return falcon_n_vocab(ctx_w->ptr);
        });
    m.def("falcon_n_ctx", [](struct falcon_context_wrapper * ctx_w) {
        return falcon_n_ctx(ctx_w->ptr);
        });
    m.def("falcon_n_embd", [](struct falcon_context_wrapper * ctx_w) {
        return falcon_n_embd(ctx_w->ptr);
        });
    m.def("falcon_free", [](struct falcon_context_wrapper * ctx_w) {
        return llama_free(ctx_w->ptr);
        });
    m.def("falcon_token_to_str", [](struct falcon_context_wrapper * ctx_w, falcon_token token) {
        return py::bytes(falcon_token_to_str(ctx_w->ptr, token));
        });
    m.def("falcon_print_timings", [](struct falcon_context_wrapper * ctx_w, falcon_token token) {
        return falcon_print_timings(ctx_w->ptr);
        });
    m.def("llama_reset_timings", [](struct falcon_context_wrapper * ctx_w, falcon_token token) {
        return llama_reset_timings(ctx_w->ptr);
        });

    m.def("falcon_set_rng_seed", [](struct falcon_context_wrapper * ctx_w, int seed) {
        return llama_set_rng_seed(ctx_w->ptr, seed);
        });

    m.def("falcon_token_bos", &falcon_token_bos);
    m.def("falcon_token_eos", &falcon_token_eos);
    m.def("falcon_token_nl", &falcon_token_nl);

    m.def("falcon_print_system_info", &falcon_print_system_info);

    m.def("falcon_sample_next_token", &falcon_sample_next_token);
}