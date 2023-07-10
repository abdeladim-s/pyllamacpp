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
#include <pybind11/pybind11.h>
#include "../ggllm.cpp/libfalcon.h"

namespace py = pybind11;

struct falcon_context_wrapper;

std::vector<falcon_token> falcon_tokenize(struct falcon_context_wrapper * ctx, const std::string & text, bool add_bos);