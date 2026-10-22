#pragma once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
स्थिर ओळीत पूर्णांक rpc_log_debug_level() {
    स्थिर स्थिरांक पूर्णांक level = [] {
        स्थिरांक अक्षर * env = std::दो_पर्या("GGML_RPC_DEBUG");
        जर (env == nullptr) {
            परत 0;
        }
        जर (env[0] == '\0') {
            परत 1;
        }
        पूर्णांक मूल्य = 0;
        साठी (स्थिरांक अक्षर * p = env; *p != '\0'; p++) {
            जर (*p < '0' || *p > '9') {
                परत 1;
            }
            मूल्य = मूल्य * 10 + (*p - '0');
            जर (मूल्य > 3) {
                मूल्य = 3;
            }
        }
        परत मूल्य;
    }();
    परत level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_size(माप_प्रकार size);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(स्थिरांक ggml_tensor * tensor);

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  करा { जर (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } पर्यंत (0)
#define LOG_DBG2(...) करा { जर (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } पर्यंत (0)
#define LOG_DBG3(...) करा { जर (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } पर्यंत (0)
