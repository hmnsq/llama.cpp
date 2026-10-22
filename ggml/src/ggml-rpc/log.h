#pragma once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#அந்தர்பாவ "ggml-impl.h"

#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
நிலையான உள்வரி முழு rpc_log_debug_level() {
    நிலையான மாறிலி முழு level = [] {
        மாறிலி எழுத்து * env = std::getenv("GGML_RPC_DEBUG");
        எனில் (env == nullptr) {
            திரும்பு 0;
        }
        எனில் (env[0] == '\0') {
            திரும்பு 1;
        }
        முழு மதிப்பு = 0;
        க்கு (மாறிலி எழுத்து * p = env; *p != '\0'; p++) {
            எனில் (*p < '0' || *p > '9') {
                திரும்பு 1;
            }
            மதிப்பு = மதிப்பு * 10 + (*p - '0');
            எனில் (மதிப்பு > 3) {
                மதிப்பு = 3;
            }
        }
        திரும்பு மதிப்பு;
    }();
    திரும்பு level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_size(size_t size);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(மாறிலி ggml_tensor * tensor);

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  செய் { எனில் (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } போது (0)
#define LOG_DBG2(...) செய் { எனில் (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } போது (0)
#define LOG_DBG3(...) செய் { எனில் (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } போது (0)
