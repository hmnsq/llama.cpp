#pragma once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ rpc_log_debug_level() {
    ಸ್ಥಿರ ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ level = [] {
        ಧ್ರುವ ವರ್ಣ * env = std::getenv("GGML_RPC_DEBUG");
        ಯದಿ (env == nullptr) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }
        ಯದಿ (env[0] == '\0') {
            ಪ್ರತ್ಯಾಗಮ 1;
        }
        ಪೂರ್ಣಾಙ್ಕ ಮೌಲ್ಯ = 0;
        ಪ್ರತಿ (ಧ್ರುವ ವರ್ಣ * p = env; *p != '\0'; p++) {
            ಯದಿ (*p < '0' || *p > '9') {
                ಪ್ರತ್ಯಾಗಮ 1;
            }
            ಮೌಲ್ಯ = ಮೌಲ್ಯ * 10 + (*p - '0');
            ಯದಿ (ಮೌಲ್ಯ > 3) {
                ಮೌಲ್ಯ = 3;
            }
        }
        ಪ್ರತ್ಯಾಗಮ ಮೌಲ್ಯ;
    }();
    ಪ್ರತ್ಯಾಗಮ level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_size(size_t size);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(ಧ್ರುವ ggml_tensor * tensor);

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  ಕುರು { ಯದಿ (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } ಯಾವತ್ (0)
#define LOG_DBG2(...) ಕುರು { ಯದಿ (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } ಯಾವತ್ (0)
#define LOG_DBG3(...) ಕುರು { ಯದಿ (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } ಯಾವತ್ (0)
