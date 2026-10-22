#pragma once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
स्थिर अन्तःस्थ पूर्णाङ्क rpc_log_debug_level() {
    स्थिर ध्रुव पूर्णाङ्क level = [] {
        ध्रुव वर्ण * env = std::getenv("GGML_RPC_DEBUG");
        जइ (env == nullptr) {
            प्रत्यागम 0;
        }
        जइ (env[0] == '\0') {
            प्रत्यागम 1;
        }
        पूर्णाङ्क मानम् = 0;
        प्रति (ध्रुव वर्ण * p = env; *p != '\0'; p++) {
            जइ (*p < '0' || *p > '9') {
                प्रत्यागम 1;
            }
            मानम् = मानम् * 10 + (*p - '0');
            जइ (मानम् > 3) {
                मानम् = 3;
            }
        }
        प्रत्यागम मानम्;
    }();
    प्रत्यागम level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_size(size_t size);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(ध्रुव ggml_tensor * tensor);

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  करेहि { जइ (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } जाव (0)
#define LOG_DBG2(...) करेहि { जइ (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } जाव (0)
#define LOG_DBG3(...) करेहि { जइ (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } जाव (0)
