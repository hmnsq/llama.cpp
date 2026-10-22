#pragma once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#অন্তর্ভাৱ "ggml-impl.h"

#অন্তর্ভাৱ <cstddef>
#অন্তর্ভাৱ <cstdlib>
#অন্তর্ভাৱ <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
স্থির অন্তঃস্থ পূর্ণাঙ্ক rpc_log_debug_level() {
    স্থির ধ্রুৱ পূর্ণাঙ্ক level = [] {
        ধ্রুৱ ৱর্ণ * env = std::getenv("GGML_RPC_DEBUG");
        যদি (env == nullptr) {
            ঘূৰাই 0;
        }
        যদি (env[0] == '\0') {
            ঘূৰাই 1;
        }
        পূর্ণাঙ্ক মান = 0;
        বাবে (ধ্রুৱ ৱর্ণ * p = env; *p != '\0'; p++) {
            যদি (*p < '0' || *p > '9') {
                ঘূৰাই 1;
            }
            মান = মান * 10 + (*p - '0');
            যদি (মান > 3) {
                মান = 3;
            }
        }
        ঘূৰাই মান;
    }();
    ঘূৰাই level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_size(size_t size);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(ধ্রুৱ ggml_tensor * tensor);

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  কৰা { যদি (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } যেতিয়ালৈ (0)
#define LOG_DBG2(...) কৰা { যদি (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } যেতিয়ালৈ (0)
#define LOG_DBG3(...) কৰা { যদি (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } যেতিয়ালৈ (0)
