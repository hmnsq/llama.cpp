#pragma once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
স্থির অন্তঃরেখ পূর্ণসংখ্যা rpc_log_debug_level() {
    স্থির ধ্রুবক পূর্ণসংখ্যা level = [] {
        ধ্রুবক অক্ষর * env = std::getenv("GGML_RPC_DEBUG");
        যদি (env == nullptr) {
            ফেরত 0;
        }
        যদি (env[0] == '\0') {
            ফেরত 1;
        }
        পূর্ণসংখ্যা মান = 0;
        জন্য (ধ্রুবক অক্ষর * p = env; *p != '\0'; p++) {
            যদি (*p < '0' || *p > '9') {
                ফেরত 1;
            }
            মান = মান * 10 + (*p - '0');
            যদি (মান > 3) {
                মান = 3;
            }
        }
        ফেরত মান;
    }();
    ফেরত level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_আকার(size_t আকার_2);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(ধ্রুবক ggml_tensor * tensor);

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  করো { যদি (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } যখন (0)
#define LOG_DBG2(...) করো { যদি (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } যখন (0)
#define LOG_DBG3(...) করো { যদি (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } যখন (0)
