#pragma once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#شامل "ggml-impl.h"

#شامل <cstddef>
#شامل <cstdlib>
#شامل <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
ایستا درونخطی صحیح rpc_log_debug_level() {
    ایستا ثابت صحیح level = [] {
        ثابت نویسه * env = std::getenv("GGML_RPC_DEBUG");
        اگر (env == nullptr) {
            بازگشت 0;
        }
        اگر (env[0] == '\0') {
            بازگشت 1;
        }
        صحیح مقدار = 0;
        برای (ثابت نویسه * p = env; *p != '\0'; p++) {
            اگر (*p < '0' || *p > '9') {
                بازگشت 1;
            }
            مقدار = مقدار * 10 + (*p - '0');
            اگر (مقدار > 3) {
                مقدار = 3;
            }
        }
        بازگشت مقدار;
    }();
    بازگشت level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_size(size_t size);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(ثابت ggml_tensor * tensor);

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  انجامبده { اگر (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } وقتی (0)
#define LOG_DBG2(...) انجامبده { اگر (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } وقتی (0)
#define LOG_DBG3(...) انجامبده { اگر (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } وقتی (0)
