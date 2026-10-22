#pragma once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#ضمن "ggml-impl.h"

#ضمن <cstddef>
#ضمن <cstdlib>
#ضمن <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
ساكن مضمن صحيح rpc_log_debug_level() {
    ساكن ثابت صحيح level = [] {
        ثابت حرف * env = std::getenv("GGML_RPC_DEBUG");
        𐤀𐤌 (env == nullptr) {
            أرجع 0;
        }
        𐤀𐤌 (env[0] == '\0') {
            أرجع 1;
        }
        صحيح value = 0;
        لكل (ثابت حرف * p = env; *p != '\0'; p++) {
            𐤀𐤌 (*p < '0' || *p > '9') {
                أرجع 1;
            }
            value = value * 10 + (*p - '0');
            𐤀𐤌 (value > 3) {
                value = 3;
            }
        }
        أرجع value;
    }();
    أرجع level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_size(size_t size);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(ثابت ggml_tensor * tensor);

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  𐤐𐤏𐤋 { 𐤀𐤌 (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } طالما (0)
#define LOG_DBG2(...) 𐤐𐤏𐤋 { 𐤀𐤌 (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } طالما (0)
#define LOG_DBG3(...) 𐤐𐤏𐤋 { 𐤀𐤌 (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } طالما (0)
