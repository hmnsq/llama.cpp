#pragma once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#અન્તર્ભાવ "ggml-impl.h"

#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
સ્થિર અંતઃરેખ પૂર્ણાંક rpc_log_debug_level() {
    સ્થિર અચલ પૂર્ણાંક level = [] {
        અચલ અક્ષર * env = std::getenv("GGML_RPC_DEBUG");
        જો (env == nullptr) {
            પાછા 0;
        }
        જો (env[0] == '\0') {
            પાછા 1;
        }
        પૂર્ણાંક કિંમત = 0;
        માટે (અચલ અક્ષર * p = env; *p != '\0'; p++) {
            જો (*p < '0' || *p > '9') {
                પાછા 1;
            }
            કિંમત = કિંમત * 10 + (*p - '0');
            જો (કિંમત > 3) {
                કિંમત = 3;
            }
        }
        પાછા કિંમત;
    }();
    પાછા level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_size(size_t size);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(અચલ ggml_tensor * tensor);

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  કરો { જો (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } જ્યારે (0)
#define LOG_DBG2(...) કરો { જો (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } જ્યારે (0)
#define LOG_DBG3(...) કરો { જો (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } જ્યારે (0)
