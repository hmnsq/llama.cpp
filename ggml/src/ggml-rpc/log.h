#pragma once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#include "ggml-impl.h"

#include <cstddef>
#include <cstdlib>
#include <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
static anta_hstha ຈຳນວນເຕັມ rpc_log_debug_level() {
    static const ຈຳນວນເຕັມ level = [] {
        const var_na * env = std::getenv("GGML_RPC_DEBUG");
        ຖ້າ (env == nullptr) {
            ສົ່ງຄືນ 0;
        }
        ຖ້າ (env[0] == '\0') {
            ສົ່ງຄືນ 1;
        }
        ຈຳນວນເຕັມ value = 0;
        ສຳລັບ (const var_na * p = env; *p != '\0'; p++) {
            ຖ້າ (*p < '0' || *p > '9') {
                ສົ່ງຄືນ 1;
            }
            value = value * 10 + (*p - '0');
            ຖ້າ (value > 3) {
                value = 3;
            }
        }
        ສົ່ງຄືນ value;
    }();
    ສົ່ງຄືນ level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_size(size_t size);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(const ggml_tensor * tensor);

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  do { ຖ້າ (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } ໃນຂະນະທີ່ (0)
#define LOG_DBG2(...) do { ຖ້າ (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } ໃນຂະນະທີ່ (0)
#define LOG_DBG3(...) do { ຖ້າ (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } ໃນຂະນະທີ່ (0)
