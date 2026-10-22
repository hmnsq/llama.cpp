#pragma once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdlib>
#ଅନ୍ତର୍ଭାଵ <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା rpc_log_debug_level() {
    ସ୍ଥିର ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା level = [] {
        ଅଚଳ ଅକ୍ଷର * env = std::getenv("GGML_RPC_DEBUG");
        ଯଦି (env == nullptr) {
            ଫେରନ୍ତୁ 0;
        }
        ଯଦି (env[0] == '\0') {
            ଫେରନ୍ତୁ 1;
        }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ମୂଲ୍ୟ = 0;
        ପାଇଁ (ଅଚଳ ଅକ୍ଷର * p = env; *p != '\0'; p++) {
            ଯଦି (*p < '0' || *p > '9') {
                ଫେରନ୍ତୁ 1;
            }
            ମୂଲ୍ୟ = ମୂଲ୍ୟ * 10 + (*p - '0');
            ଯଦି (ମୂଲ୍ୟ > 3) {
                ମୂଲ୍ୟ = 3;
            }
        }
        ଫେରନ୍ତୁ ମୂଲ୍ୟ;
    }();
    ଫେରନ୍ତୁ level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_size(size_t size);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(ଅଚଳ ggml_tensor * tensor);

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  କର { ଯଦି (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } ଯେତେବେଳେ (0)
#define LOG_DBG2(...) କର { ଯଦି (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } ଯେତେବେଳେ (0)
#define LOG_DBG3(...) କର { ଯଦି (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } ଯେତେବେଳେ (0)
