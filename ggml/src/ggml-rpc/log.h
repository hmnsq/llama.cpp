#ਹਦਾਇਤ once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ rpc_log_debug_level() {
    ਸਥਿਰ ਅਚਲ ਪੂਰਨ level = [] {
        ਅਚਲ ਅੱਖਰ * env = std::getenv("GGML_RPC_DEBUG");
        ਜੇ (env == nullptr) {
            ਵਾਪਸ 0;
        }
        ਜੇ (env[0] == '\0') {
            ਵਾਪਸ 1;
        }
        ਪੂਰਨ ਮੁੱਲ = 0;
        ਲਈ (ਅਚਲ ਅੱਖਰ * p = env; *p != '\0'; p++) {
            ਜੇ (*p < '0' || *p > '9') {
                ਵਾਪਸ 1;
            }
            ਮੁੱਲ = ਮੁੱਲ * 10 + (*p - '0');
            ਜੇ (ਮੁੱਲ > 3) {
                ਮੁੱਲ = 3;
            }
        }
        ਵਾਪਸ ਮੁੱਲ;
    }();
    ਵਾਪਸ level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_ਮਾਪ(size_t ਮਾਪ);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(ਅਚਲ ggml_tensor * tensor);

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LOG_DBG(...)  ਕਰੋ { ਜੇ (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } ਜਦੋਂ (0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LOG_DBG2(...) ਕਰੋ { ਜੇ (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } ਜਦੋਂ (0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LOG_DBG3(...) ਕਰੋ { ਜੇ (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } ਜਦੋਂ (0)
