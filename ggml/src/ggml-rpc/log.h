#సూచన once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
స్థిర అంతఃవరుస పూర్ణాంకం rpc_log_debug_level() {
    స్థిర స్థిరరాశి పూర్ణాంకం level = [] {
        స్థిరరాశి అక్షరం * env = std::getenv("GGML_RPC_DEBUG");
        ఒకవేళ (env == nullptr) {
            తిరిగి 0;
        }
        ఒకవేళ (env[0] == '\0') {
            తిరిగి 1;
        }
        పూర్ణాంకం విలువ = 0;
        కొరకు (స్థిరరాశి అక్షరం * p = env; *p != '\0'; p++) {
            ఒకవేళ (*p < '0' || *p > '9') {
                తిరిగి 1;
            }
            విలువ = విలువ * 10 + (*p - '0');
            ఒకవేళ (విలువ > 3) {
                విలువ = 3;
            }
        }
        తిరిగి విలువ;
    }();
    తిరిగి level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_కొలత(size_t కొలత);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(స్థిరరాశి ggml_tensor * tensor);

#నిర్వచించు LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#నిర్వచించు LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#నిర్వచించు LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#నిర్వచించు LOG_DBG(...)  చేయి { ఒకవేళ (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } అయితే (0)
#నిర్వచించు LOG_DBG2(...) చేయి { ఒకవేళ (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } అయితే (0)
#నిర్వచించు LOG_DBG3(...) చేయి { ఒకవేళ (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } అయితే (0)
