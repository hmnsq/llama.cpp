#ہدایت once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#شامل "ggml-impl.h"

#شامل <cstddef>
#شامل <cstdlib>
#شامل <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
ساکن اندرونی صحیح rpc_log_debug_level() {
    ساکن ثابت صحیح level = [] {
        ثابت حرف * env = std::getenv("GGML_RPC_DEBUG");
        اگر (env == nullptr) {
            واپس 0;
        }
        اگر (env[0] == '\0') {
            واپس 1;
        }
        صحیح قدر = 0;
        برائے (ثابت حرف * p = env; *p != '\0'; p++) {
            اگر (*p < '0' || *p > '9') {
                واپس 1;
            }
            قدر = قدر * 10 + (*p - '0');
            اگر (قدر > 3) {
                قدر = 3;
            }
        }
        واپس قدر;
    }();
    واپس level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_سائز(size_t سائز);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(ثابت ggml_tensor * tensor);

#تعریف_کرو LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#تعریف_کرو LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#تعریف_کرو LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#تعریف_کرو LOG_DBG(...)  کرو { اگر (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } جب (0)
#تعریف_کرو LOG_DBG2(...) کرو { اگر (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } جب (0)
#تعریف_کرو LOG_DBG3(...) کرو { اگر (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } جب (0)
