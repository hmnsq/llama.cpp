#आशय once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#समावेश "ggml-impl.h"

#समावेश <cstddef>
#समावेश <cstdlib>
#समावेश <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
अटल अंतरभूत पूर्णांक rpc_log_debug_level() {
    अटल स्थिर पूर्णांक level = [] {
        स्थिर अक्षर * env = std::दो_पर्या("GGML_RPC_DEBUG");
        अगर (env == nullptr) {
            वापस 0;
        }
        अगर (env[0] == '\0') {
            वापस 1;
        }
        पूर्णांक मूल्य = 0;
        क्रम (स्थिर अक्षर * p = env; *p != '\0'; p++) {
            अगर (*p < '0' || *p > '9') {
                वापस 1;
            }
            मूल्य = मूल्य * 10 + (*p - '0');
            अगर (मूल्य > 3) {
                मूल्य = 3;
            }
        }
        वापस मूल्य;
    }();
    वापस level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_आकार(माप_प्रकार आकार);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(स्थिर ggml_tensor * tensor);

#घोषणा LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#घोषणा LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#घोषणा LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#घोषणा LOG_DBG(...)  करो { अगर (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } जबतक (0)
#घोषणा LOG_DBG2(...) करो { अगर (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } जबतक (0)
#घोषणा LOG_DBG3(...) करो { अगर (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } जबतक (0)
