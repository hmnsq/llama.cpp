#pragma once

// Log infrastructure for the RPC backend: LOG_* are unconditional, LOG_DBG* are gated by GGML_RPC_DEBUG

#അന്തര്ഭാവ "ggml-impl.h"

#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <string>

// GGML_RPC_DEBUG verbosity: unset / 0 disabled, 1 connections and allocations, 2 per-command trace,
// 3 per-message detail. Non-numeric values fall back to 1
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ rpc_log_debug_level() {
    സ്ഥിതം സ്ഥിരം പൂർണ്ണസംഖ്യ level = [] {
        സ്ഥിരം അക്ഷരം * env = std::getenv("GGML_RPC_DEBUG");
        എങ്കിൽ (env == nullptr) {
            തിരികെ 0;
        }
        എങ്കിൽ (env[0] == '\0') {
            തിരികെ 1;
        }
        പൂർണ്ണസംഖ്യ മൂല്യം = 0;
        വേണ്ടി (സ്ഥിരം അക്ഷരം * p = env; *p != '\0'; p++) {
            എങ്കിൽ (*p < '0' || *p > '9') {
                തിരികെ 1;
            }
            മൂല്യം = മൂല്യം * 10 + (*p - '0');
            എങ്കിൽ (മൂല്യം > 3) {
                മൂല്യം = 3;
            }
        }
        തിരികെ മൂല്യം;
    }();
    തിരികെ level;
}

// Byte counts in binary units, e.g. 8208 -> "8.02 KiB". Sizes below 1 KiB stay in plain bytes
std::string rpc_log_format_size(size_t size);

// Tensor identity: "blk.0.attn_q.weight [NONE, f16, [4096, 4096, 1, 1]]"
std::string rpc_log_format_tensor(സ്ഥിരം ggml_tensor * tensor);

#define LOG_INFO(...)  GGML_LOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  GGML_LOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) GGML_LOG_ERROR(__VA_ARGS__)

#define LOG_DBG(...)  ചെയ്യുക { എങ്കിൽ (rpc_log_debug_level() >= 1) GGML_LOG_DEBUG(__VA_ARGS__); } ആകുമ്പോൾ (0)
#define LOG_DBG2(...) ചെയ്യുക { എങ്കിൽ (rpc_log_debug_level() >= 2) GGML_LOG_DEBUG(__VA_ARGS__); } ആകുമ്പോൾ (0)
#define LOG_DBG3(...) ചെയ്യുക { എങ്കിൽ (rpc_log_debug_level() >= 3) GGML_LOG_DEBUG(__VA_ARGS__); } ആകുമ്പോൾ (0)
