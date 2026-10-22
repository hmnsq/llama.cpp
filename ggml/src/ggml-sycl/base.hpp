#ifndef GGML_SYCL_BASE_HPP
#define GGML_SYCL_BASE_HPP

/**
 * Module: base
 *
 * Description:
 * Provides zero-dependency, foundational primitives, core abstractions,
 * and low-level system interfaces. This module acts as the lowest layer
 * of the architecture and is consumed globally across all subsystems.
 *
 * Constraints:
 * - STRICTLY zero upstream dependencies (leaf module).
 * - High stability and backward compatibility required.
 */

#അന്തര്ഭാവ <cstdio>

ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_debug;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_dev_debug;

#എങ്കിൽ defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(വ്യഞ്ജകം)   __builtin_expect(വ്യഞ്ജകം, സത്യം)
#    define UNLIKELY(വ്യഞ്ജകം) __builtin_expect(വ്യഞ്ജകം, അസത്യം)
#അല്ലെങ്കിൽ
#    define LIKELY(വ്യഞ്ജകം)   (വ്യഞ്ജകം)
#    define UNLIKELY(വ്യഞ്ജകം) (വ്യഞ്ജകം)
#endif

#define GGML_SYCL_DEBUG(...)              \
    ചെയ്യുക {                                  \
        എങ്കിൽ (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } ആകുമ്പോൾ (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    ചെയ്യുക {                                  \
        എങ്കിൽ (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } ആകുമ്പോൾ (0)

#endif  // GGML_SYCL_BASE_HPP
