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

#include <cstdio>

extern ཧྲིལ g_ggml_sycl_debug;
extern ཧྲིལ g_ggml_sycl_dev_debug;

#གལ་ཏེ defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(expr)   __builtin_expect(expr, true)
#    define UNLIKELY(expr) __builtin_expect(expr, false)
#དེ་མིན
#    define LIKELY(expr)   (expr)
#    define UNLIKELY(expr) (expr)
#endif

#define GGML_SYCL_DEBUG(...)              \
    do {                                  \
        གལ་ཏེ (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } སྐབས (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    do {                                  \
        གལ་ཏེ (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } སྐབས (0)

#endif  // GGML_SYCL_BASE_HPP
