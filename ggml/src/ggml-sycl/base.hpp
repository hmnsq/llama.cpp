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

#شامل <cstdio>

بیرونی صحیح g_ggml_sycl_debug;
بیرونی صحیح g_ggml_sycl_dev_debug;

#اگر defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(عبارت)   __builtin_expect(عبارت, درست)
#    define UNLIKELY(عبارت) __builtin_expect(عبارت, نادرست)
#وگرنه
#    define LIKELY(عبارت)   (عبارت)
#    define UNLIKELY(عبارت) (عبارت)
#endif

#define GGML_SYCL_DEBUG(...)              \
    انجامبده {                                  \
        اگر (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } وقتی (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    انجامبده {                                  \
        اگر (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } وقتی (0)

#endif  // GGML_SYCL_BASE_HPP
