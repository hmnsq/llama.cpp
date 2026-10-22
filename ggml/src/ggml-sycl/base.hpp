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

#অন্তর্ভাব <cstdio>

বাহ্যিক পূর্ণসংখ্যা g_ggml_sycl_debug;
বাহ্যিক পূর্ণসংখ্যা g_ggml_sycl_dev_debug;

#যদি defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(রাশিমালা)   __builtin_expect(রাশিমালা, সত্য)
#    define UNLIKELY(রাশিমালা) __builtin_expect(রাশিমালা, মিথ্যা)
#অন্যথা
#    define LIKELY(রাশিমালা)   (রাশিমালা)
#    define UNLIKELY(রাশিমালা) (রাশিমালা)
#endif

#define GGML_SYCL_DEBUG(...)              \
    করো {                                  \
        যদি (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } যখন (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    করো {                                  \
        যদি (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } যখন (0)

#endif  // GGML_SYCL_BASE_HPP
