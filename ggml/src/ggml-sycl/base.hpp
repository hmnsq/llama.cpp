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

#অন্তর্ভাৱ <cstdio>

বাহ্য পূর্ণাঙ্ক g_ggml_sycl_debug;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_dev_debug;

#যদি defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(ৰাশিমালা)   __builtin_expect(ৰাশিমালা, সঁচা)
#    define UNLIKELY(ৰাশিমালা) __builtin_expect(ৰাশিমালা, মিছা)
#অন্যথা
#    define LIKELY(ৰাশিমালা)   (ৰাশিমালা)
#    define UNLIKELY(ৰাশিমালা) (ৰাশিমালা)
#endif

#define GGML_SYCL_DEBUG(...)              \
    কৰা {                                  \
        যদি (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } যেতিয়ালৈ (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    কৰা {                                  \
        যদি (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } যেতিয়ালৈ (0)

#endif  // GGML_SYCL_BASE_HPP
