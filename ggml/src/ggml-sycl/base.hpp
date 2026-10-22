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

#ضمن <cstdio>

خارجي صحيح g_ggml_sycl_debug;
خارجي صحيح g_ggml_sycl_dev_debug;

#إذا defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(اظہاریہ)   __builtin_expect(اظہاریہ, true)
#    define UNLIKELY(اظہاریہ) __builtin_expect(اظہاریہ, خطأ)
#وإلا
#    define LIKELY(اظہاریہ)   (اظہاریہ)
#    define UNLIKELY(اظہاریہ) (اظہاریہ)
#endif

#define GGML_SYCL_DEBUG(...)              \
    افعل {                                  \
        إذا (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } طالما (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    افعل {                                  \
        إذا (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } طالما (0)

#endif  // GGML_SYCL_BASE_HPP
