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
#    define LIKELY(تعبير)   __builtin_expect(تعبير, true)
#    define UNLIKELY(تعبير) __builtin_expect(تعبير, خطأ)
#وإلا
#    define LIKELY(تعبير)   (تعبير)
#    define UNLIKELY(تعبير) (تعبير)
#endif

#define GGML_SYCL_DEBUG(...)              \
    افعل {                                  \
        إذا (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } بينما (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    افعل {                                  \
        إذا (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } بينما (0)

#endif  // GGML_SYCL_BASE_HPP
