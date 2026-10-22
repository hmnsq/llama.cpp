#اگر_تعریف_نشده GGML_SYCL_پایه_HPP
#تعریف_کن GGML_SYCL_پایه_HPP

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
#    define LIKELY(عبارت)   __builtin_expect(عبارت, درست)
#    define UNLIKELY(عبارت) __builtin_expect(عبارت, خطأ)
#وإلا
#    define LIKELY(عبارت)   (عبارت)
#    define UNLIKELY(عبارت) (عبارت)
#پایان_اگر

#تعریف_کن GGML_SYCL_DEBUG(...)              \
    افعل {                                  \
        إذا (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } طالما (0)

#تعریف_کن GGML_SYCL_DEV_DEBUG(...)          \
    افعل {                                  \
        إذا (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } طالما (0)

#پایان_اگر  // GGML_SYCL_BASE_HPP
