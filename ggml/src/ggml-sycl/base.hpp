#که_نه_تعریف_شوی GGML_SYCL_بنسټ_HPP
#تعریف_کړه GGML_SYCL_بنسټ_HPP

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
#    define LIKELY(بیان)   __builtin_expect(بیان, ریښتیا)
#    define UNLIKELY(بیان) __builtin_expect(بیان, خطأ)
#وإلا
#    define LIKELY(بیان)   (بیان)
#    define UNLIKELY(بیان) (بیان)
#د_که_پای

#تعریف_کړه GGML_SYCL_DEBUG(...)              \
    افعل {                                  \
        إذا (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } طالما (0)

#تعریف_کړه GGML_SYCL_DEV_DEBUG(...)          \
    افعل {                                  \
        إذا (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } طالما (0)

#د_که_پای  // GGML_SYCL_BASE_HPP
