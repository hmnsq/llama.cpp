#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_بنياد_HPP
#وصف_ڪريو GGML_SYCL_بنياد_HPP

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
#    define LIKELY(اظهار)   __builtin_expect(اظهار, سچ)
#    define UNLIKELY(اظهار) __builtin_expect(اظهار, خطأ)
#وإلا
#    define LIKELY(اظهار)   (اظهار)
#    define UNLIKELY(اظهار) (اظهار)
#جيڪڏهن_پورو

#وصف_ڪريو GGML_SYCL_DEBUG(...)              \
    افعل {                                  \
        إذا (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } طالما (0)

#وصف_ڪريو GGML_SYCL_DEV_DEBUG(...)          \
    افعل {                                  \
        إذا (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } طالما (0)

#جيڪڏهن_پورو  // GGML_SYCL_BASE_HPP
