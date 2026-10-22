#اگر_غیر_تعریف_شدہ GGML_SYCL_بنیاد_HPP
#تعریف_کرو GGML_SYCL_بنیاد_HPP

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

خارجی صحیح g_ggml_sycl_debug;
خارجی صحیح g_ggml_sycl_dev_debug;

#اگر defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(اظہاریہ)   __builtin_expect(اظہاریہ, سچ)
#    define UNLIKELY(اظہاریہ) __builtin_expect(اظہاریہ, جھوٹ)
#ورنہ
#    define LIKELY(اظہاریہ)   (اظہاریہ)
#    define UNLIKELY(اظہاریہ) (اظہاریہ)
#اگر_ختم

#تعریف_کرو GGML_SYCL_DEBUG(...)              \
    کرو {                                  \
        اگر (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } جب (0)

#تعریف_کرو GGML_SYCL_DEV_DEBUG(...)          \
    کرو {                                  \
        اگر (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } جب (0)

#اگر_ختم  // GGML_SYCL_BASE_HPP
