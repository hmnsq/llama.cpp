#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_ਅਧਾਰ_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_ਅਧਾਰ_HPP

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

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>

ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_debug;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_dev_debug;

#ਜੇ defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(ਵਿਅੰਜਕ)   __builtin_expect(ਵਿਅੰਜਕ, ਸੱਚ)
#    define UNLIKELY(ਵਿਅੰਜਕ) __builtin_expect(ਵਿਅੰਜਕ, ਝੂਠ)
#ਨਹੀਂਤਾਂ
#    define LIKELY(ਵਿਅੰਜਕ)   (ਵਿਅੰਜਕ)
#    define UNLIKELY(ਵਿਅੰਜਕ) (ਵਿਅੰਜਕ)
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_DEBUG(...)              \
    ਕਰੋ {                                  \
        ਜੇ (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } ਜਦੋਂ (0)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_DEV_DEBUG(...)          \
    ਕਰੋ {                                  \
        ਜੇ (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } ਜਦੋਂ (0)

#ਜੇ_ਅੰਤ  // GGML_SYCL_BASE_HPP
