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

#અન્તર્ભાવ <cstdio>

બાહ્ય પૂર્ણાંક g_ggml_sycl_debug;
બાહ્ય પૂર્ણાંક g_ggml_sycl_dev_debug;

#જો defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(પદાવલિ)   __builtin_expect(પદાવલિ, સાચું)
#    define UNLIKELY(પદાવલિ) __builtin_expect(પદાવલિ, ખોટું)
#નહીંતર
#    define LIKELY(પદાવલિ)   (પદાવલિ)
#    define UNLIKELY(પદાવલિ) (પદાવલિ)
#endif

#define GGML_SYCL_DEBUG(...)              \
    કરો {                                  \
        જો (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } જ્યારે (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    કરો {                                  \
        જો (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } જ્યારે (0)

#endif  // GGML_SYCL_BASE_HPP
