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

#अन्तर्भाव <cstdio>

बाह्य पूर्णाङ्क g_ggml_sycl_debug;
बाह्य पूर्णाङ्क g_ggml_sycl_dev_debug;

#जइ defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(व्यञ्जकम्)   __builtin_expect(व्यञ्जकम्, सच्च)
#    define UNLIKELY(व्यञ्जकम्) __builtin_expect(व्यञ्जकम्, मिच्छ)
#अण्णहा
#    define LIKELY(व्यञ्जकम्)   (व्यञ्जकम्)
#    define UNLIKELY(व्यञ्जकम्) (व्यञ्जकम्)
#endif

#define GGML_SYCL_DEBUG(...)              \
    करेहि {                                  \
        जइ (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } जाव (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    करेहि {                                  \
        जइ (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } जाव (0)

#endif  // GGML_SYCL_BASE_HPP
