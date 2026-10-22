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

#यदि defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(व्यंजक)   __builtin_expect(व्यंजक, सत्य)
#    define UNLIKELY(व्यंजक) __builtin_expect(व्यंजक, असत्य)
#अन्यथा
#    define LIKELY(व्यंजक)   (व्यंजक)
#    define UNLIKELY(व्यंजक) (व्यंजक)
#endif

#define GGML_SYCL_DEBUG(...)              \
    कुरु {                                  \
        यदि (UNLIKELY(g_ggml_sycl_debug))  \
            ख_लिखो(मानक_त्रुटि, __VA_ARGS__); \
    } यावत् (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    कुरु {                                  \
        यदि (UNLIKELY(g_ggml_sycl_dev_debug))  \
            ख_लिखो(मानक_त्रुटि, __VA_ARGS__); \
    } यावत् (0)

#endif  // GGML_SYCL_BASE_HPP
