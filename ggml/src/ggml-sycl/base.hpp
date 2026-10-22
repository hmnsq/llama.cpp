#अगर_अघोषित GGML_SYCL_BASE_HPP
#घोषणा GGML_SYCL_BASE_HPP

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

#समावेश <cstdio>

बाह्य पूर्णांक g_ggml_sycl_debug;
बाह्य पूर्णांक g_ggml_sycl_dev_debug;

#अगर defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(व्यंजक)   __builtin_expect(व्यंजक, true)
#    define UNLIKELY(व्यंजक) __builtin_expect(व्यंजक, false)
#अन्यथा
#    define LIKELY(व्यंजक)   (व्यंजक)
#    define UNLIKELY(व्यंजक) (व्यंजक)
#पूर्ण_अगर

#घोषणा GGML_SYCL_DEBUG(...)              \
    करो {                                  \
        अगर (UNLIKELY(g_ggml_sycl_debug))  \
            ख_लिखो(मानक_त्रुटि, __VA_ARGS__); \
    } जबतक (0)

#घोषणा GGML_SYCL_DEV_DEBUG(...)          \
    करो {                                  \
        अगर (UNLIKELY(g_ggml_sycl_dev_debug))  \
            ख_लिखो(मानक_त्रुटि, __VA_ARGS__); \
    } जबतक (0)

#पूर्ण_अगर  // GGML_SYCL_BASE_HPP
