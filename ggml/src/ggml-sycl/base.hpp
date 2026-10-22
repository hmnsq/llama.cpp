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

#அந்தர்பாவ <cstdio>

வெளி முழு g_ggml_sycl_debug;
வெளி முழு g_ggml_sycl_dev_debug;

#எனில் defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(கோவை)   __builtin_expect(கோவை, உண்மை)
#    define UNLIKELY(கோவை) __builtin_expect(கோவை, பொய்)
#இல்லையெனில்
#    define LIKELY(கோவை)   (கோவை)
#    define UNLIKELY(கோவை) (கோவை)
#endif

#define GGML_SYCL_DEBUG(...)              \
    செய் {                                  \
        எனில் (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } போது (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    செய் {                                  \
        எனில் (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } போது (0)

#endif  // GGML_SYCL_BASE_HPP
