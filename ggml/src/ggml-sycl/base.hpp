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

#ಅನ್ತರ್ಭಾವ <cstdio>

ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_debug;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_dev_debug;

#ಯದಿ defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(ಅಭಿವ್ಯಕ್ತಿ)   __builtin_expect(ಅಭಿವ್ಯಕ್ತಿ, ಸತ್ಯ)
#    define UNLIKELY(ಅಭಿವ್ಯಕ್ತಿ) __builtin_expect(ಅಭಿವ್ಯಕ್ತಿ, ಅಸತ್ಯ)
#ಅನ್ಯಥಾ
#    define LIKELY(ಅಭಿವ್ಯಕ್ತಿ)   (ಅಭಿವ್ಯಕ್ತಿ)
#    define UNLIKELY(ಅಭಿವ್ಯಕ್ತಿ) (ಅಭಿವ್ಯಕ್ತಿ)
#endif

#define GGML_SYCL_DEBUG(...)              \
    ಕುರು {                                  \
        ಯದಿ (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } ಯಾವತ್ (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    ಕುರು {                                  \
        ಯದಿ (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } ಯಾವತ್ (0)

#endif  // GGML_SYCL_BASE_HPP
