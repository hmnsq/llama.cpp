#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_ತಳ_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_ತಳ_HPP

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

ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_debug;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_dev_debug;

#ಆದರೆ defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(ಅಭಿವ್ಯಕ್ತಿ)   __builtin_expect(ಅಭಿವ್ಯಕ್ತಿ, ನಿಜ)
#    define UNLIKELY(ಅಭಿವ್ಯಕ್ತಿ) __builtin_expect(ಅಭಿವ್ಯಕ್ತಿ, ಸುಳ್ಳು)
#ಇಲ್ಲದಿದ್ದರೆ
#    define LIKELY(ಅಭಿವ್ಯಕ್ತಿ)   (ಅಭಿವ್ಯಕ್ತಿ)
#    define UNLIKELY(ಅಭಿವ್ಯಕ್ತಿ) (ಅಭಿವ್ಯಕ್ತಿ)
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_DEBUG(...)              \
    ಮಾಡು {                                  \
        ಆದರೆ (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } ಆಗ (0)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_DEV_DEBUG(...)          \
    ಮಾಡು {                                  \
        ಆದರೆ (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } ಆಗ (0)

#ಆದರೆ_ಅಂತ್ಯ  // GGML_SYCL_BASE_HPP
