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

#ଅନ୍ତର୍ଭାଵ <cstdio>

ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_debug;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_dev_debug;

#ଯଦି defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(ପଦସମଷ୍ଟି)   __builtin_expect(ପଦସମଷ୍ଟି, ସତ୍ୟ)
#    define UNLIKELY(ପଦସମଷ୍ଟି) __builtin_expect(ପଦସମଷ୍ଟି, ମିଥ୍ୟା)
#ଅନ୍ୟଥା
#    define LIKELY(ପଦସମଷ୍ଟି)   (ପଦସମଷ୍ଟି)
#    define UNLIKELY(ପଦସମଷ୍ଟି) (ପଦସମଷ୍ଟି)
#endif

#define GGML_SYCL_DEBUG(...)              \
    କର {                                  \
        ଯଦି (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } ଯେତେବେଳେ (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    କର {                                  \
        ଯଦି (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } ଯେତେବେଳେ (0)

#endif  // GGML_SYCL_BASE_HPP
