#నిర్వచితం_కాకపోతే GGML_SYCL_పునాది_HPP
#నిర్వచించు GGML_SYCL_పునాది_HPP

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

#అన్తర్భావ <cstdio>

బాహ్య పూర్ణాంకం g_ggml_sycl_debug;
బాహ్య పూర్ణాంకం g_ggml_sycl_dev_debug;

#ఒకవేళ defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(సమాసం)   __builtin_expect(సమాసం, నిజం)
#    define UNLIKELY(సమాసం) __builtin_expect(సమాసం, అబద్ధం)
#లేకపోతే
#    define LIKELY(సమాసం)   (సమాసం)
#    define UNLIKELY(సమాసం) (సమాసం)
#అయితే_ముగింపు

#నిర్వచించు GGML_SYCL_DEBUG(...)              \
    చేయి {                                  \
        ఒకవేళ (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } అయితే (0)

#నిర్వచించు GGML_SYCL_DEV_DEBUG(...)          \
    చేయి {                                  \
        ఒకవేళ (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } అయితే (0)

#అయితే_ముగింపు  // GGML_SYCL_BASE_HPP
