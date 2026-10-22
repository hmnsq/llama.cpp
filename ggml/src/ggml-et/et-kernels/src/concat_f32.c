//******************************************************************************
// Concat F32 Kernel
// Concatenates two F32 tensors along a specified dimension.
// All copies are aligned to cacheline boundaries (64 bytes = 16 floats).
//
// For dim >= 1, entire rows are copied from src0 or src1 into dst.
// For dim == 0, use:
// - a fast vector path when both source row segments are cacheline-aligned
// - a scalar stride-aware path otherwise
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>

ಸಂರಚನಾ ggml_et_concat_params {
    ಸಂರಚನಾ ggml_tensor src0;  // F32 input tensor 0
    ಸಂರಚನಾ ggml_tensor src1;  // F32 input tensor 1
    ಸಂರಚನಾ ggml_tensor dst;   // F32 output tensor
    int32_t            dim;   // Concatenation dimension
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ copy_row_aligned(ಭಿನ್ನ * dst, ಧ್ರುವ ಭಿನ್ನ * src, int32_t n) {
    ಪ್ರತಿ (int32_t i = 0; i < n; i += 8) {
        __asm__ ಚಞ್ಚಲ(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(ಭಿನ್ನ (*)[8]) & dst[i])
            : [src_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src[i])
            : "f11");
    }
}

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_concat_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * src1 = &params->src1;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;
    int32_t              dim  = params->dim;

    ಯದಿ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಭಿನ್ನ * src0_data = (ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ * src1_data = (ಭಿನ್ನ *) src1->data;
    ಭಿನ್ನ * dst_data  = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!src0_data || !src1_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ಧ್ರುವ int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    ಧ್ರುವ int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src strides in bytes
    ಧ್ರುವ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ಧ್ರುವ size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    // dst strides in bytes
    ಧ್ರುವ size_t dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Total rows across all higher dimensions
    ಧ್ರುವ int64_t total_rows = ne1 * ne2 * ne3;

    // Generic slow path for dim==0 when either source segment is not suitable for
    // aligned vector copies. Threading is done by cacheline-aligned row groups,
    // so writers do not share destination cache lines.
    ಯದಿ (dim == 0 && (ne00 % 16 != 0 || ne10 % 16 != 0 || nb00 != ಪರಿಮಾಣ(ಭಿನ್ನ) || nb10 != ಪರಿಮಾಣ(ಭಿನ್ನ))) {
        ಧ್ರುವ int64_t rows_per_group = et_rows_per_cacheline_group(ne0, ಪರಿಮಾಣ(ಭಿನ್ನ));
        ಧ್ರುವ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

        ಪ್ರತಿ (int64_t grp = thread_id; grp < total_groups; grp += ಅಂಕೆ_threads) {
            ಧ್ರುವ int64_t row_start = grp * rows_per_group;
            int64_t       row_end   = row_start + rows_per_group;
            ಯದಿ (row_end > total_rows) {
                row_end = total_rows;
            }

            ಪ್ರತಿ (int64_t row = row_start; row < row_end; row++) {
                int64_t i1 = row % ne1;
                int64_t i2 = (row / ne1) % ne2;
                int64_t i3 = row / (ne1 * ne2);

                ಭಿನ್ನ * dst_row = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

                ಧ್ರುವ ವರ್ಣ * s0_base = (ಧ್ರುವ ವರ್ಣ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03;
                ಪ್ರತಿ (int64_t i0 = 0; i0 < ne00; i0++) {
                    dst_row[i0] = *(ಧ್ರುವ ಭಿನ್ನ *) (s0_base + i0 * nb00);
                }

                ಧ್ರುವ ವರ್ಣ * s1_base = (ಧ್ರುವ ವರ್ಣ *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13;
                ಪ್ರತಿ (int64_t i0 = 0; i0 < ne10; i0++) {
                    dst_row[ne00 + i0] = *(ಧ್ರುವ ಭಿನ್ನ *) (s1_base + i0 * nb10);
                }
            }
        }
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    // Standard path: ne0 % 16 == 0, aligned rows
    ಪ್ರತಿ (int64_t row = thread_id; row < total_rows; row += ಅಂಕೆ_threads) {
        // Decompose linear row index into (i1, i2, i3)
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        ಭಿನ್ನ * dst_row = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        ಯದಿ (dim == 0) {
            // Concat along innermost dimension: [src0_row | src1_row]
            // Both ne00 and ne10 are multiples of 16 (cacheline-aligned)
            ಧ್ರುವ ಭಿನ್ನ * s0_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            ಧ್ರುವ ಭಿನ್ನ * s1_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13);

            copy_row_aligned(dst_row, s0_row, (int32_t) ne00);
            copy_row_aligned(dst_row + ne00, s1_row, (int32_t) ne10);

        } ಅನ್ಯಥಾ ಯದಿ (dim == 1) {
            // Concat along dim 1: first ne01 rows from src0, rest from src1
            ಯದಿ (i1 < ne01) {
                ಧ್ರುವ ಭಿನ್ನ * s0_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t) ne0);
            } ಅನ್ಯಥಾ {
                ಧ್ರುವ ಭಿನ್ನ * s1_row =
                    (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + (i1 - ne01) * nb11 + i2 * nb12 + i3 * nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t) ne0);
            }

        } ಅನ್ಯಥಾ ಯದಿ (dim == 2) {
            // Concat along dim 2: first ne02 slices from src0, rest from src1
            ಯದಿ (i2 < ne02) {
                ಧ್ರುವ ಭಿನ್ನ * s0_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t) ne0);
            } ಅನ್ಯಥಾ {
                ಧ್ರುವ ಭಿನ್ನ * s1_row =
                    (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + i1 * nb11 + (i2 - ne02) * nb12 + i3 * nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t) ne0);
            }

        } ಅನ್ಯಥಾ {
            // dim == 3: first ne03 batches from src0, rest from src1
            ಯದಿ (i3 < ne03) {
                ಧ್ರುವ ಭಿನ್ನ * s0_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t) ne0);
            } ಅನ್ಯಥಾ {
                ಧ್ರುವ ಭಿನ್ನ * s1_row =
                    (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + i1 * nb11 + i2 * nb12 + (i3 - ne03) * nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t) ne0);
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
