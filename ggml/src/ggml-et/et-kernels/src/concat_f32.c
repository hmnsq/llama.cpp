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

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

ସଂରଚନା ggml_et_concat_params {
    ସଂରଚନା ggml_tensor src0;  // F32 input tensor 0
    ସଂରଚନା ggml_tensor src1;  // F32 input tensor 1
    ସଂରଚନା ggml_tensor dst;   // F32 output tensor
    int32_t            dim;   // Concatenation dimension
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ copy_row_aligned(ଦଶମିକ * dst, ଅଚଳ ଦଶମିକ * src, int32_t n) {
    ପାଇଁ (int32_t i = 0; i < n; i += 8) {
        __asm__ ଅସ୍ଥିର(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(ଦଶମିକ (*)[8]) & dst[i])
            : [src_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & src[i])
            : "f11");
    }
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_concat_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ 0;
    }

    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ସଂରଚନା ggml_tensor * src0 = &params->src0;
    ସଂରଚନା ggml_tensor * src1 = &params->src1;
    ସଂରଚନା ggml_tensor * dst  = &params->dst;
    int32_t              dim  = params->dim;

    ଯଦି (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଦଶମିକ * src0_data = (ଦଶମିକ *) src0->data;
    ଦଶମିକ * src1_data = (ଦଶମିକ *) src1->data;
    ଦଶମିକ * dst_data  = (ଦଶମିକ *) dst->data;

    ଯଦି (!src0_data || !src1_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ଅଚଳ int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    ଅଚଳ int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src strides in bytes
    ଅଚଳ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ଅଚଳ size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    // dst strides in bytes
    ଅଚଳ size_t dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Total rows across all higher dimensions
    ଅଚଳ int64_t total_rows = ne1 * ne2 * ne3;

    // Generic slow path for dim==0 when either source segment is not suitable for
    // aligned vector copies. Threading is done by cacheline-aligned row groups,
    // so writers do not share destination cache lines.
    ଯଦି (dim == 0 && (ne00 % 16 != 0 || ne10 % 16 != 0 || nb00 != ଆକାର(ଦଶମିକ) || nb10 != ଆକାର(ଦଶମିକ))) {
        ଅଚଳ int64_t rows_per_group = et_rows_per_cacheline_group(ne0, ଆକାର(ଦଶମିକ));
        ଅଚଳ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

        ପାଇଁ (int64_t grp = thread_id; grp < total_groups; grp += ଅଙ୍କ_threads) {
            ଅଚଳ int64_t row_start = grp * rows_per_group;
            int64_t       row_end   = row_start + rows_per_group;
            ଯଦି (row_end > total_rows) {
                row_end = total_rows;
            }

            ପାଇଁ (int64_t row = row_start; row < row_end; row++) {
                int64_t i1 = row % ne1;
                int64_t i2 = (row / ne1) % ne2;
                int64_t i3 = row / (ne1 * ne2);

                ଦଶମିକ * dst_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

                ଅଚଳ ଅକ୍ଷର * s0_base = (ଅଚଳ ଅକ୍ଷର *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03;
                ପାଇଁ (int64_t i0 = 0; i0 < ne00; i0++) {
                    dst_row[i0] = *(ଅଚଳ ଦଶମିକ *) (s0_base + i0 * nb00);
                }

                ଅଚଳ ଅକ୍ଷର * s1_base = (ଅଚଳ ଅକ୍ଷର *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13;
                ପାଇଁ (int64_t i0 = 0; i0 < ne10; i0++) {
                    dst_row[ne00 + i0] = *(ଅଚଳ ଦଶମିକ *) (s1_base + i0 * nb10);
                }
            }
        }
        ଫେରନ୍ତୁ 0;
    }

    // Standard path: ne0 % 16 == 0, aligned rows
    ପାଇଁ (int64_t row = thread_id; row < total_rows; row += ଅଙ୍କ_threads) {
        // Decompose linear row index into (i1, i2, i3)
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        ଦଶମିକ * dst_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        ଯଦି (dim == 0) {
            // Concat along innermost dimension: [src0_row | src1_row]
            // Both ne00 and ne10 are multiples of 16 (cacheline-aligned)
            ଅଚଳ ଦଶମିକ * s0_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            ଅଚଳ ଦଶମିକ * s1_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13);

            copy_row_aligned(dst_row, s0_row, (int32_t) ne00);
            copy_row_aligned(dst_row + ne00, s1_row, (int32_t) ne10);

        } ଅନ୍ୟଥା ଯଦି (dim == 1) {
            // Concat along dim 1: first ne01 rows from src0, rest from src1
            ଯଦି (i1 < ne01) {
                ଅଚଳ ଦଶମିକ * s0_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t) ne0);
            } ଅନ୍ୟଥା {
                ଅଚଳ ଦଶମିକ * s1_row =
                    (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + (i1 - ne01) * nb11 + i2 * nb12 + i3 * nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t) ne0);
            }

        } ଅନ୍ୟଥା ଯଦି (dim == 2) {
            // Concat along dim 2: first ne02 slices from src0, rest from src1
            ଯଦି (i2 < ne02) {
                ଅଚଳ ଦଶମିକ * s0_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t) ne0);
            } ଅନ୍ୟଥା {
                ଅଚଳ ଦଶମିକ * s1_row =
                    (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + i1 * nb11 + (i2 - ne02) * nb12 + i3 * nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t) ne0);
            }

        } ଅନ୍ୟଥା {
            // dim == 3: first ne03 batches from src0, rest from src1
            ଯଦି (i3 < ne03) {
                ଅଚଳ ଦଶମିକ * s0_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t) ne0);
            } ଅନ୍ୟଥା {
                ଅଚଳ ଦଶମିକ * s1_row =
                    (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + i1 * nb11 + i2 * nb12 + (i3 - ne03) * nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t) ne0);
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}
