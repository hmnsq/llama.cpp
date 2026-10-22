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

#অন্তর্ভাৱ "ggml_tensor.h"
#অন্তর্ভাৱ "platform.h"

#অন্তর্ভাৱ <stdint.h>
#অন্তর্ভাৱ <string.h>

সংরচনা ggml_et_concat_params {
    সংরচনা ggml_tensor src0;  // F32 input tensor 0
    সংরচনা ggml_tensor src1;  // F32 input tensor 1
    সংরচনা ggml_tensor dst;   // F32 output tensor
    int32_t            dim;   // Concatenation dimension
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
স্থির অন্তঃস্থ শূন্য copy_row_aligned(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [src_vec] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i])
            : "f11");
    }
}

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_concat_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ঘূৰাই -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ঘূৰাই 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ঘূৰাই -1;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * src1 = &params->src1;
    সংরচনা ggml_tensor * dst  = &params->dst;
    int32_t              dim  = params->dim;

    যদি (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ঘূৰাই -1;
    }

    ভিন্ন * src0_data = (ভিন্ন *) src0->data;
    ভিন্ন * src1_data = (ভিন্ন *) src1->data;
    ভিন্ন * dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !src1_data || !dst_data) {
        ঘূৰাই -1;
    }

    ধ্রুৱ int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ধ্রুৱ int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    ধ্রুৱ int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src strides in bytes
    ধ্রুৱ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ধ্রুৱ size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    // dst strides in bytes
    ধ্রুৱ size_t dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Total rows across all higher dimensions
    ধ্রুৱ int64_t total_rows = ne1 * ne2 * ne3;

    // Generic slow path for dim==0 when either source segment is not suitable for
    // aligned vector copies. Threading is done by cacheline-aligned row groups,
    // so writers do not share destination cache lines.
    যদি (dim == 0 && (ne00 % 16 != 0 || ne10 % 16 != 0 || nb00 != পরিমাণ(ভিন্ন) || nb10 != পরিমাণ(ভিন্ন))) {
        ধ্রুৱ int64_t rows_per_group = et_rows_per_cacheline_group(ne0, পরিমাণ(ভিন্ন));
        ধ্রুৱ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

        বাবে (int64_t grp = thread_id; grp < total_groups; grp += অঙ্ক_threads) {
            ধ্রুৱ int64_t row_start = grp * rows_per_group;
            int64_t       row_end   = row_start + rows_per_group;
            যদি (row_end > total_rows) {
                row_end = total_rows;
            }

            বাবে (int64_t row = row_start; row < row_end; row++) {
                int64_t i1 = row % ne1;
                int64_t i2 = (row / ne1) % ne2;
                int64_t i3 = row / (ne1 * ne2);

                ভিন্ন * dst_row = (ভিন্ন *) ((ৱর্ণ *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

                ধ্রুৱ ৱর্ণ * s0_base = (ধ্রুৱ ৱর্ণ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03;
                বাবে (int64_t i0 = 0; i0 < ne00; i0++) {
                    dst_row[i0] = *(ধ্রুৱ ভিন্ন *) (s0_base + i0 * nb00);
                }

                ধ্রুৱ ৱর্ণ * s1_base = (ধ্রুৱ ৱর্ণ *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13;
                বাবে (int64_t i0 = 0; i0 < ne10; i0++) {
                    dst_row[ne00 + i0] = *(ধ্রুৱ ভিন্ন *) (s1_base + i0 * nb10);
                }
            }
        }
        ঘূৰাই 0;
    }

    // Standard path: ne0 % 16 == 0, aligned rows
    বাবে (int64_t row = thread_id; row < total_rows; row += অঙ্ক_threads) {
        // Decompose linear row index into (i1, i2, i3)
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        ভিন্ন * dst_row = (ভিন্ন *) ((ৱর্ণ *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        যদি (dim == 0) {
            // Concat along innermost dimension: [src0_row | src1_row]
            // Both ne00 and ne10 are multiples of 16 (cacheline-aligned)
            ধ্রুৱ ভিন্ন * s0_row = (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            ধ্রুৱ ভিন্ন * s1_row = (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13);

            copy_row_aligned(dst_row, s0_row, (int32_t) ne00);
            copy_row_aligned(dst_row + ne00, s1_row, (int32_t) ne10);

        } অন্যথা যদি (dim == 1) {
            // Concat along dim 1: first ne01 rows from src0, rest from src1
            যদি (i1 < ne01) {
                ধ্রুৱ ভিন্ন * s0_row = (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t) ne0);
            } অন্যথা {
                ধ্রুৱ ভিন্ন * s1_row =
                    (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src1_data + (i1 - ne01) * nb11 + i2 * nb12 + i3 * nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t) ne0);
            }

        } অন্যথা যদি (dim == 2) {
            // Concat along dim 2: first ne02 slices from src0, rest from src1
            যদি (i2 < ne02) {
                ধ্রুৱ ভিন্ন * s0_row = (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t) ne0);
            } অন্যথা {
                ধ্রুৱ ভিন্ন * s1_row =
                    (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src1_data + i1 * nb11 + (i2 - ne02) * nb12 + i3 * nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t) ne0);
            }

        } অন্যথা {
            // dim == 3: first ne03 batches from src0, rest from src1
            যদি (i3 < ne03) {
                ধ্রুৱ ভিন্ন * s0_row = (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t) ne0);
            } অন্যথা {
                ধ্রুৱ ভিন্ন * s1_row =
                    (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src1_data + i1 * nb11 + i2 * nb12 + (i3 - ne03) * nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t) ne0);
            }
        }
    }

    ঘূৰাই 0;
}
