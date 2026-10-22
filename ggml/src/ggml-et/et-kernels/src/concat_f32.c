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

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <string.h>

સંરચના ggml_et_concat_params {
    સંરચના ggml_tensor src0;  // F32 input tensor 0
    સંરચના ggml_tensor src1;  // F32 input tensor 1
    સંરચના ggml_tensor dst;   // F32 output tensor
    int32_t            dim;   // Concatenation dimension
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
સ્થિર અંતઃરેખ શૂન્ય copy_row_aligned(દશાંશ * dst, અચલ દશાંશ * src, int32_t n) {
    માટે (int32_t i = 0; i < n; i += 8) {
        __asm__ અસ્થિર(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(દશાંશ (*)[8]) & dst[i])
            : [src_vec] "m"(*(અચલ દશાંશ (*)[8]) & src[i])
            : "f11");
    }
}

પૂર્ણાંક entry_point(સંરચના ggml_et_concat_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;
    }

    સંરચના ggml_tensor * src0 = &params->src0;
    સંરચના ggml_tensor * src1 = &params->src1;
    સંરચના ggml_tensor * dst  = &params->dst;
    int32_t              dim  = params->dim;

    જો (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    દશાંશ * src0_data = (દશાંશ *) src0->data;
    દશાંશ * src1_data = (દશાંશ *) src1->data;
    દશાંશ * dst_data  = (દશાંશ *) dst->data;

    જો (!src0_data || !src1_data || !dst_data) {
        પાછા -1;
    }

    અચલ int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    અચલ int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    અચલ int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src strides in bytes
    અચલ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    અચલ size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    // dst strides in bytes
    અચલ size_t dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Total rows across all higher dimensions
    અચલ int64_t total_rows = ne1 * ne2 * ne3;

    // Generic slow path for dim==0 when either source segment is not suitable for
    // aligned vector copies. Threading is done by cacheline-aligned row groups,
    // so writers do not share destination cache lines.
    જો (dim == 0 && (ne00 % 16 != 0 || ne10 % 16 != 0 || nb00 != કદ(દશાંશ) || nb10 != કદ(દશાંશ))) {
        અચલ int64_t rows_per_group = et_rows_per_cacheline_group(ne0, કદ(દશાંશ));
        અચલ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

        માટે (int64_t grp = thread_id; grp < total_groups; grp += અંક_threads) {
            અચલ int64_t row_start = grp * rows_per_group;
            int64_t       row_end   = row_start + rows_per_group;
            જો (row_end > total_rows) {
                row_end = total_rows;
            }

            માટે (int64_t row = row_start; row < row_end; row++) {
                int64_t i1 = row % ne1;
                int64_t i2 = (row / ne1) % ne2;
                int64_t i3 = row / (ne1 * ne2);

                દશાંશ * dst_row = (દશાંશ *) ((અક્ષર *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

                અચલ અક્ષર * s0_base = (અચલ અક્ષર *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03;
                માટે (int64_t i0 = 0; i0 < ne00; i0++) {
                    dst_row[i0] = *(અચલ દશાંશ *) (s0_base + i0 * nb00);
                }

                અચલ અક્ષર * s1_base = (અચલ અક્ષર *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13;
                માટે (int64_t i0 = 0; i0 < ne10; i0++) {
                    dst_row[ne00 + i0] = *(અચલ દશાંશ *) (s1_base + i0 * nb10);
                }
            }
        }
        પાછા 0;
    }

    // Standard path: ne0 % 16 == 0, aligned rows
    માટે (int64_t row = thread_id; row < total_rows; row += અંક_threads) {
        // Decompose linear row index into (i1, i2, i3)
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        દશાંશ * dst_row = (દશાંશ *) ((અક્ષર *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        જો (dim == 0) {
            // Concat along innermost dimension: [src0_row | src1_row]
            // Both ne00 and ne10 are multiples of 16 (cacheline-aligned)
            અચલ દશાંશ * s0_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            અચલ દશાંશ * s1_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13);

            copy_row_aligned(dst_row, s0_row, (int32_t) ne00);
            copy_row_aligned(dst_row + ne00, s1_row, (int32_t) ne10);

        } નહીંતર જો (dim == 1) {
            // Concat along dim 1: first ne01 rows from src0, rest from src1
            જો (i1 < ne01) {
                અચલ દશાંશ * s0_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t) ne0);
            } નહીંતર {
                અચલ દશાંશ * s1_row =
                    (અચલ દશાંશ *) ((અચલ અક્ષર *) src1_data + (i1 - ne01) * nb11 + i2 * nb12 + i3 * nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t) ne0);
            }

        } નહીંતર જો (dim == 2) {
            // Concat along dim 2: first ne02 slices from src0, rest from src1
            જો (i2 < ne02) {
                અચલ દશાંશ * s0_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t) ne0);
            } નહીંતર {
                અચલ દશાંશ * s1_row =
                    (અચલ દશાંશ *) ((અચલ અક્ષર *) src1_data + i1 * nb11 + (i2 - ne02) * nb12 + i3 * nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t) ne0);
            }

        } નહીંતર {
            // dim == 3: first ne03 batches from src0, rest from src1
            જો (i3 < ne03) {
                અચલ દશાંશ * s0_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_row_aligned(dst_row, s0_row, (int32_t) ne0);
            } નહીંતર {
                અચલ દશાંશ * s1_row =
                    (અચલ દશાંશ *) ((અચલ અક્ષર *) src1_data + i1 * nb11 + i2 * nb12 + (i3 - ne03) * nb13);
                copy_row_aligned(dst_row, s1_row, (int32_t) ne0);
            }
        }
    }

    પાછા 0;
}
