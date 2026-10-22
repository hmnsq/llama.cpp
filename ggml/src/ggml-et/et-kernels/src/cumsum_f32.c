//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_cumsum_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor dst;
};

પૂર્ણાંક entry_point(સંરચના ggml_et_cumsum_params * params, શૂન્ય * env) {
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
    સંરચના ggml_tensor * dst  = &params->dst;

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    દશાંશ * src0_data = (દશાંશ *) src0->data;
    દશાંશ * dst_data  = (દશાંશ *) dst->data;

    જો (!src0_data || !dst_data) {
        પાછા -1;
    }

    અચલ int64_t ne0 = src0->ne[0];
    અચલ int64_t ne1 = src0->ne[1];
    અચલ int64_t ne2 = src0->ne[2];
    અચલ int64_t ne3 = src0->ne[3];

    અચલ size_t snb0 = src0->nb[0];
    અચલ size_t snb1 = src0->nb[1];
    અચલ size_t snb2 = src0->nb[2];
    અચલ size_t snb3 = src0->nb[3];

    અચલ size_t dnb0 = dst->nb[0];
    અચલ size_t dnb1 = dst->nb[1];
    અચલ size_t dnb2 = dst->nb[2];
    અચલ size_t dnb3 = dst->nb[3];

    જો (snb0 != કદ(દશાંશ) || dnb0 != કદ(દશાંશ)) {
        પાછા -1;
    }

    અચલ int64_t total_rows     = ne1 * ne2 * ne3;
    અચલ int64_t rows_per_group = et_rows_per_cacheline_group(ne0, કદ(દશાંશ));
    અચલ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    માટે (int64_t grp = thread_id; grp < total_groups; grp += અંક_threads) {
        અચલ int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        જો (row_end > total_rows) {
            row_end = total_rows;
        }

        માટે (int64_t row = row_start; row < row_end; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            અચલ દશાંશ * src_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i1 * snb1 + i2 * snb2 + i3 * snb3);
            દશાંશ *       dst_row = (દશાંશ *) ((અક્ષર *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            દશાંશ acc = 0.0f;
            માટે (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    પાછા 0;
}
