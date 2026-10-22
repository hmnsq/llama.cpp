#ہدایت once

#شامل <stddef.h>
#شامل <stdint.h>

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

// Amount of wdata to reserve for tiled workspaces
size_t ggml_tiled_wdata_سائز(صحیح n_tasks, ساخت ggml_tensor * dst);

// tiled K-quant matmul; returns true if the op was computed here,
// false to fall through to the stock path
bool ggml_compute_forward_mul_mat_tiled(ثابت ساخت ggml_compute_params * params,
                                        ساخت ggml_tensor * dst);

// MUL_MAT_ID (MoE) path, one expert; returns true if the expert was computed here, per expert
// eligibility (type gate, batch floor) is decided inside. expert_rows points at the expert's
// row of the matrix_rows table of (expert slot, batch row) int32 pairs; scratch is the
// per-thread tiled_ws region reserved in wdata, n_tasks of ggml_tiled_ws_size() bytes
bool ggml_compute_forward_mul_mat_id_tiled(ثابت ساخت ggml_compute_params * params,
                                           ساخت ggml_tensor *               dst,
                                           int64_t                            cur_a,
                                           int64_t                            cne1,
                                           ثابت int32_t *                    expert_صفیں,
                                           حرف *                             scratch);

#اگر_تعریف_شدہ __cplusplus
}
#اگر_ختم
