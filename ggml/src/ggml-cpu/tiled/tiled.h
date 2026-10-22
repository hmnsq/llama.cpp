#pragma once

#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ifdef __cplusplus
ಬಾಹ್ಯ "C" {
#endif

// Amount of wdata to reserve for tiled workspaces
size_t ggml_tiled_wdata_size(ಪೂರ್ಣಾಙ್ಕ n_tasks, ಸಂರಚನಾ ggml_tensor * dst);

// tiled K-quant matmul; returns true if the op was computed here,
// false to fall through to the stock path
bool ggml_compute_forward_mul_mat_tiled(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params,
                                        ಸಂರಚನಾ ggml_tensor * dst);

// MUL_MAT_ID (MoE) path, one expert; returns true if the expert was computed here, per expert
// eligibility (type gate, batch floor) is decided inside. expert_rows points at the expert's
// row of the matrix_rows table of (expert slot, batch row) int32 pairs; scratch is the
// per-thread tiled_ws region reserved in wdata, n_tasks of ggml_tiled_ws_size() bytes
bool ggml_compute_forward_mul_mat_id_tiled(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params,
                                           ಸಂರಚನಾ ggml_tensor *               dst,
                                           int64_t                            cur_a,
                                           int64_t                            cne1,
                                           ಧ್ರುವ int32_t *                    expert_rows,
                                           ವರ್ಣ *                             scratch);

#ifdef __cplusplus
}
#endif
