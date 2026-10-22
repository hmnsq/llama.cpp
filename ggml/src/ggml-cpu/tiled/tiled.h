#pragma once

#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdint.h>

#ifdef __cplusplus
বাহ্যিক "C" {
#endif

// Amount of wdata to reserve for tiled workspaces
size_t ggml_tiled_wdata_আকার(পূর্ণসংখ্যা n_tasks, কাঠামো ggml_tensor * dst);

// tiled K-quant matmul; returns true if the op was computed here,
// false to fall through to the stock path
bool ggml_compute_forward_mul_mat_tiled(ধ্রুবক কাঠামো ggml_compute_params * params,
                                        কাঠামো ggml_tensor * dst);

// MUL_MAT_ID (MoE) path, one expert; returns true if the expert was computed here, per expert
// eligibility (type gate, batch floor) is decided inside. expert_rows points at the expert's
// row of the matrix_rows table of (expert slot, batch row) int32 pairs; scratch is the
// per-thread tiled_ws region reserved in wdata, n_tasks of ggml_tiled_ws_size() bytes
bool ggml_compute_forward_mul_mat_id_tiled(ধ্রুবক কাঠামো ggml_compute_params * params,
                                           কাঠামো ggml_tensor *               dst,
                                           int64_t                            cur_a,
                                           int64_t                            cne1,
                                           ধ্রুবক int32_t *                    expert_rows,
                                           অক্ষর *                             scratch);

#ifdef __cplusplus
}
#endif
