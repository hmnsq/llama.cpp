#आशय once

#समावेश <stddef.h>
#समावेश <stdint.h>

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

// Amount of wdata to reserve for tiled workspaces
माप_प्रकार ggml_tiled_wdata_आकार(पूर्णांक n_tasks, काष्ठा ggml_tensor * dst);

// tiled K-quant matmul; returns true if the op was computed here,
// false to fall through to the stock path
bool ggml_compute_forward_mul_mat_tiled(स्थिर काष्ठा ggml_compute_params * params,
                                        काष्ठा ggml_tensor * dst);

// MUL_MAT_ID (MoE) path, one expert; returns true if the expert was computed here, per expert
// eligibility (type gate, batch floor) is decided inside. expert_rows points at the expert's
// row of the matrix_rows table of (expert slot, batch row) int32 pairs; scratch is the
// per-thread tiled_ws region reserved in wdata, n_tasks of ggml_tiled_ws_size() bytes
bool ggml_compute_forward_mul_mat_id_tiled(स्थिर काष्ठा ggml_compute_params * params,
                                           काष्ठा ggml_tensor *               dst,
                                           int64_t                            cur_a,
                                           int64_t                            cne1,
                                           स्थिर int32_t *                    expert_rows,
                                           अक्षर *                             scratch);

#अगर_घोषित __cplusplus
}
#पूर्ण_अगर
