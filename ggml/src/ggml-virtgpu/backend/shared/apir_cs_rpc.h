#ಸೂಚನೆ once

// clang-format off
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <unordered_set>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <cstdint>
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
ರಚನೆ apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t ಬಫರ್;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / ಗಾತ್ರ(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t ದತ್ತಾಂಶ;
    ಅಕ್ಷರ     name[GGML_MAX_NAME];

    ಅಕ್ಷರ padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);

ಶೂನ್ಯ apir_serialize_ನಕ್ಷೆ(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, std::ಸದಿಶ<uint8_t> & output);

/* backend */

ಶೂನ್ಯ                                      apir_track_backend_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್);
bool                                      apir_untrack_backend_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್);
std::unordered_ಗಣ<ggml_backend_ಬಫರ್_t> apir_get_track_backend_buffers();

ಶೂನ್ಯ apir_ಕೂಡಿಸು_tensor(ggml_tensor *                       tensor,
                     std::ಸದಿಶ<apir_rpc_tensor> &      tensors,
                     std::unordered_ಗಣ<ggml_tensor *> & ಭೇಟಿಯಾದ);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ಸ್ಥಿರಾಂಕ apir_rpc_tensor * tensor);

ggml_tensor * apir_create_ಗಂಟು(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ಸ್ಥಿರಾಂಕ std::unordered_map<uint64_t, ಸ್ಥಿರಾಂಕ apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_ನಕ್ಷೆ(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ಸ್ಥಿರಾಂಕ apir_rpc_tensor * tensors,
                                     ಸ್ಥಿರಾಂಕ uint64_t *        nodes);
