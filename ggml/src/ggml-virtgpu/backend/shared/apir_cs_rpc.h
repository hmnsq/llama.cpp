#pragma once

// clang-format off
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <unordered_set>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <cstdint>
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
ಸಂರಚನಾ apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / ಪರಿಮಾಣ(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    ವರ್ಣ     name[GGML_MAX_NAME];

    ವರ್ಣ padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(ಧ್ರುವ ggml_tensor * tensor);

ಶೂನ್ಯ apir_serialize_graph(ಧ್ರುವ ggml_cgraph * cgraph, std::vector<uint8_t> & output);

/* backend */

ಶೂನ್ಯ                                      apir_track_backend_buffer(ggml_backend_buffer_t buffer);
bool                                      apir_untrack_backend_buffer(ggml_backend_buffer_t buffer);
std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers();

ಶೂನ್ಯ apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ಧ್ರುವ apir_rpc_tensor * tensor);

ggml_tensor * apir_create_ಗಂಟು(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ಧ್ರುವ std::unordered_map<uint64_t, ಧ್ರುವ apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ಧ್ರುವ apir_rpc_tensor * tensors,
                                     ಧ್ರುವ uint64_t *        nodes);
