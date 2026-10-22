#pragma once

// clang-format off
#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "ggml-backend-impl.h"

#અન્તર્ભાવ <unordered_map>
#અન્તર્ભાવ <unordered_set>
#અન્તર્ભાવ <vector>
#અન્તર્ભાવ <cstdint>
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
સંરચના apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / કદ(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    અક્ષર     name[GGML_MAX_NAME];

    અક્ષર padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(અચલ ggml_tensor * tensor);

શૂન્ય apir_serialize_graph(અચલ ggml_cgraph * cgraph, std::vector<uint8_t> & output);

/* backend */

શૂન્ય                                      apir_track_backend_buffer(ggml_backend_buffer_t buffer);
bool                                      apir_untrack_backend_buffer(ggml_backend_buffer_t buffer);
std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers();

શૂન્ય apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, અચલ apir_rpc_tensor * tensor);

ggml_tensor * apir_create_ગાંઠ(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               અચલ std::unordered_map<uint64_t, અચલ apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     અચલ apir_rpc_tensor * tensors,
                                     અચલ uint64_t *        nodes);
