#pragma once

// clang-format off
#ضمن "ggml.h"
#ضمن "ggml-backend-impl.h"

#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <vector>
#ضمن <cstdint>
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
بنية apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / حجم(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    حرف     name[GGML_MAX_NAME];

    حرف padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(ثابت ggml_tensor * tensor);

فراغ apir_serialize_graph(ثابت ggml_cgraph * cgraph, std::vector<uint8_t> & output);

/* backend */

فراغ                                      apir_track_backend_buffer(ggml_backend_buffer_t buffer);
bool                                      apir_untrack_backend_buffer(ggml_backend_buffer_t buffer);
std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers();

فراغ apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ثابت apir_rpc_tensor * tensor);

ggml_tensor * apir_create_تۈگۈن(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ثابت std::unordered_map<uint64_t, ثابت apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ثابت apir_rpc_tensor * tensors,
                                     ثابت uint64_t *        nodes);
