#pragma once

// clang-format off
#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "ggml-backend-impl.h"

#அந்தர்பாவ <unordered_map>
#அந்தர்பாவ <unordered_set>
#அந்தர்பாவ <vector>
#அந்தர்பாவ <cstdint>
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
கட்டமைப்பு apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / அளவு(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    எழுத்து     name[GGML_MAX_NAME];

    எழுத்து padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(மாறிலி ggml_tensor * tensor);

வெறுமை apir_serialize_graph(மாறிலி ggml_cgraph * cgraph, std::vector<uint8_t> & output);

/* backend */

வெறுமை                                      apir_track_backend_buffer(ggml_backend_buffer_t buffer);
bool                                      apir_untrack_backend_buffer(ggml_backend_buffer_t buffer);
std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers();

வெறுமை apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, மாறிலி apir_rpc_tensor * tensor);

ggml_tensor * apir_create_முனை(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               மாறிலி std::unordered_map<uint64_t, மாறிலி apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     மாறிலி apir_rpc_tensor * tensors,
                                     மாறிலி uint64_t *        nodes);
