#pragma once

// clang-format off
#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "ggml-backend-impl.h"

#অন্তর্ভাৱ <unordered_map>
#অন্তর্ভাৱ <unordered_set>
#অন্তর্ভাৱ <vector>
#অন্তর্ভাৱ <cstdint>
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
সংরচনা apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / পরিমাণ(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    ৱর্ণ     name[GGML_MAX_NAME];

    ৱর্ণ padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(ধ্রুৱ ggml_tensor * tensor);

শূন্য apir_serialize_graph(ধ্রুৱ ggml_cgraph * cgraph, std::vector<uint8_t> & output);

/* backend */

শূন্য                                      apir_track_backend_buffer(ggml_backend_buffer_t buffer);
bool                                      apir_untrack_backend_buffer(ggml_backend_buffer_t buffer);
std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers();

শূন্য apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ধ্রুৱ apir_rpc_tensor * tensor);

ggml_tensor * apir_create_গ্ৰন্থি(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ধ্রুৱ std::unordered_map<uint64_t, ধ্রুৱ apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ধ্রুৱ apir_rpc_tensor * tensors,
                                     ধ্রুৱ uint64_t *        nodes);
