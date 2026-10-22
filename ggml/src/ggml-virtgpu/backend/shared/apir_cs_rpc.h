#pragma once

// clang-format off
#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "ggml-backend-impl.h"

#অন্তর্ভাব <unordered_map>
#অন্তর্ভাব <unordered_set>
#অন্তর্ভাব <vector>
#অন্তর্ভাব <cstdint>
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
কাঠামো apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t বাফার;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / আকার(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t তথ্য;
    অক্ষর     name[GGML_MAX_NAME];

    অক্ষর padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(ধ্রুবক ggml_tensor * tensor);

শূন্য apir_serialize_graph(ধ্রুবক ggml_cgraph * cgraph, std::vector<uint8_t> & output);

/* backend */

শূন্য                                      apir_track_backend_বাফার(ggml_backend_বাফার_t বাফার);
bool                                      apir_untrack_backend_বাফার(ggml_backend_বাফার_t বাফার);
std::unordered_set<ggml_backend_বাফার_t> apir_get_track_backend_buffers();

শূন্য apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ধ্রুবক apir_rpc_tensor * tensor);

ggml_tensor * apir_create_গ্রন্থি(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ধ্রুবক std::unordered_map<uint64_t, ধ্রুবক apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ধ্রুবক apir_rpc_tensor * tensors,
                                     ধ্রুবক uint64_t *        nodes);
