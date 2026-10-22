#pragma once

// clang-format off
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"

#ଅନ୍ତର୍ଭାଵ <unordered_map>
#ଅନ୍ତର୍ଭାଵ <unordered_set>
#ଅନ୍ତର୍ଭାଵ <vector>
#ଅନ୍ତର୍ଭାଵ <cstdint>
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
ସଂରଚନା apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / ଆକାର(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    ଅକ୍ଷର     name[GGML_MAX_NAME];

    ଅକ୍ଷର padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(ଅଚଳ ggml_tensor * tensor);

ଶୂନ୍ୟ apir_serialize_graph(ଅଚଳ ggml_cgraph * cgraph, std::vector<uint8_t> & output);

/* backend */

ଶୂନ୍ୟ                                      apir_track_backend_buffer(ggml_backend_buffer_t buffer);
bool                                      apir_untrack_backend_buffer(ggml_backend_buffer_t buffer);
std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers();

ଶୂନ୍ୟ apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ଅଚଳ apir_rpc_tensor * tensor);

ggml_tensor * apir_create_ଗଣ୍ଠି(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ଅଚଳ std::unordered_map<uint64_t, ଅଚଳ apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ଅଚଳ apir_rpc_tensor * tensors,
                                     ଅଚଳ uint64_t *        nodes);
