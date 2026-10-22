#pragma once

// clang-format off
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"

#അന്തര്ഭാവ <unordered_map>
#അന്തര്ഭാവ <unordered_set>
#അന്തര്ഭാവ <vector>
#അന്തര്ഭാവ <cstdint>
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
ഘടന apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / വലുപ്പം(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    അക്ഷരം     name[GGML_MAX_NAME];

    അക്ഷരം padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(സ്ഥിരം ggml_tensor * tensor);

ശൂന്യം apir_serialize_graph(സ്ഥിരം ggml_cgraph * cgraph, std::vector<uint8_t> & output);

/* backend */

ശൂന്യം                                      apir_track_backend_buffer(ggml_backend_buffer_t buffer);
bool                                      apir_untrack_backend_buffer(ggml_backend_buffer_t buffer);
std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers();

ശൂന്യം apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, സ്ഥിരം apir_rpc_tensor * tensor);

ggml_tensor * apir_create_കണ്ണി(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               സ്ഥിരം std::unordered_map<uint64_t, സ്ഥിരം apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     സ്ഥിരം apir_rpc_tensor * tensors,
                                     സ്ഥിരം uint64_t *        nodes);
