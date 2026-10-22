#आशय once

// clang-format off
#समावेश "ggml.h"
#समावेश "ggml-backend-impl.h"

#समावेश <unordered_map>
#समावेश <unordered_set>
#समावेश <vector>
#समावेश <cstdint>
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
काष्ठा apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t बफर;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / माप(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t तथ्य;
    अक्षर     name[GGML_MAX_NAME];

    अक्षर padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(स्थिर ggml_tensor * tensor);

व्योम apir_serialize_graph(स्थिर ggml_cgraph * cgraph, std::vector<uint8_t> & output);

/* backend */

व्योम                                      apir_track_backend_बफर(ggml_backend_बफर_t बफर);
bool                                      apir_untrack_backend_बफर(ggml_backend_बफर_t बफर);
std::unordered_set<ggml_backend_बफर_t> apir_get_track_backend_buffers();

व्योम apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, स्थिर apir_rpc_tensor * tensor);

ggml_tensor * apir_create_गाँठ(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               स्थिर std::unordered_map<uint64_t, स्थिर apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     स्थिर apir_rpc_tensor * tensors,
                                     स्थिर uint64_t *        nodes);
