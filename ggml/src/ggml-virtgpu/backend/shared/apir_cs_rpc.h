#निर्देशः once

// clang-format off
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-backend-impl.h"

#अन्तर्भाव <unordered_map>
#अन्तर्भाव <unordered_set>
#अन्तर्भाव <vector>
#अन्तर्भाव <cstdint>
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
संरचना apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t बफर;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / परिमाण(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t दत्त;
    वर्ण     name[GGML_MAX_NAME];

    वर्ण padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(ध्रुव ggml_tensor * tensor);

शून्य apir_serialize_graph(ध्रुव ggml_cgraph * cgraph, std::सदिशम्<uint8_t> & output);

/* backend */

शून्य                                      apir_track_backend_बफर(ggml_backend_बफर_t बफर);
bool                                      apir_untrack_backend_बफर(ggml_backend_बफर_t बफर);
std::unordered_समुच्चयः<ggml_backend_बफर_t> apir_get_track_backend_buffers();

शून्य apir_सङ्कलय_tensor(ggml_tensor *                       tensor,
                     std::सदिशम्<apir_rpc_tensor> &      tensors,
                     std::unordered_समुच्चयः<ggml_tensor *> & visited);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ध्रुव apir_rpc_tensor * tensor);

ggml_tensor * apir_create_ग्रन्थिः(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ध्रुव std::unordered_map<uint64_t, ध्रुव apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ध्रुव apir_rpc_tensor * tensors,
                                     ध्रुव uint64_t *        nodes);
