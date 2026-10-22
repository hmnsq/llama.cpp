#ਹਦਾਇਤ once

// clang-format off
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_set>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
ਬਣਤਰ apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t ਬਫਰ;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / ਆਕਾਰ(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t ਅੰਕੜੇ;
    ਅੱਖਰ     name[GGML_MAX_NAME];

    ਅੱਖਰ padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(ਅਚਲ ggml_tensor * tensor);

ਖਾਲੀ apir_serialize_ਗ੍ਰਾਫ(ਅਚਲ ggml_cgraph * cgraph, std::ਸਦਿਸ਼<uint8_t> & output);

/* backend */

ਖਾਲੀ                                      apir_track_backend_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ);
bool                                      apir_untrack_backend_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ);
std::unordered_ਸੈੱਟ<ggml_backend_ਬਫਰ_t> apir_get_track_backend_buffers();

ਖਾਲੀ apir_ਜੋੜੋ_tensor(ggml_tensor *                       tensor,
                     std::ਸਦਿਸ਼<apir_rpc_tensor> &      tensors,
                     std::unordered_ਸੈੱਟ<ggml_tensor *> & ਦੇਖਿਆ);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ਅਚਲ apir_rpc_tensor * tensor);

ggml_tensor * apir_create_ਗੰਢ(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ਅਚਲ std::unordered_map<uint64_t, ਅਚਲ apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_ਗ੍ਰਾਫ(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ਅਚਲ apir_rpc_tensor * tensors,
                                     ਅਚਲ uint64_t *        nodes);
