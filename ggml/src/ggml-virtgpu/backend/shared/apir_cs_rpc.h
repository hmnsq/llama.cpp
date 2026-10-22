#ہدایت once

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
    uint64_t بفر;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / حجم(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t انکڑے;
    حرف     name[GGML_MAX_NAME];

    حرف padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(ثابت ggml_tensor * tensor);

فراغ apir_serialize_گراف(ثابت ggml_cgraph * cgraph, std::سمتیہ<uint8_t> & output);

/* backend */

فراغ                                      apir_track_backend_بفر(ggml_backend_بفر_t بفر);
bool                                      apir_untrack_backend_بفر(ggml_backend_بفر_t بفر);
std::unordered_سیٹ<ggml_backend_بفر_t> apir_get_track_backend_buffers();

فراغ apir_جوڑو_tensor(ggml_tensor *                       tensor,
                     std::سمتیہ<apir_rpc_tensor> &      tensors,
                     std::unordered_سیٹ<ggml_tensor *> & ویکھیا);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ثابت apir_rpc_tensor * tensor);

ggml_tensor * apir_create_گرہ(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ثابت std::unordered_map<uint64_t, ثابت apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_گراف(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ثابت apir_rpc_tensor * tensors,
                                     ثابت uint64_t *        nodes);
