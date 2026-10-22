#సూచన once

// clang-format off
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ggml-backend-impl.h"

#అన్తర్భావ <unordered_map>
#అన్తర్భావ <unordered_set>
#అన్తర్భావ <vector>
#అన్తర్భావ <cstdint>
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
నిర్మాణం apir_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t బఫరు;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / పరిమాణం(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t దత్తాంశం;
    అక్షరం     name[GGML_MAX_NAME];

    అక్షరం padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(స్థిరరాశి ggml_tensor * tensor);

శూన్య apir_serialize_లేఖాచిత్రం(స్థిరరాశి ggml_cgraph * cgraph, std::సదిశ<uint8_t> & output);

/* backend */

శూన్య                                      apir_track_backend_బఫరు(ggml_backend_బఫరు_t బఫరు);
bool                                      apir_untrack_backend_బఫరు(ggml_backend_బఫరు_t బఫరు);
std::unordered_సమితి<ggml_backend_బఫరు_t> apir_get_track_backend_buffers();

శూన్య apir_కూడు_tensor(ggml_tensor *                       tensor,
                     std::సదిశ<apir_rpc_tensor> &      tensors,
                     std::unordered_సమితి<ggml_tensor *> & సందర్శించిన);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, స్థిరరాశి apir_rpc_tensor * tensor);

ggml_tensor * apir_create_కణుపు(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               స్థిరరాశి std::unordered_map<uint64_t, స్థిరరాశి apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_లేఖాచిత్రం(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     స్థిరరాశి apir_rpc_tensor * tensors,
                                     స్థిరరాశి uint64_t *        nodes);
