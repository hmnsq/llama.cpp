#ਅਨ੍ਤਰ੍ਭਾਵ "backend/shared/apir_cs_rpc.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-remoting.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cinttypes>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_set>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

apir_rpc_tensor apir_serialize_tensor(ਅਚਲ ggml_tensor * tensor) {
    apir_rpc_tensor ਨਤੀਜਾ;
    ਨਤੀਜਾ.id   = reinterpret_cast<uint64_t>(tensor);
    ਨਤੀਜਾ.type = tensor->type;
    ਜੇ (tensor->ਬਫਰ) {
        ggml_backend_ਬਫਰ_t ਬਫਰ = tensor->ਬਫਰ;

        ਨਤੀਜਾ.ਬਫਰ = ਬਫਰ_ਮੰਜ਼ਿਲ_HOST_HANDLE(ਬਫਰ);
    } ਨਹੀਂਤਾਂ {
        ਨਤੀਜਾ.ਬਫਰ = 0;
    }
    ਲਈ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ਨਤੀਜਾ.ne[i] = tensor->ne[i];
        ਨਤੀਜਾ.nb[i] = tensor->nb[i];
    }
    ਨਤੀਜਾ.op = tensor->op;
    ਲਈ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / ਆਕਾਰ(int32_t); i++) {
        ਨਤੀਜਾ.op_params[i] = tensor->op_params[i];
    }
    ਨਤੀਜਾ.flags = tensor->flags;
    ਲਈ (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        ਨਤੀਜਾ.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    ਨਤੀਜਾ.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    ਨਤੀਜਾ.view_offs = tensor->view_offs;
    ਨਤੀਜਾ.ਅੰਕੜੇ      = reinterpret_cast<uint64_t>(tensor->ਅੰਕੜੇ);
    ਜੇ (tensor->ਅੰਕੜੇ) {
        ਜੇ (!tensor->ਬਫਰ) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        ਨਤੀਜਾ.ਅੰਕੜੇ -= reinterpret_cast<uint64_t>(ਬਫਰ_ਮੰਜ਼ਿਲ_GGML_CONTEXT(tensor->ਬਫਰ)->ਅਧਾਰ);
    }
    snprintf(ਨਤੀਜਾ.name, GGML_MAX_NAME, "%s", tensor->name);
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਖਾਲੀ apir_ਜੋੜੋ_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & ਦੇਖਿਆ) {
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }
    ਜੇ (ਦੇਖਿਆ.find(tensor) != ਦੇਖਿਆ.end()) {
        ਵਾਪਸ;
    }
    ਦੇਖਿਆ.insert(tensor);
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        apir_ਜੋੜੋ_tensor(tensor->src[i], tensors, ਦੇਖਿਆ);
    }
    apir_ਜੋੜੋ_tensor(tensor->view_src, tensors, ਦੇਖਿਆ);
    tensors.push_back(apir_serialize_tensor(tensor));
}

ਖਾਲੀ apir_serialize_ਗ੍ਰਾਫ(ਅਚਲ ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> ਦੇਖਿਆ;
    ਲਈ (uint32_t i = 0; i < n_nodes; i++) {
        apir_ਜੋੜੋ_tensor(cgraph->nodes[i], tensors, ਦੇਖਿਆ);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    ਪੂਰਨ      output_ਮਾਪ =
        ਆਕਾਰ(uint32_t) + n_nodes * ਆਕਾਰ(uint64_t) + ਆਕਾਰ(uint32_t) + n_tensors * ਆਕਾਰ(apir_rpc_tensor);
    output.resize(output_ਮਾਪ, 0);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(output.ਅੰਕੜੇ(), &n_nodes, ਆਕਾਰ(n_nodes));
    ਲਈ (uint32_t i = 0; i < n_nodes; i++) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(output.ਅੰਕੜੇ() + ਆਕਾਰ(n_nodes) + i * ਆਕਾਰ(uint64_t), &cgraph->nodes[i], ਆਕਾਰ(uint64_t));
    }
    uint32_t * ਬਾਹਰਲਾ_ntensors = (uint32_t *) (output.ਅੰਕੜੇ() + ਆਕਾਰ(n_nodes) + n_nodes * ਆਕਾਰ(uint64_t));
    *ਬਾਹਰਲਾ_ntensors           = n_tensors;
    apir_rpc_tensor * ਬਾਹਰਲਾ_tensors =
        (apir_rpc_tensor *) (output.ਅੰਕੜੇ() + ਆਕਾਰ(n_nodes) + n_nodes * ਆਕਾਰ(uint64_t) + ਆਕਾਰ(uint32_t));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਬਾਹਰਲਾ_tensors, tensors.ਅੰਕੜੇ(), n_tensors * ਆਕਾਰ(apir_rpc_tensor));
}
