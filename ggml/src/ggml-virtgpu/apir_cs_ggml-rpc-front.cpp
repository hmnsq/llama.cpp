#ಅನ್ತರ್ಭಾವ "backend/shared/apir_cs_rpc.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"

#ಅನ್ತರ್ಭಾವ <cinttypes>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <unordered_set>
#ಅನ್ತರ್ಭಾವ <vector>

apir_rpc_tensor apir_serialize_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    apir_rpc_tensor ಫಲಿತಾಂಶ;
    ಫಲಿತಾಂಶ.id   = reinterpret_cast<uint64_t>(tensor);
    ಫಲಿತಾಂಶ.type = tensor->type;
    ಆದರೆ (tensor->ಬಫರ್) {
        ggml_backend_ಬಫರ್_t ಬಫರ್ = tensor->ಬಫರ್;

        ಫಲಿತಾಂಶ.ಬಫರ್ = ಬಫರ್_ಗಮ್ಯ_HOST_HANDLE(ಬಫರ್);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಫಲಿತಾಂಶ.ಬಫರ್ = 0;
    }
    ಗಾಗಿ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ಫಲಿತಾಂಶ.ne[i] = tensor->ne[i];
        ಫಲಿತಾಂಶ.nb[i] = tensor->nb[i];
    }
    ಫಲಿತಾಂಶ.op = tensor->op;
    ಗಾಗಿ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / ಗಾತ್ರ(int32_t); i++) {
        ಫಲಿತಾಂಶ.op_params[i] = tensor->op_params[i];
    }
    ಫಲಿತಾಂಶ.flags = tensor->flags;
    ಗಾಗಿ (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        ಫಲಿತಾಂಶ.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    ಫಲಿತಾಂಶ.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    ಫಲಿತಾಂಶ.view_offs = tensor->view_offs;
    ಫಲಿತಾಂಶ.ದತ್ತಾಂಶ      = reinterpret_cast<uint64_t>(tensor->ದತ್ತಾಂಶ);
    ಆದರೆ (tensor->ದತ್ತಾಂಶ) {
        ಆದರೆ (!tensor->ಬಫರ್) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        ಫಲಿತಾಂಶ.ದತ್ತಾಂಶ -= reinterpret_cast<uint64_t>(ಬಫರ್_ಗಮ್ಯ_GGML_CONTEXT(tensor->ಬಫರ್)->ತಳ);
    }
    snprintf(ಫಲಿತಾಂಶ.name, GGML_MAX_NAME, "%s", tensor->name);
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಶೂನ್ಯ apir_ಕೂಡಿಸು_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & ಭೇಟಿಯಾದ) {
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (ಭೇಟಿಯಾದ.find(tensor) != ಭೇಟಿಯಾದ.end()) {
        ಹಿಂತಿರುಗಿ;
    }
    ಭೇಟಿಯಾದ.insert(tensor);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        apir_ಕೂಡಿಸು_tensor(tensor->src[i], tensors, ಭೇಟಿಯಾದ);
    }
    apir_ಕೂಡಿಸು_tensor(tensor->view_src, tensors, ಭೇಟಿಯಾದ);
    tensors.push_back(apir_serialize_tensor(tensor));
}

ಶೂನ್ಯ apir_serialize_ನಕ್ಷೆ(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> ಭೇಟಿಯಾದ;
    ಗಾಗಿ (uint32_t i = 0; i < n_nodes; i++) {
        apir_ಕೂಡಿಸು_tensor(cgraph->nodes[i], tensors, ಭೇಟಿಯಾದ);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    ಪೂರ್ಣಾಂಕ      output_ಗಾತ್ರ =
        ಗಾತ್ರ(uint32_t) + n_nodes * ಗಾತ್ರ(uint64_t) + ಗಾತ್ರ(uint32_t) + n_tensors * ಗಾತ್ರ(apir_rpc_tensor);
    output.resize(output_ಗಾತ್ರ, 0);
    ಸ್ಮರಣೆ_ನಕಲು(output.ದತ್ತಾಂಶ(), &n_nodes, ಗಾತ್ರ(n_nodes));
    ಗಾಗಿ (uint32_t i = 0; i < n_nodes; i++) {
        ಸ್ಮರಣೆ_ನಕಲು(output.ದತ್ತಾಂಶ() + ಗಾತ್ರ(n_nodes) + i * ಗಾತ್ರ(uint64_t), &cgraph->nodes[i], ಗಾತ್ರ(uint64_t));
    }
    uint32_t * ಹೊರಹೋಗು_ntensors = (uint32_t *) (output.ದತ್ತಾಂಶ() + ಗಾತ್ರ(n_nodes) + n_nodes * ಗಾತ್ರ(uint64_t));
    *ಹೊರಹೋಗು_ntensors           = n_tensors;
    apir_rpc_tensor * ಹೊರಹೋಗು_tensors =
        (apir_rpc_tensor *) (output.ದತ್ತಾಂಶ() + ಗಾತ್ರ(n_nodes) + n_nodes * ಗಾತ್ರ(uint64_t) + ಗಾತ್ರ(uint32_t));
    ಸ್ಮರಣೆ_ನಕಲು(ಹೊರಹೋಗು_tensors, tensors.ದತ್ತಾಂಶ(), n_tensors * ಗಾತ್ರ(apir_rpc_tensor));
}
