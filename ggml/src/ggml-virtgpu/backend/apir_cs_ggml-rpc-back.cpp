#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "shared/apir_cs_rpc.h"

#ಅನ್ತರ್ಭಾವ <cinttypes>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <unordered_set>
#ಅನ್ತರ್ಭಾವ <vector>

std::unordered_set<ggml_backend_ಬಫರ್_t> backend_buffers;

ಶೂನ್ಯ apir_track_backend_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    backend_buffers.insert(ಬಫರ್);
}

ತಾರ್ಕಿಕ apir_untrack_backend_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಸ್ವಯಂಚಾಲಿತ it = backend_buffers.find(ಬಫರ್);
    ಆದರೆ (it == backend_buffers.end()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    backend_buffers.erase(it);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

std::unordered_set<ggml_backend_ಬಫರ್_t> apir_get_track_backend_buffers() {
    ಹಿಂತಿರುಗಿ backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ಸ್ಥಿರಾಂಕ apir_rpc_tensor * tensor) {
    ggml_tensor * ಫಲಿತಾಂಶ =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    ಗಾಗಿ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ಫಲಿತಾಂಶ->nb[i] = tensor->nb[i];
    }
    ಫಲಿತಾಂಶ->ಬಫರ್ = reinterpret_cast<ggml_backend_ಬಫರ್_t>(tensor->ಬಫರ್);
    ಆದರೆ (ಫಲಿತಾಂಶ->ಬಫರ್ && backend_buffers.find(ಫಲಿತಾಂಶ->ಬಫರ್) == backend_buffers.end()) {
        ಬರೆ("WARNING: HOST BUFFER NOT FOUND | %p\n", (ಶೂನ್ಯ *) ಫಲಿತಾಂಶ->ಬಫರ್);
        ಫಲಿತಾಂಶ->ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
    }

    uint64_t tensor_ದತ್ತಾಂಶ = tensor->ದತ್ತಾಂಶ;
    ಆದರೆ (ಫಲಿತಾಂಶ->ಬಫರ್) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_ಗಾತ್ರ  = (uint64_t) ggml_nbytes(ಫಲಿತಾಂಶ);
        uint64_t ಬಫರ್_ಆರಂಭ = (uint64_t) ggml_backend_ಬಫರ್_get_ತಳ(ಫಲಿತಾಂಶ->ಬಫರ್);
        uint64_t ಬಫರ್_ಗಾತ್ರ  = (uint64_t) ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(ಫಲಿತಾಂಶ->ಬಫರ್);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_ದತ್ತಾಂಶ += ಬಫರ್_ಆರಂಭ;

        GGML_ASSERT(tensor_ದತ್ತಾಂಶ + tensor_ಗಾತ್ರ >= tensor_ದತ್ತಾಂಶ);  // check for overflow
        GGML_ASSERT(tensor_ದತ್ತಾಂಶ >= ಬಫರ್_ಆರಂಭ && tensor_ದತ್ತಾಂಶ + tensor_ಗಾತ್ರ <= ಬಫರ್_ಆರಂಭ + ಬಫರ್_ಗಾತ್ರ);
    }

    ಫಲಿತಾಂಶ->op = (ggml_op) tensor->op;
    ಗಾಗಿ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / ಗಾತ್ರ(int32_t); i++) {
        ಫಲಿತಾಂಶ->op_params[i] = tensor->op_params[i];
    }
    ಫಲಿತಾಂಶ->flags = tensor->flags;
    ಫಲಿತಾಂಶ->ದತ್ತಾಂಶ  = reinterpret_cast<ಶೂನ್ಯ *>(tensor_ದತ್ತಾಂಶ);
    ggml_ಗಣ_name(ಫಲಿತಾಂಶ, tensor->name);
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ggml_tensor * apir_create_ಗಂಟು(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ಸ್ಥಿರಾಂಕ std::unordered_map<uint64_t, ಸ್ಥಿರಾಂಕ apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    ಆದರೆ (id == 0) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (tensor_map.find(id) != tensor_map.end()) {
        ಹಿಂತಿರುಗಿ tensor_map[id];
    }
    ಸ್ಥಿರಾಂಕ apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           ಫಲಿತಾಂಶ = apir_deserialize_tensor(ctx, tensor);
    ಆದರೆ (ಫಲಿತಾಂಶ == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    tensor_map[id] = ಫಲಿತಾಂಶ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಫಲಿತಾಂಶ->src[i] = apir_create_ಗಂಟು(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    ಫಲಿತಾಂಶ->view_src  = apir_create_ಗಂಟು(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    ಫಲಿತಾಂಶ->view_offs = tensor->view_offs;
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ggml_cgraph * apir_deserialize_ನಕ್ಷೆ(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ಸ್ಥಿರಾಂಕ apir_rpc_tensor * tensors,
                                     ಸ್ಥಿರಾಂಕ uint64_t *        nodes) {
    size_t buf_ಗಾತ್ರ = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_ನಕ್ಷೆ_overhead_custom(n_nodes, ಸುಳ್ಳು);
    ggml_init_params params = {
        /*.mem_size   =*/buf_ಗಾತ್ರ,
        /*.mem_buffer =*/ಖಾಲಿ,
        /*.no_alloc   =*/ನಿಜ,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_ನಕ್ಷೆ_custom(ctx, n_nodes, ಸುಳ್ಳು);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, ಸ್ಥಿರಾಂಕ apir_rpc_tensor *> tensor_ptrs;
    ಗಾಗಿ (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    ಗಾಗಿ (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        ಸ್ಮರಣೆ_ನಕಲು(&id, &nodes[i], ಗಾತ್ರ(id));
        graph->nodes[i] = apir_create_ಗಂಟು(id, ctx, tensor_ptrs, tensor_map);
    }

    ಹಿಂತಿರುಗಿ graph;
}
