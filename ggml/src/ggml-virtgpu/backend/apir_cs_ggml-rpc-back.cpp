#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "shared/apir_cs_rpc.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cinttypes>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_set>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

std::unordered_set<ggml_backend_ਬਫਰ_t> backend_buffers;

ਖਾਲੀ apir_track_backend_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    backend_buffers.insert(ਬਫਰ);
}

ਤਾਰਕਿਕ apir_untrack_backend_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਸਵੈਚਾਲਿਤ it = backend_buffers.find(ਬਫਰ);
    ਜੇ (it == backend_buffers.end()) {
        ਵਾਪਸ ਝੂਠ;
    }

    backend_buffers.erase(it);
    ਵਾਪਸ ਸੱਚ;
}

std::unordered_set<ggml_backend_ਬਫਰ_t> apir_get_track_backend_buffers() {
    ਵਾਪਸ backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ਅਚਲ apir_rpc_tensor * tensor) {
    ggml_tensor * ਨਤੀਜਾ =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    ਲਈ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ਨਤੀਜਾ->nb[i] = tensor->nb[i];
    }
    ਨਤੀਜਾ->ਬਫਰ = reinterpret_cast<ggml_backend_ਬਫਰ_t>(tensor->ਬਫਰ);
    ਜੇ (ਨਤੀਜਾ->ਬਫਰ && backend_buffers.find(ਨਤੀਜਾ->ਬਫਰ) == backend_buffers.end()) {
        ਲਿਖੋ("WARNING: HOST BUFFER NOT FOUND | %p\n", (ਖਾਲੀ *) ਨਤੀਜਾ->ਬਫਰ);
        ਨਤੀਜਾ->ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
    }

    uint64_t tensor_ਅੰਕੜੇ = tensor->ਅੰਕੜੇ;
    ਜੇ (ਨਤੀਜਾ->ਬਫਰ) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_ਮਾਪ  = (uint64_t) ggml_nbytes(ਨਤੀਜਾ);
        uint64_t ਬਫਰ_ਸ਼ੁਰੂ = (uint64_t) ggml_backend_ਬਫਰ_get_ਅਧਾਰ(ਨਤੀਜਾ->ਬਫਰ);
        uint64_t ਬਫਰ_ਮਾਪ  = (uint64_t) ggml_backend_ਬਫਰ_get_ਮਾਪ(ਨਤੀਜਾ->ਬਫਰ);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_ਅੰਕੜੇ += ਬਫਰ_ਸ਼ੁਰੂ;

        GGML_ASSERT(tensor_ਅੰਕੜੇ + tensor_ਮਾਪ >= tensor_ਅੰਕੜੇ);  // check for overflow
        GGML_ASSERT(tensor_ਅੰਕੜੇ >= ਬਫਰ_ਸ਼ੁਰੂ && tensor_ਅੰਕੜੇ + tensor_ਮਾਪ <= ਬਫਰ_ਸ਼ੁਰੂ + ਬਫਰ_ਮਾਪ);
    }

    ਨਤੀਜਾ->op = (ggml_op) tensor->op;
    ਲਈ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / ਆਕਾਰ(int32_t); i++) {
        ਨਤੀਜਾ->op_params[i] = tensor->op_params[i];
    }
    ਨਤੀਜਾ->flags = tensor->flags;
    ਨਤੀਜਾ->ਅੰਕੜੇ  = reinterpret_cast<ਖਾਲੀ *>(tensor_ਅੰਕੜੇ);
    ggml_ਸੈੱਟ_name(ਨਤੀਜਾ, tensor->name);
    ਵਾਪਸ ਨਤੀਜਾ;
}

ggml_tensor * apir_create_ਗੰਢ(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ਅਚਲ std::unordered_map<uint64_t, ਅਚਲ apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    ਜੇ (id == 0) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (tensor_map.find(id) != tensor_map.end()) {
        ਵਾਪਸ tensor_map[id];
    }
    ਅਚਲ apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           ਨਤੀਜਾ = apir_deserialize_tensor(ctx, tensor);
    ਜੇ (ਨਤੀਜਾ == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    tensor_map[id] = ਨਤੀਜਾ;
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        ਨਤੀਜਾ->src[i] = apir_create_ਗੰਢ(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    ਨਤੀਜਾ->view_src  = apir_create_ਗੰਢ(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    ਨਤੀਜਾ->view_offs = tensor->view_offs;
    ਵਾਪਸ ਨਤੀਜਾ;
}

ggml_cgraph * apir_deserialize_ਗ੍ਰਾਫ(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ਅਚਲ apir_rpc_tensor * tensors,
                                     ਅਚਲ uint64_t *        nodes) {
    size_t buf_ਮਾਪ = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_ਗ੍ਰਾਫ_overhead_custom(n_nodes, ਝੂਠ);
    ggml_init_params params = {
        /*.mem_size   =*/buf_ਮਾਪ,
        /*.mem_buffer =*/ਨਿਰਾ,
        /*.no_alloc   =*/ਸੱਚ,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_ਗ੍ਰਾਫ_custom(ctx, n_nodes, ਝੂਠ);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, ਅਚਲ apir_rpc_tensor *> tensor_ptrs;
    ਲਈ (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    ਲਈ (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&id, &nodes[i], ਆਕਾਰ(id));
        graph->nodes[i] = apir_create_ਗੰਢ(id, ctx, tensor_ptrs, tensor_map);
    }

    ਵਾਪਸ graph;
}
