#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "shared/apir_cs_rpc.h"

#അന്തര്ഭാവ <cinttypes>
#അന്തര്ഭാവ <unordered_map>
#അന്തര്ഭാവ <unordered_set>
#അന്തര്ഭാവ <vector>

std::unordered_set<ggml_backend_buffer_t> backend_buffers;

ശൂന്യം apir_track_backend_buffer(ggml_backend_buffer_t buffer) {
    backend_buffers.insert(buffer);
}

തർക്കം apir_untrack_backend_buffer(ggml_backend_buffer_t buffer) {
    auto it = backend_buffers.find(buffer);
    എങ്കിൽ (it == backend_buffers.end()) {
        തിരികെ അസത്യം;
    }

    backend_buffers.erase(it);
    തിരികെ സത്യം;
}

std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers() {
    തിരികെ backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, സ്ഥിരം apir_rpc_tensor * tensor) {
    ggml_tensor * result =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    വേണ്ടി (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    എങ്കിൽ (result->buffer && backend_buffers.find(result->buffer) == backend_buffers.end()) {
        printf("WARNING: HOST BUFFER NOT FOUND | %p\n", (ശൂന്യം *) result->buffer);
        result->buffer = nullptr;
    }

    uint64_t tensor_data = tensor->data;
    എങ്കിൽ (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size  = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size  = (uint64_t) ggml_backend_buffer_get_size(result->buffer);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_data += buffer_start;

        GGML_ASSERT(tensor_data + tensor_size >= tensor_data);  // check for overflow
        GGML_ASSERT(tensor_data >= buffer_start && tensor_data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op) tensor->op;
    വേണ്ടി (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / വലുപ്പം(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data  = reinterpret_cast<ശൂന്യം *>(tensor_data);
    ggml_set_name(result, tensor->name);
    തിരികെ result;
}

ggml_tensor * apir_create_കണ്ണി(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               സ്ഥിരം std::unordered_map<uint64_t, സ്ഥിരം apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    എങ്കിൽ (id == 0) {
        തിരികെ nullptr;
    }
    എങ്കിൽ (tensor_map.find(id) != tensor_map.end()) {
        തിരികെ tensor_map[id];
    }
    സ്ഥിരം apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           result = apir_deserialize_tensor(ctx, tensor);
    എങ്കിൽ (result == nullptr) {
        തിരികെ nullptr;
    }
    tensor_map[id] = result;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
        result->src[i] = apir_create_കണ്ണി(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    result->view_src  = apir_create_കണ്ണി(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    result->view_offs = tensor->view_offs;
    തിരികെ result;
}

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     സ്ഥിരം apir_rpc_tensor * tensors,
                                     സ്ഥിരം uint64_t *        nodes) {
    size_t buf_size = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, അസത്യം);
    ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/ശൂന്യ,
        /*.no_alloc   =*/സത്യം,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_graph_custom(ctx, n_nodes, അസത്യം);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, സ്ഥിരം apir_rpc_tensor *> tensor_ptrs;
    വേണ്ടി (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    വേണ്ടി (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], വലുപ്പം(id));
        graph->nodes[i] = apir_create_കണ്ണി(id, ctx, tensor_ptrs, tensor_map);
    }

    തിരികെ graph;
}
