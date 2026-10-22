#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "shared/apir_cs_rpc.h"

#ଅନ୍ତର୍ଭାଵ <cinttypes>
#ଅନ୍ତର୍ଭାଵ <unordered_map>
#ଅନ୍ତର୍ଭାଵ <unordered_set>
#ଅନ୍ତର୍ଭାଵ <vector>

std::unordered_set<ggml_backend_buffer_t> backend_buffers;

ଶୂନ୍ୟ apir_track_backend_buffer(ggml_backend_buffer_t buffer) {
    backend_buffers.insert(buffer);
}

ତାର୍କିକ apir_untrack_backend_buffer(ggml_backend_buffer_t buffer) {
    auto it = backend_buffers.find(buffer);
    ଯଦି (it == backend_buffers.end()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    backend_buffers.erase(it);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers() {
    ଫେରନ୍ତୁ backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ଅଚଳ apir_rpc_tensor * tensor) {
    ggml_tensor * result =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    ପାଇଁ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    ଯଦି (result->buffer && backend_buffers.find(result->buffer) == backend_buffers.end()) {
        printf("WARNING: HOST BUFFER NOT FOUND | %p\n", (ଶୂନ୍ୟ *) result->buffer);
        result->buffer = nullptr;
    }

    uint64_t tensor_data = tensor->data;
    ଯଦି (result->buffer) {
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
    ପାଇଁ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / ଆକାର(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data  = reinterpret_cast<ଶୂନ୍ୟ *>(tensor_data);
    ggml_set_name(result, tensor->name);
    ଫେରନ୍ତୁ result;
}

ggml_tensor * apir_create_ଗଣ୍ଠି(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ଅଚଳ std::unordered_map<uint64_t, ଅଚଳ apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    ଯଦି (id == 0) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଯଦି (tensor_map.find(id) != tensor_map.end()) {
        ଫେରନ୍ତୁ tensor_map[id];
    }
    ଅଚଳ apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           result = apir_deserialize_tensor(ctx, tensor);
    ଯଦି (result == nullptr) {
        ଫେରନ୍ତୁ nullptr;
    }
    tensor_map[id] = result;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
        result->src[i] = apir_create_ଗଣ୍ଠି(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    result->view_src  = apir_create_ଗଣ୍ଠି(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    result->view_offs = tensor->view_offs;
    ଫେରନ୍ତୁ result;
}

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ଅଚଳ apir_rpc_tensor * tensors,
                                     ଅଚଳ uint64_t *        nodes) {
    size_t buf_size = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, ମିଥ୍ୟା);
    ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/ଖାଲି,
        /*.no_alloc   =*/ସତ୍ୟ,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_graph_custom(ctx, n_nodes, ମିଥ୍ୟା);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, ଅଚଳ apir_rpc_tensor *> tensor_ptrs;
    ପାଇଁ (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    ପାଇଁ (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], ଆକାର(id));
        graph->nodes[i] = apir_create_ଗଣ୍ଠି(id, ctx, tensor_ptrs, tensor_map);
    }

    ଫେରନ୍ତୁ graph;
}
