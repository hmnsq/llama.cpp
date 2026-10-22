#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "shared/apir_cs_rpc.h"

#अन्तर्भाव <cinttypes>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <unordered_set>
#अन्तर्भाव <vector>

std::unordered_set<ggml_backend_buffer_t> backend_buffers;

शून्य apir_track_backend_buffer(ggml_backend_buffer_t buffer) {
    backend_buffers.insert(buffer);
}

तार्किक apir_untrack_backend_buffer(ggml_backend_buffer_t buffer) {
    स्वतः it = backend_buffers.find(buffer);
    जर (it == backend_buffers.end()) {
        परत असत्य;
    }

    backend_buffers.erase(it);
    परत सत्य;
}

std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers() {
    परत backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, स्थिरांक apir_rpc_tensor * tensor) {
    ggml_tensor * result =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    साठी (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    जर (result->buffer && backend_buffers.find(result->buffer) == backend_buffers.end()) {
        म_लिखो("WARNING: HOST BUFFER NOT FOUND | %p\n", (शून्य *) result->buffer);
        result->buffer = nullptr;
    }

    uint64_t tensor_data = tensor->data;
    जर (result->buffer) {
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
    साठी (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / आकार(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data  = reinterpret_cast<शून्य *>(tensor_data);
    ggml_set_name(result, tensor->name);
    परत result;
}

ggml_tensor * apir_create_गाठ(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               स्थिरांक std::unordered_map<uint64_t, स्थिरांक apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    जर (id == 0) {
        परत nullptr;
    }
    जर (tensor_map.find(id) != tensor_map.end()) {
        परत tensor_map[id];
    }
    स्थिरांक apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           result = apir_deserialize_tensor(ctx, tensor);
    जर (result == nullptr) {
        परत nullptr;
    }
    tensor_map[id] = result;
    साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        result->src[i] = apir_create_गाठ(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    result->view_src  = apir_create_गाठ(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    result->view_offs = tensor->view_offs;
    परत result;
}

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     स्थिरांक apir_rpc_tensor * tensors,
                                     स्थिरांक uint64_t *        nodes) {
    माप_प्रकार buf_size = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, असत्य);
    ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/रिक्त,
        /*.no_alloc   =*/सत्य,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_graph_custom(ctx, n_nodes, असत्य);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, स्थिरांक apir_rpc_tensor *> tensor_ptrs;
    साठी (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    साठी (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        स_नकल(&id, &nodes[i], आकार(id));
        graph->nodes[i] = apir_create_गाठ(id, ctx, tensor_ptrs, tensor_map);
    }

    परत graph;
}
