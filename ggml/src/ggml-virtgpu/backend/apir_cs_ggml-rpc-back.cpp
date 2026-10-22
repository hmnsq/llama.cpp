#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "shared/apir_cs_rpc.h"

#अन्तर्भाव <cinttypes>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <unordered_set>
#अन्तर्भाव <vector>

std::unordered_set<ggml_backend_बफर_t> backend_buffers;

शून्य apir_track_backend_बफर(ggml_backend_बफर_t बफर) {
    backend_buffers.insert(बफर);
}

तार्किक apir_untrack_backend_बफर(ggml_backend_बफर_t बफर) {
    स्वचालित it = backend_buffers.find(बफर);
    यदि (it == backend_buffers.end()) {
        प्रत्यागम असत्य;
    }

    backend_buffers.erase(it);
    प्रत्यागम सत्य;
}

std::unordered_set<ggml_backend_बफर_t> apir_get_track_backend_buffers() {
    प्रत्यागम backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ध्रुव apir_rpc_tensor * tensor) {
    ggml_tensor * परिणाम =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    प्रति (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        परिणाम->nb[i] = tensor->nb[i];
    }
    परिणाम->बफर = reinterpret_cast<ggml_backend_बफर_t>(tensor->बफर);
    यदि (परिणाम->बफर && backend_buffers.find(परिणाम->बफर) == backend_buffers.end()) {
        प्रारूप_लिख("WARNING: HOST BUFFER NOT FOUND | %p\n", (शून्य *) परिणाम->बफर);
        परिणाम->बफर = शून्यसूचकः;
    }

    uint64_t tensor_दत्त = tensor->दत्त;
    यदि (परिणाम->बफर) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_आकार  = (uint64_t) ggml_nbytes(परिणाम);
        uint64_t बफर_start = (uint64_t) ggml_backend_बफर_get_आधारः(परिणाम->बफर);
        uint64_t बफर_आकार  = (uint64_t) ggml_backend_बफर_get_आकार(परिणाम->बफर);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_दत्त += बफर_start;

        GGML_ASSERT(tensor_दत्त + tensor_आकार >= tensor_दत्त);  // check for overflow
        GGML_ASSERT(tensor_दत्त >= बफर_start && tensor_दत्त + tensor_आकार <= बफर_start + बफर_आकार);
    }

    परिणाम->op = (ggml_op) tensor->op;
    प्रति (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / परिमाण(int32_t); i++) {
        परिणाम->op_params[i] = tensor->op_params[i];
    }
    परिणाम->flags = tensor->flags;
    परिणाम->दत्त  = reinterpret_cast<शून्य *>(tensor_दत्त);
    ggml_समुच्चयः_name(परिणाम, tensor->name);
    प्रत्यागम परिणाम;
}

ggml_tensor * apir_create_ग्रन्थिः(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ध्रुव std::unordered_map<uint64_t, ध्रुव apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    यदि (id == 0) {
        प्रत्यागम शून्यसूचकः;
    }
    यदि (tensor_map.find(id) != tensor_map.end()) {
        प्रत्यागम tensor_map[id];
    }
    ध्रुव apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           परिणाम = apir_deserialize_tensor(ctx, tensor);
    यदि (परिणाम == शून्यसूचकः) {
        प्रत्यागम शून्यसूचकः;
    }
    tensor_map[id] = परिणाम;
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        परिणाम->src[i] = apir_create_ग्रन्थिः(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    परिणाम->view_src  = apir_create_ग्रन्थिः(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    परिणाम->view_offs = tensor->view_offs;
    प्रत्यागम परिणाम;
}

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ध्रुव apir_rpc_tensor * tensors,
                                     ध्रुव uint64_t *        nodes) {
    size_t buf_आकार = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, असत्य);
    ggml_init_params params = {
        /*.mem_size   =*/buf_आकार,
        /*.mem_buffer =*/रिक्त,
        /*.no_alloc   =*/सत्य,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_graph_custom(ctx, n_nodes, असत्य);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, ध्रुव apir_rpc_tensor *> tensor_ptrs;
    प्रति (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    प्रति (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        स्मृतिं_प्रतिलिख(&id, &nodes[i], परिमाण(id));
        graph->nodes[i] = apir_create_ग्रन्थिः(id, ctx, tensor_ptrs, tensor_map);
    }

    प्रत्यागम graph;
}
