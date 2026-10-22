#समावेश "ggml-backend-impl.h"
#समावेश "ggml-impl.h"
#समावेश "shared/apir_cs_rpc.h"

#समावेश <cinttypes>
#समावेश <unordered_map>
#समावेश <unordered_set>
#समावेश <vector>

std::unordered_set<ggml_backend_बफर_t> backend_buffers;

व्योम apir_track_backend_बफर(ggml_backend_बफर_t बफर) {
    backend_buffers.insert(बफर);
}

bool apir_untrack_backend_बफर(ggml_backend_बफर_t बफर) {
    स्वतः it = backend_buffers.find(बफर);
    अगर (it == backend_buffers.end()) {
        वापस false;
    }

    backend_buffers.erase(it);
    वापस true;
}

std::unordered_set<ggml_backend_बफर_t> apir_get_track_backend_buffers() {
    वापस backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, स्थिर apir_rpc_tensor * tensor) {
    ggml_tensor * परिणाम =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    क्रम (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        परिणाम->nb[i] = tensor->nb[i];
    }
    परिणाम->बफर = reinterpret_cast<ggml_backend_बफर_t>(tensor->बफर);
    अगर (परिणाम->बफर && backend_buffers.find(परिणाम->बफर) == backend_buffers.end()) {
        म_लिखो("WARNING: HOST BUFFER NOT FOUND | %p\n", (व्योम *) परिणाम->बफर);
        परिणाम->बफर = nullptr;
    }

    uint64_t tensor_तथ्य = tensor->तथ्य;
    अगर (परिणाम->बफर) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_आकार  = (uint64_t) ggml_nbytes(परिणाम);
        uint64_t बफर_start = (uint64_t) ggml_backend_बफर_get_base(परिणाम->बफर);
        uint64_t बफर_आकार  = (uint64_t) ggml_backend_बफर_get_आकार(परिणाम->बफर);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_तथ्य += बफर_start;

        GGML_ASSERT(tensor_तथ्य + tensor_आकार >= tensor_तथ्य);  // check for overflow
        GGML_ASSERT(tensor_तथ्य >= बफर_start && tensor_तथ्य + tensor_आकार <= बफर_start + बफर_आकार);
    }

    परिणाम->op = (ggml_op) tensor->op;
    क्रम (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / माप(int32_t); i++) {
        परिणाम->op_params[i] = tensor->op_params[i];
    }
    परिणाम->flags = tensor->flags;
    परिणाम->तथ्य  = reinterpret_cast<व्योम *>(tensor_तथ्य);
    ggml_set_name(परिणाम, tensor->name);
    वापस परिणाम;
}

ggml_tensor * apir_create_गाँठ(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               स्थिर std::unordered_map<uint64_t, स्थिर apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    अगर (id == 0) {
        वापस nullptr;
    }
    अगर (tensor_map.find(id) != tensor_map.end()) {
        वापस tensor_map[id];
    }
    स्थिर apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           परिणाम = apir_deserialize_tensor(ctx, tensor);
    अगर (परिणाम == nullptr) {
        वापस nullptr;
    }
    tensor_map[id] = परिणाम;
    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        परिणाम->src[i] = apir_create_गाँठ(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    परिणाम->view_src  = apir_create_गाँठ(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    परिणाम->view_offs = tensor->view_offs;
    वापस परिणाम;
}

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     स्थिर apir_rpc_tensor * tensors,
                                     स्थिर uint64_t *        nodes) {
    माप_प्रकार buf_आकार = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    ggml_init_params params = {
        /*.mem_size   =*/buf_आकार,
        /*.mem_buffer =*/शून्य,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, स्थिर apir_rpc_tensor *> tensor_ptrs;
    क्रम (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    क्रम (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        स_नकल(&id, &nodes[i], माप(id));
        graph->nodes[i] = apir_create_गाँठ(id, ctx, tensor_ptrs, tensor_map);
    }

    वापस graph;
}
