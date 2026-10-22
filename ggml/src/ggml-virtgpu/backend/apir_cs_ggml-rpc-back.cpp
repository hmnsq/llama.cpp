#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"
#ضمن "shared/apir_cs_rpc.h"

#ضمن <cinttypes>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <vector>

std::unordered_set<ggml_backend_buffer_t> backend_buffers;

فراغ apir_track_backend_buffer(ggml_backend_buffer_t buffer) {
    backend_buffers.insert(buffer);
}

منطقي apir_untrack_backend_buffer(ggml_backend_buffer_t buffer) {
    auto it = backend_buffers.find(buffer);
    𐡄𐡍 (it == backend_buffers.end()) {
        أرجع خطأ;
    }

    backend_buffers.erase(it);
    أرجع 𐡒𐡔𐡈;
}

std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers() {
    أرجع backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ثابت apir_rpc_tensor * tensor) {
    ggml_tensor * result =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    لكل (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    𐡄𐡍 (result->buffer && backend_buffers.find(result->buffer) == backend_buffers.end()) {
        printf("WARNING: HOST BUFFER NOT FOUND | %p\n", (فراغ *) result->buffer);
        result->buffer = nullptr;
    }

    uint64_t tensor_data = tensor->data;
    𐡄𐡍 (result->buffer) {
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
    لكل (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data  = reinterpret_cast<فراغ *>(tensor_data);
    ggml_set_name(result, tensor->name);
    أرجع result;
}

ggml_tensor * apir_create_node(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ثابت std::unordered_map<uint64_t, ثابت apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    𐡄𐡍 (id == 0) {
        أرجع nullptr;
    }
    𐡄𐡍 (tensor_map.find(id) != tensor_map.end()) {
        أرجع tensor_map[id];
    }
    ثابت apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           result = apir_deserialize_tensor(ctx, tensor);
    𐡄𐡍 (result == nullptr) {
        أرجع nullptr;
    }
    tensor_map[id] = result;
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        result->src[i] = apir_create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    result->view_src  = apir_create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    result->view_offs = tensor->view_offs;
    أرجع result;
}

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ثابت apir_rpc_tensor * tensors,
                                     ثابت uint64_t *        nodes) {
    size_t buf_size = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, خطأ);
    ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/عدم,
        /*.no_alloc   =*/𐡒𐡔𐡈,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_graph_custom(ctx, n_nodes, خطأ);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, ثابت apir_rpc_tensor *> tensor_ptrs;
    لكل (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], حجم(id));
        graph->nodes[i] = apir_create_node(id, ctx, tensor_ptrs, tensor_map);
    }

    أرجع graph;
}
