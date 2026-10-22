#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "shared/apir_cs_rpc.h"

#অন্তর্ভাৱ <cinttypes>
#অন্তর্ভাৱ <unordered_map>
#অন্তর্ভাৱ <unordered_set>
#অন্তর্ভাৱ <vector>

std::unordered_set<ggml_backend_buffer_t> backend_buffers;

শূন্য apir_track_backend_buffer(ggml_backend_buffer_t buffer) {
    backend_buffers.insert(buffer);
}

তার্কিক apir_untrack_backend_buffer(ggml_backend_buffer_t buffer) {
    auto it = backend_buffers.find(buffer);
    যদি (it == backend_buffers.end()) {
        ঘূৰাই মিছা;
    }

    backend_buffers.erase(it);
    ঘূৰাই সঁচা;
}

std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers() {
    ঘূৰাই backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ধ্রুৱ apir_rpc_tensor * tensor) {
    ggml_tensor * result =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    বাবে (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    যদি (result->buffer && backend_buffers.find(result->buffer) == backend_buffers.end()) {
        printf("WARNING: HOST BUFFER NOT FOUND | %p\n", (শূন্য *) result->buffer);
        result->buffer = nullptr;
    }

    uint64_t tensor_data = tensor->data;
    যদি (result->buffer) {
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
    বাবে (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / পরিমাণ(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data  = reinterpret_cast<শূন্য *>(tensor_data);
    ggml_set_name(result, tensor->name);
    ঘূৰাই result;
}

ggml_tensor * apir_create_গ্ৰন্থি(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ধ্রুৱ std::unordered_map<uint64_t, ধ্রুৱ apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    যদি (id == 0) {
        ঘূৰাই nullptr;
    }
    যদি (tensor_map.find(id) != tensor_map.end()) {
        ঘূৰাই tensor_map[id];
    }
    ধ্রুৱ apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           result = apir_deserialize_tensor(ctx, tensor);
    যদি (result == nullptr) {
        ঘূৰাই nullptr;
    }
    tensor_map[id] = result;
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
        result->src[i] = apir_create_গ্ৰন্থি(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    result->view_src  = apir_create_গ্ৰন্থি(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    result->view_offs = tensor->view_offs;
    ঘূৰাই result;
}

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ধ্রুৱ apir_rpc_tensor * tensors,
                                     ধ্রুৱ uint64_t *        nodes) {
    size_t buf_size = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, মিছা);
    ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/রিক্ত,
        /*.no_alloc   =*/সঁচা,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_graph_custom(ctx, n_nodes, মিছা);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, ধ্রুৱ apir_rpc_tensor *> tensor_ptrs;
    বাবে (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    বাবে (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], পরিমাণ(id));
        graph->nodes[i] = apir_create_গ্ৰন্থি(id, ctx, tensor_ptrs, tensor_map);
    }

    ঘূৰাই graph;
}
