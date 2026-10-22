#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "shared/apir_cs_rpc.h"

#অন্তর্ভাব <cinttypes>
#অন্তর্ভাব <unordered_map>
#অন্তর্ভাব <unordered_set>
#অন্তর্ভাব <vector>

std::unordered_set<ggml_backend_বাফার_t> backend_buffers;

শূন্য apir_track_backend_বাফার(ggml_backend_বাফার_t বাফার) {
    backend_buffers.insert(বাফার);
}

যৌক্তিক apir_untrack_backend_বাফার(ggml_backend_বাফার_t বাফার) {
    auto it = backend_buffers.find(বাফার);
    যদি (it == backend_buffers.end()) {
        ফেরত মিথ্যা;
    }

    backend_buffers.erase(it);
    ফেরত সত্য;
}

std::unordered_set<ggml_backend_বাফার_t> apir_get_track_backend_buffers() {
    ফেরত backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ধ্রুবক apir_rpc_tensor * tensor) {
    ggml_tensor * ফলাফল =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    জন্য (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ফলাফল->nb[i] = tensor->nb[i];
    }
    ফলাফল->বাফার = reinterpret_cast<ggml_backend_বাফার_t>(tensor->বাফার);
    যদি (ফলাফল->বাফার && backend_buffers.find(ফলাফল->বাফার) == backend_buffers.end()) {
        printf("WARNING: HOST BUFFER NOT FOUND | %p\n", (শূন্য *) ফলাফল->বাফার);
        ফলাফল->বাফার = nullptr;
    }

    uint64_t tensor_তথ্য = tensor->তথ্য;
    যদি (ফলাফল->বাফার) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_আকার  = (uint64_t) ggml_nbytes(ফলাফল);
        uint64_t বাফার_start = (uint64_t) ggml_backend_বাফার_get_base(ফলাফল->বাফার);
        uint64_t বাফার_আকার  = (uint64_t) ggml_backend_বাফার_get_আকার(ফলাফল->বাফার);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_তথ্য += বাফার_start;

        GGML_ASSERT(tensor_তথ্য + tensor_আকার >= tensor_তথ্য);  // check for overflow
        GGML_ASSERT(tensor_তথ্য >= বাফার_start && tensor_তথ্য + tensor_আকার <= বাফার_start + বাফার_আকার);
    }

    ফলাফল->op = (ggml_op) tensor->op;
    জন্য (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / আকার(int32_t); i++) {
        ফলাফল->op_params[i] = tensor->op_params[i];
    }
    ফলাফল->flags = tensor->flags;
    ফলাফল->তথ্য  = reinterpret_cast<শূন্য *>(tensor_তথ্য);
    ggml_set_name(ফলাফল, tensor->name);
    ফেরত ফলাফল;
}

ggml_tensor * apir_create_গ্রন্থি(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ধ্রুবক std::unordered_map<uint64_t, ধ্রুবক apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    যদি (id == 0) {
        ফেরত nullptr;
    }
    যদি (tensor_map.find(id) != tensor_map.end()) {
        ফেরত tensor_map[id];
    }
    ধ্রুবক apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           ফলাফল = apir_deserialize_tensor(ctx, tensor);
    যদি (ফলাফল == nullptr) {
        ফেরত nullptr;
    }
    tensor_map[id] = ফলাফল;
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
        ফলাফল->src[i] = apir_create_গ্রন্থি(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    ফলাফল->view_src  = apir_create_গ্রন্থি(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    ফলাফল->view_offs = tensor->view_offs;
    ফেরত ফলাফল;
}

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ধ্রুবক apir_rpc_tensor * tensors,
                                     ধ্রুবক uint64_t *        nodes) {
    size_t buf_আকার = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, মিথ্যা);
    ggml_init_params params = {
        /*.mem_size   =*/buf_আকার,
        /*.mem_buffer =*/ফাঁকা,
        /*.no_alloc   =*/সত্য,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_graph_custom(ctx, n_nodes, মিথ্যা);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, ধ্রুবক apir_rpc_tensor *> tensor_ptrs;
    জন্য (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    জন্য (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], আকার(id));
        graph->nodes[i] = apir_create_গ্রন্থি(id, ctx, tensor_ptrs, tensor_map);
    }

    ফেরত graph;
}
