#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "shared/apir_cs_rpc.h"

#అన్తర్భావ <cinttypes>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <unordered_set>
#అన్తర్భావ <vector>

std::unordered_set<ggml_backend_బఫరు_t> backend_buffers;

శూన్య apir_track_backend_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    backend_buffers.insert(బఫరు);
}

తార్కిక apir_untrack_backend_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    స్వయంచాలక it = backend_buffers.find(బఫరు);
    ఒకవేళ (it == backend_buffers.end()) {
        తిరిగి అబద్ధం;
    }

    backend_buffers.erase(it);
    తిరిగి నిజం;
}

std::unordered_set<ggml_backend_బఫరు_t> apir_get_track_backend_buffers() {
    తిరిగి backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, స్థిరరాశి apir_rpc_tensor * tensor) {
    ggml_tensor * ఫలితం =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    కొరకు (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ఫలితం->nb[i] = tensor->nb[i];
    }
    ఫలితం->బఫరు = reinterpret_cast<ggml_backend_బఫరు_t>(tensor->బఫరు);
    ఒకవేళ (ఫలితం->బఫరు && backend_buffers.find(ఫలితం->బఫరు) == backend_buffers.end()) {
        రాయి("WARNING: HOST BUFFER NOT FOUND | %p\n", (శూన్య *) ఫలితం->బఫరు);
        ఫలితం->బఫరు = శూన్యసూచిక;
    }

    uint64_t tensor_దత్తాంశం = tensor->దత్తాంశం;
    ఒకవేళ (ఫలితం->బఫరు) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_కొలత  = (uint64_t) ggml_nbytes(ఫలితం);
        uint64_t బఫరు_ప్రారంభం = (uint64_t) ggml_backend_బఫరు_get_పునాది(ఫలితం->బఫరు);
        uint64_t బఫరు_కొలత  = (uint64_t) ggml_backend_బఫరు_get_కొలత(ఫలితం->బఫరు);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_దత్తాంశం += బఫరు_ప్రారంభం;

        GGML_ASSERT(tensor_దత్తాంశం + tensor_కొలత >= tensor_దత్తాంశం);  // check for overflow
        GGML_ASSERT(tensor_దత్తాంశం >= బఫరు_ప్రారంభం && tensor_దత్తాంశం + tensor_కొలత <= బఫరు_ప్రారంభం + బఫరు_కొలత);
    }

    ఫలితం->op = (ggml_op) tensor->op;
    కొరకు (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / పరిమాణం(int32_t); i++) {
        ఫలితం->op_params[i] = tensor->op_params[i];
    }
    ఫలితం->flags = tensor->flags;
    ఫలితం->దత్తాంశం  = reinterpret_cast<శూన్య *>(tensor_దత్తాంశం);
    ggml_సమితి_name(ఫలితం, tensor->name);
    తిరిగి ఫలితం;
}

ggml_tensor * apir_create_కణుపు(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               స్థిరరాశి std::unordered_map<uint64_t, స్థిరరాశి apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    ఒకవేళ (id == 0) {
        తిరిగి శూన్యసూచిక;
    }
    ఒకవేళ (tensor_map.find(id) != tensor_map.end()) {
        తిరిగి tensor_map[id];
    }
    స్థిరరాశి apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           ఫలితం = apir_deserialize_tensor(ctx, tensor);
    ఒకవేళ (ఫలితం == శూన్యసూచిక) {
        తిరిగి శూన్యసూచిక;
    }
    tensor_map[id] = ఫలితం;
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        ఫలితం->src[i] = apir_create_కణుపు(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    ఫలితం->view_src  = apir_create_కణుపు(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    ఫలితం->view_offs = tensor->view_offs;
    తిరిగి ఫలితం;
}

ggml_cgraph * apir_deserialize_లేఖాచిత్రం(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     స్థిరరాశి apir_rpc_tensor * tensors,
                                     స్థిరరాశి uint64_t *        nodes) {
    size_t buf_కొలత = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_లేఖాచిత్రం_overhead_custom(n_nodes, అబద్ధం);
    ggml_init_params params = {
        /*.mem_size   =*/buf_కొలత,
        /*.mem_buffer =*/ఖాళీ,
        /*.no_alloc   =*/నిజం,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_లేఖాచిత్రం_custom(ctx, n_nodes, అబద్ధం);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, స్థిరరాశి apir_rpc_tensor *> tensor_ptrs;
    కొరకు (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    కొరకు (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        జ్ఞాపకం_నకలు(&id, &nodes[i], పరిమాణం(id));
        graph->nodes[i] = apir_create_కణుపు(id, ctx, tensor_ptrs, tensor_map);
    }

    తిరిగి graph;
}
