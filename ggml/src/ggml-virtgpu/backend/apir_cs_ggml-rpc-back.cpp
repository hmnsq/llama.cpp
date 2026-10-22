#شامل "ggml-backend-impl.h"
#شامل "ggml-impl.h"
#شامل "shared/apir_cs_rpc.h"

#شامل <cinttypes>
#شامل <unordered_map>
#شامل <unordered_set>
#شامل <vector>

std::unordered_set<ggml_backend_بفر_t> backend_buffers;

خالی apir_track_backend_بفر(ggml_backend_بفر_t بفر) {
    backend_buffers.insert(بفر);
}

منطقی apir_untrack_backend_بفر(ggml_backend_بفر_t بفر) {
    خودکار it = backend_buffers.find(بفر);
    اگر (it == backend_buffers.end()) {
        واپس جھوٹ;
    }

    backend_buffers.erase(it);
    واپس سچ;
}

std::unordered_set<ggml_backend_بفر_t> apir_get_track_backend_buffers() {
    واپس backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ثابت apir_rpc_tensor * tensor) {
    ggml_tensor * نتیجہ =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    برائے (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        نتیجہ->nb[i] = tensor->nb[i];
    }
    نتیجہ->بفر = reinterpret_cast<ggml_backend_بفر_t>(tensor->بفر);
    اگر (نتیجہ->بفر && backend_buffers.find(نتیجہ->بفر) == backend_buffers.end()) {
        لکھو("WARNING: HOST BUFFER NOT FOUND | %p\n", (خالی *) نتیجہ->بفر);
        نتیجہ->بفر = خالی_اشاریہ;
    }

    uint64_t tensor_ڈیٹا = tensor->ڈیٹا;
    اگر (نتیجہ->بفر) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_سائز  = (uint64_t) ggml_nbytes(نتیجہ);
        uint64_t بفر_آغاز = (uint64_t) ggml_backend_بفر_get_بنیاد(نتیجہ->بفر);
        uint64_t بفر_سائز  = (uint64_t) ggml_backend_بفر_get_سائز(نتیجہ->بفر);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_ڈیٹا += بفر_آغاز;

        GGML_ASSERT(tensor_ڈیٹا + tensor_سائز >= tensor_ڈیٹا);  // check for overflow
        GGML_ASSERT(tensor_ڈیٹا >= بفر_آغاز && tensor_ڈیٹا + tensor_سائز <= بفر_آغاز + بفر_سائز);
    }

    نتیجہ->op = (ggml_op) tensor->op;
    برائے (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        نتیجہ->op_params[i] = tensor->op_params[i];
    }
    نتیجہ->flags = tensor->flags;
    نتیجہ->ڈیٹا  = reinterpret_cast<خالی *>(tensor_ڈیٹا);
    ggml_سیٹ_name(نتیجہ, tensor->name);
    واپس نتیجہ;
}

ggml_tensor * apir_create_گرہ(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ثابت std::unordered_map<uint64_t, ثابت apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    اگر (id == 0) {
        واپس خالی_اشاریہ;
    }
    اگر (tensor_map.find(id) != tensor_map.end()) {
        واپس tensor_map[id];
    }
    ثابت apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           نتیجہ = apir_deserialize_tensor(ctx, tensor);
    اگر (نتیجہ == خالی_اشاریہ) {
        واپس خالی_اشاریہ;
    }
    tensor_map[id] = نتیجہ;
    برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        نتیجہ->src[i] = apir_create_گرہ(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    نتیجہ->view_src  = apir_create_گرہ(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    نتیجہ->view_offs = tensor->view_offs;
    واپس نتیجہ;
}

ggml_cgraph * apir_deserialize_گراف(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ثابت apir_rpc_tensor * tensors,
                                     ثابت uint64_t *        nodes) {
    size_t buf_سائز = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_گراف_overhead_custom(n_nodes, جھوٹ);
    ggml_init_params params = {
        /*.mem_size   =*/buf_سائز,
        /*.mem_buffer =*/تہی,
        /*.no_alloc   =*/سچ,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_گراف_custom(ctx, n_nodes, جھوٹ);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, ثابت apir_rpc_tensor *> tensor_ptrs;
    برائے (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    برائے (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        حافظہ_نقل(&id, &nodes[i], حجم(id));
        graph->nodes[i] = apir_create_گرہ(id, ctx, tensor_ptrs, tensor_map);
    }

    واپس graph;
}
