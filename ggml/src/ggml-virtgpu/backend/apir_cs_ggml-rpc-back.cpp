#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"
#ضمن "shared/apir_cs_rpc.h"

#ضمن <cinttypes>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <vector>

std::unordered_set<ggml_backend_بفر_t> backend_buffers;

فراغ apir_track_backend_بفر(ggml_backend_بفر_t بفر) {
    backend_buffers.insert(بفر);
}

منطقي apir_untrack_backend_بفر(ggml_backend_بفر_t بفر) {
    خودڪار it = backend_buffers.find(بفر);
    إذا (it == backend_buffers.end()) {
        أرجع خطأ;
    }

    backend_buffers.erase(it);
    أرجع سچ;
}

std::unordered_set<ggml_backend_بفر_t> apir_get_track_backend_buffers() {
    أرجع backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ثابت apir_rpc_tensor * tensor) {
    ggml_tensor * نتيجو =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    لكل (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        نتيجو->nb[i] = tensor->nb[i];
    }
    نتيجو->بفر = reinterpret_cast<ggml_backend_بفر_t>(tensor->بفر);
    إذا (نتيجو->بفر && backend_buffers.find(نتيجو->بفر) == backend_buffers.end()) {
        لکو("WARNING: HOST BUFFER NOT FOUND | %p\n", (فراغ *) نتيجو->بفر);
        نتيجو->بفر = خالي_اشارو;
    }

    uint64_t tensor_معلومات = tensor->معلومات;
    إذا (نتيجو->بفر) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_سائيز  = (uint64_t) ggml_nbytes(نتيجو);
        uint64_t بفر_شروعات = (uint64_t) ggml_backend_بفر_get_بنياد(نتيجو->بفر);
        uint64_t بفر_سائيز  = (uint64_t) ggml_backend_بفر_get_سائيز(نتيجو->بفر);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_معلومات += بفر_شروعات;

        GGML_ASSERT(tensor_معلومات + tensor_سائيز >= tensor_معلومات);  // check for overflow
        GGML_ASSERT(tensor_معلومات >= بفر_شروعات && tensor_معلومات + tensor_سائيز <= بفر_شروعات + بفر_سائيز);
    }

    نتيجو->op = (ggml_op) tensor->op;
    لكل (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        نتيجو->op_params[i] = tensor->op_params[i];
    }
    نتيجو->flags = tensor->flags;
    نتيجو->معلومات  = reinterpret_cast<فراغ *>(tensor_معلومات);
    ggml_سيٽ_name(نتيجو, tensor->name);
    أرجع نتيجو;
}

ggml_tensor * apir_create_ڳنڍ(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ثابت std::unordered_map<uint64_t, ثابت apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    إذا (id == 0) {
        أرجع خالي_اشارو;
    }
    إذا (tensor_map.find(id) != tensor_map.end()) {
        أرجع tensor_map[id];
    }
    ثابت apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           نتيجو = apir_deserialize_tensor(ctx, tensor);
    إذا (نتيجو == خالي_اشارو) {
        أرجع خالي_اشارو;
    }
    tensor_map[id] = نتيجو;
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        نتيجو->src[i] = apir_create_ڳنڍ(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    نتيجو->view_src  = apir_create_ڳنڍ(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    نتيجو->view_offs = tensor->view_offs;
    أرجع نتيجو;
}

ggml_cgraph * apir_deserialize_گراف(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ثابت apir_rpc_tensor * tensors,
                                     ثابت uint64_t *        nodes) {
    size_t buf_سائيز = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_گراف_overhead_custom(n_nodes, خطأ);
    ggml_init_params params = {
        /*.mem_size   =*/buf_سائيز,
        /*.mem_buffer =*/عدم,
        /*.no_alloc   =*/سچ,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_گراف_custom(ctx, n_nodes, خطأ);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, ثابت apir_rpc_tensor *> tensor_ptrs;
    لكل (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        يادگيري_نقل(&id, &nodes[i], حجم(id));
        graph->nodes[i] = apir_create_ڳنڍ(id, ctx, tensor_ptrs, tensor_map);
    }

    أرجع graph;
}
