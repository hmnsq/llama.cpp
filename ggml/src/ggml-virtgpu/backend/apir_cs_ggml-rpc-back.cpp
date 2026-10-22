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
    خودکار it = backend_buffers.find(بفر);
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
    ggml_tensor * نتیجہ =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    لكل (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        نتیجہ->nb[i] = tensor->nb[i];
    }
    نتیجہ->بفر = reinterpret_cast<ggml_backend_بفر_t>(tensor->بفر);
    إذا (نتیجہ->بفر && backend_buffers.find(نتیجہ->بفر) == backend_buffers.end()) {
        لکھو("WARNING: HOST BUFFER NOT FOUND | %p\n", (فراغ *) نتیجہ->بفر);
        نتیجہ->بفر = خالی_سوچک;
    }

    uint64_t tensor_انکڑے = tensor->انکڑے;
    إذا (نتیجہ->بفر) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_ناپ  = (uint64_t) ggml_nbytes(نتیجہ);
        uint64_t بفر_شروع = (uint64_t) ggml_backend_بفر_get_ادھار(نتیجہ->بفر);
        uint64_t بفر_ناپ  = (uint64_t) ggml_backend_بفر_get_ناپ(نتیجہ->بفر);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_انکڑے += بفر_شروع;

        GGML_ASSERT(tensor_انکڑے + tensor_ناپ >= tensor_انکڑے);  // check for overflow
        GGML_ASSERT(tensor_انکڑے >= بفر_شروع && tensor_انکڑے + tensor_ناپ <= بفر_شروع + بفر_ناپ);
    }

    نتیجہ->op = (ggml_op) tensor->op;
    لكل (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        نتیجہ->op_params[i] = tensor->op_params[i];
    }
    نتیجہ->flags = tensor->flags;
    نتیجہ->انکڑے  = reinterpret_cast<فراغ *>(tensor_انکڑے);
    ggml_سیٹ_name(نتیجہ, tensor->name);
    أرجع نتیجہ;
}

ggml_tensor * apir_create_گرہ(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ثابت std::unordered_map<uint64_t, ثابت apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    إذا (id == 0) {
        أرجع خالی_سوچک;
    }
    إذا (tensor_map.find(id) != tensor_map.end()) {
        أرجع tensor_map[id];
    }
    ثابت apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           نتیجہ = apir_deserialize_tensor(ctx, tensor);
    إذا (نتیجہ == خالی_سوچک) {
        أرجع خالی_سوچک;
    }
    tensor_map[id] = نتیجہ;
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        نتیجہ->src[i] = apir_create_گرہ(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    نتیجہ->view_src  = apir_create_گرہ(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    نتیجہ->view_offs = tensor->view_offs;
    أرجع نتیجہ;
}

ggml_cgraph * apir_deserialize_گراف(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ثابت apir_rpc_tensor * tensors,
                                     ثابت uint64_t *        nodes) {
    size_t buf_ناپ = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_گراف_overhead_custom(n_nodes, خطأ);
    ggml_init_params params = {
        /*.mem_size   =*/buf_ناپ,
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
        حافظہ_نقل(&id, &nodes[i], حجم(id));
        graph->nodes[i] = apir_create_گرہ(id, ctx, tensor_ptrs, tensor_map);
    }

    أرجع graph;
}
