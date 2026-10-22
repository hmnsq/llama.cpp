#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"
#ضمن "shared/apir_cs_rpc.h"

#ضمن <cinttypes>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <vector>

std::unordered_set<ggml_backend_میانگیر_t> backend_buffers;

فراغ apir_track_backend_میانگیر(ggml_backend_میانگیر_t میانگیر) {
    backend_buffers.insert(میانگیر);
}

منطقي apir_untrack_backend_میانگیر(ggml_backend_میانگیر_t میانگیر) {
    خودکار it = backend_buffers.find(میانگیر);
    إذا (it == backend_buffers.end()) {
        أرجع خطأ;
    }

    backend_buffers.erase(it);
    أرجع درست;
}

std::unordered_set<ggml_backend_میانگیر_t> apir_get_track_backend_buffers() {
    أرجع backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ثابت apir_rpc_tensor * tensor) {
    ggml_tensor * نتیجه =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    لكل (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        نتیجه->nb[i] = tensor->nb[i];
    }
    نتیجه->میانگیر = reinterpret_cast<ggml_backend_میانگیر_t>(tensor->میانگیر);
    إذا (نتیجه->میانگیر && backend_buffers.find(نتیجه->میانگیر) == backend_buffers.end()) {
        چاپ_کن("WARNING: HOST BUFFER NOT FOUND | %p\n", (فراغ *) نتیجه->میانگیر);
        نتیجه->میانگیر = اشاره_تهی;
    }

    uint64_t tensor_داده = tensor->داده;
    إذا (نتیجه->میانگیر) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_اندازه  = (uint64_t) ggml_nbytes(نتیجه);
        uint64_t میانگیر_شروع = (uint64_t) ggml_backend_میانگیر_get_پایه(نتیجه->میانگیر);
        uint64_t میانگیر_اندازه  = (uint64_t) ggml_backend_میانگیر_get_اندازه(نتیجه->میانگیر);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_داده += میانگیر_شروع;

        GGML_ASSERT(tensor_داده + tensor_اندازه >= tensor_داده);  // check for overflow
        GGML_ASSERT(tensor_داده >= میانگیر_شروع && tensor_داده + tensor_اندازه <= میانگیر_شروع + میانگیر_اندازه);
    }

    نتیجه->op = (ggml_op) tensor->op;
    لكل (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        نتیجه->op_params[i] = tensor->op_params[i];
    }
    نتیجه->flags = tensor->flags;
    نتیجه->داده  = reinterpret_cast<فراغ *>(tensor_داده);
    ggml_مجموعه_name(نتیجه, tensor->name);
    أرجع نتیجه;
}

ggml_tensor * apir_create_گره(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ثابت std::unordered_map<uint64_t, ثابت apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    إذا (id == 0) {
        أرجع اشاره_تهی;
    }
    إذا (tensor_map.find(id) != tensor_map.end()) {
        أرجع tensor_map[id];
    }
    ثابت apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           نتیجه = apir_deserialize_tensor(ctx, tensor);
    إذا (نتیجه == اشاره_تهی) {
        أرجع اشاره_تهی;
    }
    tensor_map[id] = نتیجه;
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        نتیجه->src[i] = apir_create_گره(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    نتیجه->view_src  = apir_create_گره(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    نتیجه->view_offs = tensor->view_offs;
    أرجع نتیجه;
}

ggml_cgraph * apir_deserialize_گراف(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ثابت apir_rpc_tensor * tensors,
                                     ثابت uint64_t *        nodes) {
    size_t buf_اندازه = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_گراف_overhead_custom(n_nodes, خطأ);
    ggml_init_params params = {
        /*.mem_size   =*/buf_اندازه,
        /*.mem_buffer =*/عدم,
        /*.no_alloc   =*/درست,
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
        کپی_حافظه(&id, &nodes[i], حجم(id));
        graph->nodes[i] = apir_create_گره(id, ctx, tensor_ptrs, tensor_map);
    }

    أرجع graph;
}
