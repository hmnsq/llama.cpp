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
    خپلکاری it = backend_buffers.find(بفر);
    إذا (it == backend_buffers.end()) {
        أرجع خطأ;
    }

    backend_buffers.erase(it);
    أرجع ریښتیا;
}

std::unordered_set<ggml_backend_بفر_t> apir_get_track_backend_buffers() {
    أرجع backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, ثابت apir_rpc_tensor * tensor) {
    ggml_tensor * پایله =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    لكل (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        پایله->nb[i] = tensor->nb[i];
    }
    پایله->بفر = reinterpret_cast<ggml_backend_بفر_t>(tensor->بفر);
    إذا (پایله->بفر && backend_buffers.find(پایله->بفر) == backend_buffers.end()) {
        چاپ_کړه("WARNING: HOST BUFFER NOT FOUND | %p\n", (فراغ *) پایله->بفر);
        پایله->بفر = تش_اشاره;
    }

    uint64_t tensor_معلومات = tensor->معلومات;
    إذا (پایله->بفر) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_اندازه  = (uint64_t) ggml_nbytes(پایله);
        uint64_t بفر_پیل = (uint64_t) ggml_backend_بفر_get_بنسټ(پایله->بفر);
        uint64_t بفر_اندازه  = (uint64_t) ggml_backend_بفر_get_اندازه(پایله->بفر);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_معلومات += بفر_پیل;

        GGML_ASSERT(tensor_معلومات + tensor_اندازه >= tensor_معلومات);  // check for overflow
        GGML_ASSERT(tensor_معلومات >= بفر_پیل && tensor_معلومات + tensor_اندازه <= بفر_پیل + بفر_اندازه);
    }

    پایله->op = (ggml_op) tensor->op;
    لكل (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        پایله->op_params[i] = tensor->op_params[i];
    }
    پایله->flags = tensor->flags;
    پایله->معلومات  = reinterpret_cast<فراغ *>(tensor_معلومات);
    ggml_سېټ_name(پایله, tensor->name);
    أرجع پایله;
}

ggml_tensor * apir_create_غوټه(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               ثابت std::unordered_map<uint64_t, ثابت apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    إذا (id == 0) {
        أرجع تش_اشاره;
    }
    إذا (tensor_map.find(id) != tensor_map.end()) {
        أرجع tensor_map[id];
    }
    ثابت apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           پایله = apir_deserialize_tensor(ctx, tensor);
    إذا (پایله == تش_اشاره) {
        أرجع تش_اشاره;
    }
    tensor_map[id] = پایله;
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        پایله->src[i] = apir_create_غوټه(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    پایله->view_src  = apir_create_غوټه(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    پایله->view_offs = tensor->view_offs;
    أرجع پایله;
}

ggml_cgraph * apir_deserialize_ګراف(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     ثابت apir_rpc_tensor * tensors,
                                     ثابت uint64_t *        nodes) {
    size_t buf_اندازه = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_ګراف_overhead_custom(n_nodes, خطأ);
    ggml_init_params params = {
        /*.mem_size   =*/buf_اندازه,
        /*.mem_buffer =*/عدم,
        /*.no_alloc   =*/ریښتیا,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_ګراف_custom(ctx, n_nodes, خطأ);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, ثابت apir_rpc_tensor *> tensor_ptrs;
    لكل (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        حافظه_نقل(&id, &nodes[i], حجم(id));
        graph->nodes[i] = apir_create_غوټه(id, ctx, tensor_ptrs, tensor_map);
    }

    أرجع graph;
}
