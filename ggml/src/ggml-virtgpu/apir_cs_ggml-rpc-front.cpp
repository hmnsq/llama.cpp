#ضمن "backend/shared/apir_cs_rpc.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-remoting.h"

#ضمن <cinttypes>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <vector>

apir_rpc_tensor apir_serialize_tensor(ثابت ggml_tensor * tensor) {
    apir_rpc_tensor پایله;
    پایله.id   = reinterpret_cast<uint64_t>(tensor);
    پایله.type = tensor->type;
    إذا (tensor->بفر) {
        ggml_backend_بفر_t بفر = tensor->بفر;

        پایله.بفر = بفر_منزل_HOST_HANDLE(بفر);
    } وإلا {
        پایله.بفر = 0;
    }
    لكل (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        پایله.ne[i] = tensor->ne[i];
        پایله.nb[i] = tensor->nb[i];
    }
    پایله.op = tensor->op;
    لكل (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        پایله.op_params[i] = tensor->op_params[i];
    }
    پایله.flags = tensor->flags;
    لكل (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        پایله.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    پایله.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    پایله.view_offs = tensor->view_offs;
    پایله.معلومات      = reinterpret_cast<uint64_t>(tensor->معلومات);
    إذا (tensor->معلومات) {
        إذا (!tensor->بفر) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        پایله.معلومات -= reinterpret_cast<uint64_t>(بفر_منزل_GGML_CONTEXT(tensor->بفر)->بنسټ);
    }
    snprintf(پایله.name, GGML_MAX_NAME, "%s", tensor->name);
    أرجع پایله;
}

فراغ apir_جمع_کړه_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & لیدل_شوی) {
    إذا (tensor == تش_اشاره) {
        أرجع;
    }
    إذا (لیدل_شوی.find(tensor) != لیدل_شوی.end()) {
        أرجع;
    }
    لیدل_شوی.insert(tensor);
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        apir_جمع_کړه_tensor(tensor->src[i], tensors, لیدل_شوی);
    }
    apir_جمع_کړه_tensor(tensor->view_src, tensors, لیدل_شوی);
    tensors.push_back(apir_serialize_tensor(tensor));
}

فراغ apir_serialize_ګراف(ثابت ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> لیدل_شوی;
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        apir_جمع_کړه_tensor(cgraph->nodes[i], tensors, لیدل_شوی);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    صحيح      output_اندازه =
        حجم(uint32_t) + n_nodes * حجم(uint64_t) + حجم(uint32_t) + n_tensors * حجم(apir_rpc_tensor);
    output.resize(output_اندازه, 0);
    حافظه_نقل(output.معلومات(), &n_nodes, حجم(n_nodes));
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        حافظه_نقل(output.معلومات() + حجم(n_nodes) + i * حجم(uint64_t), &cgraph->nodes[i], حجم(uint64_t));
    }
    uint32_t * بهر_ntensors = (uint32_t *) (output.معلومات() + حجم(n_nodes) + n_nodes * حجم(uint64_t));
    *بهر_ntensors           = n_tensors;
    apir_rpc_tensor * بهر_tensors =
        (apir_rpc_tensor *) (output.معلومات() + حجم(n_nodes) + n_nodes * حجم(uint64_t) + حجم(uint32_t));
    حافظه_نقل(بهر_tensors, tensors.معلومات(), n_tensors * حجم(apir_rpc_tensor));
}
