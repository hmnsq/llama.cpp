#ضمن "backend/shared/apir_cs_rpc.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-remoting.h"

#ضمن <cinttypes>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <vector>

apir_rpc_tensor apir_serialize_tensor(ثابت ggml_tensor * tensor) {
    apir_rpc_tensor نتيجو;
    نتيجو.id   = reinterpret_cast<uint64_t>(tensor);
    نتيجو.type = tensor->type;
    إذا (tensor->بفر) {
        ggml_backend_بفر_t بفر = tensor->بفر;

        نتيجو.بفر = بفر_منزل_HOST_HANDLE(بفر);
    } وإلا {
        نتيجو.بفر = 0;
    }
    لكل (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        نتيجو.ne[i] = tensor->ne[i];
        نتيجو.nb[i] = tensor->nb[i];
    }
    نتيجو.op = tensor->op;
    لكل (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        نتيجو.op_params[i] = tensor->op_params[i];
    }
    نتيجو.flags = tensor->flags;
    لكل (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        نتيجو.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    نتيجو.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    نتيجو.view_offs = tensor->view_offs;
    نتيجو.معلومات      = reinterpret_cast<uint64_t>(tensor->معلومات);
    إذا (tensor->معلومات) {
        إذا (!tensor->بفر) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        نتيجو.معلومات -= reinterpret_cast<uint64_t>(بفر_منزل_GGML_CONTEXT(tensor->بفر)->بنياد);
    }
    snprintf(نتيجو.name, GGML_MAX_NAME, "%s", tensor->name);
    أرجع نتيجو;
}

فراغ apir_جوڙيو_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & گهميل) {
    إذا (tensor == خالي_اشارو) {
        أرجع;
    }
    إذا (گهميل.find(tensor) != گهميل.end()) {
        أرجع;
    }
    گهميل.insert(tensor);
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        apir_جوڙيو_tensor(tensor->src[i], tensors, گهميل);
    }
    apir_جوڙيو_tensor(tensor->view_src, tensors, گهميل);
    tensors.push_back(apir_serialize_tensor(tensor));
}

فراغ apir_serialize_گراف(ثابت ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> گهميل;
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        apir_جوڙيو_tensor(cgraph->nodes[i], tensors, گهميل);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    صحيح      output_سائيز =
        حجم(uint32_t) + n_nodes * حجم(uint64_t) + حجم(uint32_t) + n_tensors * حجم(apir_rpc_tensor);
    output.resize(output_سائيز, 0);
    يادگيري_نقل(output.معلومات(), &n_nodes, حجم(n_nodes));
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        يادگيري_نقل(output.معلومات() + حجم(n_nodes) + i * حجم(uint64_t), &cgraph->nodes[i], حجم(uint64_t));
    }
    uint32_t * ٻاهريون_ntensors = (uint32_t *) (output.معلومات() + حجم(n_nodes) + n_nodes * حجم(uint64_t));
    *ٻاهريون_ntensors           = n_tensors;
    apir_rpc_tensor * ٻاهريون_tensors =
        (apir_rpc_tensor *) (output.معلومات() + حجم(n_nodes) + n_nodes * حجم(uint64_t) + حجم(uint32_t));
    يادگيري_نقل(ٻاهريون_tensors, tensors.معلومات(), n_tensors * حجم(apir_rpc_tensor));
}
