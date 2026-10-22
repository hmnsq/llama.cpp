#ضمن "backend/shared/apir_cs_rpc.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-remoting.h"

#ضمن <cinttypes>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <vector>

apir_rpc_tensor apir_serialize_tensor(ثابت ggml_tensor * tensor) {
    apir_rpc_tensor نتیجه;
    نتیجه.id   = reinterpret_cast<uint64_t>(tensor);
    نتیجه.type = tensor->type;
    إذا (tensor->میانگیر) {
        ggml_backend_میانگیر_t میانگیر = tensor->میانگیر;

        نتیجه.میانگیر = میانگیر_مقصد_HOST_HANDLE(میانگیر);
    } وإلا {
        نتیجه.میانگیر = 0;
    }
    لكل (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        نتیجه.ne[i] = tensor->ne[i];
        نتیجه.nb[i] = tensor->nb[i];
    }
    نتیجه.op = tensor->op;
    لكل (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        نتیجه.op_params[i] = tensor->op_params[i];
    }
    نتیجه.flags = tensor->flags;
    لكل (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        نتیجه.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    نتیجه.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    نتیجه.view_offs = tensor->view_offs;
    نتیجه.داده      = reinterpret_cast<uint64_t>(tensor->داده);
    إذا (tensor->داده) {
        إذا (!tensor->میانگیر) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        نتیجه.داده -= reinterpret_cast<uint64_t>(میانگیر_مقصد_GGML_CONTEXT(tensor->میانگیر)->پایه);
    }
    snprintf(نتیجه.name, GGML_MAX_NAME, "%s", tensor->name);
    أرجع نتیجه;
}

فراغ apir_جمع_کن_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & دیدهشده) {
    إذا (tensor == اشاره_تهی) {
        أرجع;
    }
    إذا (دیدهشده.find(tensor) != دیدهشده.end()) {
        أرجع;
    }
    دیدهشده.insert(tensor);
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        apir_جمع_کن_tensor(tensor->src[i], tensors, دیدهشده);
    }
    apir_جمع_کن_tensor(tensor->view_src, tensors, دیدهشده);
    tensors.push_back(apir_serialize_tensor(tensor));
}

فراغ apir_serialize_گراف(ثابت ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> دیدهشده;
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        apir_جمع_کن_tensor(cgraph->nodes[i], tensors, دیدهشده);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    صحيح      output_اندازه =
        حجم(uint32_t) + n_nodes * حجم(uint64_t) + حجم(uint32_t) + n_tensors * حجم(apir_rpc_tensor);
    output.resize(output_اندازه, 0);
    کپی_حافظه(output.داده(), &n_nodes, حجم(n_nodes));
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        کپی_حافظه(output.داده() + حجم(n_nodes) + i * حجم(uint64_t), &cgraph->nodes[i], حجم(uint64_t));
    }
    uint32_t * بیرون_ntensors = (uint32_t *) (output.داده() + حجم(n_nodes) + n_nodes * حجم(uint64_t));
    *بیرون_ntensors           = n_tensors;
    apir_rpc_tensor * بیرون_tensors =
        (apir_rpc_tensor *) (output.داده() + حجم(n_nodes) + n_nodes * حجم(uint64_t) + حجم(uint32_t));
    کپی_حافظه(بیرون_tensors, tensors.داده(), n_tensors * حجم(apir_rpc_tensor));
}
