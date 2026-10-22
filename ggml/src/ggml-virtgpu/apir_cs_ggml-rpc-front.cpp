#ضمن "backend/shared/apir_cs_rpc.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-remoting.h"

#ضمن <cinttypes>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <vector>

apir_rpc_tensor apir_serialize_tensor(ثابت ggml_tensor * tensor) {
    apir_rpc_tensor نتیجہ;
    نتیجہ.id   = reinterpret_cast<uint64_t>(tensor);
    نتیجہ.type = tensor->type;
    إذا (tensor->بفر) {
        ggml_backend_بفر_t بفر = tensor->بفر;

        نتیجہ.بفر = بفر_منزل_HOST_HANDLE(بفر);
    } وإلا {
        نتیجہ.بفر = 0;
    }
    لكل (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        نتیجہ.ne[i] = tensor->ne[i];
        نتیجہ.nb[i] = tensor->nb[i];
    }
    نتیجہ.op = tensor->op;
    لكل (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        نتیجہ.op_params[i] = tensor->op_params[i];
    }
    نتیجہ.flags = tensor->flags;
    لكل (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        نتیجہ.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    نتیجہ.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    نتیجہ.view_offs = tensor->view_offs;
    نتیجہ.انکڑے      = reinterpret_cast<uint64_t>(tensor->انکڑے);
    إذا (tensor->انکڑے) {
        إذا (!tensor->بفر) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        نتیجہ.انکڑے -= reinterpret_cast<uint64_t>(بفر_منزل_GGML_CONTEXT(tensor->بفر)->ادھار);
    }
    snprintf(نتیجہ.name, GGML_MAX_NAME, "%s", tensor->name);
    أرجع نتیجہ;
}

فراغ apir_جوڑو_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & ویکھیا) {
    إذا (tensor == خالی_سوچک) {
        أرجع;
    }
    إذا (ویکھیا.find(tensor) != ویکھیا.end()) {
        أرجع;
    }
    ویکھیا.insert(tensor);
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        apir_جوڑو_tensor(tensor->src[i], tensors, ویکھیا);
    }
    apir_جوڑو_tensor(tensor->view_src, tensors, ویکھیا);
    tensors.push_back(apir_serialize_tensor(tensor));
}

فراغ apir_serialize_گراف(ثابت ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> ویکھیا;
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        apir_جوڑو_tensor(cgraph->nodes[i], tensors, ویکھیا);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    صحيح      output_ناپ =
        حجم(uint32_t) + n_nodes * حجم(uint64_t) + حجم(uint32_t) + n_tensors * حجم(apir_rpc_tensor);
    output.resize(output_ناپ, 0);
    حافظہ_نقل(output.انکڑے(), &n_nodes, حجم(n_nodes));
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        حافظہ_نقل(output.انکڑے() + حجم(n_nodes) + i * حجم(uint64_t), &cgraph->nodes[i], حجم(uint64_t));
    }
    uint32_t * باہرلا_ntensors = (uint32_t *) (output.انکڑے() + حجم(n_nodes) + n_nodes * حجم(uint64_t));
    *باہرلا_ntensors           = n_tensors;
    apir_rpc_tensor * باہرلا_tensors =
        (apir_rpc_tensor *) (output.انکڑے() + حجم(n_nodes) + n_nodes * حجم(uint64_t) + حجم(uint32_t));
    حافظہ_نقل(باہرلا_tensors, tensors.انکڑے(), n_tensors * حجم(apir_rpc_tensor));
}
