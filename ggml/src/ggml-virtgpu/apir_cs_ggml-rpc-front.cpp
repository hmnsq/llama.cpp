#شامل "backend/shared/apir_cs_rpc.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-impl.h"
#شامل "ggml-remoting.h"

#شامل <cinttypes>
#شامل <unordered_map>
#شامل <unordered_set>
#شامل <vector>

apir_rpc_tensor apir_serialize_tensor(ثابت ggml_tensor * tensor) {
    apir_rpc_tensor نتیجہ;
    نتیجہ.id   = reinterpret_cast<uint64_t>(tensor);
    نتیجہ.type = tensor->type;
    اگر (tensor->بفر) {
        ggml_backend_بفر_t بفر = tensor->بفر;

        نتیجہ.بفر = بفر_مقصد_HOST_HANDLE(بفر);
    } ورنہ {
        نتیجہ.بفر = 0;
    }
    برائے (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        نتیجہ.ne[i] = tensor->ne[i];
        نتیجہ.nb[i] = tensor->nb[i];
    }
    نتیجہ.op = tensor->op;
    برائے (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        نتیجہ.op_params[i] = tensor->op_params[i];
    }
    نتیجہ.flags = tensor->flags;
    برائے (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        نتیجہ.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    نتیجہ.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    نتیجہ.view_offs = tensor->view_offs;
    نتیجہ.ڈیٹا      = reinterpret_cast<uint64_t>(tensor->ڈیٹا);
    اگر (tensor->ڈیٹا) {
        اگر (!tensor->بفر) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        نتیجہ.ڈیٹا -= reinterpret_cast<uint64_t>(بفر_مقصد_GGML_CONTEXT(tensor->بفر)->بنیاد);
    }
    snprintf(نتیجہ.name, GGML_MAX_NAME, "%s", tensor->name);
    واپس نتیجہ;
}

خالی apir_جوڑو_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & دیکھا) {
    اگر (tensor == خالی_اشاریہ) {
        واپس;
    }
    اگر (دیکھا.find(tensor) != دیکھا.end()) {
        واپس;
    }
    دیکھا.insert(tensor);
    برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        apir_جوڑو_tensor(tensor->src[i], tensors, دیکھا);
    }
    apir_جوڑو_tensor(tensor->view_src, tensors, دیکھا);
    tensors.push_back(apir_serialize_tensor(tensor));
}

خالی apir_serialize_گراف(ثابت ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> دیکھا;
    برائے (uint32_t i = 0; i < n_nodes; i++) {
        apir_جوڑو_tensor(cgraph->nodes[i], tensors, دیکھا);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    صحیح      output_سائز =
        حجم(uint32_t) + n_nodes * حجم(uint64_t) + حجم(uint32_t) + n_tensors * حجم(apir_rpc_tensor);
    output.resize(output_سائز, 0);
    حافظہ_نقل(output.ڈیٹا(), &n_nodes, حجم(n_nodes));
    برائے (uint32_t i = 0; i < n_nodes; i++) {
        حافظہ_نقل(output.ڈیٹا() + حجم(n_nodes) + i * حجم(uint64_t), &cgraph->nodes[i], حجم(uint64_t));
    }
    uint32_t * برآمد_ntensors = (uint32_t *) (output.ڈیٹا() + حجم(n_nodes) + n_nodes * حجم(uint64_t));
    *برآمد_ntensors           = n_tensors;
    apir_rpc_tensor * برآمد_tensors =
        (apir_rpc_tensor *) (output.ڈیٹا() + حجم(n_nodes) + n_nodes * حجم(uint64_t) + حجم(uint32_t));
    حافظہ_نقل(برآمد_tensors, tensors.ڈیٹا(), n_tensors * حجم(apir_rpc_tensor));
}
