#অন্তর্ভাব "backend/shared/apir_cs_rpc.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-remoting.h"

#অন্তর্ভাব <cinttypes>
#অন্তর্ভাব <unordered_map>
#অন্তর্ভাব <unordered_set>
#অন্তর্ভাব <vector>

apir_rpc_tensor apir_serialize_tensor(ধ্রুবক ggml_tensor * tensor) {
    apir_rpc_tensor ফলাফল;
    ফলাফল.id   = reinterpret_cast<uint64_t>(tensor);
    ফলাফল.type = tensor->type;
    যদি (tensor->বাফার) {
        ggml_backend_বাফার_t বাফার = tensor->বাফার;

        ফলাফল.বাফার = বাফার_TO_HOST_HANDLE(বাফার);
    } অন্যথা {
        ফলাফল.বাফার = 0;
    }
    জন্য (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ফলাফল.ne[i] = tensor->ne[i];
        ফলাফল.nb[i] = tensor->nb[i];
    }
    ফলাফল.op = tensor->op;
    জন্য (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / আকার(int32_t); i++) {
        ফলাফল.op_params[i] = tensor->op_params[i];
    }
    ফলাফল.flags = tensor->flags;
    জন্য (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        ফলাফল.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    ফলাফল.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    ফলাফল.view_offs = tensor->view_offs;
    ফলাফল.তথ্য      = reinterpret_cast<uint64_t>(tensor->তথ্য);
    যদি (tensor->তথ্য) {
        যদি (!tensor->বাফার) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        ফলাফল.তথ্য -= reinterpret_cast<uint64_t>(বাফার_TO_GGML_CONTEXT(tensor->বাফার)->base);
    }
    snprintf(ফলাফল.name, GGML_MAX_NAME, "%s", tensor->name);
    ফেরত ফলাফল;
}

শূন্য apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited) {
    যদি (tensor == nullptr) {
        ফেরত;
    }
    যদি (visited.find(tensor) != visited.end()) {
        ফেরত;
    }
    visited.insert(tensor);
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
        apir_add_tensor(tensor->src[i], tensors, visited);
    }
    apir_add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(apir_serialize_tensor(tensor));
}

শূন্য apir_serialize_graph(ধ্রুবক ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> visited;
    জন্য (uint32_t i = 0; i < n_nodes; i++) {
        apir_add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    পূর্ণসংখ্যা      output_আকার =
        আকার(uint32_t) + n_nodes * আকার(uint64_t) + আকার(uint32_t) + n_tensors * আকার(apir_rpc_tensor);
    output.resize(output_আকার, 0);
    memcpy(output.তথ্য(), &n_nodes, আকার(n_nodes));
    জন্য (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.তথ্য() + আকার(n_nodes) + i * আকার(uint64_t), &cgraph->nodes[i], আকার(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *) (output.তথ্য() + আকার(n_nodes) + n_nodes * আকার(uint64_t));
    *out_ntensors           = n_tensors;
    apir_rpc_tensor * out_tensors =
        (apir_rpc_tensor *) (output.তথ্য() + আকার(n_nodes) + n_nodes * আকার(uint64_t) + আকার(uint32_t));
    memcpy(out_tensors, tensors.তথ্য(), n_tensors * আকার(apir_rpc_tensor));
}
