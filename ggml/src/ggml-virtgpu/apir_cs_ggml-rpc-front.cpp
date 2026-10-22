#समावेश "backend/shared/apir_cs_rpc.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-remoting.h"

#समावेश <cinttypes>
#समावेश <unordered_map>
#समावेश <unordered_set>
#समावेश <vector>

apir_rpc_tensor apir_serialize_tensor(स्थिर ggml_tensor * tensor) {
    apir_rpc_tensor परिणाम;
    परिणाम.id   = reinterpret_cast<uint64_t>(tensor);
    परिणाम.type = tensor->type;
    अगर (tensor->बफर) {
        ggml_backend_बफर_t बफर = tensor->बफर;

        परिणाम.बफर = बफर_TO_HOST_HANDLE(बफर);
    } अन्यथा {
        परिणाम.बफर = 0;
    }
    क्रम (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        परिणाम.ne[i] = tensor->ne[i];
        परिणाम.nb[i] = tensor->nb[i];
    }
    परिणाम.op = tensor->op;
    क्रम (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / माप(int32_t); i++) {
        परिणाम.op_params[i] = tensor->op_params[i];
    }
    परिणाम.flags = tensor->flags;
    क्रम (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        परिणाम.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    परिणाम.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    परिणाम.view_offs = tensor->view_offs;
    परिणाम.तथ्य      = reinterpret_cast<uint64_t>(tensor->तथ्य);
    अगर (tensor->तथ्य) {
        अगर (!tensor->बफर) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        परिणाम.तथ्य -= reinterpret_cast<uint64_t>(बफर_TO_GGML_CONTEXT(tensor->बफर)->base);
    }
    snprintf(परिणाम.name, GGML_MAX_NAME, "%s", tensor->name);
    वापस परिणाम;
}

व्योम apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited) {
    अगर (tensor == nullptr) {
        वापस;
    }
    अगर (visited.find(tensor) != visited.end()) {
        वापस;
    }
    visited.insert(tensor);
    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        apir_add_tensor(tensor->src[i], tensors, visited);
    }
    apir_add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(apir_serialize_tensor(tensor));
}

व्योम apir_serialize_graph(स्थिर ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> visited;
    क्रम (uint32_t i = 0; i < n_nodes; i++) {
        apir_add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    पूर्णांक      output_आकार =
        माप(uint32_t) + n_nodes * माप(uint64_t) + माप(uint32_t) + n_tensors * माप(apir_rpc_tensor);
    output.resize(output_आकार, 0);
    स_नकल(output.तथ्य(), &n_nodes, माप(n_nodes));
    क्रम (uint32_t i = 0; i < n_nodes; i++) {
        स_नकल(output.तथ्य() + माप(n_nodes) + i * माप(uint64_t), &cgraph->nodes[i], माप(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *) (output.तथ्य() + माप(n_nodes) + n_nodes * माप(uint64_t));
    *out_ntensors           = n_tensors;
    apir_rpc_tensor * out_tensors =
        (apir_rpc_tensor *) (output.तथ्य() + माप(n_nodes) + n_nodes * माप(uint64_t) + माप(uint32_t));
    स_नकल(out_tensors, tensors.तथ्य(), n_tensors * माप(apir_rpc_tensor));
}
