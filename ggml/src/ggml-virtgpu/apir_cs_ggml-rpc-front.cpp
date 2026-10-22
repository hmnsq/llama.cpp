#अन्तर्भाव "backend/shared/apir_cs_rpc.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-remoting.h"

#अन्तर्भाव <cinttypes>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <unordered_set>
#अन्तर्भाव <vector>

apir_rpc_tensor apir_serialize_tensor(ध्रुव ggml_tensor * tensor) {
    apir_rpc_tensor परिणाम;
    परिणाम.id   = reinterpret_cast<uint64_t>(tensor);
    परिणाम.type = tensor->type;
    यदि (tensor->बफर) {
        ggml_backend_बफर_t बफर = tensor->बफर;

        परिणाम.बफर = बफर_गन्तव्यम्_HOST_HANDLE(बफर);
    } अन्यथा {
        परिणाम.बफर = 0;
    }
    प्रति (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        परिणाम.ne[i] = tensor->ne[i];
        परिणाम.nb[i] = tensor->nb[i];
    }
    परिणाम.op = tensor->op;
    प्रति (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / परिमाण(int32_t); i++) {
        परिणाम.op_params[i] = tensor->op_params[i];
    }
    परिणाम.flags = tensor->flags;
    प्रति (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        परिणाम.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    परिणाम.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    परिणाम.view_offs = tensor->view_offs;
    परिणाम.दत्त      = reinterpret_cast<uint64_t>(tensor->दत्त);
    यदि (tensor->दत्त) {
        यदि (!tensor->बफर) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        परिणाम.दत्त -= reinterpret_cast<uint64_t>(बफर_गन्तव्यम्_GGML_CONTEXT(tensor->बफर)->आधारः);
    }
    snprintf(परिणाम.name, GGML_MAX_NAME, "%s", tensor->name);
    प्रत्यागम परिणाम;
}

शून्य apir_सङ्कलय_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited) {
    यदि (tensor == शून्यसूचकः) {
        प्रत्यागम;
    }
    यदि (visited.find(tensor) != visited.end()) {
        प्रत्यागम;
    }
    visited.insert(tensor);
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        apir_सङ्कलय_tensor(tensor->src[i], tensors, visited);
    }
    apir_सङ्कलय_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(apir_serialize_tensor(tensor));
}

शून्य apir_serialize_graph(ध्रुव ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> visited;
    प्रति (uint32_t i = 0; i < n_nodes; i++) {
        apir_सङ्कलय_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    पूर्णाङ्क      output_आकार =
        परिमाण(uint32_t) + n_nodes * परिमाण(uint64_t) + परिमाण(uint32_t) + n_tensors * परिमाण(apir_rpc_tensor);
    output.resize(output_आकार, 0);
    स्मृतिं_प्रतिलिख(output.दत्त(), &n_nodes, परिमाण(n_nodes));
    प्रति (uint32_t i = 0; i < n_nodes; i++) {
        स्मृतिं_प्रतिलिख(output.दत्त() + परिमाण(n_nodes) + i * परिमाण(uint64_t), &cgraph->nodes[i], परिमाण(uint64_t));
    }
    uint32_t * बहिर्गमनम्_ntensors = (uint32_t *) (output.दत्त() + परिमाण(n_nodes) + n_nodes * परिमाण(uint64_t));
    *बहिर्गमनम्_ntensors           = n_tensors;
    apir_rpc_tensor * बहिर्गमनम्_tensors =
        (apir_rpc_tensor *) (output.दत्त() + परिमाण(n_nodes) + n_nodes * परिमाण(uint64_t) + परिमाण(uint32_t));
    स्मृतिं_प्रतिलिख(बहिर्गमनम्_tensors, tensors.दत्त(), n_tensors * परिमाण(apir_rpc_tensor));
}
