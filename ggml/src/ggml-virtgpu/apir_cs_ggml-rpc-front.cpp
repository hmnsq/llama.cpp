#అన్తర్భావ "backend/shared/apir_cs_rpc.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-remoting.h"

#అన్తర్భావ <cinttypes>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <unordered_set>
#అన్తర్భావ <vector>

apir_rpc_tensor apir_serialize_tensor(స్థిరరాశి ggml_tensor * tensor) {
    apir_rpc_tensor ఫలితం;
    ఫలితం.id   = reinterpret_cast<uint64_t>(tensor);
    ఫలితం.type = tensor->type;
    ఒకవేళ (tensor->బఫరు) {
        ggml_backend_బఫరు_t బఫరు = tensor->బఫరు;

        ఫలితం.బఫరు = బఫరు_గమ్యం_HOST_HANDLE(బఫరు);
    } లేకపోతే {
        ఫలితం.బఫరు = 0;
    }
    కొరకు (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ఫలితం.ne[i] = tensor->ne[i];
        ఫలితం.nb[i] = tensor->nb[i];
    }
    ఫలితం.op = tensor->op;
    కొరకు (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / పరిమాణం(int32_t); i++) {
        ఫలితం.op_params[i] = tensor->op_params[i];
    }
    ఫలితం.flags = tensor->flags;
    కొరకు (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        ఫలితం.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    ఫలితం.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    ఫలితం.view_offs = tensor->view_offs;
    ఫలితం.దత్తాంశం      = reinterpret_cast<uint64_t>(tensor->దత్తాంశం);
    ఒకవేళ (tensor->దత్తాంశం) {
        ఒకవేళ (!tensor->బఫరు) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        ఫలితం.దత్తాంశం -= reinterpret_cast<uint64_t>(బఫరు_గమ్యం_GGML_CONTEXT(tensor->బఫరు)->పునాది);
    }
    snprintf(ఫలితం.name, GGML_MAX_NAME, "%s", tensor->name);
    తిరిగి ఫలితం;
}

శూన్య apir_కూడు_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & సందర్శించిన) {
    ఒకవేళ (tensor == శూన్యసూచిక) {
        తిరిగి;
    }
    ఒకవేళ (సందర్శించిన.find(tensor) != సందర్శించిన.end()) {
        తిరిగి;
    }
    సందర్శించిన.insert(tensor);
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        apir_కూడు_tensor(tensor->src[i], tensors, సందర్శించిన);
    }
    apir_కూడు_tensor(tensor->view_src, tensors, సందర్శించిన);
    tensors.push_back(apir_serialize_tensor(tensor));
}

శూన్య apir_serialize_లేఖాచిత్రం(స్థిరరాశి ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> సందర్శించిన;
    కొరకు (uint32_t i = 0; i < n_nodes; i++) {
        apir_కూడు_tensor(cgraph->nodes[i], tensors, సందర్శించిన);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    పూర్ణాంకం      output_కొలత =
        పరిమాణం(uint32_t) + n_nodes * పరిమాణం(uint64_t) + పరిమాణం(uint32_t) + n_tensors * పరిమాణం(apir_rpc_tensor);
    output.resize(output_కొలత, 0);
    జ్ఞాపకం_నకలు(output.దత్తాంశం(), &n_nodes, పరిమాణం(n_nodes));
    కొరకు (uint32_t i = 0; i < n_nodes; i++) {
        జ్ఞాపకం_నకలు(output.దత్తాంశం() + పరిమాణం(n_nodes) + i * పరిమాణం(uint64_t), &cgraph->nodes[i], పరిమాణం(uint64_t));
    }
    uint32_t * బహిర్గతం_ntensors = (uint32_t *) (output.దత్తాంశం() + పరిమాణం(n_nodes) + n_nodes * పరిమాణం(uint64_t));
    *బహిర్గతం_ntensors           = n_tensors;
    apir_rpc_tensor * బహిర్గతం_tensors =
        (apir_rpc_tensor *) (output.దత్తాంశం() + పరిమాణం(n_nodes) + n_nodes * పరిమాణం(uint64_t) + పరిమాణం(uint32_t));
    జ్ఞాపకం_నకలు(బహిర్గతం_tensors, tensors.దత్తాంశం(), n_tensors * పరిమాణం(apir_rpc_tensor));
}
