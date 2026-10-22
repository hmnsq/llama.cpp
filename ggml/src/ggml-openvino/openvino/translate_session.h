#pragma once

#അന്തര്ഭാവ "input_model.h"
#അന്തര്ഭാവ "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
പൊതു:
    TranslateSession(സ്ഥിരം frontend::നിവേശം_Model::Ptr & നിവേശം_model,
                     സ്ഥിരം std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(സ്ഥിരം frontend::നിവേശം_Model::Ptr & നിവേശം_model);

സ്വകാര്യ:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    സ്ഥിരം frontend::നിവേശം_Model::Ptr m_നിവേശം_model;
    സ്ഥിരം std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
