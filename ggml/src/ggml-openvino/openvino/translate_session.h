#pragma once

#અન્તર્ભાવ "input_model.h"
#અન્તર્ભાવ "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
જાહેર:
    TranslateSession(અચલ frontend::આગત_Model::Ptr & આગત_model,
                     અચલ std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(અચલ frontend::આગત_Model::Ptr & આગત_model);

ખાનગી:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    અચલ frontend::આગત_Model::Ptr m_આગત_model;
    અચલ std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
