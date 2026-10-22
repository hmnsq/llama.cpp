#pragma once

#अन्तर्भाव "input_model.h"
#अन्तर्भाव "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
सार्वजनिक:
    TranslateSession(स्थिरांक frontend::आदान_Model::Ptr & आदान_model,
                     स्थिरांक std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(स्थिरांक frontend::आदान_Model::Ptr & आदान_model);

खाजगी:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    स्थिरांक frontend::आदान_Model::Ptr m_आदान_model;
    स्थिरांक std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
