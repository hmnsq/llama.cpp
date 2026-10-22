#pragma once

#অন্তর্ভাব "input_model.h"
#অন্তর্ভাব "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
সার্বজনিক:
    TranslateSession(ধ্রুব frontend::প্রবেশ_Model::Ptr & প্রবেশ_model,
                     ধ্রুব std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(ধ্রুব frontend::প্রবেশ_Model::Ptr & প্রবেশ_model);

গুপ্ত:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    ধ্রুব frontend::প্রবেশ_Model::Ptr m_প্রবেশ_model;
    ধ্রুব std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
