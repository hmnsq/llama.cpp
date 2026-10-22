#pragma once

#অন্তর্ভাৱ "input_model.h"
#অন্তর্ভাৱ "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
সার্ৱজনিক:
    TranslateSession(ধ্রুৱ frontend::প্ৰবেশ_Model::Ptr & প্ৰবেশ_model,
                     ধ্রুৱ std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(ধ্রুৱ frontend::প্ৰবেশ_Model::Ptr & প্ৰবেশ_model);

গুপ্ত:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    ধ্রুৱ frontend::প্ৰবেশ_Model::Ptr m_প্ৰবেশ_model;
    ধ্রুৱ std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
