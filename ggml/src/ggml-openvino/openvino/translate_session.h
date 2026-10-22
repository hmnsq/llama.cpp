#pragma once

#ଅନ୍ତର୍ଭାଵ "input_model.h"
#ଅନ୍ତର୍ଭାଵ "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
ସର୍ବସାଧାରଣ:
    TranslateSession(ଅଚଳ frontend::ନିବେଶ_Model::Ptr & ନିବେଶ_model,
                     ଅଚଳ std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(ଅଚଳ frontend::ନିବେଶ_Model::Ptr & ନିବେଶ_model);

ବ୍ୟକ୍ତିଗତ:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    ଅଚଳ frontend::ନିବେଶ_Model::Ptr m_ନିବେଶ_model;
    ଅଚଳ std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
