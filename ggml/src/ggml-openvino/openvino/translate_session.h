#pragma once

#ಅನ್ತರ್ಭಾವ "input_model.h"
#ಅನ್ತರ್ಭಾವ "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
ಸಾರ್ವಜನಿಕ:
    TranslateSession(ಧ್ರುವ frontend::ಒಳಹರಿವು_Model::Ptr & ಒಳಹರಿವು_model,
                     ಧ್ರುವ std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(ಧ್ರುವ frontend::ಒಳಹರಿವು_Model::Ptr & ಒಳಹರಿವು_model);

ಗುಪ್ತ:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    ಧ್ರುವ frontend::ಒಳಹರಿವು_Model::Ptr m_ಒಳಹರಿವು_model;
    ಧ್ರುವ std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
