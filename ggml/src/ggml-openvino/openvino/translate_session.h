#आशय once

#समावेश "input_model.h"
#समावेश "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
खुला:
    TranslateSession(स्थिर frontend::निवेश_Model::Ptr & निवेश_model,
                     स्थिर std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(स्थिर frontend::निवेश_Model::Ptr & निवेश_model);

निजी:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    स्थिर frontend::निवेश_Model::Ptr m_निवेश_model;
    स्थिर std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
