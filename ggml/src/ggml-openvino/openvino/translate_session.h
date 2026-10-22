#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "input_model.h"
#ಅನ್ತರ್ಭಾವ "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
ಸಾರ್ವಜನಿಕ:
    TranslateSession(ಸ್ಥಿರಾಂಕ frontend::ಒಳಹರಿವು_Model::Ptr & ಒಳಹರಿವು_model,
                     ಸ್ಥಿರಾಂಕ std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_ನಕ್ಷೆ(ಸ್ಥಿರಾಂಕ frontend::ಒಳಹರಿವು_Model::Ptr & ಒಳಹರಿವು_model);

ಖಾಸಗಿ:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    ಸ್ಥಿರಾಂಕ frontend::ಒಳಹರಿವು_Model::Ptr m_ಒಳಹರಿವು_model;
    ಸ್ಥಿರಾಂಕ std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
