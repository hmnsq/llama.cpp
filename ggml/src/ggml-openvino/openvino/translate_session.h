#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "input_model.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
ਜਨਤਕ:
    TranslateSession(ਅਚਲ frontend::ਨਿਵੇਸ਼_Model::Ptr & ਨਿਵੇਸ਼_model,
                     ਅਚਲ std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_ਗ੍ਰਾਫ(ਅਚਲ frontend::ਨਿਵੇਸ਼_Model::Ptr & ਨਿਵੇਸ਼_model);

ਨਿਜੀ:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    ਅਚਲ frontend::ਨਿਵੇਸ਼_Model::Ptr m_ਨਿਵੇਸ਼_model;
    ਅਚਲ std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
