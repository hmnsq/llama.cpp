#لارښوونه once

#ضمن "input_model.h"
#ضمن "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
عام:
    TranslateSession(ثابت frontend::ننوت_Model::Ptr & ننوت_model,
                     ثابت std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_ګراف(ثابت frontend::ننوت_Model::Ptr & ننوت_model);

خاص:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    ثابت frontend::ننوت_Model::Ptr m_ننوت_model;
    ثابت std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
