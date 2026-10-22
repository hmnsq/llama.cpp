#pragma once

#ضمن "input_model.h"
#ضمن "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
عام:
    TranslateSession(ثابت frontend::كىرگۈزۈش_Model::Ptr & كىرگۈزۈش_model,
                     ثابت std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(ثابت frontend::كىرگۈزۈش_Model::Ptr & كىرگۈزۈش_model);

خاص:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    ثابت frontend::كىرگۈزۈش_Model::Ptr m_كىرگۈزۈش_model;
    ثابت std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
