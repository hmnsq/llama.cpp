#pragma once

#அந்தர்பாவ "input_model.h"
#அந்தர்பாவ "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
பொது:
    TranslateSession(மாறிலி frontend::உள்ளீடு_Model::Ptr & உள்ளீடு_model,
                     மாறிலி std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(மாறிலி frontend::உள்ளீடு_Model::Ptr & உள்ளீடு_model);

தனிப்பட்ட:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    மாறிலி frontend::உள்ளீடு_Model::Ptr m_உள்ளீடு_model;
    மாறிலி std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
