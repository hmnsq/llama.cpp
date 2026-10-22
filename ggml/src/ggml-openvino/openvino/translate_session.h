#సూచన once

#అన్తర్భావ "input_model.h"
#అన్తర్భావ "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
బహిరంగ:
    TranslateSession(స్థిరరాశి frontend::ఆదానం_Model::Ptr & ఆదానం_model,
                     స్థిరరాశి std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_లేఖాచిత్రం(స్థిరరాశి frontend::ఆదానం_Model::Ptr & ఆదానం_model);

అంతరంగ:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    స్థిరరాశి frontend::ఆదానం_Model::Ptr m_ఆదానం_model;
    స్థిరరాశి std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
