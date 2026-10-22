#అన్తర్భావ "frontend.h"

#అన్తర్భావ "input_model.h"
#అన్తర్భావ "op_table.h"
#అన్తర్భావ "translate_session.h"
#అన్తర్భావ <openvino/core/type.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {

Front_చివర::Front_చివర() {}

std::shared_ptr<Model> Front_చివర::convert(స్థిరరాశి ఆదానం_Model::Ptr & model, తార్కిక naive) {
    స్వయంచాలక ggml_model = ov::as_type_ptr<ggml::ఆదానం_Model>(model);
    FRONT_చివర_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    స్థిరరాశి స్వయంచాలక & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    తిరిగి converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
