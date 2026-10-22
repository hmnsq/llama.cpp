#شامل "frontend.h"

#شامل "input_model.h"
#شامل "op_table.h"
#شامل "translate_session.h"
#شامل <openvino/core/type.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(ثابت درون_داد_Model::Ptr & model, منطقی naive) {
    auto ggml_model = ov::as_type_ptr<ggml::درون_داد_Model>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    ثابت auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    بازگشت converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
