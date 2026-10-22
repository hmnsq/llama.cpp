#ضمن "frontend.h"

#ضمن "input_model.h"
#ضمن "op_table.h"
#ضمن "translate_session.h"
#ضمن <openvino/core/type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(ثابت مدخل_Model::Ptr & model, منطقي naive) {
    auto ggml_model = ov::as_type_ptr<ggml::مدخل_Model>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    ثابت auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    أرجع converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
