#ضمن "frontend.h"

#ضمن "input_model.h"
#ضمن "op_table.h"
#ضمن "translate_session.h"
#ضمن <openvino/core/type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {

Front_انتها::Front_انتها() {}

std::shared_ptr<Model> Front_انتها::convert(ثابت درون_داد_Model::Ptr & model, منطقي naive) {
    خودکار ggml_model = ov::as_type_ptr<ggml::درون_داد_Model>(model);
    FRONT_انتها_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    ثابت خودکار & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    أرجع converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
