#شامل "frontend.h"

#شامل "input_model.h"
#شامل "op_table.h"
#شامل "translate_session.h"
#شامل <openvino/core/type.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {

Front_انتہا::Front_انتہا() {}

std::shared_ptr<Model> Front_انتہا::convert(ثابت داخلہ_Model::Ptr & model, منطقی naive) {
    خودکار ggml_model = ov::as_type_ptr<ggml::داخلہ_Model>(model);
    FRONT_انتہا_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    ثابت خودکار & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    واپس converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
