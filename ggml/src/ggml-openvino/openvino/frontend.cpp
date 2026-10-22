#அந்தர்பாவ "frontend.h"

#அந்தர்பாவ "input_model.h"
#அந்தர்பாவ "op_table.h"
#அந்தர்பாவ "translate_session.h"
#அந்தர்பாவ <openvino/core/type.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(மாறிலி உள்ளீடு_Model::Ptr & model, தருக்க naive) {
    auto ggml_model = ov::as_type_ptr<ggml::உள்ளீடு_Model>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    மாறிலி auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    திரும்பு converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
