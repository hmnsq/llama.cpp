#ਅਨ੍ਤਰ੍ਭਾਵ "frontend.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "input_model.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "translate_session.h"
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/type.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {

Front_ਸਿਰਾ::Front_ਸਿਰਾ() {}

std::shared_ptr<Model> Front_ਸਿਰਾ::convert(ਅਚਲ ਨਿਵੇਸ਼_Model::Ptr & model, ਤਾਰਕਿਕ naive) {
    ਸਵੈਚਾਲਿਤ ggml_model = ov::as_type_ptr<ggml::ਨਿਵੇਸ਼_Model>(model);
    FRONT_ਸਿਰਾ_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    ਅਚਲ ਸਵੈਚਾਲਿਤ & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    ਵਾਪਸ converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
