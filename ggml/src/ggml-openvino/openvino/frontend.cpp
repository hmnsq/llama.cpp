#অন্তর্ভাব "frontend.h"

#অন্তর্ভাব "input_model.h"
#অন্তর্ভাব "op_table.h"
#অন্তর্ভাব "translate_session.h"
#অন্তর্ভাব <openvino/core/type.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(ধ্রুবক প্রবেশ_Model::Ptr & model, যৌক্তিক naive) {
    auto ggml_model = ov::as_type_ptr<ggml::প্রবেশ_Model>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    ধ্রুবক auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    ফেরত converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
