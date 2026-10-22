#অন্তর্ভাৱ "frontend.h"

#অন্তর্ভাৱ "input_model.h"
#অন্তর্ভাৱ "op_table.h"
#অন্তর্ভাৱ "translate_session.h"
#অন্তর্ভাৱ <openvino/core/type.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(ধ্রুৱ প্ৰবেশ_Model::Ptr & model, তার্কিক naive) {
    auto ggml_model = ov::as_type_ptr<ggml::প্ৰবেশ_Model>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    ধ্রুৱ auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    ঘূৰাই converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
