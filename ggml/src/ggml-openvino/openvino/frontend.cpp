#समावेश "frontend.h"

#समावेश "input_model.h"
#समावेश "op_table.h"
#समावेश "translate_session.h"
#समावेश <openvino/core/type.hpp>

namespace ov {
namespace frontend {
namespace ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(स्थिर निवेश_Model::Ptr & model, bool naive) {
    स्वतः ggml_model = ov::as_type_ptr<ggml::निवेश_Model>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    स्थिर स्वतः & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    वापस converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
