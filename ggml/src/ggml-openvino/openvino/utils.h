#رهنمود once

#ضمن "node_context.h"

#ضمن <memory>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <utility>

namespace ov {
namespace frontend {
namespace ggml {

std::string get_Current_زمان();

فراغ رقم_inputs_check(ثابت گره_Context & context, size_t min_inputs, size_t max_inputs);

صحيح non_cont_dim(std::وکتور<size_t> ne, std::وکتور<size_t> nb);

قالب <typename T> std::وکتور<صحيح> argsort_descend(ثابت std::وکتور<T> & v) {
    std::وکتور<صحيح> idx(v.اندازه());
    std::iota(idx.begin(), idx.انتها(), 0);
    std::مرتب_سازی(idx.begin(), idx.انتها(), [&v](صحيح i1, صحيح i2) { أرجع v[i1] > v[i2]; });
    أرجع idx;
}

قالب <typename T> std::وکتور<T> sorted_descend(std::وکتور<T> v) {
    std::مرتب_سازی(v.begin(), v.انتها(), [](T a, T b) { أرجع a > b; });
    أرجع v;
}

قالب <typename T> bool is_permuted(ثابت std::وکتور<T> & strides) {
    لكل (size_t i = 0; i < strides.اندازه() - 1; ++i) {
        إذا (strides[i] < strides[i + 1]) {
            أرجع true;
        }
    }
    أرجع false;
}

قالب <typename T> std::وکتور<T> جایگشت_2(ثابت std::وکتور<T> & x, ثابت std::وکتور<صحيح> & perm) {
    std::وکتور<T> نتیجه;
    نتیجه.reserve(perm.اندازه());
    لكل (صحيح i : perm) {
        نتیجه.افزودن_پشت(x[i]);
    }
    أرجع نتیجه;
}

std::shared_ptr<ov::گره_2> get_dimensions(ثابت std::shared_ptr<ov::op::v3::Shape_از> & shape,
                                         ثابت std::وکتور<صحيح> & dims);
std::shared_ptr<ov::گره_2> get_dimensions(ثابت std::shared_ptr<ov::گره_2> & گره, ثابت std::وکتور<صحيح> & dims);

Output_وکتور rename_outputs_with_suffix(ثابت Output_وکتور & outputs, ثابت std::string & suffix);

std::pair<ov::Output<گره_2>, ov::Output<گره_2>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::گره_2> inp_pos,
                                                           std::shared_ptr<ov::گره_2> rope_freqs_وزن = nullptr,
                                                           صحيح نما = 0,
                                                           bool stateful = false,
                                                           int64_t سر_dim = 0);

ov::Output<ov::گره_2> process_view_درون_داد(ثابت گره_Context & context, صحيح درون_داد_شاخص, صحيح برش_len = 0, صحيح axis = -1);

ov::Output<ov::گره_2> process_view_درون_داد_new(ثابت گره_Context & context, صحيح درون_داد_شاخص);

namespace op {
قالب <typename T> Output_وکتور translate_1to1_تطابق_2_inputs(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);
    خودکار درون_داد_0 = process_view_درون_داد_new(context, 0);
    خودکار درون_داد_1 = process_view_درون_داد_new(context, 1);

    خودکار output_type = context.get_output_type();
    إذا (درون_داد_0.get_element_type() != درون_داد_1.get_element_type()) {
        إذا (درون_داد_0.get_element_type() != ov::element::f32) {
            درون_داد_0 = std::make_shared<ov::op::v0::Convert>(درون_داد_0, ov::element::f32);
        }
        إذا (درون_داد_1.get_element_type() != ov::element::f32) {
            درون_داد_1 = std::make_shared<ov::op::v0::Convert>(درون_داد_1, ov::element::f32);
        }
    }

    ov::Output<ov::گره_2> res = std::make_shared<T>(درون_داد_0, درون_داد_1);
    إذا (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

قالب <typename T> Output_وکتور translate_1to1_تطابق_1_درون_داد(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    خودکار res = std::make_shared<T>(درون_داد);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}
}  // namespace op

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
