#شامل "fuse_argsort_topk.h"

#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/topk.hpp>
#شامل <openvino/pass/pattern/op/wrap_type.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا pass {

Fuse_Argsort_چوٹی_K::Fuse_Argsort_چوٹی_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    خودکار pattern = ov::pass::pattern::wrap_type<ov::op::v11::چوٹی_K>();

    ثابت خودکار callback = [](ov::pass::pattern::Matcher & m) {
        خودکار topk = ov::as_type_ptr<ov::op::v11::چوٹی_K>(m.get_مطابقت_جڑ());
        اگر (!topk->output(0).get_ہدف_inputs().خالی_خانے() || topk->output(1).get_ہدف_inputs().خالی_خانے() ||
            topk->get_ترتیب_دو_type() != ov::op::v11::چوٹی_K::ترتیب_دو_Type::ترتیب_دو_اقدار) {
            واپس جھوٹ;
        }

        ثابت خودکار shape = topk->get_output_partial_shape(1);
        اگر (shape.rank().is_dynamic()) {
            واپس جھوٹ;
        }
        ثابت int64_t rank = shape.rank().get_لمبائی();
        ثابت int64_t axis = topk->get_axis();
        اگر (shape[axis].is_dynamic()) {
            واپس جھوٹ;
        }

        int64_t prefix = -1;
        برائے (ثابت خودکار & داخلہ : topk->output(1).get_ہدف_inputs()) {
            ثابت خودکار * قاش_2 = ov::as_type<ov::op::v8::قاش>(داخلہ.get_گرہ());
            اگر (!قاش_2 || داخلہ.get_اشاریہ() != 0 || قاش_2->get_داخلہ_سائز() != 5) {
                واپس جھوٹ;
            }
            int64_t اقدار[4];
            برائے (size_t i = 0; i < 4; ++i) {
                خودکار value = ov::as_type_ptr<ov::op::v0::Constant>(قاش_2->get_داخلہ_گرہ_shared_ptr(i + 1));
                اگر (!value || ov::shape_سائز(value->get_shape()) != 1) {
                    واپس جھوٹ;
                }
                اقدار[i] = value->cast_سمتیہ<int64_t>()[0];
            }
            ثابت int64_t قاش_axis = اقدار[3] < 0 ? اقدار[3] + rank : اقدار[3];
            اگر (اقدار[0] != 0 || اقدار[2] != 1 || قاش_axis != axis || اقدار[1] <= 0 ||
                اقدار[1] >= shape[axis].get_لمبائی() || (prefix != -1 && prefix != اقدار[1])) {
                واپس جھوٹ;
            }
            prefix = اقدار[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->سیٹ_دلیل(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        واپس سچ;
    };

    رجسٹر_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
