#ضمن "fuse_moe_router.h"

#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/core/rt_info.hpp>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/clamp.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/reduce_sum.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/softmax.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/tile.hpp>
#ضمن <openvino/op/topk.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

نطاق {

منطقي constant_is(ثابت ov::Output<ov::Node> & output, ثابت std::vector<int64_t> & اقدار) {
    خودکار گرہ = ov::as_type_ptr<ov::op::v0::Constant>(output.get_گرہ_shared_ptr());
    أرجع گرہ && گرہ->get_element_type().is_integral_نمبر() && گرہ->cast_سمتیہ<int64_t>() == اقدار;
}

}  // namespace

Fuse_Moe_راؤٹر::Fuse_Moe_راؤٹر() {
    ورتوں نطاق ov::pass::pattern;
    ورتوں نطاق ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    خودکار logits = wrap_type<v0::MatMul>();
    خودکار softmax = wrap_type<v8::Softmax>({logits});
    خودکار topk = wrap_type<v11::چوٹی_K>({softmax, any_داخلہ()});
    topk->سیٹ_output_ناپ(2);
    خودکار ids = wrap_type<v8::ٹکڑا>({topk->output(1), any_داخلہ(), any_داخلہ(), any_داخلہ(), any_داخلہ()});
    خودکار ids_2d = wrap_type<v0::Squeeze>({ids, any_داخلہ()});
    خودکار probs = wrap_type<v1::Reshape>({softmax, any_داخلہ()});
    خودکار انکڑے = wrap_type<v0::Squeeze>({probs, any_داخلہ()});
    خودکار انکڑے_batch = wrap_type<v8::Gather>({wrap_type<v3::Shape_دا>({انکڑے}), any_داخلہ(), any_داخلہ()});
    خودکار ids_گنتی = wrap_type<v8::Gather>({wrap_type<v3::Shape_دا>({ids_2d}), any_داخلہ(), any_داخلہ()});
    خودکار نشانہ = wrap_type<v0::Concat>({انکڑے_batch, ids_گنتی});
    خودکار نشر_2 = wrap_type<v3::نشر>({ids_2d, نشانہ});
    خودکار gather = wrap_type<v8::Gather>({انکڑے, نشر_2, any_داخلہ()});
    خودکار weights_4d = wrap_type<v0::Unsqueeze>({gather, any_داخلہ()});
    خودکار weights = wrap_type<v1::Reshape>({weights_4d, any_داخلہ()});
    خودکار جوڑ = wrap_type<v1::Reduce_جوڑ>({weights, any_داخلہ()});
    خودکار clamp = wrap_type<v0::Clamp>({جوڑ});
    خودکار tile = wrap_type<v0::Tile>({clamp, any_داخلہ()});
    خودکار norm = wrap_type<v1::Divide>({weights, tile});

    ثابت خودکار callback = [=](Matcher & m) {
        ثابت خودکار & pm = m.get_نمونہ_قدر_map();
        ثابت خودکار گرہ = [&](ثابت std::shared_ptr<ov::Node> & p) { أرجع pm.at(p).get_گرہ_shared_ptr(); };
        ثابت خودکار داخلہ_is = [&](ثابت std::shared_ptr<ov::Node> & p, size_t i, ثابت std::vector<int64_t> & v) {
            أرجع constant_is(گرہ(p)->داخلہ_قدر(i), v);
        };
        خودکار mm = ov::as_type_ptr<v0::MatMul>(گرہ(logits));
        خودکار sm = ov::as_type_ptr<v8::Softmax>(گرہ(softmax));
        خودکار tk = ov::as_type_ptr<v11::چوٹی_K>(گرہ(topk));
        خودکار reduce = ov::as_type_ptr<v1::Reduce_جوڑ>(گرہ(جوڑ));
        خودکار حد_بندی = ov::as_type_ptr<v0::Clamp>(گرہ(clamp));
        ثابت خودکار shape = mm->get_output_partial_shape(0);
        إذا (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_داخلہ_partial_shape(0).rank() != 4 ||
            mm->get_داخلہ_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_منوال() != v11::چوٹی_K::منوال_2::MAX ||
            tk->get_ترتیب_دیو_type() != v11::چوٹی_K::ترتیب_دیو_Type::ترتیب_دیو_اقدار || tk->get_stable() ||
            tk->get_اشاریہ_element_type() != ov::element::i32 ||
            !tk->output(0).get_نشانہ_inputs().خالی_تھاواں()) {
            أرجع خطأ;
        }
        ثابت int64_t experts = shape[3].get_لمائی();
        خودکار end = ov::as_type_ptr<v0::Constant>(گرہ(ids)->get_داخلہ_گرہ_shared_ptr(2));
        إذا (!end || !end->get_element_type().is_integral_نمبر() || ov::shape_ناپ(end->get_shape()) != 1) {
            أرجع خطأ;
        }
        ثابت int64_t k = end->cast_سمتیہ<int64_t>()[0];
        ثابت خودکار sorted_shape = tk->get_output_partial_shape(1);
        إذا (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_لمائی() < k ||
            !داخلہ_is(probs, 1, {1, -1, experts, 1}) ||
            !داخلہ_is(انکڑے, 1, {0}) || !داخلہ_is(ids_2d, 1, {0, 1}) ||
            !داخلہ_is(weights_4d, 1, {0}) || !داخلہ_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(گرہ(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(گرہ(weights))->get_special_zero() ||
            !داخلہ_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(گرہ(gather))->get_batch_dims() != 1 ||
            !داخلہ_is(انکڑے_batch, 1, {0}) || !داخلہ_is(انکڑے_batch, 2, {0}) ||
            !داخلہ_is(ids_گنتی, 1, {1}) || !داخلہ_is(ids_گنتی, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(گرہ(انکڑے_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(گرہ(ids_گنتی))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(گرہ(نشانہ))->get_axis() != 0 ||
            ov::as_type_ptr<v3::نشر>(گرہ(نشر_2))->get_نشر_spec().m_type != ov::op::نشر_Type::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!داخلہ_is(جوڑ, 1, {-1}) && !داخلہ_is(جوڑ, 1, {3})) ||
            !داخلہ_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(گرہ(norm))->get_autob().m_type != ov::op::Auto_نشر_Type::NUMPY) {
            أرجع خطأ;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        إذا (!(حد_بندی->get_min() <= 0.5 * مزدوج(k) / experts && حد_بندی->get_max() >= 2.0)) {
            أرجع خطأ;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        لكل (ثابت خودکار & داخلہ : tk->output(1).get_نشانہ_inputs()) {
            خودکار ٹکڑا_2 = ov::as_type_ptr<v8::ٹکڑا>(داخلہ.get_گرہ()->shared_from_this());
            إذا (!ٹکڑا_2 || داخلہ.get_اشاریہ() != 0 || ٹکڑا_2->get_داخلہ_ناپ() != 5 ||
                !constant_is(ٹکڑا_2->داخلہ_قدر(1), {0}) || !constant_is(ٹکڑا_2->داخلہ_قدر(2), {k}) ||
                !constant_is(ٹکڑا_2->داخلہ_قدر(3), {1}) ||
                (!constant_is(ٹکڑا_2->داخلہ_قدر(4), {3}) && !constant_is(ٹکڑا_2->داخلہ_قدر(4), {-1}))) {
                أرجع خطأ;
            }
            slices.push_back(ٹکڑا_2);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        خودکار axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        خودکار hidden = std::make_shared<v0::Squeeze>(mm->داخلہ_قدر(0), axis0);
        خودکار routing = std::make_shared<v0::MatMul>(hidden, mm->داخلہ_قدر(1), خطأ, mm->get_transpose_b());
        خودکار probabilities = std::make_shared<v8::Softmax>(routing, -1);
        خودکار selected = std::make_shared<v11::چوٹی_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::چوٹی_K::منوال_2::MAX, v11::چوٹی_K::ترتیب_دیو_Type::ترتیب_دیو_اقدار, tk->get_اشاریہ_element_type(), خطأ);
        خودکار کل = std::make_shared<v1::Reduce_جوڑ>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), سچ);
        خودکار normalized = std::make_shared<v1::Divide>(selected->output(0), کل);
        خودکار weights_باہرلا = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        خودکار ids_باہرلا = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, گرہ(norm)}, {hidden, routing, probabilities, selected, کل, normalized, weights_باہرلا, ids_باہرلا});
        ov::replace_گرہ(گرہ(norm), weights_باہرلا);
        لكل (ثابت خودکار & ٹکڑا_2 : slices) {
            ov::replace_گرہ(ٹکڑا_2, ids_باہرلا);
        }
        أرجع سچ;
    };
    رجسٹر_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
