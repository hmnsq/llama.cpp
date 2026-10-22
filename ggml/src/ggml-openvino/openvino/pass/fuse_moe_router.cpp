#شامل "fuse_moe_router.h"

#شامل <openvino/core/graph_util.hpp>
#شامل <openvino/core/rt_info.hpp>
#شامل <openvino/op/broadcast.hpp>
#شامل <openvino/op/clamp.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/divide.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/matmul.hpp>
#شامل <openvino/op/reduce_sum.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/softmax.hpp>
#شامل <openvino/op/squeeze.hpp>
#شامل <openvino/op/tile.hpp>
#شامل <openvino/op/topk.hpp>
#شامل <openvino/op/unsqueeze.hpp>
#شامل <openvino/pass/pattern/op/wrap_type.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا pass {

نامخلا {

منطقی constant_is(ثابت ov::Output<ov::Node> & output, ثابت std::vector<int64_t> & اقدار) {
    خودکار گرہ = ov::as_type_ptr<ov::op::v0::Constant>(output.get_گرہ_shared_ptr());
    واپس گرہ && گرہ->get_element_type().is_integral_نمبر() && گرہ->cast_سمتیہ<int64_t>() == اقدار;
}

}  // namespace

Fuse_Moe_رہنما_آلہ::Fuse_Moe_رہنما_آلہ() {
    استعمال نامخلا ov::pass::pattern;
    استعمال نامخلا ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    خودکار logits = wrap_type<v0::MatMul>();
    خودکار softmax = wrap_type<v8::Softmax>({logits});
    خودکار topk = wrap_type<v11::چوٹی_K>({softmax, any_داخلہ()});
    topk->سیٹ_output_سائز(2);
    خودکار ids = wrap_type<v8::قاش>({topk->output(1), any_داخلہ(), any_داخلہ(), any_داخلہ(), any_داخلہ()});
    خودکار ids_2d = wrap_type<v0::Squeeze>({ids, any_داخلہ()});
    خودکار probs = wrap_type<v1::Reshape>({softmax, any_داخلہ()});
    خودکار ڈیٹا = wrap_type<v0::Squeeze>({probs, any_داخلہ()});
    خودکار ڈیٹا_batch = wrap_type<v8::Gather>({wrap_type<v3::Shape_کا>({ڈیٹا}), any_داخلہ(), any_داخلہ()});
    خودکار ids_گنتی = wrap_type<v8::Gather>({wrap_type<v3::Shape_کا>({ids_2d}), any_داخلہ(), any_داخلہ()});
    خودکار ہدف = wrap_type<v0::Concat>({ڈیٹا_batch, ids_گنتی});
    خودکار نشر_2 = wrap_type<v3::نشر>({ids_2d, ہدف});
    خودکار gather = wrap_type<v8::Gather>({ڈیٹا, نشر_2, any_داخلہ()});
    خودکار weights_4d = wrap_type<v0::Unsqueeze>({gather, any_داخلہ()});
    خودکار weights = wrap_type<v1::Reshape>({weights_4d, any_داخلہ()});
    خودکار مجموع = wrap_type<v1::Reduce_مجموع>({weights, any_داخلہ()});
    خودکار clamp = wrap_type<v0::Clamp>({مجموع});
    خودکار tile = wrap_type<v0::Tile>({clamp, any_داخلہ()});
    خودکار norm = wrap_type<v1::Divide>({weights, tile});

    ثابت خودکار callback = [=](Matcher & m) {
        ثابت خودکار & pm = m.get_نمونہ_قدر_map();
        ثابت خودکار گرہ = [&](ثابت std::shared_ptr<ov::Node> & p) { واپس pm.at(p).get_گرہ_shared_ptr(); };
        ثابت خودکار داخلہ_is = [&](ثابت std::shared_ptr<ov::Node> & p, size_t i, ثابت std::vector<int64_t> & v) {
            واپس constant_is(گرہ(p)->داخلہ_قدر(i), v);
        };
        خودکار mm = ov::as_type_ptr<v0::MatMul>(گرہ(logits));
        خودکار sm = ov::as_type_ptr<v8::Softmax>(گرہ(softmax));
        خودکار tk = ov::as_type_ptr<v11::چوٹی_K>(گرہ(topk));
        خودکار reduce = ov::as_type_ptr<v1::Reduce_مجموع>(گرہ(مجموع));
        خودکار حد_بندی = ov::as_type_ptr<v0::Clamp>(گرہ(clamp));
        ثابت خودکار shape = mm->get_output_partial_shape(0);
        اگر (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_داخلہ_partial_shape(0).rank() != 4 ||
            mm->get_داخلہ_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_منوال() != v11::چوٹی_K::منوال_2::MAX ||
            tk->get_ترتیب_دو_type() != v11::چوٹی_K::ترتیب_دو_Type::ترتیب_دو_اقدار || tk->get_stable() ||
            tk->get_اشاریہ_element_type() != ov::element::i32 ||
            !tk->output(0).get_ہدف_inputs().خالی_خانے()) {
            واپس جھوٹ;
        }
        ثابت int64_t experts = shape[3].get_لمبائی();
        خودکار end = ov::as_type_ptr<v0::Constant>(گرہ(ids)->get_داخلہ_گرہ_shared_ptr(2));
        اگر (!end || !end->get_element_type().is_integral_نمبر() || ov::shape_سائز(end->get_shape()) != 1) {
            واپس جھوٹ;
        }
        ثابت int64_t k = end->cast_سمتیہ<int64_t>()[0];
        ثابت خودکار sorted_shape = tk->get_output_partial_shape(1);
        اگر (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_لمبائی() < k ||
            !داخلہ_is(probs, 1, {1, -1, experts, 1}) ||
            !داخلہ_is(ڈیٹا, 1, {0}) || !داخلہ_is(ids_2d, 1, {0, 1}) ||
            !داخلہ_is(weights_4d, 1, {0}) || !داخلہ_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(گرہ(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(گرہ(weights))->get_special_zero() ||
            !داخلہ_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(گرہ(gather))->get_batch_dims() != 1 ||
            !داخلہ_is(ڈیٹا_batch, 1, {0}) || !داخلہ_is(ڈیٹا_batch, 2, {0}) ||
            !داخلہ_is(ids_گنتی, 1, {1}) || !داخلہ_is(ids_گنتی, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(گرہ(ڈیٹا_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(گرہ(ids_گنتی))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(گرہ(ہدف))->get_axis() != 0 ||
            ov::as_type_ptr<v3::نشر>(گرہ(نشر_2))->get_نشر_spec().m_type != ov::op::نشر_Type::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!داخلہ_is(مجموع, 1, {-1}) && !داخلہ_is(مجموع, 1, {3})) ||
            !داخلہ_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(گرہ(norm))->get_autob().m_type != ov::op::Auto_نشر_Type::NUMPY) {
            واپس جھوٹ;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        اگر (!(حد_بندی->get_min() <= 0.5 * دوہرا(k) / experts && حد_بندی->get_max() >= 2.0)) {
            واپس جھوٹ;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        برائے (ثابت خودکار & داخلہ : tk->output(1).get_ہدف_inputs()) {
            خودکار قاش_2 = ov::as_type_ptr<v8::قاش>(داخلہ.get_گرہ()->shared_from_this());
            اگر (!قاش_2 || داخلہ.get_اشاریہ() != 0 || قاش_2->get_داخلہ_سائز() != 5 ||
                !constant_is(قاش_2->داخلہ_قدر(1), {0}) || !constant_is(قاش_2->داخلہ_قدر(2), {k}) ||
                !constant_is(قاش_2->داخلہ_قدر(3), {1}) ||
                (!constant_is(قاش_2->داخلہ_قدر(4), {3}) && !constant_is(قاش_2->داخلہ_قدر(4), {-1}))) {
                واپس جھوٹ;
            }
            slices.push_back(قاش_2);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        خودکار axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        خودکار hidden = std::make_shared<v0::Squeeze>(mm->داخلہ_قدر(0), axis0);
        خودکار routing = std::make_shared<v0::MatMul>(hidden, mm->داخلہ_قدر(1), جھوٹ, mm->get_transpose_b());
        خودکار probabilities = std::make_shared<v8::Softmax>(routing, -1);
        خودکار selected = std::make_shared<v11::چوٹی_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::چوٹی_K::منوال_2::MAX, v11::چوٹی_K::ترتیب_دو_Type::ترتیب_دو_اقدار, tk->get_اشاریہ_element_type(), جھوٹ);
        خودکار کل = std::make_shared<v1::Reduce_مجموع>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), سچ);
        خودکار normalized = std::make_shared<v1::Divide>(selected->output(0), کل);
        خودکار weights_برآمد = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        خودکار ids_برآمد = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, گرہ(norm)}, {hidden, routing, probabilities, selected, کل, normalized, weights_برآمد, ids_برآمد});
        ov::replace_گرہ(گرہ(norm), weights_برآمد);
        برائے (ثابت خودکار & قاش_2 : slices) {
            ov::replace_گرہ(قاش_2, ids_برآمد);
        }
        واپس سچ;
    };
    رجسٹر_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
