#অন্তর্ভাব "fuse_moe_router.h"

#অন্তর্ভাব <openvino/core/graph_util.hpp>
#অন্তর্ভাব <openvino/core/rt_info.hpp>
#অন্তর্ভাব <openvino/op/broadcast.hpp>
#অন্তর্ভাব <openvino/op/clamp.hpp>
#অন্তর্ভাব <openvino/op/concat.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/divide.hpp>
#অন্তর্ভাব <openvino/op/gather.hpp>
#অন্তর্ভাব <openvino/op/matmul.hpp>
#অন্তর্ভাব <openvino/op/reduce_sum.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/shape_of.hpp>
#অন্তর্ভাব <openvino/op/slice.hpp>
#অন্তর্ভাব <openvino/op/softmax.hpp>
#অন্তর্ভাব <openvino/op/squeeze.hpp>
#অন্তর্ভাব <openvino/op/tile.hpp>
#অন্তর্ভাব <openvino/op/topk.hpp>
#অন্তর্ভাব <openvino/op/unsqueeze.hpp>
#অন্তর্ভাব <openvino/pass/pattern/op/wrap_type.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ pass {

নামাকাশ {

যৌক্তিক constant_is(ধ্রুবক ov::Output<ov::Node> & output, ধ্রুবক std::vector<int64_t> & মানসমূহ) {
    auto গ্রন্থি = ov::as_type_ptr<ov::op::v0::Constant>(output.get_গ্রন্থি_shared_ptr());
    ফেরত গ্রন্থি && গ্রন্থি->get_element_type().is_integral_সংখ্যা() && গ্রন্থি->cast_vector<int64_t>() == মানসমূহ;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using নামাকাশ ov::pass::pattern;
    using নামাকাশ ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    auto logits = wrap_type<v0::MatMul>();
    auto softmax = wrap_type<v8::Softmax>({logits});
    auto topk = wrap_type<v11::শীর্ষ_K>({softmax, any_প্রবেশ()});
    topk->set_output_আকার(2);
    auto ids = wrap_type<v8::Slice>({topk->output(1), any_প্রবেশ(), any_প্রবেশ(), any_প্রবেশ(), any_প্রবেশ()});
    auto ids_2d = wrap_type<v0::Squeeze>({ids, any_প্রবেশ()});
    auto probs = wrap_type<v1::Reshape>({softmax, any_প্রবেশ()});
    auto তথ্য = wrap_type<v0::Squeeze>({probs, any_প্রবেশ()});
    auto তথ্য_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({তথ্য}), any_প্রবেশ(), any_প্রবেশ()});
    auto ids_গুনতি = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_প্রবেশ(), any_প্রবেশ()});
    auto target = wrap_type<v0::Concat>({তথ্য_batch, ids_গুনতি});
    auto broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    auto gather = wrap_type<v8::Gather>({তথ্য, broadcast, any_প্রবেশ()});
    auto weights_4d = wrap_type<v0::Unsqueeze>({gather, any_প্রবেশ()});
    auto weights = wrap_type<v1::Reshape>({weights_4d, any_প্রবেশ()});
    auto যোগ_2 = wrap_type<v1::Reduce_যোগ>({weights, any_প্রবেশ()});
    auto clamp = wrap_type<v0::Clamp>({যোগ_2});
    auto tile = wrap_type<v0::Tile>({clamp, any_প্রবেশ()});
    auto norm = wrap_type<v1::Divide>({weights, tile});

    ধ্রুবক auto callback = [=](Matcher & m) {
        ধ্রুবক auto & pm = m.get_pattern_মান_map();
        ধ্রুবক auto গ্রন্থি = [&](ধ্রুবক std::shared_ptr<ov::Node> & p) { ফেরত pm.at(p).get_গ্রন্থি_shared_ptr(); };
        ধ্রুবক auto প্রবেশ_is = [&](ধ্রুবক std::shared_ptr<ov::Node> & p, size_t i, ধ্রুবক std::vector<int64_t> & v) {
            ফেরত constant_is(গ্রন্থি(p)->প্রবেশ_মান(i), v);
        };
        auto mm = ov::as_type_ptr<v0::MatMul>(গ্রন্থি(logits));
        auto sm = ov::as_type_ptr<v8::Softmax>(গ্রন্থি(softmax));
        auto tk = ov::as_type_ptr<v11::শীর্ষ_K>(গ্রন্থি(topk));
        auto reduce = ov::as_type_ptr<v1::Reduce_যোগ>(গ্রন্থি(যোগ_2));
        auto limit = ov::as_type_ptr<v0::Clamp>(গ্রন্থি(clamp));
        ধ্রুবক auto shape = mm->get_output_partial_shape(0);
        যদি (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_প্রবেশ_partial_shape(0).rank() != 4 ||
            mm->get_প্রবেশ_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::শীর্ষ_K::Mode::MAX ||
            tk->get_sort_type() != v11::শীর্ষ_K::SortType::SORT_মানসমূহ || tk->get_stable() ||
            tk->get_সূচক_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            ফেরত মিথ্যা;
        }
        ধ্রুবক int64_t experts = shape[3].get_দৈর্ঘ্য();
        auto end = ov::as_type_ptr<v0::Constant>(গ্রন্থি(ids)->get_প্রবেশ_গ্রন্থি_shared_ptr(2));
        যদি (!end || !end->get_element_type().is_integral_সংখ্যা() || ov::shape_আকার(end->get_shape()) != 1) {
            ফেরত মিথ্যা;
        }
        ধ্রুবক int64_t k = end->cast_vector<int64_t>()[0];
        ধ্রুবক auto sorted_shape = tk->get_output_partial_shape(1);
        যদি (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_দৈর্ঘ্য() < k ||
            !প্রবেশ_is(probs, 1, {1, -1, experts, 1}) ||
            !প্রবেশ_is(তথ্য, 1, {0}) || !প্রবেশ_is(ids_2d, 1, {0, 1}) ||
            !প্রবেশ_is(weights_4d, 1, {0}) || !প্রবেশ_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(গ্রন্থি(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(গ্রন্থি(weights))->get_special_zero() ||
            !প্রবেশ_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(গ্রন্থি(gather))->get_batch_dims() != 1 ||
            !প্রবেশ_is(তথ্য_batch, 1, {0}) || !প্রবেশ_is(তথ্য_batch, 2, {0}) ||
            !প্রবেশ_is(ids_গুনতি, 1, {1}) || !প্রবেশ_is(ids_গুনতি, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(গ্রন্থি(তথ্য_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(গ্রন্থি(ids_গুনতি))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(গ্রন্থি(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(গ্রন্থি(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!প্রবেশ_is(যোগ_2, 1, {-1}) && !প্রবেশ_is(যোগ_2, 1, {3})) ||
            !প্রবেশ_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(গ্রন্থি(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            ফেরত মিথ্যা;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        যদি (!(limit->get_min() <= 0.5 * দ্বিগুণ(k) / experts && limit->get_max() >= 2.0)) {
            ফেরত মিথ্যা;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        জন্য (ধ্রুবক auto & প্রবেশ : tk->output(1).get_target_inputs()) {
            auto slice = ov::as_type_ptr<v8::Slice>(প্রবেশ.get_গ্রন্থি()->shared_from_this());
            যদি (!slice || প্রবেশ.get_সূচক() != 0 || slice->get_প্রবেশ_আকার() != 5 ||
                !constant_is(slice->প্রবেশ_মান(1), {0}) || !constant_is(slice->প্রবেশ_মান(2), {k}) ||
                !constant_is(slice->প্রবেশ_মান(3), {1}) ||
                (!constant_is(slice->প্রবেশ_মান(4), {3}) && !constant_is(slice->প্রবেশ_মান(4), {-1}))) {
                ফেরত মিথ্যা;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto hidden = std::make_shared<v0::Squeeze>(mm->প্রবেশ_মান(0), axis0);
        auto routing = std::make_shared<v0::MatMul>(hidden, mm->প্রবেশ_মান(1), মিথ্যা, mm->get_transpose_b());
        auto probabilities = std::make_shared<v8::Softmax>(routing, -1);
        auto selected = std::make_shared<v11::শীর্ষ_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::শীর্ষ_K::Mode::MAX, v11::শীর্ষ_K::SortType::SORT_মানসমূহ, tk->get_সূচক_element_type(), মিথ্যা);
        auto যোগ = std::make_shared<v1::Reduce_যোগ>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), সত্য);
        auto normalized = std::make_shared<v1::Divide>(selected->output(0), যোগ);
        auto weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        auto ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, গ্রন্থি(norm)}, {hidden, routing, probabilities, selected, যোগ, normalized, weights_out, ids_out});
        ov::replace_গ্রন্থি(গ্রন্থি(norm), weights_out);
        জন্য (ধ্রুবক auto & slice : slices) {
            ov::replace_গ্রন্থি(slice, ids_out);
        }
        ফেরত সত্য;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
