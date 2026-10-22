#ಅನ್ತರ್ಭಾವ "fuse_moe_router.h"

#ಅನ್ತರ್ಭಾವ <openvino/core/graph_util.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/rt_info.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/clamp.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/divide.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/matmul.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reduce_sum.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/softmax.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/squeeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/tile.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/topk.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/wrap_type.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ pass {

ನಾಮಾಕಾಶ {

ತಾರ್ಕಿಕ constant_is(ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & output, ಸ್ಥಿರಾಂಕ std::vector<int64_t> & ಮೌಲ್ಯಗಳು) {
    ಸ್ವಯಂಚಾಲಿತ ಗಂಟು = ov::as_type_ptr<ov::op::v0::Constant>(output.get_ಗಂಟು_shared_ptr());
    ಹಿಂತಿರುಗಿ ಗಂಟು && ಗಂಟು->get_element_type().is_integral_ಸಂಖ್ಯೆ() && ಗಂಟು->cast_ಸದಿಶ<int64_t>() == ಮೌಲ್ಯಗಳು;
}

}  // namespace

Fuse_Moe_ಮಾರ್ಗನಿರ್ದೇಶಕ::Fuse_Moe_ಮಾರ್ಗನಿರ್ದೇಶಕ() {
    ಬಳಸಿ ನಾಮಾಕಾಶ ov::pass::pattern;
    ಬಳಸಿ ನಾಮಾಕಾಶ ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    ಸ್ವಯಂಚಾಲಿತ logits = wrap_type<v0::MatMul>();
    ಸ್ವಯಂಚಾಲಿತ softmax = wrap_type<v8::Softmax>({logits});
    ಸ್ವಯಂಚಾಲಿತ topk = wrap_type<v11::ತುದಿ_K>({softmax, any_ಒಳಹರಿವು()});
    topk->ಗಣ_output_ಗಾತ್ರ(2);
    ಸ್ವಯಂಚಾಲಿತ ids = wrap_type<v8::ತುಂಡು>({topk->output(1), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು()});
    ಸ್ವಯಂಚಾಲಿತ ids_2d = wrap_type<v0::Squeeze>({ids, any_ಒಳಹರಿವು()});
    ಸ್ವಯಂಚಾಲಿತ probs = wrap_type<v1::Reshape>({softmax, any_ಒಳಹರಿವು()});
    ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ = wrap_type<v0::Squeeze>({probs, any_ಒಳಹರಿವು()});
    ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_batch = wrap_type<v8::Gather>({wrap_type<v3::Shape_ದ>({ದತ್ತಾಂಶ}), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು()});
    ಸ್ವಯಂಚಾಲಿತ ids_ಎಣಿಕೆ = wrap_type<v8::Gather>({wrap_type<v3::Shape_ದ>({ids_2d}), any_ಒಳಹರಿವು(), any_ಒಳಹರಿವು()});
    ಸ್ವಯಂಚಾಲಿತ ಗುರಿ = wrap_type<v0::Concat>({ದತ್ತಾಂಶ_batch, ids_ಎಣಿಕೆ});
    ಸ್ವಯಂಚಾಲಿತ ಪ್ರಸಾರ_2 = wrap_type<v3::ಪ್ರಸಾರ>({ids_2d, ಗುರಿ});
    ಸ್ವಯಂಚಾಲಿತ gather = wrap_type<v8::Gather>({ದತ್ತಾಂಶ, ಪ್ರಸಾರ_2, any_ಒಳಹರಿವು()});
    ಸ್ವಯಂಚಾಲಿತ weights_4d = wrap_type<v0::Unsqueeze>({gather, any_ಒಳಹರಿವು()});
    ಸ್ವಯಂಚಾಲಿತ weights = wrap_type<v1::Reshape>({weights_4d, any_ಒಳಹರಿವು()});
    ಸ್ವಯಂಚಾಲಿತ ಮೊತ್ತ = wrap_type<v1::Reduce_ಮೊತ್ತ>({weights, any_ಒಳಹರಿವು()});
    ಸ್ವಯಂಚಾಲಿತ clamp = wrap_type<v0::Clamp>({ಮೊತ್ತ});
    ಸ್ವಯಂಚಾಲಿತ tile = wrap_type<v0::Tile>({clamp, any_ಒಳಹರಿವು()});
    ಸ್ವಯಂಚಾಲಿತ norm = wrap_type<v1::Divide>({weights, tile});

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ callback = [=](Matcher & m) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & pm = m.get_ನಮೂನೆ_ಮೌಲ್ಯ_map();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಗಂಟು = [&](ಸ್ಥಿರಾಂಕ std::shared_ptr<ov::Node> & p) { ಹಿಂತಿರುಗಿ pm.at(p).get_ಗಂಟು_shared_ptr(); };
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_is = [&](ಸ್ಥಿರಾಂಕ std::shared_ptr<ov::Node> & p, size_t i, ಸ್ಥಿರಾಂಕ std::vector<int64_t> & v) {
            ಹಿಂತಿರುಗಿ constant_is(ಗಂಟು(p)->ಒಳಹರಿವು_ಮೌಲ್ಯ(i), v);
        };
        ಸ್ವಯಂಚಾಲಿತ mm = ov::as_type_ptr<v0::MatMul>(ಗಂಟು(logits));
        ಸ್ವಯಂಚಾಲಿತ sm = ov::as_type_ptr<v8::Softmax>(ಗಂಟು(softmax));
        ಸ್ವಯಂಚಾಲಿತ tk = ov::as_type_ptr<v11::ತುದಿ_K>(ಗಂಟು(topk));
        ಸ್ವಯಂಚಾಲಿತ reduce = ov::as_type_ptr<v1::Reduce_ಮೊತ್ತ>(ಗಂಟು(ಮೊತ್ತ));
        ಸ್ವಯಂಚಾಲಿತ ಎಲ್ಲೆ = ov::as_type_ptr<v0::Clamp>(ಗಂಟು(clamp));
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ shape = mm->get_output_partial_shape(0);
        ಆದರೆ (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_ಒಳಹರಿವು_partial_shape(0).rank() != 4 ||
            mm->get_ಒಳಹರಿವು_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_ಬಹುಲಕ() != v11::ತುದಿ_K::ಬಹುಲಕ_2::MAX ||
            tk->get_ವಿಂಗಡಿಸು_type() != v11::ತುದಿ_K::ವಿಂಗಡಿಸು_Type::ವಿಂಗಡಿಸು_ಮೌಲ್ಯಗಳು || tk->get_stable() ||
            tk->get_ಸೂಚ್ಯಂಕ_element_type() != ov::element::i32 ||
            !tk->output(0).get_ಗುರಿ_inputs().ಖಾಲಿಗಳು()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ int64_t experts = shape[3].get_ಉದ್ದಳತೆ();
        ಸ್ವಯಂಚಾಲಿತ end = ov::as_type_ptr<v0::Constant>(ಗಂಟು(ids)->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(2));
        ಆದರೆ (!end || !end->get_element_type().is_integral_ಸಂಖ್ಯೆ() || ov::shape_ಗಾತ್ರ(end->get_shape()) != 1) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ int64_t k = end->cast_ಸದಿಶ<int64_t>()[0];
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ sorted_shape = tk->get_output_partial_shape(1);
        ಆದರೆ (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_ಉದ್ದಳತೆ() < k ||
            !ಒಳಹರಿವು_is(probs, 1, {1, -1, experts, 1}) ||
            !ಒಳಹರಿವು_is(ದತ್ತಾಂಶ, 1, {0}) || !ಒಳಹರಿವು_is(ids_2d, 1, {0, 1}) ||
            !ಒಳಹರಿವು_is(weights_4d, 1, {0}) || !ಒಳಹರಿವು_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(ಗಂಟು(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(ಗಂಟು(weights))->get_special_zero() ||
            !ಒಳಹರಿವು_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(ಗಂಟು(gather))->get_batch_dims() != 1 ||
            !ಒಳಹರಿವು_is(ದತ್ತಾಂಶ_batch, 1, {0}) || !ಒಳಹರಿವು_is(ದತ್ತಾಂಶ_batch, 2, {0}) ||
            !ಒಳಹರಿವು_is(ids_ಎಣಿಕೆ, 1, {1}) || !ಒಳಹರಿವು_is(ids_ಎಣಿಕೆ, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(ಗಂಟು(ದತ್ತಾಂಶ_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(ಗಂಟು(ids_ಎಣಿಕೆ))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(ಗಂಟು(ಗುರಿ))->get_axis() != 0 ||
            ov::as_type_ptr<v3::ಪ್ರಸಾರ>(ಗಂಟು(ಪ್ರಸಾರ_2))->get_ಪ್ರಸಾರ_spec().m_type != ov::op::ಪ್ರಸಾರ_Type::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!ಒಳಹರಿವು_is(ಮೊತ್ತ, 1, {-1}) && !ಒಳಹರಿವು_is(ಮೊತ್ತ, 1, {3})) ||
            !ಒಳಹರಿವು_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(ಗಂಟು(norm))->get_autob().m_type != ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        ಆದರೆ (!(ಎಲ್ಲೆ->get_min() <= 0.5 * ದ್ವಿಗುಣ(k) / experts && ಎಲ್ಲೆ->get_max() >= 2.0)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಒಳಹರಿವು : tk->output(1).get_ಗುರಿ_inputs()) {
            ಸ್ವಯಂಚಾಲಿತ ತುಂಡು_2 = ov::as_type_ptr<v8::ತುಂಡು>(ಒಳಹರಿವು.get_ಗಂಟು()->shared_from_this());
            ಆದರೆ (!ತುಂಡು_2 || ಒಳಹರಿವು.get_ಸೂಚ್ಯಂಕ() != 0 || ತುಂಡು_2->get_ಒಳಹರಿವು_ಗಾತ್ರ() != 5 ||
                !constant_is(ತುಂಡು_2->ಒಳಹರಿವು_ಮೌಲ್ಯ(1), {0}) || !constant_is(ತುಂಡು_2->ಒಳಹರಿವು_ಮೌಲ್ಯ(2), {k}) ||
                !constant_is(ತುಂಡು_2->ಒಳಹರಿವು_ಮೌಲ್ಯ(3), {1}) ||
                (!constant_is(ತುಂಡು_2->ಒಳಹರಿವು_ಮೌಲ್ಯ(4), {3}) && !constant_is(ತುಂಡು_2->ಒಳಹರಿವು_ಮೌಲ್ಯ(4), {-1}))) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            slices.push_back(ತುಂಡು_2);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        ಸ್ವಯಂಚಾಲಿತ axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        ಸ್ವಯಂಚಾಲಿತ hidden = std::make_shared<v0::Squeeze>(mm->ಒಳಹರಿವು_ಮೌಲ್ಯ(0), axis0);
        ಸ್ವಯಂಚಾಲಿತ routing = std::make_shared<v0::MatMul>(hidden, mm->ಒಳಹರಿವು_ಮೌಲ್ಯ(1), ಸುಳ್ಳು, mm->get_transpose_b());
        ಸ್ವಯಂಚಾಲಿತ probabilities = std::make_shared<v8::Softmax>(routing, -1);
        ಸ್ವಯಂಚಾಲಿತ selected = std::make_shared<v11::ತುದಿ_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::ತುದಿ_K::ಬಹುಲಕ_2::MAX, v11::ತುದಿ_K::ವಿಂಗಡಿಸು_Type::ವಿಂಗಡಿಸು_ಮೌಲ್ಯಗಳು, tk->get_ಸೂಚ್ಯಂಕ_element_type(), ಸುಳ್ಳು);
        ಸ್ವಯಂಚಾಲಿತ ಒಟ್ಟು = std::make_shared<v1::Reduce_ಮೊತ್ತ>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), ನಿಜ);
        ಸ್ವಯಂಚಾಲಿತ normalized = std::make_shared<v1::Divide>(selected->output(0), ಒಟ್ಟು);
        ಸ್ವಯಂಚಾಲಿತ weights_ಹೊರಹೋಗು = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        ಸ್ವಯಂಚಾಲಿತ ids_ಹೊರಹೋಗು = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, ಗಂಟು(norm)}, {hidden, routing, probabilities, selected, ಒಟ್ಟು, normalized, weights_ಹೊರಹೋಗು, ids_ಹೊರಹೋಗು});
        ov::replace_ಗಂಟು(ಗಂಟು(norm), weights_ಹೊರಹೋಗು);
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ತುಂಡು_2 : slices) {
            ov::replace_ಗಂಟು(ತುಂಡು_2, ids_ಹೊರಹೋಗು);
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    };
    ರಿಜಿಸ್ಟರ್_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
