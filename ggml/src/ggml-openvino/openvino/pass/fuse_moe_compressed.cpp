#अन्तर्भाव "fuse_moe_compressed.h"

#अन्तर्भाव <cstring>
#अन्तर्भाव <limits>
#अन्तर्भाव <set>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/graph_util.hpp>
#अन्तर्भाव <openvino/core/rt_info.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/gelu.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/reduce_sum.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/subtract.hpp>
#अन्तर्भाव <openvino/op/swish.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <openvino/pass/constant_folding.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/optional.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/wrap_type.hpp>

#अन्तर्भाव "../op/gather_matmul.hpp"
#अन्तर्भाव "../op/moe_compressed.hpp"

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश pass {

नामाकाश {

संरचना dequant_inputs {
    ov::Output<ov::Node> weight;
    ov::Output<ov::Node> scale;
    ov::Output<ov::Node> zp;
    तार्किक has_zp = असत्य;
    तार्किक ok = असत्य;
};

// Peel the chain built by make_int4_weights/make_int8_weights back to its Constant inputs.
// Grouped weights keep the pre-Reshape rank-4 form [n_expert, n, k/group, group] with scale
// and zp at [n_expert, n, k/group, 1], which is the layout MOECompressed expects. Channel-wise
// weights stay rank-3 with a rank-3 scale and carry no zp.
dequant_inputs unwrap_dequant(अचर ov::Output<ov::Node> & b) {
    dequant_inputs res;

    स्वतः गाँठो = b.get_गाँठो_shared_ptr();
    जबसम्म (ov::is_type<ov::op::v0::Convert>(गाँठो) || ov::is_type<ov::op::v1::Reshape>(गाँठो)) {
        गाँठो = गाँठो->get_आगत_गाँठो_shared_ptr(0);
    }

    स्वतः mul = ov::as_type_ptr<ov::op::v1::Multiply>(गाँठो);
    यदि (!mul) {
        फर्काऊ res;
    }
    res.scale = mul->आगत_मान(1);

    स्वतः lhs = mul->get_आगत_गाँठो_shared_ptr(0);
    यदि (स्वतः sub = ov::as_type_ptr<ov::op::v1::Subtract>(lhs)) {
        // Take the zero point down to its Constant: an integer zp is wrapped in a Convert to f16,
        // and the op wants the integer form. A natively quantized expert instead carries an exact
        // f16 zp (-min/scale) with no integer behind it, which the MoE kernel does not accept.
        स्वतः zp_गाँठो = sub->get_आगत_गाँठो_shared_ptr(1);
        जबसम्म (ov::is_type<ov::op::v0::Convert>(zp_गाँठो)) {
            zp_गाँठो = zp_गाँठो->get_आगत_गाँठो_shared_ptr(0);
        }
        res.zp = zp_गाँठो->output(0);
        res.has_zp = सत्य;
        lhs = sub->get_आगत_गाँठो_shared_ptr(0);
    }
    जबसम्म (ov::is_type<ov::op::v0::Convert>(lhs)) {
        lhs = lhs->get_आगत_गाँठो_shared_ptr(0);
    }
    यदि (!ov::is_type<ov::op::v0::Constant>(lhs)) {
        फर्काऊ res;
    }

    res.weight = lhs->output(0);
    res.ok = res.scale.get_partial_shape().is_static() && res.weight.get_partial_shape().is_static();
    फर्काऊ res;
}

माप_प्रकार logical_k(अचर ov::Shape & shape) {
    फर्काऊ shape.size() == 4 ? shape[2] * shape[3] : shape.back();
}

// Slice a [n_expert, m, ...] compressed weight/scale/zp Constant along axis 1 (m), [begin, end).
// Built by copying raw bytes directly instead of a graph Slice op: CommonOptimizations rewrites
// v8::Slice into v1::StridedSlice for constant folding regardless of disable_constant_folding
// (the rewrite doesn't carry the marking over), and that reference evaluator crashes on
// sub-byte (u4/i4) element types. Each axis-1 "row" (the trailing dims) is confirmed
// byte-aligned here -- weight/scale/zp trailing sizes are always whole groups -- so a
// byte-range memcpy per row is exact for both regular and sub-byte element types.
ov::Output<ov::Node> slice_experts_dim1(अचर ov::Output<ov::Node> & tensor, int64_t begin, int64_t end) {
    स्वतः constant = ov::as_type_ptr<ov::op::v0::Constant>(tensor.get_गाँठो_shared_ptr());
    OPENVINO_ASSERT(constant, "slice_experts_dim1 expects a Constant input");

    अचर स्वतः & shape = constant->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "slice_experts_dim1 expects rank >= 2");
    अचर स्वतः & type = constant->get_element_type();

    माप_प्रकार row_elems = 1;
    लागि (माप_प्रकार i = 2; i < shape.size(); ++i) {
        row_elems *= shape[i];
    }
    अचर माप_प्रकार row_bits = row_elems * type.bitwidth();
    OPENVINO_ASSERT(row_bits % 8 == 0, "slice_experts_dim1: row is not byte-aligned");
    अचर माप_प्रकार row_bytes = row_bits / 8;

    ov::Shape out_shape = shape;
    out_shape[1] = static_cast<माप_प्रकार>(end - begin);

    std::vector<uint8_t> out_data(shape[0] * static_cast<माप_प्रकार>(end - begin) * row_bytes);
    अचर स्वतः * src = static_cast<अचर uint8_t *>(constant->get_data_ptr());
    अचर माप_प्रकार src_row_bytes = shape[1] * row_bytes;
    लागि (माप_प्रकार e = 0; e < shape[0]; ++e) {
        std::स_नकल(out_data.data() + e * static_cast<माप_प्रकार>(end - begin) * row_bytes,
                   src + e * src_row_bytes + static_cast<माप_प्रकार>(begin) * row_bytes,
                   static_cast<माप_प्रकार>(end - begin) * row_bytes);
    }

    फर्काऊ std::make_shared<ov::op::v0::Constant>(type, out_shape, out_data.data());
}

}  // namespace

FuseMoeCompressed::FuseMoeCompressed() {
    using नामाकाश ov::pass::pattern;

    // The gate and up projections each get their own Reshape/Transpose of the hidden state and
    // their own Reshape of the routing ids, so every branch needs its own sub-pattern. On GPU
    // mul_mat_id also converts the activations to f16 before the op and back to f32 after it,
    // so those Converts are matched as optional.
    स्वतः hidden_gate_m = any_आगत();
    स्वतः a_gate_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_gate_m, any_आगत() });
    स्वतः a_gate_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_gate_reshape_m }), any_आगत() });
    स्वतः hidden_up_m = any_आगत();
    स्वतः a_up_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ hidden_up_m, any_आगत() }) }),
          any_आगत() });

    स्वतः gate_w_m = any_आगत();
    स्वतः up_w_m = any_आगत();
    स्वतः down_w_m = any_आगत();
    स्वतः ids_gate_m = any_आगत();
    स्वतः ids_up_m = any_आगत();
    स्वतः ids_down_m = any_आगत();

    स्वतः bgm_gate_m = wrap_type<ov::op::internal::GatherMatmul>({ a_gate_m, gate_w_m, ids_gate_m, any_आगत() });
    स्वतः gate_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_gate_m, any_आगत() }), any_आगत() }) });

    स्वतः silu_m = wrap_type<ov::op::v4::Swish>({ gate_u_m });

    स्वतः bgm_up_m = wrap_type<ov::op::internal::GatherMatmul>({ a_up_m, up_w_m, ids_up_m, any_आगत() });
    स्वतः up_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_up_m, any_आगत() }), any_आगत() }) });
    स्वतः swiglu_m = wrap_type<ov::op::v1::Multiply>({ silu_m, up_u_m });

    स्वतः d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ swiglu_m, any_आगत() }) }),
          any_आगत() });
    स्वतः bgm_down_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, down_w_m, ids_down_m, any_आगत() });
    स्वतः down_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_down_m, any_आगत() }), any_आगत() }) });

    स्वतः routing_m = any_आगत();
    स्वतः weighted_m = wrap_type<ov::op::v1::Multiply>({ down_u_m, routing_m });
    स्वतः जरा_m = wrap_type<ov::op::v1::ReduceSum>({ weighted_m, any_आगत() });

    अचर स्वतः callback = [=](Matcher & m) {
        स्वतः & pm = m.get_pattern_मान_map();

        अचर स्वतः gate = unwrap_dequant(pm.at(gate_w_m));
        अचर स्वतः up = unwrap_dequant(pm.at(up_w_m));
        अचर स्वतः down = unwrap_dequant(pm.at(down_w_m));
        यदि (!gate.ok || !up.ok || !down.ok) {
            फर्काऊ असत्य;
        }

        अचर स्वतः gate_shape = gate.weight.get_shape();
        अचर स्वतः up_shape = up.weight.get_shape();
        अचर स्वतः down_shape = down.weight.get_shape();
        यदि (gate_shape != up_shape || gate_shape.size() < 3 || down_shape.size() < 3) {
            फर्काऊ असत्य;
        }

        // MOECompressed carries one group_size and one has_zp for all three projections, so a
        // model whose down-proj is quantized differently from gate/up cannot be described. This
        // happens when ggml requantizes Q5_K/Q6_K experts to channel-wise int8.
        यदि (gate.has_zp != down.has_zp || gate_shape.size() != down_shape.size()) {
            फर्काऊ असत्य;
        }

        // The kernel only takes an integer zero point (moe_3gemm_swiglu_opt validate_impl).
        यदि (gate.has_zp) {
            स्थिर अचर std::set<ov::element::Type> int_zp_types = { ov::element::u4, ov::element::i4,
                                                                     ov::element::u8, ov::element::i8 };
            यदि (int_zp_types.count(gate.zp.get_element_type()) == 0 ||
                int_zp_types.count(down.zp.get_element_type()) == 0) {
                फर्काऊ असत्य;
            }
        }

        // Config holds a single group_size for all three projections.
        अचर स्वतः group_of = [](अचर dequant_inputs & w) {
            अचर स्वतः s = w.weight.get_shape();
            फर्काऊ s.size() == 4 ? s[3] : logical_k(s);
        };
        यदि (group_of(gate) != group_of(up) || group_of(gate) != group_of(down)) {
            फर्काऊ असत्य;
        }

        // all three branches must route the same hidden state through the same experts
        यदि (pm.at(hidden_gate_m) != pm.at(hidden_up_m)) {
            फर्काऊ असत्य;
        }

        स्वतः ids = pm.at(ids_down_m);
        अचर स्वतः ids_pshape = ids.get_partial_shape();
        यदि (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_length() - 1].is_dynamic()) {
            फर्काऊ असत्य;
        }
        अचर माप_प्रकार शीर्ष_k = ids_pshape[ids_pshape.rank().get_length() - 1].get_length();

        // routing weights arrive as [1, n_tokens, top_k, 1]; the op wants [..., top_k]
        स्वतः routing = pm.at(routing_m);
        अचर स्वतः routing_pshape = routing.get_partial_shape();
        यदि (routing_pshape.rank().is_dynamic() || routing_pshape.rank().get_length() != 4 ||
            routing_pshape[3] != 1) {
            फर्काऊ असत्य;
        }
        // MOE requires routing weights and ids to have the same shape. Drop the trailing 1 of the
        // routing weights and give the ids the leading batch dim, so both become [1, n_tokens, top_k].
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        यदि (ids_pshape.rank().get_length() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        यदि (routing.get_partial_shape() != ids.get_partial_shape()) {
            फर्काऊ असत्य;
        }

        अचर माप_प्रकार down_k = logical_k(down_shape);
        अचर स्वतः down_scale_shape = down.scale.get_shape();
        अचर माप_प्रकार down_groups = down_scale_shape.size() >= 3 ? down_scale_shape[2] : 1;

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_size = logical_k(gate_shape);
        config.inter_size = gate_shape[1];
        config.अङ्क_expert = gate_shape[0];
        config.अङ्क_shared_expert = 0;
        config.शीर्ष_k = शीर्ष_k;
        config.group_size = down_groups <= 1 ? std::numeric_limits<माप_प्रकार>::max() : down_k / down_groups;
        config.has_batch_dim = सत्य;
        config.has_zp = gate.has_zp;
        // dynamic makes the output follow the hidden state, so the plugin can lower this region
        // to f16 together with the rest of the graph
        config.out_type = ov::element::dynamic;

        स्वतः absent_zp = [] {
            स्वतः zp = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{ 0 });
            ov::pass::disable_constant_folding(zp);
            फर्काऊ zp->output(0);
        };

        // MOE takes its output type from the hidden state. Transpose the activations before the
        // f16 Convert that mul_mat_id adds on GPU, so the op stays f32 like the block it replaces
        // and the plugin can lower the whole region uniformly.
        अचर स्वतः a_transpose = pm.at(a_gate_m).get_गाँठो_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_gate_reshape_m), a_transpose->आगत_मान(1));

        अचर ov::OutputVector args = {
            hidden, routing, ids,
            gate.weight,  gate.scale,   gate.has_zp ? gate.zp : absent_zp(),
            up.weight,    up.scale,     up.has_zp ? up.zp : absent_zp(),
            down.weight,  down.scale,   down.has_zp ? down.zp : absent_zp(),
        };

        स्वतः moe = std::make_shared<ov::op::internal::MOECompressed>(args, config);

        // MOE takes its output type from the hidden state, which is f16 on GPU, while the rest of
        // the ggml graph works in f32.
        ov::Output<ov::Node> result = moe->output(0);
        अचर स्वतः जरा_type = m.get_match_जरा()->get_output_element_type(0);
        यदि (result.get_element_type() != जरा_type) {
            result = std::make_shared<ov::op::v0::Convert>(result, जरा_type);
        }

        result.get_गाँठो_shared_ptr()->set_friendly_name(m.get_match_जरा()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result.get_गाँठो_shared_ptr());
        ov::replace_गाँठो(m.get_match_जरा(), result.get_गाँठो_shared_ptr());
        register_new_गाँठो(moe);
        फर्काऊ सत्य;
    };

    register_matcher(std::make_shared<Matcher>(जरा_m, "ov::frontend::ggml::pass::FuseMoeCompressed"), callback);
}

FuseMoeCompressedFusedGateUp::FuseMoeCompressedFusedGateUp() {
    using नामाकाश ov::pass::pattern;

    // A single GatherMatmul computes the fused gate+up projection; the gate/up split happens
    // AFTER the GEMM via two Slice ops on the last (feature) axis (see get_glu_inputs in
    // op/glu_geglu.cpp), unlike FuseMoeCompressed's two-separate-GatherMatmul models.
    स्वतः hidden_m = any_आगत();
    स्वतः a_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_m, any_आगत() });
    स्वतः a_m = wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_reshape_m }), any_आगत() });

    स्वतः gate_up_w_m = any_आगत();
    स्वतः ids_gate_up_m = any_आगत();
    स्वतः bgm_fused_m = wrap_type<ov::op::internal::GatherMatmul>({ a_m, gate_up_w_m, ids_gate_up_m, any_आगत() });
    // Two graph shapes reach here. The rank-4 (stateless) graph restores the batch dim with an
    // Unsqueeze after the Transpose and may Convert afterwards:
    //     GatherMatmul -> Transpose -> Unsqueeze -> [Convert] -> Slice
    // The rank-3 (stateful) graph never drops to a batch dim at all, so there is no Unsqueeze
    // here - it appears later, between the GEGLU and the down Reshape - and the Convert sits on
    // the other side of the Transpose:
    //     GatherMatmul -> Convert -> Transpose -> Slice
    // optional<T>({a, b}) matches T(a, b) or bare a, so one pattern covers both.
    स्वतः gu_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_fused_m }), any_आगत() });
    स्वतः gu_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ gu_t_m, any_आगत() }) });

    स्वतः gate_slice_m = wrap_type<ov::op::v8::Slice>({ gu_u_m, any_आगत(), any_आगत(), any_आगत(), any_आगत() });
    स्वतः up_slice_m = wrap_type<ov::op::v8::Slice>({ gu_u_m, any_आगत(), any_आगत(), any_आगत(), any_आगत() });
    स्वतः gelu_m = wrap_type<ov::op::v7::Gelu>({ gate_slice_m });
    स्वतः geglu_m = wrap_type<ov::op::v1::Multiply>({ gelu_m, up_slice_m });

    // The rank-3 graph inserts the Unsqueeze the gate/up branch lacked right here, before the
    // Reshape that feeds the down projection; the rank-4 graph goes straight from the GEGLU
    // into the Reshape.
    स्वतः d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>(
              { optional<ov::op::v0::Unsqueeze>({ geglu_m, any_आगत() }), any_आगत() }) }),
          any_आगत() });
    स्वतः down_w_m = any_आगत();
    स्वतः ids_down_m = any_आगत();
    स्वतः bgm_down_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, down_w_m, ids_down_m, any_आगत() });
    // Same two shapes as the gate/up branch above.
    स्वतः down_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_down_m }), any_आगत() });
    स्वतः down_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ down_t_m, any_आगत() }) });

    // gemma-4 applies an extra per-expert output scale to the down projection before the
    // router-weight multiply (llama-graph.cpp's ffn_down_exps.scale); FuseMoeCompressed's
    // Qwen-shaped models have no such scale, so this node is specific to this pattern.
    स्वतः down_scale_m = any_आगत();
    स्वतः down_scaled_m = wrap_type<ov::op::v1::Multiply>({ down_u_m, down_scale_m });

    स्वतः routing_m = any_आगत();
    स्वतः weighted_m = wrap_type<ov::op::v1::Multiply>({ down_scaled_m, routing_m });
    स्वतः जरा_m = wrap_type<ov::op::v1::ReduceSum>({ weighted_m, any_आगत() });

    अचर स्वतः callback = [=](Matcher & m) {
        स्वतः & pm = m.get_pattern_मान_map();

        अचर स्वतः gate_up = unwrap_dequant(pm.at(gate_up_w_m));
        अचर स्वतः down = unwrap_dequant(pm.at(down_w_m));
        यदि (!gate_up.ok || !down.ok) {
            फर्काऊ असत्य;
        }

        // oneDNN's weight-decompression GEMM needs a real zero point, so a symmetric (no zp)
        // projection cannot be fused here -- e.g. the GPU-only Q4_0 fallback in
        // ggml_openvino_get_extracted_layout for grouped 8-bit expert weights.
        यदि (!gate_up.has_zp || !down.has_zp) {
            फर्काऊ असत्य;
        }

        अचर स्वतः gate_up_shape = gate_up.weight.get_shape();
        अचर स्वतः down_shape = down.weight.get_shape();
        यदि (gate_up_shape.size() < 3 || down_shape.size() < 3 || gate_up_shape.size() != down_shape.size()) {
            फर्काऊ असत्य;
        }

        // The op reads the zero point straight off a weight port, so it must already be an
        // integer Constant. Requantized experts (GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all)
        // are; a native Q4_K expert keeps an exact f16 zp and is left to the unfused path
        // rather than rounded here -- rounding would put a live node on that port and the
        // expert GEMM would read the wrong zero point.
        स्थिर अचर std::set<ov::element::Type> int_types = { ov::element::u4, ov::element::i4,
                                                               ov::element::u8, ov::element::i8 };
        यदि (int_types.count(gate_up.weight.get_element_type()) == 0 ||
            int_types.count(down.weight.get_element_type()) == 0 ||
            int_types.count(gate_up.zp.get_element_type()) == 0 ||
            int_types.count(down.zp.get_element_type()) == 0) {
            फर्काऊ असत्य;
        }

        अचर स्वतः group_of = [](अचर dequant_inputs & w) {
            अचर स्वतः s = w.weight.get_shape();
            फर्काऊ s.size() == 4 ? s[3] : logical_k(s);
        };
        यदि (group_of(gate_up) != group_of(down)) {
            फर्काऊ असत्य;
        }

        स्वतः gelu_गाँठो = ov::as_type_ptr<ov::op::v7::Gelu>(pm.at(gelu_m).get_गाँठो_shared_ptr());
        यदि (!gelu_गाँठो || gelu_गाँठो->get_approximation_mode() != ov::op::GeluApproximationMode::ERF) {
            फर्काऊ असत्य;
        }

        // Read the actual gate/up split point from the matched Slice nodes instead of assuming
        // inter_size/2, so this stays correct if the model's ffn_dim convention ever changes.
        स्वतः gate_slice = ov::as_type_ptr<ov::op::v8::Slice>(pm.at(gate_slice_m).get_गाँठो_shared_ptr());
        स्वतः up_slice = ov::as_type_ptr<ov::op::v8::Slice>(pm.at(up_slice_m).get_गाँठो_shared_ptr());
        स्वतः gate_end_c = ov::as_type_ptr<ov::op::v0::Constant>(gate_slice->आगत_मान(2).get_गाँठो_shared_ptr());
        स्वतः up_begin_c = ov::as_type_ptr<ov::op::v0::Constant>(up_slice->आगत_मान(1).get_गाँठो_shared_ptr());
        यदि (!gate_end_c || !up_begin_c) {
            फर्काऊ असत्य;
        }
        अचर स्वतः gate_end = gate_end_c->cast_vector<int64_t>().at(0);
        अचर स्वतः up_begin = up_begin_c->cast_vector<int64_t>().at(0);
        यदि (gate_end != up_begin || gate_end <= 0 || gate_end >= static_cast<int64_t>(gate_up_shape[1])) {
            फर्काऊ असत्य;
        }
        अचर int64_t mid = gate_end;
        अचर int64_t full = static_cast<int64_t>(gate_up_shape[1]);

        स्वतः ids = pm.at(ids_down_m);
        अचर स्वतः ids_pshape = ids.get_partial_shape();
        यदि (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_length() - 1].is_dynamic()) {
            फर्काऊ असत्य;
        }
        अचर माप_प्रकार शीर्ष_k = ids_pshape[ids_pshape.rank().get_length() - 1].get_length();

        // Routing weights arrive as [1, n_tokens, top_k, 1] on the rank-4 (stateless) graph and
        // as [n_tokens, top_k, 1] on the rank-3 (stateful) one, which is the same thing without
        // the leading batch dim. Normalise the rank-3 form up to the rank-4 one so everything
        // below - and the op's own config - stays in the shape that is already validated on the
        // stateless path; the batch dim is taken back off the result at the end.
        अचर स्वतः routing_in = pm.at(routing_m);
        अचर स्वतः routing_pshape = routing_in.get_partial_shape();
        अचर स्वतः routing_rank = routing_pshape.rank();
        यदि (routing_rank.is_dynamic() || (routing_rank.get_length() != 4 && routing_rank.get_length() != 3) ||
            routing_pshape[routing_rank.get_length() - 1] != 1) {
            फर्काऊ असत्य;
        }
        अचर तार्किक batchless = routing_rank.get_length() == 3;

        // Fold gemma-4's per-expert output scale into the routing weights: the reduction is
        // sum_e(routing[e] * scale[e] * down_out[e]), and MOECompressed only takes one
        // per-expert weight, so pre-multiply it into routing here (same [.., top_k, 1] shape).
        स्वतः down_scale = pm.at(down_scale_m);
        यदि (down_scale.get_partial_shape() != routing_pshape) {
            फर्काऊ असत्य;
        }
        ov::Output<ov::Node> routing = std::make_shared<ov::op::v1::Multiply>(routing_in, down_scale);
        यदि (batchless) {
            routing = std::make_shared<ov::op::v0::Unsqueeze>(
                routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        यदि (ids_pshape.rank().get_length() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        यदि (routing.get_partial_shape() != ids.get_partial_shape()) {
            फर्काऊ असत्य;
        }

        अचर माप_प्रकार down_k = logical_k(down_shape);
        अचर स्वतः down_scale_shape = down.scale.get_shape();
        अचर माप_प्रकार down_groups = down_scale_shape.size() >= 3 ? down_scale_shape[2] : 1;

        // Split the fused gate_up weight/scale/zp on the output axis. slice_experts_dim1 copies
        // raw bytes out of the Constant, so each half stays a Constant -- which is what the op
        // needs on its weight ports.
        अचर ov::Output<ov::Node> gate_weight = slice_experts_dim1(gate_up.weight, 0, mid);
        अचर ov::Output<ov::Node> up_weight = slice_experts_dim1(gate_up.weight, mid, full);
        अचर ov::Output<ov::Node> gate_scale = slice_experts_dim1(gate_up.scale, 0, mid);
        अचर ov::Output<ov::Node> up_scale = slice_experts_dim1(gate_up.scale, mid, full);
        अचर ov::Output<ov::Node> gate_zp = slice_experts_dim1(gate_up.zp, 0, mid);
        अचर ov::Output<ov::Node> up_zp = slice_experts_dim1(gate_up.zp, mid, full);

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::GEGLU_ERF;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_size = logical_k(gate_up_shape);
        config.inter_size = static_cast<माप_प्रकार>(mid);
        config.अङ्क_expert = gate_up_shape[0];
        config.अङ्क_shared_expert = 0;
        config.शीर्ष_k = शीर्ष_k;
        config.group_size = down_groups <= 1 ? std::numeric_limits<माप_प्रकार>::max() : down_k / down_groups;
        config.has_batch_dim = सत्य;
        config.has_zp = सत्य;
        config.out_type = ov::element::dynamic;

        // Rebuild the activations Transpose before mul_mat_id's f16 GPU Convert, so the op
        // stays f32 like the block it replaces (mirrors FuseMoeCompressed).
        अचर स्वतः a_transpose = pm.at(a_m).get_गाँठो_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_reshape_m), a_transpose->आगत_मान(1));

        // w0 is the activated (gate) lane and w1 the multiplied (up) lane -- see
        // moe_3gemm_swiglu_opt.cpp: "scratch.up = up(x) * silu(gate(x))".
        अचर ov::OutputVector args = {
            hidden,      routing,     ids,
            gate_weight, gate_scale,  gate_zp,
            up_weight,   up_scale,    up_zp,
            down.weight, down.scale,  down.zp,
        };

        स्वतः moe = std::make_shared<ov::op::internal::MOECompressed>(args, config);

        ov::Output<ov::Node> result = moe->output(0);
        // The op was fed the batched form, so it produces [1, n_tokens, hidden]. On the rank-3
        // graph the ReduceSum being replaced is [n_tokens, hidden], so drop the batch dim again.
        यदि (batchless) {
            result = std::make_shared<ov::op::v0::Squeeze>(
                result, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        अचर स्वतः जरा_type = m.get_match_जरा()->get_output_element_type(0);
        यदि (result.get_element_type() != जरा_type) {
            result = std::make_shared<ov::op::v0::Convert>(result, जरा_type);
        }
        यदि (result.get_partial_shape() != m.get_match_जरा()->get_output_partial_shape(0)) {
            फर्काऊ असत्य;
        }

        result.get_गाँठो_shared_ptr()->set_friendly_name(m.get_match_जरा()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result.get_गाँठो_shared_ptr());
        ov::replace_गाँठो(m.get_match_जरा(), result.get_गाँठो_shared_ptr());
        register_new_गाँठो(moe);
        फर्काऊ सत्य;
    };

    register_matcher(
        std::make_shared<Matcher>(जरा_m, "ov::frontend::ggml::pass::FuseMoeCompressedFusedGateUp"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
