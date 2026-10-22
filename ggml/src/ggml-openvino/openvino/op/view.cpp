#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/slice.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_view(ثابت NodeContext & context) {
    num_inputs_check(context, 1, 1);

    ܐܢ (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        ܦܢܐ {context.get_input(0)};
    }

    ܐܢ (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        ثابت std::string & vname = context.get_name();
        ܐܢ (vname.find("ffn_moe_weighted") != std::string::npos) {
            auto src_ps = context.get_input_shape(0);
            auto dst_ps = context.get_output_shape();
            ܐܢ (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                auto sst = context.get_input_stride(0);
                auto dst = context.get_output_stride();
                size_t voff = context.get_output_op_offset();
                auto ss = src_ps.to_shape();
                auto dd = dst_ps.to_shape();
                ثابت size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                ثابت auto in_rank = context.get_input(0).get_partial_shape().rank();
                ثابت صحيح axis_shift =
                    in_rank.is_static() ? (صحيح) nd - (صحيح) in_rank.get_length() : 0;
                منطقي shift_ok = axis_shift >= 0 && (size_t) axis_shift < nd;
                لكل (صحيح a = 0; a < axis_shift && shift_ok; ++a) {
                    shift_ok = (ss[a] == 1 && dd[a] == 1);
                }
                ܐܢ (shift_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<منطقي> used(nd, ܕܓܠܐ);
                    std::vector<صحيح> dst_to_src(nd, -1);
                    منطقي ok = ܫܪܪܐ;
                    لكل (size_t d = 0; d < nd; ++d) {
                        ܐܢ (dd[d] == 1) {
                            استمر;
                        }
                        صحيح found = -1;
                        لكل (size_t s = 0; s < nd; ++s) {
                            ܐܢ (!used[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                found = (صحيح) s;
                                اكسر;
                            }
                        }
                        ܐܢ (found < 0) {
                            ok = ܕܓܠܐ;
                            اكسر;
                        }
                        used[found] = ܫܪܪܐ;
                        dst_to_src[d] = found;
                    }
                    صحيح dropped = -1;
                    ܐܢ (ok) {
                        لكل (size_t s = 0; s < nd; ++s) {
                            ܐܢ (!used[s] && ss[s] > 1) {
                                ܐܢ (dropped >= 0) {
                                    ok = ܕܓܠܐ;
                                    اكسر;
                                }
                                dropped = (صحيح) s;
                            }
                        }
                    }
                    ܐܢ (ok && dropped >= 0) {
                        ثابت size_t dstr = sst[dropped];
                        ثابت int64_t dsz = (int64_t) ss[dropped];
                        ܐܢ (dstr > 0 && voff % dstr == 0) {
                            ثابت int64_t sel = (int64_t) (voff / dstr);
                            ܐܢ (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::Slice>(
                                    context.get_input(0),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel + 1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {dropped - axis_shift}));
                                // Build the reshape target from the (concrete) dst shape, but
                                // keep the dynamic token axis dynamic instead of freezing it
                                // to the captured n_tokens. Without this the constant dst
                                // shape bakes in the prefill token count and the static value
                                // flows downstream, turning every later decoder layer static
                                // (the GPU in-place-concat KV-cache bug). The token axis is
                                // PERMUTED between the sliced input and the dst (e.g. input
                                // [1,tok,expert,emb] -> dst [1,1,tok,emb]), so special_zero
                                // (which copies the same-position dim) is not enough: pull the
                                // dynamic dim from the correct SOURCE axis via ShapeOf+Gather
                                // and place it at the dst token position.
                                ثابت int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                صحيح dst_ov_axis = (dyn != -1) ? ((صحيح) nd - 1 - (صحيح) dyn) : -1;
                                صحيح src_ov_axis = (dst_ov_axis >= axis_shift && dst_ov_axis < (صحيح) nd)
                                                      ? dst_to_src[dst_ov_axis]
                                                      : -1;
                                ܐܢ (dst_ov_axis >= 0 && src_ov_axis >= axis_shift) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    auto sl_shape = std::make_shared<ov::op::v3::ShapeOf>(sl, ov::element::i64);
                                    auto tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_shift}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::OutputVector parts;
                                    لكل (صحيح a = axis_shift; a < (صحيح) nd; ++a) {
                                        ܐܢ (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } ܐܚܪܢܐ {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    auto dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, ܕܓܠܐ);
                                    ܦܢܐ rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_shift, dd.end());
                                auto dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, ܕܓܠܐ);
                                ܦܢܐ rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        ܦܢܐ {context.get_input(0)};
    }

    auto input = context.get_input(0);
    auto src_shape = context.get_input_shape(0);
    auto dst_shape = context.get_output_shape();

    ܐܢ (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        ܦܢܐ {input};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    لكل (int64_t i = 0; i < src_shape.rank().get_length(); ++i) {
        ܐܢ (src_shape[i].is_dynamic()) {
            ܦܢܐ {input};
        }
        src_elems *= src_shape[i].get_length();
    }
    لكل (int64_t i = 0; i < dst_shape.rank().get_length(); ++i) {
        ܐܢ (dst_shape[i].is_dynamic()) {
            ܦܢܐ {input};
        }
        dst_elems *= dst_shape[i].get_length();
    }

    ܐܢ (dst_elems >= src_elems) {
        ܦܢܐ {input};
    }

    auto src_stride = context.get_input_stride(0);
    auto dst_stride = context.get_output_stride();
    size_t view_offset = context.get_output_op_offset();

    منطقي same_stride = (src_stride.size() == dst_stride.size());
    ܐܢ (same_stride) {
        لكل (size_t i = 0; i < src_stride.size(); ++i) {
            ܐܢ (src_stride[i] != dst_stride[i]) {
                same_stride = ܕܓܠܐ;
                اكسر;
            }
        }
    }

    ܐܢ (!same_stride) {
        ܦܢܐ {input};
    }

    auto src_ov_shape = src_shape.to_shape();
    auto dst_ov_shape = dst_shape.to_shape();
    size_t ndims = src_ov_shape.size();
    ܐܢ (dst_ov_shape.size() != ndims) {
        ܦܢܐ {input};
    }

    std::vector<صحيح> diff_dims;
    لكل (size_t i = 0; i < ndims; ++i) {
        ܐܢ (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<صحيح>(i));
        }
    }

    ܐܢ (diff_dims.size() != 1) {
        ܦܢܐ {input};
    }

    صحيح slice_dim = diff_dims[0];
    int64_t dim_size = static_cast<int64_t>(src_ov_shape[slice_dim]);

    size_t ov_stride_for_dim = 1;
    لكل (size_t i = slice_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_size = src_stride.back();
    ܐܢ (elem_size == 0) {
        elem_size = 1;
    }

    int64_t begin_val = 0;
    ܐܢ (ov_stride_for_dim > 0 && elem_size > 0) {
        begin_val = static_cast<int64_t>((view_offset / elem_size) / ov_stride_for_dim);
    }
    int64_t end_val = begin_val + static_cast<int64_t>(dst_ov_shape[slice_dim]);

    ܐܢ (begin_val < 0 || end_val > dim_size) {
        ܦܢܐ {input};
    }

    auto sliced =
        std::make_shared<ov::op::v8::Slice>(input, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

    sliced->set_friendly_name(context.get_output_name());
    ܦܢܐ {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
