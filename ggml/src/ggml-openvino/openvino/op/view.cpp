#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <openvino/op/concat.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/gather.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/shape_of.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_view(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    எனில் (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        திரும்பு {context.get_உள்ளீடு(0)};
    }

    எனில் (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        மாறிலி std::string & vname = context.get_name();
        எனில் (vname.find("ffn_moe_weighted") != std::string::npos) {
            auto src_ps = context.get_உள்ளீடு_shape(0);
            auto dst_ps = context.get_output_shape();
            எனில் (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                auto sst = context.get_உள்ளீடு_stride(0);
                auto dst = context.get_output_stride();
                size_t voff = context.get_output_op_offset();
                auto ss = src_ps.to_shape();
                auto dd = dst_ps.to_shape();
                மாறிலி size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                மாறிலி auto in_rank = context.get_உள்ளீடு(0).get_partial_shape().rank();
                மாறிலி முழு axis_shift =
                    in_rank.is_static() ? (முழு) nd - (முழு) in_rank.get_length() : 0;
                தருக்க shift_ok = axis_shift >= 0 && (size_t) axis_shift < nd;
                க்கு (முழு a = 0; a < axis_shift && shift_ok; ++a) {
                    shift_ok = (ss[a] == 1 && dd[a] == 1);
                }
                எனில் (shift_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<தருக்க> used(nd, பொய்);
                    std::vector<முழு> dst_to_src(nd, -1);
                    தருக்க ok = உண்மை;
                    க்கு (size_t d = 0; d < nd; ++d) {
                        எனில் (dd[d] == 1) {
                            தொடர்;
                        }
                        முழு found = -1;
                        க்கு (size_t s = 0; s < nd; ++s) {
                            எனில் (!used[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                found = (முழு) s;
                                நிறுத்து;
                            }
                        }
                        எனில் (found < 0) {
                            ok = பொய்;
                            நிறுத்து;
                        }
                        used[found] = உண்மை;
                        dst_to_src[d] = found;
                    }
                    முழு dropped = -1;
                    எனில் (ok) {
                        க்கு (size_t s = 0; s < nd; ++s) {
                            எனில் (!used[s] && ss[s] > 1) {
                                எனில் (dropped >= 0) {
                                    ok = பொய்;
                                    நிறுத்து;
                                }
                                dropped = (முழு) s;
                            }
                        }
                    }
                    எனில் (ok && dropped >= 0) {
                        மாறிலி size_t dstr = sst[dropped];
                        மாறிலி int64_t dsz = (int64_t) ss[dropped];
                        எனில் (dstr > 0 && voff % dstr == 0) {
                            மாறிலி int64_t sel = (int64_t) (voff / dstr);
                            எனில் (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::Slice>(
                                    context.get_உள்ளீடு(0),
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
                                மாறிலி int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                முழு dst_ov_axis = (dyn != -1) ? ((முழு) nd - 1 - (முழு) dyn) : -1;
                                முழு src_ov_axis = (dst_ov_axis >= axis_shift && dst_ov_axis < (முழு) nd)
                                                      ? dst_to_src[dst_ov_axis]
                                                      : -1;
                                எனில் (dst_ov_axis >= 0 && src_ov_axis >= axis_shift) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    auto sl_shape = std::make_shared<ov::op::v3::ShapeOf>(sl, ov::element::i64);
                                    auto tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_shift}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::OutputVector parts;
                                    க்கு (முழு a = axis_shift; a < (முழு) nd; ++a) {
                                        எனில் (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } இல்லையெனில் {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    auto dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, பொய்);
                                    திரும்பு rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_shift, dd.end());
                                auto dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, பொய்);
                                திரும்பு rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        திரும்பு {context.get_உள்ளீடு(0)};
    }

    auto உள்ளீடு = context.get_உள்ளீடு(0);
    auto src_shape = context.get_உள்ளீடு_shape(0);
    auto dst_shape = context.get_output_shape();

    எனில் (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        திரும்பு {உள்ளீடு};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    க்கு (int64_t i = 0; i < src_shape.rank().get_length(); ++i) {
        எனில் (src_shape[i].is_dynamic()) {
            திரும்பு {உள்ளீடு};
        }
        src_elems *= src_shape[i].get_length();
    }
    க்கு (int64_t i = 0; i < dst_shape.rank().get_length(); ++i) {
        எனில் (dst_shape[i].is_dynamic()) {
            திரும்பு {உள்ளீடு};
        }
        dst_elems *= dst_shape[i].get_length();
    }

    எனில் (dst_elems >= src_elems) {
        திரும்பு {உள்ளீடு};
    }

    auto src_stride = context.get_உள்ளீடு_stride(0);
    auto dst_stride = context.get_output_stride();
    size_t view_offset = context.get_output_op_offset();

    தருக்க same_stride = (src_stride.size() == dst_stride.size());
    எனில் (same_stride) {
        க்கு (size_t i = 0; i < src_stride.size(); ++i) {
            எனில் (src_stride[i] != dst_stride[i]) {
                same_stride = பொய்;
                நிறுத்து;
            }
        }
    }

    எனில் (!same_stride) {
        திரும்பு {உள்ளீடு};
    }

    auto src_ov_shape = src_shape.to_shape();
    auto dst_ov_shape = dst_shape.to_shape();
    size_t ndims = src_ov_shape.size();
    எனில் (dst_ov_shape.size() != ndims) {
        திரும்பு {உள்ளீடு};
    }

    std::vector<முழு> diff_dims;
    க்கு (size_t i = 0; i < ndims; ++i) {
        எனில் (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<முழு>(i));
        }
    }

    எனில் (diff_dims.size() != 1) {
        திரும்பு {உள்ளீடு};
    }

    முழு slice_dim = diff_dims[0];
    int64_t dim_size = static_cast<int64_t>(src_ov_shape[slice_dim]);

    size_t ov_stride_for_dim = 1;
    க்கு (size_t i = slice_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_size = src_stride.back();
    எனில் (elem_size == 0) {
        elem_size = 1;
    }

    int64_t begin_val = 0;
    எனில் (ov_stride_for_dim > 0 && elem_size > 0) {
        begin_val = static_cast<int64_t>((view_offset / elem_size) / ov_stride_for_dim);
    }
    int64_t end_val = begin_val + static_cast<int64_t>(dst_ov_shape[slice_dim]);

    எனில் (begin_val < 0 || end_val > dim_size) {
        திரும்பு {உள்ளீடு};
    }

    auto sliced =
        std::make_shared<ov::op::v8::Slice>(உள்ளீடு, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

    sliced->set_friendly_name(context.get_output_name());
    திரும்பு {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
