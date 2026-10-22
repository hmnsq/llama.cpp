#अन्तर्भाव "op_table.h"

#अन्तर्भाव "utils.h"

#अन्तर्भाव <openvino/op/abs.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/ceiling.hpp>
#अन्तर्भाव <openvino/op/cos.hpp>
#अन्तर्भाव <openvino/op/divide.hpp>
#अन्तर्भाव <openvino/op/exp.hpp>
#अन्तर्भाव <openvino/op/floor.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/gelu.hpp>
#अन्तर्भाव <openvino/op/hswish.hpp>
#अन्तर्भाव <openvino/op/log.hpp>
#अन्तर्भाव <openvino/op/matmul.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/negative.hpp>
#अन्तर्भाव <openvino/op/relu.hpp>
#अन्तर्भाव <openvino/op/sigmoid.hpp>
#अन्तर्भाव <openvino/op/sign.hpp>
#अन्तर्भाव <openvino/op/sin.hpp>
#अन्तर्भाव <openvino/op/softplus.hpp>
#अन्तर्भाव <openvino/op/subtract.hpp>
#अन्तर्भाव <openvino/op/swish.hpp>
#अन्तर्भाव <openvino/op/tanh.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {

std::unordered_map<std::string, CreatorFunction> get_supported_ops() {
    using नामाकाश ov::op;
    प्रत्यागम {
        {"GGML_OP_ADD",             op::translate_add                              },
        {"GGML_OP_ADD1",            op::translate_1to1_match_2_inputs<v1::Add>     },
        {"GGML_OP_ADD_ID",          op::translate_add_id                           },
        {"GGML_OP_CONCAT",          op::translate_concat                           },
        {"GGML_OP_CONT",            op::translate_cont                             },
        {"GGML_OP_DUP",             op::translate_cont                             },
        {"GGML_OP_DIV",             op::translate_div                              },
        {"GGML_OP_FILL",            op::translate_fill                             },
        {"GGML_OP_GET_ROWS",        op::translate_get_rows                         },
        {"GGML_OP_IM2COL",          op::translate_im2col                           },
        {"GGML_OP_IM2COL_3D",       op::translate_im2col_3d                        },
        {"GGML_OP_MUL",             op::translate_1to1_match_2_inputs<v1::Multiply>},
        {"GGML_OP_MUL_MAT",         op::translate_mulmat                           },
        {"GGML_OP_MUL_MAT_ID",      op::translate_mul_mat_id                       },
        {"GGML_OP_PERMUTE",         op::translate_permute                          },
        {"GGML_OP_RESHAPE",         op::translate_reshape                          },
        {"GGML_OP_RMS_NORM",        op::translate_rms_norm                         },
        {"GGML_OP_NORM",            op::translate_norm                             },
        {"GGML_OP_L2_NORM",         op::translate_l2_norm                          },
        {"GGML_OP_SUM_ROWS",        op::translate_sum_rows                         },
        {"GGML_OP_ROPE",            op::translate_rope                             },
        {"GGML_OP_SCALE",           op::translate_scale                            },
        {"GGML_OP_SQR",             op::translate_sqr                              },
        {"GGML_OP_SQRT",            op::translate_sqrt                             },
        {"GGML_OP_SOFT_MAX",        op::translate_soft_max                         },
        {"GGML_OP_ARGSORT",         op::translate_argsort                          },
        {"GGML_OP_SUB",             op::translate_1to1_match_2_inputs<v1::Subtract>},
        {"GGML_OP_TRANSPOSE",       op::translate_transpose                        },
        {"GGML_OP_SIN",             op::translate_1to1_match_1_आदान<v0::Sin>      },
        {"GGML_OP_COS",             op::translate_1to1_match_1_आदान<v0::Cos>      },
        {"GGML_OP_LOG",             op::translate_1to1_match_1_आदान<v0::Log>      },
        {"GGML_OP_MEAN",            op::translate_mean                             },
        {"GGML_OP_SUM",             op::translate_sum                              },
        {"GGML_UNARY_OP_GELU",      op::translate_unary_gelu                       },
        {"GGML_UNARY_OP_GELU_ERF",  op::translate_unary_gelu_erf                   },
        {"GGML_UNARY_OP_GELU_QUICK", op::translate_unary_gelu_quick                },
        {"GGML_UNARY_OP_ELU",       op::translate_unary_elu                        },
        {"GGML_UNARY_OP_HARDSWISH", op::translate_1to1_match_1_आदान<v4::HSwish>   },
        {"GGML_UNARY_OP_HARDSIGMOID", op::translate_unary_hardsigmoid              },
        {"GGML_UNARY_OP_STEP",      op::translate_unary_step                       },
        {"GGML_UNARY_OP_ABS",       op::translate_1to1_match_1_आदान<v0::Abs>      },
        {"GGML_UNARY_OP_SGN",       op::translate_1to1_match_1_आदान<v0::Sign>     },
        {"GGML_UNARY_OP_FLOOR",     op::translate_1to1_match_1_आदान<v0::Floor>    },
        {"GGML_UNARY_OP_CEIL",      op::translate_1to1_match_1_आदान<v0::Ceiling>  },
        {"GGML_UNARY_OP_ROUND",     op::translate_unary_round                      },
        {"GGML_UNARY_OP_EXPM1",     op::translate_unary_expm1                      },
        {"GGML_UNARY_OP_SIGMOID",   op::translate_1to1_match_1_आदान<v0::Sigmoid>  },
        {"GGML_UNARY_OP_SILU",      op::translate_1to1_match_1_आदान<v4::Swish>    },
        {"GGML_UNARY_OP_SOFTPLUS",  op::translate_unary_softplus                   },
        {"GGML_UNARY_OP_TANH",      op::translate_1to1_match_1_आदान<v0::Tanh>     },
        {"GGML_UNARY_OP_EXP",       op::translate_1to1_match_1_आदान<v0::Exp>      },
        {"GGML_UNARY_OP_NEG",       op::translate_1to1_match_1_आदान<v0::Negative> },
        {"GGML_UNARY_OP_RELU",      op::translate_1to1_match_1_आदान<v0::Relu>     },
        {"GGML_OP_VIEW",            op::translate_view                             },
        {"GGML_GLU_OP_SWIGLU",      op::translate_glu_swiglu                       },
        {"GGML_GLU_OP_SWIGLU_OAI",  op::translate_glu_swiglu_oai                   },
        {"GGML_GLU_OP_SWIGLU_CLAMP", op::translate_glu_swiglu_clamp                },
        {"GGML_GLU_OP_GEGLU",       op::translate_glu_geglu                        },
        {"GGML_GLU_OP_GEGLU_QUICK", op::translate_glu_geglu_quick                  },
        {"GGML_OP_SET_ROWS",        op::translate_set_rows                         },
        {"GGML_OP_CPY",             op::translate_cpy                              },
        {"GGML_OP_FLASH_ATTN_EXT",  op::translate_flash_attn_ext                   },
        {"GGML_OP_CLAMP",           op::translate_clamp                            },
        {"GGML_OP_PAD",             op::translate_pad                              },
        {"GGML_OP_SSM_CONV",        op::translate_ssm_conv                         },
        {"GGML_OP_GATED_DELTA_NET", op::translate_gated_delta_net                  },
        {"GGML_OP_REPEAT",          op::translate_repeat                           },
        {"GGML_OP_CUMSUM",          op::translate_cumsum                           },
        {"GGML_OP_FILL",            op::translate_fill                             },
        {"GGML_OP_DIAG",            op::translate_diag                             },
        {"GGML_OP_TRI",             op::translate_tri                              },
        {"GGML_OP_SET",             op::translate_set                              },
        {"GGML_OP_POOL_2D",         op::translate_pool_2d                          },
        {"GGML_OP_ROLL",            op::translate_roll                             },
        {"GGML_OP_UPSCALE",         op::translate_upscale                          },
        {"GGML_OP_CONV_2D",         op::translate_conv_2d                          },
        {"GGML_OP_CONV_2D_DW",      op::translate_conv_2d_dw                       },
        {"GGML_OP_CONV_TRANSPOSE_1D", op::translate_conv_transpose_1d             },
        {"GGML_OP_CONV_TRANSPOSE_2D", op::translate_conv_transpose_2d             },
        {"GGML_OP_CONV_3D",         op::translate_conv_3d                          },
        // solve_tri has accuracy issues on GPU
        // {"GGML_OP_SOLVE_TRI",       op::translate_solve_tri                        },
    };
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
