#ضمن "ggml-vulkan-common.h"

ساكن vk_device_architecture get_device_architecture(ثابت vk::PhysicalDevice& device) {
    vk::PhysicalDeviceProperties props = device.getProperties();

    إذا (props.vendorID == VK_VENDOR_ID_AMD) {
        ثابت std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        منطقي amd_shader_core_properties = خطأ;
        منطقي integer_dot_حاصلضرب = خطأ;
        منطقي subgroup_اندازه_control = خطأ;
        منطقي shader_float8 = خطأ;

        لكل (ثابت خودکار& properties : ext_props) {
            إذا (strcmp("VK_AMD_shader_core_properties", properties.extensionName) == 0) {
                amd_shader_core_properties = درست;
            } وإلا إذا (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_حاصلضرب = درست;
            } وإلا إذا (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_اندازه_control = درست;
            } وإلا إذا (strcmp("VK_EXT_shader_float8", properties.extensionName) == 0) {
                shader_float8 = درست;
            }
        }

        إذا (!amd_shader_core_properties || !integer_dot_حاصلضرب || !subgroup_اندازه_control) {
            أرجع vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceShaderCorePropertiesAMD shader_core_props_amd;
        vk::Physical_Device_Shader_Integer_Dot_حاصلضرب_Properties_KHR integer_dot_props;
        vk::Physical_Device_Subgroup_اندازه_Control_Properties_EXT subgroup_اندازه_control_props;

        props2.p_بعدی = &shader_core_props_amd;
        shader_core_props_amd.p_بعدی = &integer_dot_props;
        integer_dot_props.p_بعدی = &subgroup_اندازه_control_props;

        device.getProperties2(&props2);

        إذا (subgroup_اندازه_control_props.max_Subgroup_اندازه == 64 && subgroup_اندازه_control_props.min_Subgroup_اندازه == 64) {
            أرجع vk_device_architecture::AMD_GCN;
        }
        إذا (subgroup_اندازه_control_props.max_Subgroup_اندازه == 64 && subgroup_اندازه_control_props.min_Subgroup_اندازه == 32) {
            // RDNA
            إذا (shader_core_props_amd.wavefronts_در_هر_Simd == 20) {
                أرجع vk_device_architecture::AMD_RDNA1;
            }
            إذا (shader_float8) {
                أرجع vk_device_architecture::AMD_RDNA4;
            }
            إذا (integer_dot_props.integer_Dot_Product4x8_بیت_Packed_Mixed_Signedness_Accelerated) {
                أرجع vk_device_architecture::AMD_RDNA3;
            }
            أرجع vk_device_architecture::AMD_RDNA2;
        }
    } وإلا إذا (props.vendorID == VK_VENDOR_ID_INTEL) {
        ثابت std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        منطقي subgroup_اندازه_control = خطأ;
        منطقي integer_dot_حاصلضرب = خطأ;

        لكل (ثابت خودکار& properties : ext_props) {
            إذا (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_اندازه_control = درست;
            } وإلا إذا (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_حاصلضرب = درست;
            }
        }

        إذا (!subgroup_اندازه_control || !integer_dot_حاصلضرب) {
            أرجع vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::Physical_Device_Subgroup_اندازه_Control_Properties_EXT subgroup_اندازه_control_props;
        vk::Physical_Device_Shader_Integer_Dot_حاصلضرب_Properties_KHR integer_dot_props;

        props2.p_بعدی = &subgroup_اندازه_control_props;
        subgroup_اندازه_control_props.p_بعدی = &integer_dot_props;
        device.getProperties2(&props2);

        إذا (subgroup_اندازه_control_props.min_Subgroup_اندازه == 16) {
            // Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8.
            // Minimum subgroup size matches the SIMD width so we distinguish architecture by checking this value.
            // https://www.intel.com/content/www/us/en/content-details/824434/2024-intel-tech-tour-xe2-and-lunar-lake-s-gpu.html
            // https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
            أرجع vk_device_architecture::INTEL_XE2;
        } وإلا إذا (subgroup_اندازه_control_props.min_Subgroup_اندازه == 8 &&
                 integer_dot_حاصلضرب && integer_dot_props.integer_Dot_Product4x8_بیت_Packed_Signed_Accelerated) {
            أرجع vk_device_architecture::INTEL_XE1;
        }
    } وإلا إذا (props.vendorID == VK_VENDOR_ID_NVIDIA) {
        ثابت std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        منطقي cooperative_ماتریس = خطأ;
        منطقي sm_builtins = خطأ;

        // Detect "pre-turing" based on lack of coopmat support.
        لكل (ثابت خودکار& properties : ext_props) {
            إذا (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_ماتریس = درست;
            } وإلا إذا (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = درست;
            }
        }

        إذا (!cooperative_ماتریس) {
            أرجع vk_device_architecture::NVIDIA_PRE_TURING;
        }

        إذا (sm_builtins) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;

            props2.p_بعدی = &sm_props;

            device.getProperties2(&props2);

            // Turing has 32, following architectures have 48
            إذا (sm_props.shader_Warps_در_هر_SM == 32) {
                أرجع vk_device_architecture::NVIDIA_TURING;
            }
        }
    } وإلا إذا(props.vendorID == VK_VENDOR_ID_QUALCOMM){
        ثابت std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        منطقي cooperative_ماتریس = خطأ;
        منطقي cooperative_ماتریس_conversion = خطأ;

        لكل (ثابت خودکار& properties : ext_props) {
            إذا (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_ماتریس = درست;
            } وإلا إذا (strcmp("VK_QCOM_cooperative_matrix_conversion", properties.extensionName) == 0) {
                cooperative_ماتریس_conversion = درست;
            }
        }

        إذا (cooperative_ماتریس && cooperative_ماتریس_conversion) {
            أرجع vk_device_architecture::QUALCOMM_ADRENO;
        }
    }
    أرجع vk_device_architecture::OTHER;
}

منطقي ggml_vk_lightning_indexer_k_type_supported(ggml_type type) {
    أرجع std::find(lightning_indexer_k_types.begin(), lightning_indexer_k_types.end(), type) != lightning_indexer_k_types.end();
}
فراغ ggml_vk_print_device_fault_info(ثابت vk_device& device) {
    إذا (!device->device_fault || !device->pfn_vkGetDeviceFaultInfoEXT) {
        أرجع;
    }

    VkDeviceFaultCountsEXT fault_counts {};
    fault_counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    Vk_نتیجه res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, اشاره_تهی);
    إذا (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (counts) failed: %d\n", res);
        أرجع;
    }

    std::vector<Vk_Device_Fault_آدرس_Info_EXT> آدرس_infos(fault_counts.آدرس_Info_شمار);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor_infos(fault_counts.vendor_Info_شمار);

    VkDeviceFaultInfoEXT fault_info {};
    fault_info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    fault_info.p_آدرس_Infos = آدرس_infos.داده();
    fault_info.pVendorInfos = vendor_infos.داده();

    res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, &fault_info);
    إذا (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (info) failed: %d\n", res);
        أرجع;
    }

    إذا (fault_counts.آدرس_Info_شمار == 0 && fault_counts.vendor_Info_شمار == 0 && fault_info.description[0] == '\0') {
        أرجع;
    }

    إذا (fault_info.description[0] != '\0') {
        GGML_LOG_ERROR("ggml_vulkan: device fault on %s: %s\n", device->name.c_str(), fault_info.description);
    }

    لكل (uint32_t i = 0; i < fault_counts.آدرس_Info_شمار; i++) {
        ثابت خودکار& info = آدرس_infos[i];
        GGML_LOG_CONT("  address fault %u: type=%d address=0x%llx precision=0x%llx\n",
                i, (صحيح)info.آدرس_Type,
                (بیعلامت دراز دراز)info.reported_آدرس,
                (بیعلامت دراز دراز)info.آدرس_Precision);
    }
    لكل (uint32_t i = 0; i < fault_counts.vendor_Info_شمار; i++) {
        ثابت خودکار& info = vendor_infos[i];
        GGML_LOG_CONT("  vendor fault %u: %s (code=0x%llx data=0x%llx)\n",
                i, info.description,
                (بیعلامت دراز دراز)info.vendor_Fault_کد,
                (بیعلامت دراز دراز)info.vendor_Fault_داده);
    }
}
uint64_t ggml_vk_get_گره_flops(ثابت ggml_tensor * گره) {
    إذا (گره->op == GGML_OP_MUL_MAT || گره->op == GGML_OP_MUL_MAT_ID) {
        ثابت uint64_t m     = گره->ne[0];
        ثابت uint64_t n     = گره->ne[1];
        ثابت uint64_t k     = گره->src[1]->ne[0];
        ثابت uint64_t batch = گره->ne[2] * گره->ne[3];
        أرجع m * n * (k + (k - 1)) * batch;
    }
    إذا (گره->op == GGML_OP_CONV_2D || گره->op == GGML_OP_CONV_TRANSPOSE_2D) {
        ثابت ggml_tensor * knl = گره->src[0];
        ثابت uint64_t Cout  = گره->ne[2];
        ثابت uint64_t اندازه_K = گره->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        ثابت uint64_t اندازه_N_2 = گره->ne[3] * گره->ne[0] * گره->ne[1];
        أرجع Cout * اندازه_N_2 * (اندازه_K + (اندازه_K - 1));
    }
    إذا (گره->op == GGML_OP_CONV_3D) {
        ثابت ggml_tensor * knl = گره->src[0];
        ثابت uint64_t OC     = ggml_get_op_params_i32(گره, 11);
        ثابت uint64_t IC     = ggml_get_op_params_i32(گره, 9);
        ثابت uint64_t اندازه_K = IC * knl->ne[0] * knl->ne[1] * knl->ne[2];
        ثابت uint64_t اندازه_N_2 = گره->ne[3] / OC * گره->ne[0] * گره->ne[1] * گره->ne[2];
        أرجع OC * اندازه_N_2 * (اندازه_K + (اندازه_K - 1));
    }
    إذا (گره->op == GGML_OP_FLASH_ATTN_EXT) {
        ثابت ggml_tensor * q = گره->src[0];
        ثابت ggml_tensor * k = گره->src[1];
        ثابت ggml_tensor * v = گره->src[2];
        أرجع 2ull * q->ne[1] * q->ne[2] * (k->ne[0] + v->ne[0]) * k->ne[1] * q->ne[3];
    }
    أرجع 0;
}
فراغ ggml_vk_print_گره_list(ثابت ggml_cgraph * cgraph, صحيح start, صحيح end) {
    uint64_t مجموع_flops = 0;
    صحيح n_ops = 0;
    لكل (صحيح j = start; j <= end && j < cgraph->n_nodes; j++) {
        uint64_t flops = ggml_vk_get_گره_flops(cgraph->nodes[j]);
        مجموع_flops += flops;
        n_ops++;
        إذا (flops > 0) {
            GGML_LOG_CONT("  node %d: %s (%s) [%.2f GFLOP]\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op),
                    flops / 1e9);
        } وإلا {
            GGML_LOG_CONT("  node %d: %s (%s)\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op));
        }
    }
    GGML_LOG_CONT("  total: %d ops, %.2f GFLOP\n", n_ops, مجموع_flops / 1e9);
}
فراغ ggml_vk_print_device_گمشده_info(ثابت vk_device& device) {
    ggml_vk_print_device_fault_info(device);
    إذا (device->serialize_submissions && device->diag_cgraph != اشاره_تهی && device->diag_prev_شروع >= 0) {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s, likely caused by previous submission (nodes %d to %d):\n",
                device->name.c_str(), device->diag_prev_شروع, device->diag_prev_انتها);
        ggml_vk_print_گره_list(device->diag_cgraph, device->diag_prev_شروع, device->diag_prev_انتها);
    } وإلا {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s\n", device->name.c_str());
    }
}
فراغ * ثابت vk_ptr_پایه = (فراغ *)(uintptr_t) 0x1000;  // NOLINT

uint64_t vk_tensor_جابه_جایی(ثابت ggml_tensor * tensor) {
    إذا (tensor->view_src) {
        أرجع (uint8_t *) tensor->view_src->داده - (uint8_t *) vk_ptr_پایه;
    }
    أرجع (uint8_t *) tensor->داده - (uint8_t *) vk_ptr_پایه;
}

size_t ggml_vk_tensor_میانگیر_جابه_جایی(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t) {
    // vk_tensor_offset() is relative to vk_ptr_base, but mapped host tensors need an offset relative to their Vulkan buffer.
    إذا (ctx->device->uma) {
        vk_میانگیر buf = اشاره_تهی;
        size_t off = 0;
        ggml_vk_host_get(ctx->device, t->داده, buf, off);
        إذا (buf) {
            أرجع off;
        }
    }
    أرجع (size_t)(vk_tensor_جابه_جایی(t) + t->view_offs);
}
size_t ggml_vk_descriptor_جابه_جایی(size_t tensor_جابه_جایی, size_t alignment, size_t type_اندازه) {
    // Move the descriptor back until its distance to the tensor is divisible by the tensor type size.
    size_t descriptor_جابه_جایی = tensor_جابه_جایی & ~(alignment - 1);
    طالما ((tensor_جابه_جایی - descriptor_جابه_جایی) % type_اندازه != 0) {
        GGML_ASSERT(descriptor_جابه_جایی >= alignment);
        descriptor_جابه_جایی -= alignment;
    }

    أرجع descriptor_جابه_جایی;
}
uint32_t get_misalign_بایتها(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t) {
    ثابت size_t tensor_جابه_جایی = ggml_vk_tensor_میانگیر_جابه_جایی(ctx, t);
    ثابت size_t descriptor_جابه_جایی = ggml_vk_descriptor_جابه_جایی(
        tensor_جابه_جایی, ctx->device->properties.limits.min_Storage_میانگیر_جابه_جایی_Alignment, ggml_type_اندازه(t->type));
    GGML_ASSERT(tensor_جابه_جایی - descriptor_جابه_جایی <= UINT32_MAX);
    أرجع tensor_جابه_جایی - descriptor_جابه_جایی;
}

uint32_t ggml_vk_concat_unit_اندازه(ggml_type type) {
    ثابت uint32_t type_اندازه = ggml_type_اندازه(type);

    إذا (!ggml_is_quantized(type)) {
        أرجع type_اندازه;
    }

    // Use the widest existing concat shader that evenly divides a quant block.
    إذا (type_اندازه % 8 == 0) {
        أرجع 8;
    }
    إذا (type_اندازه % 4 == 0) {
        أرجع 4;
    }
    إذا (type_اندازه % 2 == 0) {
        أرجع 2;
    }
    أرجع 1;
}
منطقي ggml_vk_concat_supported(ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst) {
    إذا (src0->type != src1->type || src0->type != dst->type) {
        أرجع خطأ;
    }

    إذا (!ggml_is_quantized(src0->type)) {
        ثابت size_t type_اندازه = ggml_type_اندازه(src0->type);
        أرجع type_اندازه == 1 || type_اندازه == 2 || type_اندازه == 4 || type_اندازه == 8;
    }

    // Quantized tensor rows are block-aligned when created.
    أرجع ggml_is_contiguous_ردیفها(src0) && ggml_is_contiguous_ردیفها(src1) && ggml_is_contiguous_ردیفها(dst);
}
ساكن منطقي vk_instance_initialized = خطأ;

vk_instance_t vk_instance;

ساكن Vk_Device_اندازه ggml_vk_get_max_میانگیر_range(ثابت ggml_backend_vk_context * ctx, ثابت vk_میانگیر &buf, ثابت Vk_Device_اندازه جابه_جایی) {
    ثابت Vk_Device_اندازه range = std::min(Vk_Device_اندازه{buf->size - جابه_جایی},
                                        Vk_Device_اندازه{ctx->device->properties.limits.max_Storage_میانگیر_Range});
    أرجع range;
}

فراغ ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx) {
    // Use waitForFences while most of the graph executes. Hopefully the CPU can sleep
    // during this wait.
    إذا (ctx->almost_ready_fence_pending) {
        VK_CHECK(ctx->device->device.waitForFences({ ctx->almost_ready_fence }, درست, UINT64_MAX), "almost_ready_fence", ctx->device);
        ctx->device->device.resetFences({ ctx->almost_ready_fence });
        ctx->almost_ready_fence_pending = خطأ;
    }

    // Spin (w/pause) waiting for the graph to finish executing.
    vk::نتیجه_2 نتیجه;
    لكل (;;) {
        حاول {
            نتیجه = ctx->device->device.get_Fence_وضعیت(ctx->fence);
        } امسك (vk::Device_گمشده_Error &) {
            ggml_vk_print_device_گمشده_info(ctx->device);
            GGML_LOG_ERROR("ggml_vulkan: getFenceStatus at %s:%d\n", __FILE__, __LINE__);
            ارم;
        }
        إذا (نتیجه == vk::نتیجه_2::eSuccess) {
            اكسر;
        }
        إذا (نتیجه != vk::نتیجه_2::eNotReady) {
            GGML_LOG_ERROR("ggml_vulkan: error %s at %s:%d\n", to_string(نتیجه).c_str(), __FILE__, __LINE__);
            ارم vk::SystemError(vk::make_error_code(نتیجه), "ggml_vulkan: getFenceStatus");
        }
        لكل (uint32_t i = 0; i < 100; ++i) {
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
        }
    }
    ctx->device->device.resetFences({ ctx->fence });
}

ساكن منطقي ggml_vk_strip_decode_وکتور(ثابت uint32_t * code, size_t کلمه_شمار, std::vector<uint32_t> & بیرون) {
    ساكن ثابت حرف k_Decode_وکتور_Ext[] = "SPV_NV_cooperative_matrix_decode_vector";

    إذا (کلمه_شمار < 5) {
        أرجع خطأ;
    }

    منطقي استفاده_میکند_decode_وکتور = خطأ;
    لكل (size_t pos = 5; pos < کلمه_شمار; ) {
        uint32_t کلمه = code[pos];
        uint32_t wc   = کلمه >> spv::کلمه_شمار_لغزش;
        uint32_t op   = کلمه & spv::Op_کد_ماسک;
        GGML_ASSERT(wc > 0 && pos + wc <= کلمه_شمار);
        إذا (op == spv::OpExtension && wc >= 2) {
            ثابت حرف * s = reinterpret_cast<ثابت حرف *>(&code[pos + 1]);
            إذا (strcmp(s, k_Decode_وکتور_Ext) == 0) {
                استفاده_میکند_decode_وکتور = درست;
                اكسر;
            }
        }
        pos += wc;
    }

    إذا (!استفاده_میکند_decode_وکتور) {
        أرجع خطأ;
    }

    VK_LOG_DEBUG("ggml_vk_strip_decode_vector: stripping SPV_NV_cooperative_matrix_decode_vector");

    // Bulk-copy unchanged runs and only break the run when an instruction needs to
    // be dropped or patched. Use reserve + insert/push_back so the destination buffer
    // is touched exactly once (no zero-initialization pass from resize()).
    بیرون.clear();
    بیرون.reserve(کلمه_شمار);

    size_t پیاپی_شروع = 0;
    خودکار flush_پیاپی = [&](size_t به_بالا_مقصد) {
        إذا (به_بالا_مقصد > پیاپی_شروع) {
            بیرون.insert(بیرون.end(), code + پیاپی_شروع, code + به_بالا_مقصد);
        }
    };

    لكل (size_t pos = 5; pos < کلمه_شمار; ) {
        uint32_t کلمه = code[pos];
        uint32_t wc   = کلمه >> spv::کلمه_شمار_لغزش;
        uint32_t op   = کلمه & spv::Op_کد_ماسک;
        GGML_ASSERT(wc > 0 && pos + wc <= کلمه_شمار);

        إذا (op == spv::OpExtension && wc >= 2) {
            ثابت حرف * s = reinterpret_cast<ثابت حرف *>(&code[pos + 1]);
            إذا (strcmp(s, k_Decode_وکتور_Ext) == 0) {
                flush_پیاپی(pos);
                pos += wc;
                پیاپی_شروع = pos;
                استمر;
            }
        }

        إذا (op == spv::OpCapability && wc == 2 && code[pos + 1] == k_Spv_Capability_Cooperative_ماتریس_Decode_وکتور_NV) {
            flush_پیاپی(pos);
            pos += wc;
            پیاپی_شروع = pos;
            استمر;
        }

        إذا (op == k_Spv_Op_Cooperative_ماتریس_بارگیری_Tensor_NV) {
            // [opcode/wc][ResultType][Result][Pointer][Object][TensorLayout][MemOperand mask][mem extras...][TA mask][ta extras...]
            GGML_ASSERT(wc >= 8);

            uint32_t mem_ماسک = code[pos + 6];
            size_t   cur      = pos + 7;
            // Each of these MemoryAccess bits (when set) carries one trailing operand.
            cur += (mem_ماسک & 0x2)     ? 1 : 0; // Aligned
            cur += (mem_ماسک & 0x8)     ? 1 : 0; // MakePointerAvailable
            cur += (mem_ماسک & 0x10)    ? 1 : 0; // MakePointerVisible
            cur += (mem_ماسک & 0x10000) ? 1 : 0; // AliasScopeINTELMask
            cur += (mem_ماسک & 0x20000) ? 1 : 0; // NoAliasINTELMask
            GGML_ASSERT(cur < pos + wc);

            uint32_t ta_ماسک = code[cur];
            إذا ((ta_ماسک & k_Spv_Tensor_Addressing_Decode_وکتور_Func_بیت) == 0) {
                pos += wc;
                استمر; // leave instruction inside the current unchanged run
            }

            flush_پیاپی(pos);

            // Append unchanged prefix of the instruction (header through the mem-extras).
            size_t inst_شروع = بیرون.size();
            size_t pre_n      = cur - pos;
            بیرون.insert(بیرون.end(), code + pos, code + pos + pre_n);

            // Emit TA mask with the DecodeVectorFunc bit cleared.
            بیرون.push_back(ta_ماسک & ~k_Spv_Tensor_Addressing_Decode_وکتور_Func_بیت);

            // TA extras: TensorView (0x1) and DecodeFunc (0x2) are kept verbatim;
            // DecodeVectorFunc (0x4) is dropped along with its trailing id operand.
            size_t keep_ta_extras = ((ta_ماسک & 0x1) ? 1 : 0) + ((ta_ماسک & 0x2) ? 1 : 0);
            إذا (keep_ta_extras) {
                بیرون.insert(بیرون.end(), code + cur + 1, code + cur + 1 + keep_ta_extras);
            }

            GGML_ASSERT(wc == pre_n + 1 + keep_ta_extras + 1);

            // Patch the instruction header with the new (one-shorter) word count.
            uint32_t new_wc = wc - 1;
            بیرون[inst_شروع] = (new_wc << spv::کلمه_شمار_لغزش) | op;

            pos += wc;
            پیاپی_شروع = pos;
            استمر;
        }

        pos += wc;
    }

    flush_پیاپی(کلمه_شمار);
    أرجع درست;
}

ساكن منطقي ggml_vk_roll_bk_loop(ثابت uint32_t * code, size_t کلمه_شمار, std::vector<uint32_t> & بیرون) {
    إذا (کلمه_شمار < 5) {
        أرجع خطأ;
    }

    بنية vk_spv_loop {
        size_t   سرآیند;
        size_t   end;
        uint32_t control;
    };

    std::vector<vk_spv_loop> loops;

    // Collect a list of all loops in the module.
    لكل (size_t pos = 5; pos < کلمه_شمار; ) {
        ثابت uint32_t wc = code[pos] >> spv::کلمه_شمار_لغزش;
        ثابت uint32_t op = code[pos] & spv::Op_کد_ماسک;
        إذا (wc == 0 || pos + wc > کلمه_شمار) {
            أرجع خطأ;
        }

        إذا (op == spv::Op_Loop_ادغام && wc >= 4) { loops.push_back({ pos, 0, code[pos + 3] }); }

        إذا (op == spv::OpLabel && wc >= 2) {
            لكل (خودکار & l : loops) {
                إذا (l.end == 0 && code[l.سرآیند + 1] == code[pos + 1]) { l.end = pos; }
            }
        }

        pos += wc;
    }

    خودکار encloses = [](ثابت vk_spv_loop & a, ثابت vk_spv_loop & b) {
        أرجع a.سرآیند < b.سرآیند && b.سرآیند < a.end;
    };

    // Find the BK loop.
    ثابت vk_spv_loop * bk = اشاره_تهی;
    لكل (ثابت خودکار & h : loops) {
        إذا (h.control != spv::Loop_Control_Unroll_ماسک) {
            استمر;
        }
        ثابت vk_spv_loop * والد = اشاره_تهی;
        منطقي has_فرزند = خطأ;
        لكل (ثابت خودکار & g : loops) {
            إذا (encloses(g, h) && (!والد || g.سرآیند > والد->سرآیند)) {
                والد = &g;
            }
            إذا (encloses(h, g)) {
                has_فرزند = درست;
            }
        }
        // BK loop should be the last loop nested inside the loop with no hint
        // and have at least one child loop.
        إذا (والد &&
            والد->control == spv::Loop_Control_ماسک_None &&
            has_فرزند &&
            (!bk || h.سرآیند > bk->سرآیند)) {
            bk = &h;
        }
    }
    إذا (!bk) {
        أرجع خطأ;
    }

    // set DontUnroll instead of Unroll
    بیرون.تخصیص_بده(code, code + کلمه_شمار);
    بیرون[bk->سرآیند + 3] = spv::Loop_Control_Dont_Unroll_ماسک;
    أرجع درست;
}

ساكن فراغ ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, size_t spv_اندازه, ثابت فراغ* spv_داده, ثابت std::string entrypoint,
                                         uint32_t parameter_شمار, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         منطقي disable_robustness, منطقي require_پر_subgroups, uint32_t required_subgroup_اندازه) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_شمار <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_پر_subgroups << ", " << required_subgroup_اندازه << ")");
    GGML_ASSERT(parameter_شمار > 0);
    GGML_ASSERT(parameter_شمار <= MAX_PARAMETER_شمار);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_اندازه, reinterpret_cast<ثابت uint32_t *>(spv_داده));

    // Patch SPIR-V to enable supported FP16 float controls, avoiding the need
    // for separate shader variants.
    std::vector<uint32_t> spirv;
    إذا (device->float_controls_rte_fp16 || device->float_controls_denorm_preserve_fp16) {
        ثابت uint32_t* spv_واژگان = reinterpret_cast<ثابت uint32_t *>(spv_داده);
        size_t کلمه_شمار = spv_اندازه / حجم(uint32_t);
        spirv.تخصیص_بده(spv_واژگان, spv_واژگان + کلمه_شمار);

        // Find insertion points respecting SPIR-V layout order:
        //   Header(5) -> OpCapability -> OpExtension -> ... -> OpEntryPoint -> OpExecutionMode -> ...
        size_t pos = 5; // skip header
        size_t cap_درج_pos = pos;
        size_t ext_درج_pos = pos;
        size_t exec_درج_pos = pos;
        uint32_t مدخل_نقطه_id = 0;

        طالما (pos < spirv.size()) {
            uint32_t opcode = spirv[pos] & spv::Op_کد_ماسک;
            uint32_t len    = spirv[pos] >> spv::کلمه_شمار_لغزش;
            إذا (len == 0) اكسر;

            إذا (opcode == spv::OpCapability) {
                cap_درج_pos = pos + len;
                ext_درج_pos = pos + len;
            } وإلا إذا (opcode == spv::OpExtension) {
                ext_درج_pos = pos + len;
            } وإلا إذا (opcode == spv::Op_مدخل_نقطه) {
                مدخل_نقطه_id = spirv[pos + 2];
                exec_درج_pos = pos + len;
            } وإلا إذا (opcode == spv::Op_Execution_نما || opcode == spv::Op_Execution_نما_Id) {
                exec_درج_pos = pos + len;
            } وإلا إذا (مدخل_نقطه_id != 0) {
                اكسر;
            }

            pos += len;
        }

        // Insert from latest position first so earlier indices stay valid.

        إذا (device->float_controls_rte_fp16) {
            // OpExecutionMode %entrypoint RoundingModeRTE 16
            uint32_t exec_نما[] = { (4u << spv::کلمه_شمار_لغزش) | spv::Op_Execution_نما, مدخل_نقطه_id, spv::Execution_نما_Rounding_نما_RTE, 16 };
            spirv.insert(spirv.begin() + exec_درج_pos, std::begin(exec_نما), std::end(exec_نما));
        }

        إذا (device->float_controls_denorm_preserve_fp16) {
            // OpExecutionMode %entrypoint DenormPreserve 16
            uint32_t exec_نما[] = { (4u << spv::کلمه_شمار_لغزش) | spv::Op_Execution_نما, مدخل_نقطه_id, spv::Execution_نما_Denorm_Preserve, 16 };
            spirv.insert(spirv.begin() + exec_درج_pos, std::begin(exec_نما), std::end(exec_نما));
        }

        // OpExtension "SPV_KHR_float_controls"
        ثابت حرف ext_str[] = "SPV_KHR_float_controls";
        size_t ext_str_واژگان = CEIL_DIV(حجم(ext_str), حجم(uint32_t));
        std::vector<uint32_t> extension(1 + ext_str_واژگان, 0);
        extension[0] = (uint32_t)((1 + ext_str_واژگان) << spv::کلمه_شمار_لغزش) | spv::OpExtension;
        کپی_حافظه(&extension[1], ext_str, حجم(ext_str));
        spirv.insert(spirv.begin() + ext_درج_pos, extension.begin(), extension.end());

        إذا (device->float_controls_rte_fp16) {
            // OpCapability RoundingModeRTE
            uint32_t capability[] = { (2u << spv::کلمه_شمار_لغزش) | spv::OpCapability, spv::Capability_Rounding_نما_RTE };
            spirv.insert(spirv.begin() + cap_درج_pos, std::begin(capability), std::end(capability));
        }

        إذا (device->float_controls_denorm_preserve_fp16) {
            // OpCapability DenormPreserve
            uint32_t capability[] = { (2u << spv::کلمه_شمار_لغزش) | spv::OpCapability, spv::CapabilityDenormPreserve };
            spirv.insert(spirv.begin() + cap_درج_pos, std::begin(capability), std::end(capability));
        }

        shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * حجم(uint32_t), spirv.داده());
    }

#إذا defined(GGML_VULKAN_COOPMAT2_DECODE_وکتور_GLSLC_SUPPORT)
    إذا (device->coopmat2 && !device->coopmat2_decode_وکتور) {
        ثابت uint32_t * src   = spirv.خالی() ? reinterpret_cast<ثابت uint32_t *>(spv_داده) : spirv.داده();
        size_t           src_n = spirv.خالی() ? spv_اندازه / حجم(uint32_t) : spirv.size();
        std::vector<uint32_t> stripped;
        إذا (ggml_vk_strip_decode_وکتور(src, src_n, stripped)) {
            spirv = std::move(stripped);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * حجم(uint32_t), spirv.داده());
        }
    }
#پایان_اگر

#إذا VK_سرآیند_نسخه >= 287
    // Roll the mul_mm BK loop on Asahi Linux. Skip bf16 and the mul_mmq pipelines.
    إذا (device->driver_id == vk::DriverId::eMesaHoneykrisp &&
        pipeline->name.rfind("matmul", 0) == 0 &&
        pipeline->name.find("bf16") == std::string::npos &&
        pipeline->name.find("q8_1") == std::string::npos) {
        ثابت uint32_t * src   = spirv.خالی() ? reinterpret_cast<ثابت uint32_t *>(spv_داده) : spirv.داده();
        size_t           src_n = spirv.خالی() ? spv_اندازه / حجم(uint32_t) : spirv.size();
        std::vector<uint32_t> rolled;
        إذا (ggml_vk_roll_bk_loop(src, src_n, rolled)) {
            spirv = std::move(rolled);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * حجم(uint32_t), spirv.داده());
        }
    }
#پایان_اگر

    حاول {
        pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: shader module creation failed for " << pipeline->name << ": " << e.what() << std::پایان_سطر;
        ارم;
    }

    vk::افزودن_Constant_Range pcr(
        vk::Shader_Stage_Flag_بیتها::eCompute,
        0,
        pipeline->افزودن_constant_اندازه
    );

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
    حاول {
        pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: pipeline layout creation failed for " << pipeline->name << ": " << e.what() << std::پایان_سطر;
        ارم;
    }

    std::vector<vk::Specialization_Map_مدخل> specialization_entries(specialization_constants.size());

    لكل (size_t i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].جابه_جایی = i * حجم(uint32_t);
        specialization_entries[i].size = حجم(uint32_t);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.داده(),
        specialization_constants.size() * حجم(uint32_t),
        specialization_constants.داده()
    );

    vk::PipelineShaderStageCreateFlags pipeline_shader_stage_create_flags{};

    إذا (device->subgroup_require_پر_support && require_پر_subgroups) {
        pipeline_shader_stage_create_flags |= vk::Pipeline_Shader_Stage_Create_Flag_بیتها::e_Require_پر_Subgroups_EXT;
    }

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            pipeline_shader_stage_create_flags,
            vk::Shader_Stage_Flag_بیتها::eCompute,
            pipeline->shader_module,
            entrypoint.c_str(),
            &specialization_info);

    vk::Pipeline_Shader_Stage_Required_Subgroup_اندازه_Create_Info_EXT pipeline_shader_stage_required_subgroup_اندازه_create_info;
    pipeline_shader_stage_required_subgroup_اندازه_create_info.required_Subgroup_اندازه = required_subgroup_اندازه;
    إذا (device->subgroup_اندازه_control && required_subgroup_اندازه > 0) {
        GGML_ASSERT(device->subgroup_min_اندازه <= required_subgroup_اندازه && required_subgroup_اندازه <= device->subgroup_max_اندازه);
        pipeline_shader_create_info.مجموعه_PNext(&pipeline_shader_stage_required_subgroup_اندازه_create_info);
    }

    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        device->pipeline_executable_properties_support ?
            vk::Pipeline_Create_Flag_بیتها::eCaptureStatisticsKHR :
            vk::PipelineCreateFlags{},
        pipeline_shader_create_info,
        pipeline->layout);

    vk::PipelineRobustnessCreateInfoEXT rci;

    إذا (device->pipeline_robustness && disable_robustness) {
        rci.storageBuffers = vk::Pipeline_Robustness_میانگیر_Behavior_EXT::eDisabled;
        rci.یکنواخت_Buffers = vk::Pipeline_Robustness_میانگیر_Behavior_EXT::eDisabled;
        compute_pipeline_create_info.مجموعه_PNext(&rci);
    }

#إذا defined(VK_EXT_shader_64bit_indexing)
    vk::PipelineCreateFlags2CreateInfo pipelineFlags2CreateInfo;
    إذا (pipeline->is_64b_indexing)
    {
        pipelineFlags2CreateInfo.flags = vk::PipelineCreateFlagBits2::e64_بیت_Indexing_EXT;
        إذا (device->pipeline_executable_properties_support) {
            pipelineFlags2CreateInfo.flags |= vk::PipelineCreateFlagBits2::eCaptureStatisticsKHR;
        }
        pipelineFlags2CreateInfo.مجموعه_PNext(compute_pipeline_create_info.p_بعدی);
        compute_pipeline_create_info.مجموعه_PNext(&pipelineFlags2CreateInfo);
    }
#پایان_اگر

    حاول {
        pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: compute pipeline creation failed for " << pipeline->name << ": " << e.what() << std::پایان_سطر;
        ارم;
    }

    إذا (vk_instance.debug_utils_support) {
        vk::DebugUtilsObjectNameInfoEXT duoni;
        duoni.objectType = vk::ObjectType::ePipeline;
        duoni.pObjectName = pipeline->name.c_str();
        duoni.objectHandle = /*reinterpret_cast*/(uint64_t)(static_cast<VkPipeline>(pipeline->pipeline));
        vk_instance.pfn_vk_مجموعه_Debug_Utils_Object_Name_EXT(device->device, &static_cast<VkDebugUtilsObjectNameInfoEXT &>(duoni));
    }

    إذا (device->pipeline_executable_properties_support) {
        vk::PipelineExecutableInfoKHR executableInfo;
        executableInfo.pipeline = pipeline->pipeline;

        خودکار statistics = device->device.getPipelineExecutableStatisticsKHR(executableInfo);

        منطقي print_stats = !vk_pipeline_stats_filter.خالی() &&
                           pipeline->name.find(vk_pipeline_stats_filter) != std::string::npos;
        إذا (print_stats) {
            std::cerr << "ggml_vulkan: pipeline stats for " << pipeline->name << ":" << std::پایان_سطر;
        }

        لكل (خودکار & s : statistics) {
            إذا (print_stats) {
                std::cerr << "ggml_vulkan:   " << s.name.داده() << ": ";
                بدل (s.format) {
                    حالة vk::PipelineExecutableStatisticFormatKHR::eBool32:
                        std::cerr << (s.value.b32 ? "true" : "false");
                        اكسر;
                    حالة vk::PipelineExecutableStatisticFormatKHR::eInt64:
                        std::cerr << s.value.i64;
                        اكسر;
                    حالة vk::PipelineExecutableStatisticFormatKHR::eUint64:
                        std::cerr << s.value.u64;
                        اكسر;
                    حالة vk::PipelineExecutableStatisticFormatKHR::eFloat64:
                        std::cerr << s.value.f64;
                        اكسر;
                }
                std::cerr << std::پایان_سطر;
            }
            // "Register Count" is reported by NVIDIA drivers.
            إذا (strcmp(s.name, "Register Count") == 0) {
                VK_LOG_DEBUG(pipeline->name << " " << s.name << ": " << s.value.u64 << " registers");
                pipeline->ثبات_شمار = (uint32_t)s.value.u64;
            }
        }
    }

    {
        std::lock_guard<std::mutex> guard(device->compile_mutex);
        device->all_pipelines.push_back(pipeline);
        pipeline->compiled = درست;
        pipeline->compile_pending = خطأ;
    }
    device->compile_cv.notify_all();
}

فراغ ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_pipeline_destroy_pipeline(" << pipeline->name << ")");
    device.destroyPipelineLayout(pipeline->layout);

    device.destroyShaderModule(pipeline->shader_module);

    device.destroyPipeline(pipeline->pipeline);
}

فراغ ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n) {
    VK_LOG_DEBUG("ggml_pipeline_request_descriptor_sets(" << pipeline->name << ", " << n << ")");
    ctx->pipeline_descriptor_مجموعه_requirements += n;
    إذا (!pipeline->compiled) {
        ggml_vk_بارگیری_shaders(ctx->device, pipeline);
    }
    ggml_pipeline_allocate_descriptor_sets(ctx);
}

فراغ ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {

    إذا (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_مجموعه_requirements) {
        // Enough descriptors are available
        أرجع;
    }

    vk_device& device = ctx->device;

    // Grow by 50% to avoid frequent allocations
    uint32_t needed = std::max(3 * ctx->descriptor_sets.size() / 2, size_t{ctx->pipeline_descriptor_مجموعه_requirements});
    uint32_t مقصد_alloc = needed - ctx->descriptor_sets.size();
    uint32_t pool_مانده = VK_DEVICE_DESCRIPTOR_POOL_اندازه - ctx->descriptor_sets.size() % VK_DEVICE_DESCRIPTOR_POOL_اندازه;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_اندازه;

    طالما (مقصد_alloc > 0) {
        ثابت uint32_t alloc_شمار = std::min(pool_مانده, مقصد_alloc);
        مقصد_alloc -= alloc_شمار;
        pool_مانده = VK_DEVICE_DESCRIPTOR_POOL_اندازه;

        إذا (pool_idx >= ctx->descriptor_pools.size()) {
            vk::Descriptor_Pool_اندازه descriptor_pool_اندازه(vk::DescriptorType::e_Storage_میانگیر, MAX_PARAMETER_شمار * VK_DEVICE_DESCRIPTOR_POOL_اندازه);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_اندازه, descriptor_pool_اندازه);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::Descriptor_مجموعه_Layout> layouts(alloc_شمار);
        لكل (uint32_t i = 0; i < alloc_شمار; i++) {
            layouts[i] = device->dsl;
        }
        vk::Descriptor_مجموعه_Allocate_Info descriptor_مجموعه_alloc_info(ctx->descriptor_pools[pool_idx], alloc_شمار, layouts.داده());
        std::vector<vk::Descriptor_مجموعه> sets = device->device.allocateDescriptorSets(descriptor_مجموعه_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
    ctx->descriptor_مجموعه_bindings.resize(ctx->descriptor_sets.size());
}

ساكن vk_command_میانگیر* ggml_vk_create_cmd_میانگیر(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_create_cmd_buffer()");
    vk::Command_میانگیر_Allocate_Info command_میانگیر_alloc_info(
        p.pool,
        vk::Command_میانگیر_Level::ePrimary,
        1);
    ثابت std::vector<vk::Command_میانگیر> cmd_buffers = device->device.allocateCommandBuffers(command_میانگیر_alloc_info);
    p.cmd_buffers.push_back({ cmd_buffers.front(), 0, درست });
    أرجع &p.cmd_buffers[p.cmd_buffers.size()-1];
}

فراغ ggml_vk_submit(vk_context& ctx, vk::Fence fence) {
    إذا (ctx->seqs.خالی()) {
        إذا (fence) {
            ctx->p->q->handle->submit({}, fence);
        }
        أرجع;
    }
    VK_LOG_DEBUG("ggml_vk_submit(" << ctx << ", " << fence << ")");

    std::vector<std::vector<uint64_t>> tl_wait_vals;
    std::vector<std::vector<uint64_t>> tl_signal_vals;
    std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
    std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
    std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
    std::vector<vk::SubmitInfo> submit_infos;
    صحيح idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    size_t reserve = 0;

    لكل (ثابت خودکار& sequence : ctx->seqs) {
        reserve += sequence.size();
    }

    // Pre-reserve vectors to prevent reallocation, which invalidates pointers
    tl_wait_semaphores.reserve(reserve);
    tl_wait_vals.reserve(reserve);
    tl_signal_semaphores.reserve(reserve);
    tl_signal_vals.reserve(reserve);
    tl_submit_infos.reserve(reserve);
    submit_infos.reserve(reserve);
    stage_flags.reserve(reserve);

    لكل (ثابت خودکار& sequence : ctx->seqs) {
        لكل (ثابت خودکار& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            tl_wait_vals.push_back({});
            tl_wait_semaphores.push_back({});
            tl_signal_vals.push_back({});
            tl_signal_semaphores.push_back({});
            لكل (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(ctx->p->q->stage_flags);
                tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
                tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
            }
            لكل (size_t i = 0; i < submission.signal_semaphores.size(); i++) {
                tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
                tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
            }
            tl_submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_vals[idx].داده(),
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_vals[idx].داده(),
            });
            tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
            tl_submit_infos[idx].p_بعدی = اشاره_تهی;
            vk::SubmitInfo si{
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_semaphores[idx].داده(),
                stage_flags[idx].داده(),
                1,
                &submission.میانگیر->buf,
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_semaphores[idx].داده(),
            };
            si.مجموعه_PNext(&tl_submit_infos[idx]);
            submit_infos.push_back(si);
        }
    }

    ctx->p->q->handle->submit(submit_infos, fence);

    ctx->seqs.clear();
}

uint32_t ggml_vk_پیدا_کن_صف_family_شاخص(std::vector<vk::صف_Family_Properties>& صف_family_props, ثابت vk::صف_Flags& required, ثابت vk::صف_Flags& avoid, int32_t compute_شاخص, uint32_t min_رقم_queues) {
    VK_LOG_DEBUG("ggml_vk_find_queue_family_index()");
    ثابت uint32_t qfsize = صف_family_props.size();

    // Try with avoid preferences first
    لكل (uint32_t i = 0; i < qfsize; i++) {
        إذا (صف_family_props[i].صف_شمار >= min_رقم_queues && (compute_شاخص < 0 || i != (uint32_t) compute_شاخص) && صف_family_props[i].صف_Flags_2 & required && !(صف_family_props[i].صف_Flags_2 & avoid)) {
            أرجع i;
        }
    }

    // Fall back to only required
    لكل (size_t i = 0; i < qfsize; i++) {
        إذا (صف_family_props[i].صف_شمار >= min_رقم_queues && (compute_شاخص < 0 || i != (uint32_t) compute_شاخص) && صف_family_props[i].صف_Flags_2 & required) {
            أرجع i;
        }
    }

    // Fall back to reusing compute queue
    لكل (size_t i = 0; i < qfsize; i++) {
        إذا (صف_family_props[i].صف_شمار >= min_رقم_queues && صف_family_props[i].صف_Flags_2 & required) {
            أرجع i;
        }
    }

    // Fall back to ignoring min_num_queries
    لكل (size_t i = 0; i < qfsize; i++) {
        إذا (صف_family_props[i].صف_Flags_2 & required) {
            أرجع i;
        }
    }

    // All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
    // Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
    إذا (compute_شاخص >= 0) {
        أرجع compute_شاخص;
    }

    std::cerr << "ggml_vulkan: No suitable queue family index found." << std::پایان_سطر;

    لكل(خودکار &q_family : صف_family_props) {
        std::cerr << "Queue number: "  + std::to_string(q_family.صف_شمار) << " flags: " + to_string(q_family.صف_Flags_2) << std::پایان_سطر;
    }
    abort();
}

std::unique_ptr<vk_صف> ggml_vk_create_صف(vk_device& device, uint32_t صف_family_شاخص, uint32_t صف_شاخص, vk::PipelineStageFlags&& stage_flags, منطقي transfer_only) {
    VK_LOG_DEBUG("ggml_vk_create_queue()");
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    خودکار q = std::make_unique<vk_صف>();
    q->صف_family_شاخص = صف_family_شاخص;
    q->transfer_only = transfer_only;

    std::shared_ptr<vk_صف_handle> h;
    vk::Device_صف_Info2 صف_info2{};
    صف_info2.صف_Family_شاخص = صف_family_شاخص;
    صف_info2.صف_شاخص_2 = صف_شاخص;

    إذا (device->has_internally_synchronized_queues) {
        h = std::make_shared<vk_صف_handle_unsynchronized>();
        صف_info2.flags = eInternallySynchronizedKHR;
    } وإلا {
        h = std::make_shared<vk_صف_handle_synchronized>();
    }

    h->queue = device->device.getQueue2(صف_info2);
    h->device = device;
    // Avoid concurrent submissions on NVIDIA due to driver bug.
    إذا (device->vendor_id == VK_VENDOR_ID_NVIDIA) {
        h->device_submit_mutex = &device->صف_submit_mutex;
    }
    q->handle = h;

    q->cmd_pool.init(device, q.get());

    q->stage_flags = stage_flags;
    أرجع q;
}

std::unique_ptr<vk_صف> ggml_vk_create_aliased_صف(vk_device& device, ثابت std::unique_ptr<vk_صف>& منبع) {
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    خودکار q = std::make_unique<vk_صف>();
    q->handle = منبع->handle;
    q->صف_family_شاخص = منبع->صف_family_شاخص;
    q->stage_flags = منبع->stage_flags;
    q->transfer_only = منبع->transfer_only;
    q->cmd_pool.init(device, q.get());
    أرجع q;
}

vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p) {
    vk_context نتیجه = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_context(" << نتیجه << ")");
    ctx->gc.contexts.emplace_back(نتیجه);
    نتیجه->p = &p;
    أرجع نتیجه;
}

vk_context ggml_vk_create_temporary_context(vk_command_pool& p) {
    vk_context نتیجه = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_temporary_context(" << نتیجه << ")");
    نتیجه->p = &p;
    أرجع نتیجه;
}

ساكن vk_semaphore * ggml_vk_create_دودویی_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::e_دودویی, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.مجموعه_PNext(&tci);
    vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
    ctx->gc.semaphores.push_back({ semaphore, 0 });
    أرجع &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}

ساكن vk_semaphore * ggml_vk_create_timeline_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    إذا (ctx->semaphore_idx >= ctx->gc.tl_semaphores.size()) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.مجموعه_PNext(&tci);
        vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
        ctx->gc.tl_semaphores.push_back({ semaphore, 0 });
    }
    أرجع &ctx->gc.tl_semaphores[ctx->semaphore_idx++];
}

ساكن vk::Event ggml_vk_create_event(ggml_backend_vk_context * ctx) {
    إذا (ctx->event_idx >= ctx->gc.events.size()) {
        ctx->gc.events.push_back(ctx->device->device.createEvent({}));
    }
    أرجع ctx->gc.events[ctx->event_idx++];
}

فراغ ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_command_pool_cleanup()");

    // Requires command buffers to be done
    device->device.resetCommandPool(p.pool);
    // Don't clear the command buffers and mark them as not in use.
    // This allows us to reuse them
    لكل (خودکار& cmd_میانگیر : p.cmd_buffers) {
        cmd_میانگیر.درون_use = خطأ;
    }
}

فراغ ggml_vk_صف_command_pools_cleanup(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_queue_command_pools_cleanup()");

    // Arbitrary frequency to cleanup/reuse command buffers
    ساكن constexpr uint32_t cleanup_frequency = 10;

    إذا (device->compute_صف && device->compute_صف->cmd_pool.buffers_درون_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->compute_صف->cmd_pool);
    }
    إذا (device->transfer_صف && device->transfer_صف->cmd_pool.buffers_درون_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->transfer_صف->cmd_pool);
    }
}

vk_subbuffer ggml_vk_subbuffer(ثابت ggml_backend_vk_context* ctx, ثابت vk_میانگیر& buf, size_t جابه_جایی) {
    أرجع { buf, جابه_جایی, ggml_vk_get_max_میانگیر_range(ctx, buf, جابه_جایی) };
}

فراغ ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_sync_buffers()");

    ثابت منطقي transfer_صف = subctx->p->q->transfer_only;

    إذا (ctx) {
        ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_تقسیم_k_need_sync = خطأ;
    }

    subctx->s->میانگیر->buf.pipelineBarrier(
        subctx->p->q->stage_flags,
        subctx->p->q->stage_flags,
        {},
        { {
          { !transfer_صف ? (vk::Access_Flag_بیتها::eShaderRead | vk::Access_Flag_بیتها::eShaderWrite | vk::Access_Flag_بیتها::eTransferRead | vk::Access_Flag_بیتها::eTransferWrite) : (vk::Access_Flag_بیتها::eTransferRead | vk::Access_Flag_بیتها::eTransferWrite) },
          { !transfer_صف ? (vk::Access_Flag_بیتها::eShaderRead | vk::Access_Flag_بیتها::eShaderWrite | vk::Access_Flag_بیتها::eTransferRead | vk::Access_Flag_بیتها::eTransferWrite) : (vk::Access_Flag_بیتها::eTransferRead | vk::Access_Flag_بیتها::eTransferWrite) }
        } },
        {},
        {}
    );
}

ساكن فراغ ggml_vk_reset_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->میانگیر->buf.resetEvent(
        event,
        ctx->p->q->stage_flags
    );
}

فراغ ggml_vk_مجموعه_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->میانگیر->buf.مجموعه_Event(
        event,
        ctx->p->q->stage_flags
    );
}

فراغ ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events) {
    VK_LOG_DEBUG("ggml_vk_wait_events()");
    إذا (events.خالی()) {
        أرجع;
    }

    ctx->s->میانگیر->buf.waitEvents(
        events,
        ctx->p->q->stage_flags,
        ctx->p->q->stage_flags,
        {},
        {},
        {}
    );
}

ساكن vk_fa_tuning_params get_fa_tuning_params_scalar(ثابت vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ردیفها, uint32_t n_kv, ggml_type k_type, ggml_type v_type, منطقي f32acc) {

    vk_fa_tuning_params نتیجه{};
    نتیجه.path = FA_SCALAR;

    إذا (device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Disable subgroup use due to performance issues when enforcing subgroup sizes
        نتیجه.subgroup_اندازه = 32;
        نتیجه.disable_subgroups = درست;
    } وإلا إذا (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN) {
        نتیجه.subgroup_اندازه = n_ردیفها < 4 ? 32 : device->subgroup_اندازه;
    } وإلا {
        نتیجه.subgroup_اندازه = device->subgroup_اندازه;
    }

    // Row split splits the workgroup so that synchronization only has to happen within subgroups, which avoids barriers
    uint32_t ردیف_تقسیم_max_hsk = 64;
    إذا (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN && !device->uma) {
        ردیف_تقسیم_max_hsk = n_ردیفها <= 8 ? 64 : 128;
    }
    نتیجه.ردیف_تقسیم = (n_ردیفها < 4 || hsk <= ردیف_تقسیم_max_hsk) ? 1 : 4;

    إذا (نتیجه.subgroup_اندازه > 32 && (n_ردیفها < 4 || hsk < (نتیجه.ردیف_تقسیم == 1 ? 128 : 64))) {
        نتیجه.workgroup_اندازه = نتیجه.subgroup_اندازه * 2;
    } وإلا {
        نتیجه.workgroup_اندازه = نتیجه.subgroup_اندازه * 4;
    }

    ثابت uint32_t D = hsk | hsv;

    ثابت منطقي reduce_بلوک_ردیفها = D & 8 || n_kv < 1024 || device->vendor_id == VK_VENDOR_ID_INTEL;

    إذا (n_ردیفها == 1) {
        نتیجه.بلوک_ردیفها = 1;
        نتیجه.بلوک_تعداد_ستون = 64;
    } وإلا {
        // row_split 1 means higher register use per row, so block size has to be adjusted
        إذا (نتیجه.ردیف_تقسیم == 1) {
            نتیجه.بلوک_ردیفها = n_ردیفها == 2 ? 2 : ((n_ردیفها <= 4 || reduce_بلوک_ردیفها) ? 4 : 8);
        } وإلا {
            نتیجه.بلوک_ردیفها = n_ردیفها <= 4 ? 4 : ((n_ردیفها <= 8 || reduce_بلوک_ردیفها) ? 8 : 16);
        }

        نتیجه.بلوک_تعداد_ستون = (D & 8) ? 64 : 32;
    }

    ثابت uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit

    نتیجه.d_تقسیم = std::min(std::min(نتیجه.subgroup_اندازه, 8u), D_lsb / 4);

    نتیجه.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    إذا (!reduce_بلوک_ردیفها && !ggml_vk_flash_attn_scalar_shmem_support(device, نتیجه, hsk, hsv, f32acc, k_type, v_type)) {
        نتیجه.بلوک_ردیفها /= 2;
    }

    // On AMD RDNA, for small head sizes and big batch size the shader uses few registers, so too many subgroups get scheduled
    // at once and end up thrashing the cache. Fix this by setting a large (unused) shmem buffer that reduces occupancy.
    // This targets an occupancy of 4 subgroups per SIMD.
    إذا (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.max_Compute_Shared_حافظه_اندازه == 65536) {
        إذا (device->architecture != AMD_GCN && n_ردیفها >= 64 && hsk <= 128) {
            // 30kb target for hsk > 64, 26kb for <= 64 due to smaller workgroup size
            // Values are guessed, tested on RDNA2
            نتیجه.حد_occupancy_shmem = (hsk <= 64 ? 26 : 30) * 1024 / 4 / 4;
        } وإلا إذا (device->architecture == AMD_GCN && n_ردیفها <= 8 && hsk >= 256) {
            // Same thing for GCN, with an occupancy target of 2 subgroups per SIMD.
            // Here low-batch FA with large head size is affected.
            // n_rows < 4 switch because workgroup size switches from 128 to 256 there.
            نتیجه.حد_occupancy_shmem = (n_ردیفها < 4 ? 14 : 26) * 1024 / 4 / 4;
        }
    }

    أرجع نتیجه;
}

ساكن vk_fa_tuning_params get_fa_tuning_params_coopmat1(ثابت vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ردیفها, uint32_t n_kv, ggml_type k_type, ggml_type v_type, منطقي f32acc) {
    GGML_استفادهنشده(n_ردیفها);
    GGML_استفادهنشده(n_kv);
    GGML_استفادهنشده(k_type);
    GGML_استفادهنشده(v_type);
    GGML_استفادهنشده(f32acc);

    vk_fa_tuning_params نتیجه{};
    نتیجه.path = FA_COOPMAT1;

    ثابت uint32_t D = hsk | hsv;

    ثابت uint32_t coopmat_بلوک_ردیفها = 16;
    ثابت uint32_t coopmat_بلوک_تعداد_ستون = 16;

    ثابت uint32_t رقم_subgroups = 4;

    نتیجه.بلوک_ردیفها = coopmat_بلوک_ردیفها;
    نتیجه.بلوک_تعداد_ستون = coopmat_بلوک_تعداد_ستون * رقم_subgroups;
    نتیجه.ردیف_تقسیم = رقم_subgroups;
    نتیجه.subgroup_اندازه = device->subgroup_اندازه;
    نتیجه.workgroup_اندازه = رقم_subgroups * نتیجه.subgroup_اندازه;

    ثابت uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit
    نتیجه.d_تقسیم = std::min(std::min(نتیجه.subgroup_اندازه, 8u), D_lsb / 4);

    نتیجه.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    أرجع نتیجه;
}

ساكن vk_fa_tuning_params get_fa_tuning_params_coopmat2(ثابت vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ردیفها, uint32_t n_kv, ggml_type k_type, ggml_type v_type, منطقي f32acc) {
    GGML_استفادهنشده(n_kv);
    GGML_استفادهنشده(f32acc);

    vk_fa_tuning_params نتیجه{};
    نتیجه.path = FA_COOPMAT2;

    ثابت uint32_t D = hsk | hsv;

    ثابت منطقي small_ردیفها = n_ردیفها < 32;

    إذا (small_ردیفها) {
        نتیجه.بلوک_ردیفها = 32;
        نتیجه.بلوک_تعداد_ستون = 32;
    } وإلا إذا (ggml_is_quantized(k_type) || ggml_is_quantized(v_type) || hsk >= 256 || hsv >= 256) {
        نتیجه.بلوک_ردیفها = (hsk >= 512 || hsv >= 512) ? 32 : 64;
        نتیجه.بلوک_تعداد_ستون = 32;
    } وإلا {
        نتیجه.بلوک_ردیفها = 64;
        نتیجه.بلوک_تعداد_ستون = 64;
    }

    نتیجه.subgroup_اندازه = device->subgroup_اندازه;
    نتیجه.workgroup_اندازه = (small_ردیفها && (D % 32) == 0) ? 256 : 128;

    أرجع نتیجه;
}

vk_fa_tuning_params get_fa_tuning_params(ثابت vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ردیفها, uint32_t n_kv, ggml_type k_type, ggml_type v_type, منطقي f32acc) {
    Fa_کد_مسیر path = device->coopmat2 ? FA_COOPMAT2 :
                      device->coopmat1_fa_support ? FA_COOPMAT1 : FA_SCALAR;

    إذا (path == FA_COOPMAT2 && k_type == GGML_TYPE_BF16 && !device->coopmat2_bf16_support) {
        path = FA_COOPMAT1;
    }
    إذا (path == FA_COOPMAT1 && k_type == GGML_TYPE_BF16 && !device->coopmat_bf16_support) {
        path = FA_SCALAR;
    }

    إذا (path == FA_COOPMAT1 && device->architecture == vk_device_architecture::NVIDIA_TURING) {
        // Nvidia compiler bug, see https://github.com/ggml-org/llama.cpp/pull/19075#issuecomment-3820716090
        path = FA_SCALAR;
    }

    إذا (path == FA_COOPMAT1) {
        منطقي shape_ok = (f32acc && device->coopmat_support_16x16x16_f32acc) ||
                        (!f32acc && device->coopmat_support_16x16x16_f16acc);
        ثابت vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_ردیفها, n_kv, k_type, v_type, f32acc);
        منطقي shmem_ok = ggml_vk_flash_attn_coopmat_shmem_support(device, params, hsk, hsv, f32acc, k_type, v_type);

        إذا (!shape_ok || !shmem_ok) {
            path = FA_SCALAR;
        }
    }

    // scalar is faster than coopmat when N==1
    إذا (n_ردیفها == 1 && (path == FA_COOPMAT1 || path == FA_COOPMAT2)) {
        path = FA_SCALAR;
    }

    بدل (path) {
    حالة FA_SCALAR:
        أرجع get_fa_tuning_params_scalar(device, hsk, hsv, n_ردیفها, n_kv, k_type, v_type, f32acc);
    حالة FA_COOPMAT1:
        أرجع get_fa_tuning_params_coopmat1(device, hsk, hsv, n_ردیفها, n_kv, k_type, v_type, f32acc);
    حالة FA_COOPMAT2:
        أرجع get_fa_tuning_params_coopmat2(device, hsk, hsv, n_ردیفها, n_kv, k_type, v_type, f32acc);
    افتراضي:
        ارم std::runtime_error("unsupported FaCodePath");
    }
}

vk_fa_pipeline_state get_fa_pipeline_state(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, منطقي aligned, منطقي f32acc,
                                                  منطقي use_ماسک, منطقي use_ماسک_opt, منطقي use_logit_softcap, منطقي use_sparse, ggml_type k_type, ggml_type v_type) {
    ثابت منطقي old_amd_windows = device->vendor_id == VK_VENDOR_ID_AMD && device->driver_id == vk::DriverId::eAmdProprietary &&
                                 (device->architecture == AMD_GCN || device->architecture == AMD_RDNA1 || device->architecture == AMD_RDNA2);

    uint32_t flags = (use_ماسک_opt      ? 1 : 0) |
                     (use_ماسک          ? 2 : 0) |
                     (use_logit_softcap ? 4 : 0) |
                     (old_amd_windows   ? 8 : 0) |
                     (use_sparse        ? 16 : 0);

    ثابت uint32_t subgroup_اندازه = params.disable_subgroups ? 0 : params.subgroup_اندازه;

    أرجع vk_fa_pipeline_state{hsk, hsv, params.بلوک_ردیفها, params.بلوک_تعداد_ستون, params.d_تقسیم, params.ردیف_تقسیم, params.shmem_staging, params.path, params.workgroup_اندازه, subgroup_اندازه, aligned, f32acc, flags, params.حد_occupancy_shmem, k_type, v_type};
}

ساكن uint32_t fa_بلوک_بایتها(ggml_type t) {
    إذا (t == GGML_TYPE_F32) {
        أرجع 16u;
    }
    أرجع (uint32_t) ggml_type_اندازه(t);
}

ساكن std::vector<uint32_t> get_fa_spec_constants(ثابت vk_fa_pipeline_state& state) {
    أرجع {
        /* 0 WorkGroupSize   */ state.workgroup_اندازه,
        /* 1 Br              */ state.Br,
        /* 2 Bc              */ state.Bc,
        /* 3 HSK             */ state.HSK,
        /* 4 HSV             */ state.HSV,
        /* 5 Clamp           */ static_cast<uint32_t>(!state.aligned),
        /* 6 D_split         */ state.D_تقسیم,
        /* 7 row_split       */ state.ردیف_تقسیم,
        /* 8 SubGroupSize    */ state.subgroup_اندازه,
        /* 9 SHMEM_STAGING   */ state.shmem_staging ? 1u : 0u,
        /*10 Flags           */ state.flags,
        /*11 LIMIT_OCCUPANCY_SHMEM */ state.حد_occupancy_shmem,
        /*12 FaTypeK         */ static_cast<uint32_t>(state.k_type),
        /*13 FaTypeV         */ static_cast<uint32_t>(state.v_type),
        /*14 FaBlockBytesK   */ fa_بلوک_بایتها(state.k_type),
        /*15 FaBlockBytesV   */ fa_بلوک_بایتها(state.v_type),
    };
}

ساكن منطقي ggml_vk_matmul_shmem_support(ثابت vk_device& device, ثابت std::vector<uint32_t>& warptile, منطقي mul_mat_id, ggml_type src0_type) {

    uint32_t lut_اندازه = 0;
    بدل (src0_type) {
    حالة GGML_TYPE_IQ1_S:
    حالة GGML_TYPE_IQ1_M:
        // Regular matmul uses the compact uint16_t IQ1 grid; the expanded
        // uint32_t grid is only enabled for the q8_1/int-dot vector path.
        lut_اندازه = 2*2048;
        اكسر;
    حالة GGML_TYPE_IQ2_XXS:
        lut_اندازه = 8*256;
        اكسر;
    حالة GGML_TYPE_IQ2_XS:
        lut_اندازه = 8*512;
        اكسر;
    حالة GGML_TYPE_IQ2_S:
        lut_اندازه = 8*1024;
        اكسر;
    حالة GGML_TYPE_IQ3_XXS:
        lut_اندازه = 4*256;
        اكسر;
    حالة GGML_TYPE_IQ3_S:
        lut_اندازه = 4*512;
        اكسر;
    حالة GGML_TYPE_IQ4_NL:
    حالة GGML_TYPE_IQ4_XS:
    حالة GGML_TYPE_MXFP4:
        lut_اندازه = 4*16;
        اكسر;
    حالة GGML_TYPE_NVFP4:
        // Same kvalues budget as MXFP4 plus ue4m3_fp32_lut[128] (types.glsl, DATA_A_NVFP4).
        lut_اندازه = 4*16 + 128u * (uint32_t)حجم(عائم);
        اكسر;
    افتراضي:
        اكسر;
    }

    // Needs to be kept up to date on shader changes
    // Needs to stay aligned with ggml_vk_mul_mm_spec.
    ثابت منطقي intel_shmem_stride_pad_zero = device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
                                              device->driver_id == vk::DriverId::eIntelProprietaryWindows;
    ثابت uint32_t bank_conflict_جابه_جایی = intel_shmem_stride_pad_zero ? 0 : (device->coopmat_support ? 8 : 1);
    ثابت uint32_t type_اندازه = device->fp16 ? حجم(ggml_fp16_t) : حجم(عائم);
    ثابت uint32_t warps = warptile[0] / warptile[10];

    ثابت uint32_t بارگیری_bufs = (warptile[1] + warptile[2]) * (warptile[3] + bank_conflict_جابه_جایی) * type_اندازه;
    ثابت uint32_t mmid_ردیف_ids = mul_mat_id ? (warptile[2] * 2 * حجم(uint16_t)) : 0;
    ثابت uint32_t coopmat_stage = device->coopmat_support ? warptile[7] * warptile[8] / warps * حجم(عائم) : 0;

    ثابت uint32_t مجموع_اندازه = بارگیری_bufs + mmid_ردیف_ids + coopmat_stage + lut_اندازه;
    ثابت منطقي supported = مجموع_اندازه <= device->properties.limits.max_Compute_Shared_حافظه_اندازه;

    VK_LOG_DEBUG("ggml_vk_matmul_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", supported=" << supported);

    أرجع supported;
}

ساكن منطقي ggml_vk_matmul_int_shmem_support(ثابت vk_device& device, ثابت std::vector<uint32_t>& warptile, منطقي mul_mat_id, ggml_type src0_type) {

    // FLOAT_TYPE in the shader is float16_t with fp16 support, otherwise float.
    ثابت uint32_t fp_اندازه   = device->fp16 ? 2u : 4u;
    ثابت uint32_t fp_align  = fp_اندازه;
    ثابت uint32_t fp2_اندازه  = 2u * fp_اندازه;
    ثابت uint32_t fp2_align = device->fp16 ? 4u : 8u;

    بنية member { uint32_t size, align; };
    خودکار std430_اندازه = [](std::initializer_list<member> members) {
        uint32_t off = 0, struct_align = 1;
        لكل (ثابت خودکار &m : members) {
            off = (off + m.align - 1) & ~(m.align - 1);
            off += m.size;
            struct_align = std::max(struct_align, m.align);
        }
        أرجع (off + struct_align - 1) & ~(struct_align - 1);
    };

    uint32_t بلوک_a_اندازه = 0;
    بدل (src0_type) {
        حالة GGML_TYPE_Q2_0:    بلوک_a_اندازه = std430_اندازه({{32, 4}, {fp_اندازه,  fp_align}});                  اكسر; // qs[8] + dm
        حالة GGML_TYPE_Q4_0:    بلوک_a_اندازه = std430_اندازه({{16, 4}, {fp_اندازه,  fp_align}});                  اكسر; // qs[16/4] + dm
        حالة GGML_TYPE_Q4_1:    بلوک_a_اندازه = std430_اندازه({{16, 4}, {fp2_اندازه, fp2_align}});                 اكسر; // qs[16/4] + dm(vec2)
        حالة GGML_TYPE_Q5_0:    بلوک_a_اندازه = std430_اندازه({{16, 4}, {4, 4}, {fp_اندازه,  fp_align}});          اكسر; // qs[16/4] + qh + dm
        حالة GGML_TYPE_Q5_1:    بلوک_a_اندازه = std430_اندازه({{16, 4}, {4, 4}, {fp2_اندازه, fp2_align}});         اكسر; // qs[16/4] + qh + dm(vec2)
        حالة GGML_TYPE_Q8_0:    بلوک_a_اندازه = std430_اندازه({{32, 4}, {fp_اندازه,  fp_align}});                  اكسر; // qs[8] + dm
        حالة GGML_TYPE_IQ4_XS:  بلوک_a_اندازه = std430_اندازه({{32, 4}, {fp_اندازه,  fp_align}});                  اكسر; // qs[8] + d
        حالة GGML_TYPE_MXFP4:   بلوک_a_اندازه = std430_اندازه({{32, 4}, {fp_اندازه,  fp_align}});                  اكسر; // qs[8] + d
        حالة GGML_TYPE_IQ4_NL:  بلوک_a_اندازه = std430_اندازه({{32, 4}, {fp_اندازه,  fp_align}});                  اكسر; // qs[8] + d
        حالة GGML_TYPE_NVFP4:   بلوک_a_اندازه = std430_اندازه({{32, 4}, {fp2_اندازه, fp2_align}});                 اكسر; // qs[8] + d_scales(vec2)
        حالة GGML_TYPE_Q2_K:    بلوک_a_اندازه = std430_اندازه({{ 8, 4}, {2, 2}, {fp2_اندازه, fp2_align}});         اكسر; // qs[2] + scales(u8vec2) + dm(vec2)
        حالة GGML_TYPE_Q3_K:    بلوک_a_اندازه = std430_اندازه({{16, 4}, {fp2_اندازه, fp2_align}});                 اكسر; // qs[4] + d_scales(vec2)
        حالة GGML_TYPE_Q4_K:    بلوک_a_اندازه = std430_اندازه({{16, 4}, {fp2_اندازه, fp2_align}});                 اكسر; // qs[4] + dm(vec2)
        حالة GGML_TYPE_Q5_K:    بلوک_a_اندازه = std430_اندازه({{32, 4}, {fp2_اندازه, fp2_align}});                 اكسر; // qs[8] + dm(vec2)
        حالة GGML_TYPE_Q6_K:    بلوک_a_اندازه = std430_اندازه({{32, 4}, {fp2_اندازه, fp2_align}});                 اكسر; // qs[8] + d_scales(vec2)
        حالة GGML_TYPE_IQ3_S:   بلوک_a_اندازه = std430_اندازه({{32, 4}, {fp_اندازه,  fp_align}});                  اكسر; // qs[8] + d
        افتراضي:
            أرجع خطأ;
    }

    // IQ3_S also copies its 512-entry grid into shared memory (types.glsl, init_iq_shmem)
    ثابت uint32_t lut_اندازه = (src0_type == GGML_TYPE_IQ3_S) ? 4*512 : 0;

    // block_b_cache: { int32_t qs[8]; FLOAT_TYPEV2 ds; }
    ثابت uint32_t بلوک_b_اندازه = std430_اندازه({{32, 4}, {fp2_اندازه, fp2_align}});

    ثابت uint32_t BM = warptile[1];
    ثابت uint32_t BN = warptile[2];
    // mul_mmq.comp: BK_STEP=1 for MUL_MAT_ID, 4 otherwise.
    ثابت uint32_t BK_گام = mul_mat_id ? 1u : 4u;

    ثابت uint32_t buf_a_اندازه = BM * BK_گام * بلوک_a_اندازه;
    ثابت uint32_t buf_b_اندازه = BN * BK_گام * بلوک_b_اندازه;
    ثابت uint32_t mmid_ردیف_ids = mul_mat_id ? (BN * 2u * (uint32_t)حجم(uint16_t)) : 0u;

    ثابت uint32_t مجموع_اندازه = buf_a_اندازه + buf_b_اندازه + mmid_ردیف_ids + lut_اندازه;
    ثابت منطقي supported = مجموع_اندازه <= device->properties.limits.max_Compute_Shared_حافظه_اندازه;

    VK_LOG_DEBUG("ggml_vk_matmul_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << مجموع_اندازه << ", supported=" << supported);

    أرجع supported;
}

ساكن منطقي ggml_vk_matmul_cm1_int_shmem_support(ثابت vk_device& device, ثابت std::vector<uint32_t>& warptile, منطقي mul_mat_id, ggml_type src0_type) {

    منطقي kscales2 = خطأ;    // two scale sets per block
    منطقي has_dm   = خطأ;    // d+m as vec2 + b-side sum
    منطقي has_kvalues = خطأ;
    بدل (src0_type) {
        حالة GGML_TYPE_Q4_0: حالة GGML_TYPE_Q5_0: حالة GGML_TYPE_Q8_0:
            اكسر;
        حالة GGML_TYPE_Q4_1: حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q4_K: حالة GGML_TYPE_Q5_K:
            has_dm = درست;                          اكسر;
        حالة GGML_TYPE_IQ4_NL: حالة GGML_TYPE_IQ4_XS: حالة GGML_TYPE_MXFP4:
            has_kvalues = درست;                     اكسر;
        حالة GGML_TYPE_Q3_K: حالة GGML_TYPE_Q6_K:
            kscales2 = درست;                        اكسر;
        حالة GGML_TYPE_NVFP4:
            kscales2 = درست; has_kvalues = درست;    اكسر;
        افتراضي:
            أرجع خطأ;
    }

    ثابت uint32_t BM         = warptile[1];
    ثابت uint32_t BN         = warptile[2];

    ثابت uint32_t BK      = 32;
    ثابت uint32_t BK_گام = mul_mat_id ? 2u : 4u;
    ثابت uint32_t QPITCH  = BK_گام * (BK / 4u) + 4u;
    ثابت uint32_t KSCALES = kscales2 ? 2u : 1u;

    uint32_t مجموع = 0;
    مجموع += BM * QPITCH * (uint32_t)حجم(uint32_t);   // buf_a_qs
    مجموع += BN * QPITCH * (uint32_t)حجم(uint32_t);   // buf_b_qs
    مجموع += has_dm ? (BM * BK_گام * 2u * (uint32_t)حجم(عائم))   // buf_a_dm (vec2)
                    : (BM * BK_گام * KSCALES * (uint32_t)حجم(عائم)); // buf_a_d
    مجموع += BN * BK_گام * (uint32_t)حجم(عائم);     // buf_b_d
    إذا (has_dm) {
        مجموع += BN * BK_گام * (uint32_t)حجم(عائم); // buf_b_s
    }
    إذا (has_kvalues) {
        مجموع += 16u * (uint32_t)حجم(int8_t);         // cm1_kvalues[16]
    }
    إذا (src0_type == GGML_TYPE_NVFP4 && !device->ocp_fp4) {
        مجموع += 128u * (uint32_t)حجم(عائم);         // ue4m3_fp32_lut[128]
    }
    إذا (mul_mat_id) {
        مجموع += BN * 2u * (uint32_t)حجم(uint16_t);   // row_ids[BN] (u16vec2)
    }

    ثابت منطقي supported = مجموع <= device->properties.limits.max_Compute_Shared_حافظه_اندازه;

    VK_LOG_DEBUG("ggml_vk_matmul_cm1_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << مجموع << ", supported=" << supported);

    أرجع supported;
}

ساكن ثابت std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
    {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

ساكن ثابت std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
};

ساكن std::vector<GpuPipelineConfig> gpu_pipeline_configs = {
    {
        vk_device_architecture::AMD_RDNA1,
        {
            rdna1_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_اندازه
    },
    {
        vk_device_architecture::AMD_RDNA2,
        {
            rdna2_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_اندازه
    },
};

uint32_t get_subgroup_اندازه(ثابت std::string &pipeline_name, ثابت vk_device_architecture &arch) {
    لكل (ثابت خودکار &config : gpu_pipeline_configs) {
        إذا (config.arch == arch) {
            خودکار pipIt = config.pipelines.find(pipeline_name);
            إذا (pipIt != config.pipelines.end()) {
                أرجع pipIt->second;
            }
            std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
            std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                      [](ثابت خودکار &a, ثابت خودکار &b) { أرجع a.اولی.size() > b.اولی.size(); });
            لكل (ثابت خودکار &مدخل : sorted_pipelines) {
                إذا (pipeline_name.find(مدخل.اولی) != std::string::npos) {
                    أرجع مدخل.second;
                }
            }
            أرجع config.default_subgroup_اندازه;
        }
    }
    أرجع 0; // If no matching configuration is found
}

ساكن منطقي ggml_vk_fa_type_needs_shmem(ggml_type type) {
    بدل (type) {
    حالة GGML_TYPE_IQ4_NL:
        أرجع درست;
    افتراضي:
        أرجع خطأ;
    }
}

ساكن منطقي ggml_vk_fa_scalar_استفاده_میکند_mmq(ثابت vk_device& device, ggml_type k_type, ggml_type v_type) {
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    أرجع device->integer_dot_حاصلضرب && device->subgroup_clustered &&
           !ggml_vk_fa_type_needs_shmem(v_type) &&
           (k_type == GGML_TYPE_Q4_0 || k_type == GGML_TYPE_Q4_1 ||
            k_type == GGML_TYPE_Q5_0 || k_type == GGML_TYPE_Q5_1 ||
            k_type == GGML_TYPE_Q8_0);
#وإلا
    GGML_استفادهنشده(device);
    GGML_استفادهنشده(k_type);
    GGML_استفادهنشده(v_type);
    أرجع خطأ;
#پایان_اگر
}

فراغ ggml_vk_بارگیری_shaders(vk_device& device, vk_pipeline requested) {
    VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");

    // some shaders have a minimum subgroup size
    ثابت uint32_t subgroup_اندازه_8 = std::max(device->subgroup_اندازه, 8u);
    ثابت uint32_t subgroup_اندازه_16 = std::max(device->subgroup_اندازه, 16u);
    ثابت uint32_t subgroup_اندازه_32 = std::max(device->subgroup_اندازه, 32u);

    // clamp WARP for l_/m_ warptiles so WM <= BM (breaks on subgroupSize > 64)
    ثابت uint32_t mm_warp_8  = std::min(subgroup_اندازه_8,  64u);
    ثابت uint32_t mm_warp_16 = std::min(subgroup_اندازه_16, 64u);

    ثابت uint32_t mul_mat_subgroup_اندازه = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_اندازه_control) ? device->subgroup_min_اندازه : device->subgroup_اندازه;
    ثابت uint32_t mul_mat_subgroup_اندازه_8 = std::max(mul_mat_subgroup_اندازه, 8u);
    ثابت uint32_t mul_mat_subgroup_اندازه_16 = std::max(mul_mat_subgroup_اندازه, 16u);
    ثابت uint32_t mul_mat_subgroup_اندازه_32 = std::max(mul_mat_subgroup_اندازه, 32u);
    ثابت uint32_t mul_mat_mm_warp_8  = std::min(mul_mat_subgroup_اندازه_8,  64u);
    ثابت uint32_t mul_mat_mm_warp_16 = std::min(mul_mat_subgroup_اندازه_16, 64u);

    ثابت منطقي subgroup_min_اندازه_16 = (!device->subgroup_اندازه_control && device->subgroup_اندازه >= 16) ||
                                      (device->subgroup_اندازه_control && device->subgroup_max_اندازه >= 16);

    // mulmat
    // Warptile layout (indices match mul_mm.comp constantIDs):
    //   [0..9]  : BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK
    //   [10]    : WARP / required_subgroup_size (read via WARP_SIZE_IDX)
    ساكن constexpr size_t WARP_اندازه_IDX = 10;
    std::vector<uint32_t> l_warptile, m_warptile, s_warptile,
                          l_warptile_id, m_warptile_id, s_warptile_id,
                          l_warptile_mmq, m_warptile_mmq, s_warptile_mmq,
                          l_warptile_mmq_int, m_warptile_mmq_int, s_warptile_mmq_int,
                          l_warptile_mmq_cm1_int, m_warptile_mmq_cm1_int, s_warptile_mmq_cm1_int,
                          l_warptile_mmq_cm1_int_k, m_warptile_mmq_cm1_int_k, s_warptile_mmq_cm1_int_k,
                          l_warptile_mmq_int_k, m_warptile_mmq_int_k, s_warptile_mmq_int_k,
                          l_warptile_mmq_k, m_warptile_mmq_k, s_warptile_mmq_k,
                          l_warptile_mmqid, m_warptile_mmqid, s_warptile_mmqid,
                          l_warptile_mmqid_int, m_warptile_mmqid_int, s_warptile_mmqid_int,
                          l_warptile_mmqid_int_k, m_warptile_mmqid_int_k, s_warptile_mmqid_int_k;
    std::array<uint32_t, 3> l_wg_denoms, m_wg_denoms, s_wg_denoms,
                            l_mmq_wg_denoms, m_mmq_wg_denoms, s_mmq_wg_denoms,
                            l_mmq_wg_denoms_k, m_mmq_wg_denoms_k, s_mmq_wg_denoms_k,
                            l_mmq_cm1_wg_denoms_k, m_mmq_cm1_wg_denoms_k, s_mmq_cm1_wg_denoms_k,
                            l_mmqid_wg_denoms, m_mmqid_wg_denoms, s_mmqid_wg_denoms;

    uint32_t l_align, m_align, s_align;

    // RDNA3.5 preferred wave32 here
    ثابت منطقي cm1_use_wave32 = device->vendor_id == VK_VENDOR_ID_AMD &&
                                device->subgroup_اندازه_control &&
                                device->subgroup_min_اندازه <= 32 && device->subgroup_max_اندازه >= 32;
    ثابت uint32_t cm1_sg = cm1_use_wave32 ? 32 : device->subgroup_اندازه;

    vk_pipeline wait_pipeline;
    CompileTask claimed_task {};
    منطقي has_claimed_task = خطأ;

    // The rest of the walk reads and writes shared device state, so hold the
    // lock until we're done deciding what to compile.
    std::unique_lock<std::mutex> compile_قفل(device->compile_mutex);

    إذا (device->coopmat2) {
        // spec constants and tile sizes for non-quant matmul/matmul_id
        l_warptile = { 256, 128, 256, 64, 1 };
        m_warptile = { 256, 128, 128, 64, 0 };
        s_warptile = { 128,  64,  64, 64, 0 };
        l_wg_denoms = {128, 256, 1 };
        m_wg_denoms = {128, 128, 1 };
        s_wg_denoms = { 64,  64, 1 };

        // spec constants and tile sizes for quant matmul (non-Qi_K)
        l_warptile_mmq = { 256, 128, 256, 64, 1 };
        m_warptile_mmq = { 256, 128, 128, 64, 1 };
        s_warptile_mmq = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms = { 128, 256, 1 };
        m_mmq_wg_denoms = { 128, 128, 1 };
        s_mmq_wg_denoms = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul (Qi_K)
        l_warptile_mmq_k = { 256, 128, 256, 64, 1 };
        m_warptile_mmq_k = { 256, 128, 128, 64, 1 };
        s_warptile_mmq_k = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms_k = { 128, 256, 1 };
        m_mmq_wg_denoms_k = { 128, 128, 1 };
        s_mmq_wg_denoms_k = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul_id
        ثابت uint32_t mmqid_bk = device->coopmat2_decode_وکتور ? 64u : 32u;
        l_warptile_mmqid = { 256, 128, 128, mmqid_bk, 1 };
        m_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        s_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        l_mmqid_wg_denoms = { 128, 128, 1 };
        m_mmqid_wg_denoms = { 128, 64, 1 };
        s_mmqid_wg_denoms = { 128, 64, 1 };

        l_align = 128;
        m_align =  64;
        s_align =  32;
    } وإلا {
        // Matrix cores require different warp group sizes
        ثابت uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
        ثابت uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
        ثابت uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
        ثابت uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
        ثابت uint32_t tn_m = device->coopmat_support ? device->coopmat_n : 2;
        ثابت uint32_t tn_s = device->coopmat_support ? device->coopmat_n : 2;
        ثابت uint32_t tk_l = device->coopmat_support ? device->coopmat_k : 1;
        ثابت uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
        ثابت uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;

        ثابت uint32_t itm = device->coopmat_int_m;
        ثابت uint32_t itn = device->coopmat_int_n;
        ثابت uint32_t itk = device->coopmat_int_k;

        ثابت uint32_t s_warptile_wm = device->subgroup_اندازه == 8 ? 8 : 32;

        l_warptile = { 128,             128, 128, 16, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile = { 128,              64,  64, 16, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile = { subgroup_اندازه_32, 32,  32, 16, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_اندازه_8 };

        l_warptile_mmq = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile_mmq = { 128,              64,  64, 32, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile_mmq = { subgroup_اندازه_32, 32,  32, 32, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_اندازه_8 };

        // Integer MMQ has a smaller shared memory profile, but heavier register use
        l_warptile_mmq_int = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int = { 128,              64,  64, 32, mm_warp_8,     32, 2, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int = { subgroup_اندازه_32, 32,  32, 32, s_warptile_wm, 32, 2, 2, 1, 1, subgroup_اندازه_8 };

        ثابت خودکار cm1_bs = [cm1_sg](uint32_t bm, uint32_t bn) {
            أرجع cm1_sg * (bm / std::min(cm1_sg, bm)) * (bn / 32);
        };

        l_warptile_mmq_cm1_int = { cm1_bs(128, 128), 128, 128, 32, std::min(cm1_sg, 128u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        m_warptile_mmq_cm1_int = { cm1_bs( 64,  64),  64,  64, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        s_warptile_mmq_cm1_int = { cm1_bs( 32,  32),  32,  32, 32, std::min(cm1_sg,  32u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };

        l_warptile_mmq_cm1_int_k = { cm1_bs( 64, 128),  64, 128, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        m_warptile_mmq_cm1_int_k = { cm1_bs( 64,  64),  64,  64, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        s_warptile_mmq_cm1_int_k = { cm1_bs( 32,  32),  32,  32, 32, std::min(cm1_sg,  32u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };

        l_mmq_cm1_wg_denoms_k = { l_warptile_mmq_cm1_int_k[1], l_warptile_mmq_cm1_int_k[2], 1 };
        m_mmq_cm1_wg_denoms_k = { m_warptile_mmq_cm1_int_k[1], m_warptile_mmq_cm1_int_k[2], 1 };
        s_mmq_cm1_wg_denoms_k = { s_warptile_mmq_cm1_int_k[1], s_warptile_mmq_cm1_int_k[2], 1 };

        // K-quants use even more registers, mitigate by setting WMITER to 1
        l_warptile_mmq_int_k = { 128,               128, 128, 32, mm_warp_8 * 2, 64, 1, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int_k = { 128,                64,  64, 32, mm_warp_8,     32, 1, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int_k = { subgroup_اندازه_32,   32,  32, 32, s_warptile_wm, 32, 1, 2, 1, 1, subgroup_اندازه_8 };

        l_warptile_id = { 128,                      128, 128, 16, mul_mat_mm_warp_16 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_16 };
        m_warptile_id = { 128,                       64,  64, 16, mul_mat_mm_warp_16,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_16 };
        s_warptile_id = { mul_mat_subgroup_اندازه_16,  32,  32, 16, s_warptile_wm,          32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_اندازه_16 };

        l_warptile_mmqid = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_8 };
        m_warptile_mmqid = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_8 };
        s_warptile_mmqid = { mul_mat_subgroup_اندازه_32,   32,  32, 32, s_warptile_wm,         32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_اندازه_8 };

        l_warptile_mmqid_int = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, 4, 4, 1, mul_mat_mm_warp_8 };
        m_warptile_mmqid_int = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, 2, 2, 1, mul_mat_mm_warp_8 };
        s_warptile_mmqid_int = { mul_mat_subgroup_اندازه_32,   32,  32, 32, s_warptile_wm,         32, 2, 2, 1, 1, mul_mat_subgroup_اندازه_8 };

        l_warptile_mmqid_int_k = { 128,                     128, 128, 32, mul_mat_mm_warp_16 * 2, 64, 1, 4, 4, 1, mul_mat_mm_warp_16 };
        m_warptile_mmqid_int_k = { 128,                      64,  64, 32, mul_mat_mm_warp_16,     32, 1, 2, 2, 1, mul_mat_mm_warp_16 };
        s_warptile_mmqid_int_k = { mul_mat_subgroup_اندازه_32, 32,  32, 32, s_warptile_wm,          32, 1, 2, 1, 1, mul_mat_subgroup_اندازه_16 };

        // chip specific tuning
        إذا ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
            m_warptile_mmq = m_warptile_mmq_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
            m_warptile_mmqid = m_warptile_mmqid_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
        } وإلا إذا (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary) {
            // This is intentionally using tx_m values, slight performance increase
            l_warptile = { 256, 128, 128, 16, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq_int_k = { 256, 128, 128, 32, mm_warp_16, 64, 1, 4, 2, 1, mm_warp_16 };
        } وإلا إذا (device->vendor_id == VK_VENDOR_ID_QUALCOMM && device->coopmat_support) {
            m_warptile     = { 64, 64, 64, 16, 64, 64, 1, tm_l, tn_l, tk_l, 64 };
            m_warptile_mmq = { 64, 64, 64, 32, 64, 64, 1, tm_m, tn_m, tk_m, 64 };
        }

        l_mmq_wg_denoms = l_wg_denoms = {128, 128, 1 };
        m_mmq_wg_denoms = m_wg_denoms = { 64,  64, 1 };
        s_mmq_wg_denoms = s_wg_denoms = { 32,  32, 1 };
        l_align = 128;
        m_align =  64;
        s_align =  32;

        إذا (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support) {
            // Xe1/Xe2/Xe3 with coopmat enabled - warptile performance tuning
            l_warptile = { 512, 128, 128, 16, mm_warp_8, 32, 2, tm_l, tn_l, tk_l, mm_warp_8 };
            إذا (device->architecture == INTEL_XE1) {
                l_warptile_mmq  = { 512, 256, 128, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 256, 128, 1 };
                l_align         = 32;  //set as BK
            } وإلا {
                l_warptile_mmq  = { 512, 128, 256, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 128, 256, 1 };
                l_align         = 32;  //set as BK
            }
        }

        ثابت منطقي use_cm1_int = device->coopmat_int_support &&
                                 (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);

        لكل (uint32_t i = 0; i < GGML_TYPE_شمار; ++i) {
            ggml_type t = (ggml_type)i;
            // Disable medium and large matrix multiplication if not enough shared memory is available
            // Check mmq warptiles as the largest configuration
            // Throw an error if not enough for any matrix multiplication is available
            إذا (!ggml_vk_matmul_shmem_support(device, s_warptile_mmq, خطأ, t)) {
                std::cerr << "ggml_vulkan: Error: Shared memory size too small for matrix multiplication." << std::پایان_سطر;
                ارم std::runtime_error("Shared memory size too small for matrix multiplication.");
            } وإلا إذا (!ggml_vk_matmul_shmem_support(device, m_warptile_mmq, خطأ, t)) {
                device->mul_mat_m[i] = خطأ;
                device->mul_mat_l[i] = خطأ;
            } وإلا إذا (!ggml_vk_matmul_shmem_support(device, l_warptile_mmq, خطأ, t)) {
                device->mul_mat_l[i] = خطأ;
            }

            // Disable mul_mat_id if not enough shared memory is available
            إذا (!ggml_vk_matmul_shmem_support(device, s_warptile_mmqid, درست, t)) {
                device->mul_mat_id_s[i] = خطأ;
                device->mul_mat_id_m[i] = خطأ;
                device->mul_mat_id_l[i] = خطأ;
            } وإلا إذا (!ggml_vk_matmul_shmem_support(device, m_warptile_mmqid, درست, t)) {
                device->mul_mat_id_m[i] = خطأ;
                device->mul_mat_id_l[i] = خطأ;
            } وإلا إذا (!ggml_vk_matmul_shmem_support(device, l_warptile_mmqid, درست, t)) {
                device->mul_mat_id_l[i] = خطأ;
            }

            // The q8_1 mmq path has its own (larger) shmem layout, check it separately.
            // K-quants and IQ3_S use the _int_k warptiles, others use _int.
            // cm1 splits k-tiles on the KSCALES==2 types and shares tiles between dense/id.
            ثابت منطقي is_k_quant = (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                                     t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                                     t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ3_S);
            ثابت منطقي cm1_k_tile = (t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q6_K ||
                                     t == GGML_TYPE_NVFP4);

            ثابت خودکار & s_int   = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmq_int_k     : s_warptile_mmq_int);
            ثابت خودکار & m_int   = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmq_int_k     : m_warptile_mmq_int);
            ثابت خودکار & l_int   = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmq_int_k     : l_warptile_mmq_int);
            ثابت خودکار & s_intid = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmqid_int_k   : s_warptile_mmqid_int);
            ثابت خودکار & m_intid = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmqid_int_k   : m_warptile_mmqid_int);
            ثابت خودکار & l_intid = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmqid_int_k   : l_warptile_mmqid_int);

            ثابت خودکار int_shmem_support = [&](ثابت std::vector<uint32_t>& wt, منطقي id) {
                أرجع use_cm1_int ? ggml_vk_matmul_cm1_int_shmem_support(device, wt, id, t)
                                   : ggml_vk_matmul_int_shmem_support(device, wt, id, t);
            };

            إذا (!int_shmem_support(s_int, خطأ)) {
                device->mul_mat_s_int[i] = خطأ;
                device->mul_mat_m_int[i] = خطأ;
                device->mul_mat_l_int[i] = خطأ;
            } وإلا إذا (!int_shmem_support(m_int, خطأ)) {
                device->mul_mat_m_int[i] = خطأ;
                device->mul_mat_l_int[i] = خطأ;
            } وإلا إذا (!int_shmem_support(l_int, خطأ)) {
                device->mul_mat_l_int[i] = خطأ;
            }

            إذا (!int_shmem_support(s_intid, درست)) {
                device->mul_mat_id_s_int[i] = خطأ;
                device->mul_mat_id_m_int[i] = خطأ;
                device->mul_mat_id_l_int[i] = خطأ;
            } وإلا إذا (!int_shmem_support(m_intid, درست)) {
                device->mul_mat_id_m_int[i] = خطأ;
                device->mul_mat_id_l_int[i] = خطأ;
            } وإلا إذا (!int_shmem_support(l_intid, درست)) {
                device->mul_mat_id_l_int[i] = خطأ;
            }
        }
    }

    خودکار ثابت &ggml_vk_create_pipeline = [&](vk_device& device, vk_pipeline& پایه_pipeline, ثابت حرف *name, size_t spv_اندازه, ثابت فراغ* spv_داده, ثابت حرف *entrypoint,
                                              uint32_t parameter_شمار, uint32_t افزودن_constant_اندازه, std::array<uint32_t, 3> wg_denoms, ثابت std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, منطقي disable_robustness = خطأ, منطقي require_پر_subgroups = خطأ, uint32_t required_subgroup_اندازه = 0) {

        إذا (!require_پر_subgroups && required_subgroup_اندازه == 0) {
            required_subgroup_اندازه = get_subgroup_اندازه(name, device->architecture);
        }

        vk_pipeline *ptr = &پایه_pipeline;

        صحيح رقم_pipelines = 1;
#إذا defined(VK_EXT_shader_64bit_indexing)
        إذا (device->shader_64b_indexing) {
            رقم_pipelines = 2;
        }
#پایان_اگر
        لكل (صحيح i = 0; i < رقم_pipelines; ++i, ptr = &(*ptr)->بعدی) {
            vk_pipeline &pipeline = *ptr;
            إذا (!pipeline) {
                pipeline = std::make_shared<vk_pipeline_struct>();
            }
            إذا (!pipeline->initialized) {
                pipeline->name = name;
                pipeline->parameter_شمار = parameter_شمار;
                pipeline->افزودن_constant_اندازه = افزودن_constant_اندازه;
                pipeline->wg_denoms = wg_denoms;
                pipeline->align = align;
                pipeline->initialized = درست;
#إذا defined(VK_EXT_shader_64bit_indexing)
                pipeline->is_64b_indexing = (i == 1);
#پایان_اگر
            }

            // We only care about the pipeline this call asked for; the rest
            // (including the 64-bit indexing variant) are handled by their
            // own request_descriptor_sets / load_shaders calls.
            إذا (pipeline.get() != requested.get()) {
                استمر;
            }

            إذا (pipeline->compiled) {
                استمر;
            }

            wait_pipeline = pipeline;

            إذا (!pipeline->compile_pending) {
                pipeline->compile_pending = درست;
                claimed_task.pipeline = pipeline;
                claimed_task.spv_اندازه = spv_اندازه;
                claimed_task.spv_داده = spv_داده;
                claimed_task.entrypoint = entrypoint;
                claimed_task.parameter_شمار = parameter_شمار;
                claimed_task.wg_denoms = wg_denoms;
                claimed_task.specialization_constants = specialization_constants;
                claimed_task.disable_robustness = disable_robustness;
                claimed_task.require_پر_subgroups = require_پر_subgroups;
                claimed_task.required_subgroup_اندازه = required_subgroup_اندازه;
                has_claimed_task = درست;
            }
        }
    };

    خودکار ثابت &ggml_vk_create_pipeline2 = [&](vk_device& device, vk_pipeline& pipeline, ثابت std::string &name, size_t spv_اندازه, ثابت فراغ* spv_داده, ثابت حرف *entrypoint,
                                              uint32_t parameter_شمار, uint32_t افزودن_constant_اندازه, std::array<uint32_t, 3> wg_denoms, ثابت std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, منطقي disable_robustness = خطأ, منطقي require_پر_subgroups = خطأ, uint32_t required_subgroup_اندازه = 0) {
        أرجع ggml_vk_create_pipeline(device, pipeline, name.c_str(), spv_اندازه, spv_داده, entrypoint,
                                       parameter_شمار, افزودن_constant_اندازه, wg_denoms, specialization_constants,
                                       align, disable_robustness, require_پر_subgroups, required_subgroup_اندازه);
    };

    // FA scalar has two SPIR-V modules (MMQ vs non-MMQ); FA cm1 has one. K/V
    // quant type is selected at runtime via the FaTypeK / FaTypeV spec constants.

    لكل (خودکار &fa : device->pipeline_flash_attn_f32_f16) {
        إذا (fa.اولی.path != FA_SCALAR) استمر;
        ثابت uint32_t Br = fa.اولی.Br;
        ثابت uint32_t Bc = fa.اولی.Bc;
        ثابت منطقي aligned = fa.اولی.aligned;
        ثابت منطقي f32acc = fa.اولی.f32acc;
        ثابت uint32_t fa_sgs = fa.اولی.subgroup_اندازه;
        ثابت منطقي fa_ds = fa.اولی.subgroup_اندازه == 0;

        ثابت منطقي bf16_kv = fa.اولی.k_type == GGML_TYPE_BF16;
        ثابت منطقي use_mmq = ggml_vk_fa_scalar_استفاده_میکند_mmq(device, fa.اولی.k_type, fa.اولی.v_type);
        ثابت فراغ * spv_داده = اشاره_تهی;
        size_t spv_اندازه = 0;
        ثابت حرف *name = اشاره_تهی;
        إذا (bf16_kv) {
            spv_داده = flash_attn_f32_f16_fp32_داده;
            spv_اندازه = flash_attn_f32_f16_fp32_len;
            name = aligned ? "flash_attn_f32_bf16_aligned" : "flash_attn_f32_bf16";
        } وإلا إذا (use_mmq) {
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            إذا (device->fp16) {
                إذا (f32acc) { spv_داده = flash_attn_f32_f16_int8_داده;        spv_اندازه = flash_attn_f32_f16_int8_len; }
                وإلا        { spv_داده = flash_attn_f32_f16_f16acc_int8_داده; spv_اندازه = flash_attn_f32_f16_f16acc_int8_len; }
            } وإلا {
                spv_داده = flash_attn_f32_f16_fp32_int8_داده;
                spv_اندازه = flash_attn_f32_f16_fp32_int8_len;
            }
#پایان_اگر
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        } وإلا {
            إذا (device->fp16) {
                إذا (device->dot2_f16) {
                    إذا (f32acc) { spv_داده = flash_attn_f32_f16_dot2_داده;        spv_اندازه = flash_attn_f32_f16_dot2_len; }
                    وإلا        { spv_داده = flash_attn_f32_f16_dot2_f16acc_داده; spv_اندازه = flash_attn_f32_f16_dot2_f16acc_len; }
                } وإلا {
                    إذا (f32acc) { spv_داده = flash_attn_f32_f16_داده;        spv_اندازه = flash_attn_f32_f16_len; }
                    وإلا        { spv_داده = flash_attn_f32_f16_f16acc_داده; spv_اندازه = flash_attn_f32_f16_f16acc_len; }
                }
            } وإلا {
                spv_داده = flash_attn_f32_f16_fp32_داده;
                spv_اندازه = flash_attn_f32_f16_fp32_len;
            }
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        }
        ggml_vk_create_pipeline(device, fa.second, name, spv_اندازه, spv_داده, "main", 8,
                                حجم(vk_flash_attn_افزودن_constants), {Br, 1, 1},
                                get_fa_spec_constants(fa.اولی), aligned ? Bc : 1, درست,
                                !fa_ds, !fa_ds ? fa_sgs : 0);
    }

#إذا defined(VK_KHR_cooperative_ماتریس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    إذا (device->coopmat1_fa_support) {
        لكل (خودکار &fa : device->pipeline_flash_attn_f32_f16) {
            إذا (fa.اولی.path != FA_COOPMAT1) استمر;
            ثابت uint32_t Br = fa.اولی.Br;
            ثابت uint32_t Bc = fa.اولی.Bc;
            ثابت منطقي aligned = fa.اولی.aligned;
            ثابت منطقي f32acc = fa.اولی.f32acc;
            ثابت uint32_t fa_sgs = fa.اولی.subgroup_اندازه;
            ثابت منطقي fa_ds = fa.اولی.subgroup_اندازه == 0;

            ثابت منطقي bf16_kv = fa.اولی.k_type == GGML_TYPE_BF16;

            ثابت فراغ * spv_داده;
            size_t spv_اندازه;
            ثابت حرف *name;
            إذا (bf16_kv) {
#إذا defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                إذا (!device->coopmat_bf16_support) استمر;
                spv_داده = flash_attn_f32_f16_bf16_cm1_داده;
                spv_اندازه = flash_attn_f32_f16_bf16_cm1_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm1" : "flash_attn_f32_bf16_cm1";
#وإلا
                استمر;
#پایان_اگر
            } وإلا {
                إذا (f32acc) { spv_داده = flash_attn_f32_f16_cm1_داده;        spv_اندازه = flash_attn_f32_f16_cm1_len; }
                وإلا        { spv_داده = flash_attn_f32_f16_f16acc_cm1_داده; spv_اندازه = flash_attn_f32_f16_f16acc_cm1_len; }
                name = aligned ? "flash_attn_f32_f16_aligned_cm1" : "flash_attn_f32_f16_cm1";
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_اندازه, spv_داده, "main", 8,
                                    حجم(vk_flash_attn_افزودن_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.اولی), aligned ? Bc : 1, درست,
                                    !fa_ds, !fa_ds ? fa_sgs : 0);
        }
    }
#پایان_اگر

#إذا defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    إذا (device->coopmat2) {
        لكل (خودکار &fa : device->pipeline_flash_attn_f32_f16) {
            إذا (fa.اولی.path != FA_COOPMAT2) استمر;
            ثابت uint32_t Br = fa.اولی.Br;
            ثابت uint32_t Bc = fa.اولی.Bc;
            ثابت منطقي aligned = fa.اولی.aligned;
            ثابت منطقي f32acc = fa.اولی.f32acc;

            ثابت منطقي bf16_kv = fa.اولی.k_type == GGML_TYPE_BF16;
            ثابت فراغ * spv_داده;
            size_t spv_اندازه;
            ثابت حرف * name;
            إذا (bf16_kv) {
#إذا defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                إذا (!device->coopmat2_bf16_support) استمر;
                spv_داده = flash_attn_f32_f16_bf16_cm2_داده;
                spv_اندازه = flash_attn_f32_f16_bf16_cm2_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm2" : "flash_attn_f32_bf16_cm2";
#وإلا
                استمر;
#پایان_اگر
            } وإلا إذا (aligned) {
                إذا (f32acc) { spv_داده = flash_attn_f32_f16_cm2_داده;        spv_اندازه = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_aligned_f32acc_cm2"; }
                وإلا        { spv_داده = flash_attn_f32_f16_f16acc_cm2_داده; spv_اندازه = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_aligned_f16acc_cm2"; }
            } وإلا {
                إذا (f32acc) { spv_داده = flash_attn_f32_f16_cm2_داده;        spv_اندازه = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_f32acc_cm2"; }
                وإلا        { spv_داده = flash_attn_f32_f16_f16acc_cm2_داده; spv_اندازه = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_f16acc_cm2"; }
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_اندازه, spv_داده, "main", 8,
                                    حجم(vk_flash_attn_افزودن_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.اولی), aligned ? Bc : 1, درست, خطأ, 0);
        }
    }
#پایان_اگر

    خودکار ثابت &ggml_vk_mul_mm_spec = [&device](std::vector<uint32_t> spec, منطقي aligned) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        إذا (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=12: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=13: APPLY_SLM_A_RESHAPE = true
        }
        أرجع spec;
    };

    خودکار ثابت &ggml_vk_mul_mm_spec_quant = [&device](std::vector<uint32_t> spec, منطقي aligned, uint32_t type) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        spec.push_back(type);               // constantID=12: MmTypeA
        إذا (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=13: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=14: APPLY_SLM_A_RESHAPE = true
        }
        أرجع spec;
    };

    ساكن ثابت ggml_type non_lut_quant_types[] = {
        GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
    };

#تعریف_کن FOR_EACH_LUT_TYPE_NONFP4(X) \
    X(GGML_TYPE_IQ1_S,   iq1_s)   \
    X(GGML_TYPE_IQ1_M,   iq1_m)   \
    X(GGML_TYPE_IQ2_XXS, iq2_xxs) \
    X(GGML_TYPE_IQ2_XS,  iq2_xs)  \
    X(GGML_TYPE_IQ2_S,   iq2_s)   \
    X(GGML_TYPE_IQ3_XXS, iq3_xxs) \
    X(GGML_TYPE_IQ3_S,   iq3_s)   \
    X(GGML_TYPE_IQ4_XS,  iq4_xs)  \
    X(GGML_TYPE_IQ4_NL,  iq4_nl)
#تعریف_کن FOR_EACH_LUT_FP4_TYPE(X) \
    X(GGML_TYPE_MXFP4,   mxfp4)   \
    X(GGML_TYPE_NVFP4,   nvfp4)
#تعریف_کن FOR_EACH_LUT_TYPE(X) \
    FOR_EACH_LUT_TYPE_NONFP4(X)  \
    FOR_EACH_LUT_FP4_TYPE(X)

    ثابت صحيح mul_mat_id_param_شمار = 5;

    استفاده spec_fn_t = std::function<std::vector<uint32_t>(ثابت std::vector<uint32_t>&, منطقي)>;
    خودکار ثابت &create_mm_pipelines = [&](
        ثابت vk_matmul_pipeline_کلید& کلید,
        ثابت std::vector<vk_tile_config>& tile_configs,
        ثابت std::string& shader_name, size_t spv_len, ثابت فراغ* spv_داده,
        uint32_t افزودن_constant_اندازه, uint32_t param_شمار,
        ثابت spec_fn_t& spec_fn,
        منطقي disable_robustness = خطأ, منطقي require_پر_subgroups = خطأ, uint32_t required_subgroup_اندازه = 0,
        منطقي create_aligned = درست, منطقي pin_subgroup_مقصد_warp = خطأ
    ) {
        خودکار& vec = device->pipeline_matmul[کلید];
        ثابت منطقي اولی_call = vec.خالی();
        لكل (size_t i = 0; i < tile_configs.size(); i++) {
            ثابت خودکار& tc = tile_configs[i];

            // Intel coopmat1 pins the required subgroup size to each warptile's WARP element.
            ثابت uint32_t rsgs = pin_subgroup_مقصد_warp ? tc.warptile[WARP_اندازه_IDX] : required_subgroup_اندازه;
            ثابت منطقي     rfs  = require_پر_subgroups || pin_subgroup_مقصد_warp;

            إذا (اولی_call) {
                vk_matmul_pipeline_pair pair{};
                pair.align = tc.align;
                std::string suffix = "_" + std::to_string(i);
                pair.unaligned = std::make_shared<vk_pipeline_struct>();
                إذا (create_aligned) {
                    pair.aligned = std::make_shared<vk_pipeline_struct>();
                }
                vec.push_back(pair);
            }

            ggml_vk_create_pipeline(device, vec[i].unaligned,
                vec[i].unaligned->name.خالی() ? (shader_name + "_" + std::to_string(i)).c_str() : vec[i].unaligned->name.c_str(),
                spv_len, spv_داده, "main", param_شمار, افزودن_constant_اندازه,
                tc.wg_denoms, spec_fn(tc.warptile, خطأ), 1,
                disable_robustness, rfs, rsgs);

            إذا (vec[i].aligned) {
                ggml_vk_create_pipeline(device, vec[i].aligned,
                    vec[i].aligned->name.خالی() ? (shader_name + "_aligned_" + std::to_string(i)).c_str() : vec[i].aligned->name.c_str(),
                    spv_len, spv_داده, "main", param_شمار, افزودن_constant_اندازه,
                    tc.wg_denoms, spec_fn(tc.warptile, درست), tc.align,
                    disable_robustness, rfs, rsgs);
            }
        }
    };

    خودکار filter_tc = [&](ثابت std::vector<vk_tile_config>& configs, ggml_type type, منطقي is_id, منطقي is_int = خطأ) -> std::vector<vk_tile_config> {
        std::vector<vk_tile_config> نتیجه;
        منطقي enabled[3];
        إذا (is_int) {
            enabled[0] = is_id ? device->mul_mat_id_s_int[type] : device->mul_mat_s_int[type];
            enabled[1] = is_id ? device->mul_mat_id_m_int[type] : device->mul_mat_m_int[type];
            enabled[2] = is_id ? device->mul_mat_id_l_int[type] : device->mul_mat_l_int[type];
        } وإلا {
            enabled[0] = is_id ? device->mul_mat_id_s[type] : device->mul_mat_s[type];
            enabled[1] = is_id ? device->mul_mat_id_m[type] : device->mul_mat_m[type];
            enabled[2] = is_id ? device->mul_mat_id_l[type] : device->mul_mat_l[type];
        }
        لكل (size_t i = 0; i < configs.size() && i < 3; i++) {
            إذا (enabled[i]) نتیجه.push_back(configs[i]);
        }
        أرجع نتیجه;
    };

    std::vector<vk_tile_config> tc_mm = {{s_warptile, s_wg_denoms, s_align}, {m_warptile, m_wg_denoms, m_align}, {l_warptile, l_wg_denoms, l_align}};
    std::vector<vk_tile_config> tc_mmq = {{s_warptile_mmq, s_mmq_wg_denoms, s_align}, {m_warptile_mmq, m_mmq_wg_denoms, m_align}, {l_warptile_mmq, l_mmq_wg_denoms, l_align}};

#إذا defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    إذا (device->coopmat2) {
        خودکار ثابت &ggml_vk_mul_mm_cm2_spec = [&](std::vector<uint32_t> spec, منطقي aligned, uint32_t type = UINT32_MAX) {
            spec.push_back(aligned ? 1u : 0u);        // ALIGNED
            spec.push_back(device->subgroup_اندازه);     // subgroup_size
            إذا (type != UINT32_MAX) {
                spec.push_back(type);                  // MmTypeA
                spec.push_back((uint32_t)ggml_type_اندازه((ggml_type)type)); // MmABlockBytes
            }
            أرجع spec;
        };

        std::vector<vk_tile_config> tc_mmq_k = {{s_warptile_mmq_k, s_mmq_wg_denoms_k, s_align}, {m_warptile_mmq_k, m_mmq_wg_denoms_k, m_align}, {l_warptile_mmq_k, l_mmq_wg_denoms_k, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmqid_wg_denoms, s_align}, {m_warptile_mmqid, m_mmqid_wg_denoms, m_align}, {l_warptile_mmqid, l_mmqid_wg_denoms, l_align}};

        spec_fn_t cm2_spec = [&](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_cm2_spec(wt, a); };

        // F16 x F16
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, درست},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm2_len, matmul_f16_f16acc_cm2_داده, حجم(vk_mat_mat_افزودن_constants), 3, cm2_spec, درست);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f16",        matmul_f16_cm2_len,        matmul_f16_cm2_داده,        حجم(vk_mat_mat_افزودن_constants), 3, cm2_spec, درست);
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        إذا (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, خطأ, خطأ}, tc_mm, "matmul_bf16", matmul_bf16_cm2_len, matmul_bf16_cm2_داده, حجم(vk_mat_mat_افزودن_constants), 3, cm2_spec, درست);
        }
#پایان_اگر
        لكل (ثابت خودکار type : non_lut_quant_types) {
            // regression in unified shader on Ampere
            إذا (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                استمر;
            }
            خودکار& tc = ((type >= GGML_TYPE_Q2_K && type <= GGML_TYPE_Q6_K) || type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) ? tc_mmq_k : tc_mmq;
            spec_fn_t qs = [&, type](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, خطأ, درست},  tc, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm2_len, matmul_quant_f16_f16acc_cm2_داده, حجم(vk_mat_mat_افزودن_constants), 3, qs, درست);
            create_mm_pipelines({type, GGML_TYPE_F16, خطأ, خطأ}, tc, "matmul_quant_f16",        matmul_quant_f16_cm2_len,        matmul_quant_f16_cm2_داده,        حجم(vk_mat_mat_افزودن_constants), 3, qs, درست);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, خطأ, درست},  tc_mmq_k, "matmul_q4_k_f16_f16acc", matmul_q4_k_f16_f16acc_cm2_len, matmul_q4_k_f16_f16acc_cm2_داده, حجم(vk_mat_mat_افزودن_constants), 3, cm2_spec, درست);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, خطأ, خطأ}, tc_mmq_k, "matmul_q4_k_f16",        matmul_q4_k_f16_cm2_len,        matmul_q4_k_f16_cm2_داده,        حجم(vk_mat_mat_افزودن_constants), 3, cm2_spec, درست);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, خطأ, درست},  tc_mmq_k, "matmul_q5_k_f16_f16acc", matmul_q5_k_f16_f16acc_cm2_len, matmul_q5_k_f16_f16acc_cm2_داده, حجم(vk_mat_mat_افزودن_constants), 3, cm2_spec, درست);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, خطأ, خطأ}, tc_mmq_k, "matmul_q5_k_f16",        matmul_q5_k_f16_cm2_len,        matmul_q5_k_f16_cm2_داده,        حجم(vk_mat_mat_افزودن_constants), 3, cm2_spec, درست);
#تعریف_کن X_CM2(TYPE, tstr) \
        { خودکار tc = filter_tc(tc_mmq, TYPE, خطأ); \
          إذا (!tc.خالی()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, خطأ, درست},  tc, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm2_len, matmul_##tstr##_f16_f16acc_cm2_data, حجم(vk_mat_mat_افزودن_constants), 3, cm2_spec, درست); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, خطأ, خطأ}, tc, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm2_len,        matmul_##tstr##_f16_cm2_data,        حجم(vk_mat_mat_افزودن_constants), 3, cm2_spec, درست); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2)
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        إذا (device->ocp_fp4) {
#تعریف_کن X_CM2_OCP(TYPE, tstr) \
            { خودکار tc = filter_tc(tc_mmq, TYPE, خطأ); \
              إذا (!tc.خالی()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, خطأ, درست},  tc, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm2_len, matmul_##tstr##_f16_ocp_f16acc_cm2_data, حجم(vk_mat_mat_افزودن_constants), 3, cm2_spec, درست); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, خطأ, خطأ}, tc, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm2_len,        matmul_##tstr##_f16_ocp_cm2_data,        حجم(vk_mat_mat_افزودن_constants), 3, cm2_spec, درست); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_OCP)
#لغو_تعریف X_CM2_OCP
        } وإلا
#پایان_اگر
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2)
        }
#لغو_تعریف X_CM2

        GGML_ASSERT(device->subgroup_ballot);

        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, درست, درست},  tc_mm, "matmul_id_subgroup_f16_f16acc", matmul_id_subgroup_f16_f16acc_cm2_len, matmul_id_subgroup_f16_f16acc_cm2_داده, حجم(vk_mat_mat_id_افزودن_constants), 5, cm2_spec, درست);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, درست, خطأ}, tc_mm, "matmul_id_subgroup_f16",        matmul_id_subgroup_f16_cm2_len,        matmul_id_subgroup_f16_cm2_داده,        حجم(vk_mat_mat_id_افزودن_constants), 5, cm2_spec, درست);
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        إذا (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, درست, خطأ}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm2_len, matmul_id_subgroup_bf16_cm2_داده, حجم(vk_mat_mat_id_افزودن_constants), 5, cm2_spec, درست);
        }
#پایان_اگر
        لكل (ثابت خودکار type : non_lut_quant_types) {
            إذا (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                استمر;
            }
            spec_fn_t qs_id = [&, type](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, درست, درست},  tc_mmqid, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm2_len, matmul_id_subgroup_quant_f16_f16acc_cm2_داده, حجم(vk_mat_mat_id_افزودن_constants), 5, qs_id, درست);
            create_mm_pipelines({type, GGML_TYPE_F16, درست, خطأ}, tc_mmqid, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm2_len,        matmul_id_subgroup_quant_f16_cm2_داده,        حجم(vk_mat_mat_id_افزودن_constants), 5, qs_id, درست);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, درست, درست},  tc_mmqid, "matmul_id_subgroup_q4_k_f16_f16acc", matmul_id_subgroup_q4_k_f16_f16acc_cm2_len, matmul_id_subgroup_q4_k_f16_f16acc_cm2_داده, حجم(vk_mat_mat_id_افزودن_constants), 5, cm2_spec, درست);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, درست, خطأ}, tc_mmqid, "matmul_id_subgroup_q4_k_f16",        matmul_id_subgroup_q4_k_f16_cm2_len,        matmul_id_subgroup_q4_k_f16_cm2_داده,        حجم(vk_mat_mat_id_افزودن_constants), 5, cm2_spec, درست);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, درست, درست},  tc_mmqid, "matmul_id_subgroup_q5_k_f16_f16acc", matmul_id_subgroup_q5_k_f16_f16acc_cm2_len, matmul_id_subgroup_q5_k_f16_f16acc_cm2_داده, حجم(vk_mat_mat_id_افزودن_constants), 5, cm2_spec, درست);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, درست, خطأ}, tc_mmqid, "matmul_id_subgroup_q5_k_f16",        matmul_id_subgroup_q5_k_f16_cm2_len,        matmul_id_subgroup_q5_k_f16_cm2_داده,        حجم(vk_mat_mat_id_افزودن_constants), 5, cm2_spec, درست);
#تعریف_کن X_CM2_ID(TYPE, tstr) \
        { خودکار tc = filter_tc(tc_mmqid, TYPE, درست); \
          إذا (!tc.خالی()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, درست, درست},  tc, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm2_data, حجم(vk_mat_mat_id_افزودن_constants), 5, cm2_spec, درست); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, درست, خطأ}, tc, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm2_len,        matmul_id_subgroup_##tstr##_f16_cm2_data,        حجم(vk_mat_mat_id_افزودن_constants), 5, cm2_spec, درست); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2_ID)
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        إذا (device->ocp_fp4) {
#تعریف_کن X_CM2_ID_OCP(TYPE, tstr) \
            { خودکار tc = filter_tc(tc_mmqid, TYPE, درست); \
              إذا (!tc.خالی()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, درست, درست},  tc, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_data, حجم(vk_mat_mat_id_افزودن_constants), 5, cm2_spec, درست); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, درست, خطأ}, tc, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm2_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm2_data,        حجم(vk_mat_mat_id_افزودن_constants), 5, cm2_spec, درست); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID_OCP)
#لغو_تعریف X_CM2_ID_OCP
        } وإلا
#پایان_اگر
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID)
        }
#لغو_تعریف X_CM2_ID
    } وإلا
#پایان_اگر  // defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
#إذا defined(VK_KHR_cooperative_ماتریس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    إذا (device->coopmat_support) {
        spec_fn_t cm1_spec = [&](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_spec(wt, a); };

        // Intel coopmat1 pins each pipeline's required subgroup size to its warptile WARP element.
        ثابت منطقي cm1_pin = device->vendor_id == VK_VENDOR_ID_INTEL;

        // Intel coopmat1 uses a dedicated large-tile config for quant matmul_id.
        std::vector<vk_tile_config> tc_mmq_id = tc_mmq;
        إذا (cm1_pin) {
            tc_mmq_id[2] = { { 512, 128, 128, 32, 32, 32, 2, device->coopmat_m, device->coopmat_n, device->coopmat_k, 32 }, { 128, 128, 1 }, 32 };
        }

        خودکار cm1_create = [&](vk_matmul_pipeline_کلید کلید, ثابت std::vector<vk_tile_config>& tc_پایه,
                              ثابت std::string& name, size_t len, ثابت فراغ* داده, uint32_t pc_اندازه, uint32_t pc) {
            خودکار tc = filter_tc(tc_پایه, کلید.type_a, کلید.mul_mat_id);
            إذا (!tc.خالی()) create_mm_pipelines(کلید, tc, name, len, داده, pc_اندازه, pc, cm1_spec, خطأ, درست, 0, درست, cm1_pin);
        };
        خودکار cm1_create_quant = [&](vk_matmul_pipeline_کلید کلید, ثابت std::vector<vk_tile_config>& tc_پایه,
                                    ثابت std::string& name, size_t len, ثابت فراغ* داده, uint32_t pc_اندازه, uint32_t pc) {
            spec_fn_t qs = [&, type_a=کلید.type_a](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
            خودکار tc = filter_tc(tc_پایه, کلید.type_a, کلید.mul_mat_id);
            إذا (!tc.خالی()) create_mm_pipelines(کلید, tc, name, len, داده, pc_اندازه, pc, qs, خطأ, درست, 0, درست, cm1_pin);
        };
        // int8 MMQ helper: per-type cm1 shader, warptile passed as-is (carries DEVICE_ARCH in
        // spec constant WARP_SIZE_IDX+1), subgroup size pinned to the warptile WARP element, no aligned variant.
        خودکار cm1_create_mmq = [&](vk_matmul_pipeline_کلید کلید, ثابت std::vector<vk_tile_config>& tc_پایه,
                                  ثابت std::string& name, size_t len, ثابت فراغ* داده, uint32_t pc_اندازه, uint32_t pc) {
            spec_fn_t عنصر_خنثی = [](ثابت std::vector<uint32_t>& wt, منطقي) { أرجع wt; };
            خودکار tc = filter_tc(tc_پایه, کلید.type_a, کلید.mul_mat_id, درست);
            إذا (!tc.خالی()) create_mm_pipelines(کلید, tc, name, len, داده, pc_اندازه, pc, عنصر_خنثی, خطأ, خطأ, 0, خطأ, درست);
        };

        std::vector<vk_tile_config> tc_mmq_cm1_int = {
            {s_warptile_mmq_cm1_int, s_mmq_wg_denoms, s_align},
            {m_warptile_mmq_cm1_int, m_mmq_wg_denoms, m_align},
            {l_warptile_mmq_cm1_int, l_mmq_wg_denoms, l_align},
        };
        std::vector<vk_tile_config> tc_mmq_cm1_int_k = {
            {s_warptile_mmq_cm1_int_k, s_mmq_cm1_wg_denoms_k, s_align},
            {m_warptile_mmq_cm1_int_k, m_mmq_cm1_wg_denoms_k, m_align},
            {l_warptile_mmq_cm1_int_k, l_mmq_cm1_wg_denoms_k, l_align},
        };

        // Some quants are not performant on RDNA4, those fall back to FP16 matmul
        ثابت منطقي rdna3 = device->architecture == AMD_RDNA3;
        ثابت منطقي rdna4 = device->architecture == AMD_RDNA4;

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, خطأ, خطأ}, tc_mm, "matmul_f32_f32",     matmul_f32_f32_cm1_len,     matmul_f32_f32_cm1_داده,     حجم(vk_mat_mat_افزودن_constants), 3);
        cm1_create({GGML_TYPE_F32, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f32_f16",     matmul_f32_f16_cm1_len,     matmul_f32_f16_cm1_داده,     حجم(vk_mat_mat_افزودن_constants), 3);
        إذا (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, درست},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm1_len, matmul_f16_f16acc_cm1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, خطأ, درست},  tc_mm, "matmul_f16_f32_f16acc", matmul_f16_f32_f16acc_cm1_len, matmul_f16_f32_f16acc_cm1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
        }
        إذا (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f16",      matmul_f16_cm1_len,      matmul_f16_cm1_داده,      حجم(vk_mat_mat_افزودن_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, خطأ, خطأ}, tc_mm, "matmul_f16_f32",  matmul_f16_f32_cm1_len,  matmul_f16_f32_cm1_داده,  حجم(vk_mat_mat_افزودن_constants), 3);
        }
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        إذا (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, خطأ, خطأ}, tc_mm, "matmul_bf16", matmul_bf16_cm1_len, matmul_bf16_cm1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
        }
#پایان_اگر
        لكل (ثابت خودکار type : non_lut_quant_types) {
            إذا (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, خطأ, درست},  tc_mmq, "matmul_quant_f32_f16acc", matmul_quant_f32_f16acc_cm1_len, matmul_quant_f32_f16acc_cm1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, خطأ, درست},  tc_mmq, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm1_len, matmul_quant_f16_f16acc_cm1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
            }
            إذا (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_quant_f32",        matmul_quant_f32_cm1_len,        matmul_quant_f32_cm1_داده,        حجم(vk_mat_mat_افزودن_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, خطأ, خطأ}, tc_mmq, "matmul_quant_f16",        matmul_quant_f16_cm1_len,        matmul_quant_f16_cm1_داده,        حجم(vk_mat_mat_افزودن_constants), 3);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#تعریف_کن X_CM1(TYPE, tstr) \
        إذا (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, خطأ, درست},  tc_mmq, "matmul_" #tstr "_f32_f16acc", matmul_##tstr##_f32_f16acc_cm1_len, matmul_##tstr##_f32_f16acc_cm1_data, حجم(vk_mat_mat_افزودن_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, خطأ, درست},  tc_mmq, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm1_len, matmul_##tstr##_f16_f16acc_cm1_data, حجم(vk_mat_mat_افزودن_constants), 3); \
        } \
        إذا (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_" #tstr "_f32",        matmul_##tstr##_f32_cm1_len,        matmul_##tstr##_f32_cm1_data,        حجم(vk_mat_mat_افزودن_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, خطأ, خطأ}, tc_mmq, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm1_len,        matmul_##tstr##_f16_cm1_data,        حجم(vk_mat_mat_افزودن_constants), 3); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1)
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        إذا (device->ocp_fp4) {
#تعریف_کن X_CM1_OCP(TYPE, tstr) \
            إذا (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, خطأ, درست},  tc_mmq, "matmul_" #tstr "_f32_ocp_f16acc", matmul_##tstr##_f32_ocp_f16acc_cm1_len, matmul_##tstr##_f32_ocp_f16acc_cm1_data, حجم(vk_mat_mat_افزودن_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, خطأ, درست},  tc_mmq, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm1_len, matmul_##tstr##_f16_ocp_f16acc_cm1_data, حجم(vk_mat_mat_افزودن_constants), 3); \
            } \
            إذا (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_" #tstr "_f32_ocp",        matmul_##tstr##_f32_ocp_cm1_len,        matmul_##tstr##_f32_ocp_cm1_data,        حجم(vk_mat_mat_افزودن_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, خطأ, خطأ}, tc_mmq, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm1_len,        matmul_##tstr##_f16_ocp_cm1_data,        حجم(vk_mat_mat_افزودن_constants), 3); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_OCP)
#لغو_تعریف X_CM1_OCP
        } وإلا
#پایان_اگر
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1)
        }
#لغو_تعریف X_CM1

        إذا (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q4_0_q8_1",   matmul_q4_0_q8_1_cm1_len,   matmul_q4_0_q8_1_cm1_داده,   حجم(vk_mat_mat_افزودن_constants), 3);
            إذا (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q4_1_q8_1",   matmul_q4_1_q8_1_cm1_len,   matmul_q4_1_q8_1_cm1_داده,   حجم(vk_mat_mat_افزودن_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q5_0_q8_1",   matmul_q5_0_q8_1_cm1_len,   matmul_q5_0_q8_1_cm1_داده,   حجم(vk_mat_mat_افزودن_constants), 3);
            إذا (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q5_1_q8_1",   matmul_q5_1_q8_1_cm1_len,   matmul_q5_1_q8_1_cm1_داده,   حجم(vk_mat_mat_افزودن_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q8_0_q8_1",   matmul_q8_0_q8_1_cm1_len,   matmul_q8_0_q8_1_cm1_داده,   حجم(vk_mat_mat_افزودن_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_iq4_nl_q8_1", matmul_iq4_nl_q8_1_cm1_len, matmul_iq4_nl_q8_1_cm1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_cm1_len, matmul_iq4_xs_q8_1_cm1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_mxfp4_q8_1",  matmul_mxfp4_q8_1_cm1_len,  matmul_mxfp4_q8_1_cm1_داده,  حجم(vk_mat_mat_افزودن_constants), 3);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int_k, "matmul_q3_k_q8_1",   matmul_q3_k_q8_1_cm1_len,   matmul_q3_k_q8_1_cm1_داده,   حجم(vk_mat_mat_افزودن_constants), 3);
            إذا (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q4_k_q8_1",   matmul_q4_k_q8_1_cm1_len,   matmul_q4_k_q8_1_cm1_داده,   حجم(vk_mat_mat_افزودن_constants), 3); }
            إذا (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int,   "matmul_q5_k_q8_1",   matmul_q5_k_q8_1_cm1_len,   matmul_q5_k_q8_1_cm1_داده,   حجم(vk_mat_mat_افزودن_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int_k, "matmul_q6_k_q8_1",   matmul_q6_k_q8_1_cm1_len,   matmul_q6_k_q8_1_cm1_داده,   حجم(vk_mat_mat_افزودن_constants), 3);
            إذا (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_cm1_int_k, "matmul_nvfp4_q8_1",  matmul_nvfp4_q8_1_cm1_len,  matmul_nvfp4_q8_1_cm1_داده,  حجم(vk_mat_mat_افزودن_constants), 3); }
        }

        GGML_ASSERT(device->subgroup_ballot);

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, درست, خطأ}, tc_mm, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_cm1_len, matmul_id_subgroup_f32_f32_cm1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
        إذا (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, درست, درست},  tc_mm, "matmul_id_subgroup_f16_f16acc",     matmul_id_subgroup_f16_f16acc_cm1_len,     matmul_id_subgroup_f16_f16acc_cm1_داده,     حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, درست, درست},  tc_mm, "matmul_id_subgroup_f16_f32_f16acc", matmul_id_subgroup_f16_f32_f16acc_cm1_len, matmul_id_subgroup_f16_f32_f16acc_cm1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
        }
        إذا (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, درست, خطأ}, tc_mm, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_cm1_len,     matmul_id_subgroup_f16_cm1_داده,     حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, درست, خطأ}, tc_mm, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_cm1_len, matmul_id_subgroup_f16_f32_cm1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
        }
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        إذا (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, درست, خطأ}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm1_len, matmul_id_subgroup_bf16_cm1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
        }
#پایان_اگر
        لكل (ثابت خودکار type : non_lut_quant_types) {
            إذا (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, درست, درست},  tc_mmq_id, "matmul_id_subgroup_quant_f32_f16acc", matmul_id_subgroup_quant_f32_f16acc_cm1_len, matmul_id_subgroup_quant_f32_f16acc_cm1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                cm1_create_quant({type, GGML_TYPE_F16, درست, درست},  tc_mmq_id, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm1_len, matmul_id_subgroup_quant_f16_f16acc_cm1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            }
            إذا (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, درست, خطأ}, tc_mmq_id, "matmul_id_subgroup_quant_f32",        matmul_id_subgroup_quant_f32_cm1_len,        matmul_id_subgroup_quant_f32_cm1_داده,        حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                cm1_create_quant({type, GGML_TYPE_F16, درست, خطأ}, tc_mmq_id, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm1_len,        matmul_id_subgroup_quant_f16_cm1_داده,        حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#تعریف_کن X_CM1_ID(TYPE, tstr) \
        إذا (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, درست, درست},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_f16acc", matmul_id_subgroup_##tstr##_f32_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_f16acc_cm1_data, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار); \
            cm1_create({TYPE, GGML_TYPE_F16, درست, درست},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm1_data, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار); \
        } \
        إذا (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, درست, خطأ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32",        matmul_id_subgroup_##tstr##_f32_cm1_len,        matmul_id_subgroup_##tstr##_f32_cm1_data,        حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار); \
            cm1_create({TYPE, GGML_TYPE_F16, درست, خطأ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm1_len,        matmul_id_subgroup_##tstr##_f16_cm1_data,        حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1_ID)
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        إذا (device->ocp_fp4) {
#تعریف_کن X_CM1_ID_OCP(TYPE, tstr) \
            إذا (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, درست, درست},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp_f16acc", matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_data, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار); \
                cm1_create({TYPE, GGML_TYPE_F16, درست, درست},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_data, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار); \
            } \
            إذا (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, درست, خطأ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp",        matmul_id_subgroup_##tstr##_f32_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f32_ocp_cm1_data,        حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار); \
                cm1_create({TYPE, GGML_TYPE_F16, درست, خطأ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm1_data,        حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID_OCP)
#لغو_تعریف X_CM1_ID_OCP
        } وإلا
#پایان_اگر
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID)
        }
#لغو_تعریف X_CM1_ID

        إذا (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, درست, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_0_q8_1",   matmul_id_subgroup_q4_0_q8_1_cm1_len,   matmul_id_subgroup_q4_0_q8_1_cm1_داده,   حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            cm1_create_mmq({GGML_TYPE_Q4_1,   GGML_TYPE_Q8_1, درست, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_1_q8_1",   matmul_id_subgroup_q4_1_q8_1_cm1_len,   matmul_id_subgroup_q4_1_q8_1_cm1_داده,   حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, درست, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_0_q8_1",   matmul_id_subgroup_q5_0_q8_1_cm1_len,   matmul_id_subgroup_q5_0_q8_1_cm1_داده,   حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            cm1_create_mmq({GGML_TYPE_Q5_1,   GGML_TYPE_Q8_1, درست, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_1_q8_1",   matmul_id_subgroup_q5_1_q8_1_cm1_len,   matmul_id_subgroup_q5_1_q8_1_cm1_داده,   حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, درست, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q8_0_q8_1",   matmul_id_subgroup_q8_0_q8_1_cm1_len,   matmul_id_subgroup_q8_0_q8_1_cm1_داده,   حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_nl_q8_1", matmul_id_subgroup_iq4_nl_q8_1_cm1_len, matmul_id_subgroup_iq4_nl_q8_1_cm1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_cm1_len, matmul_id_subgroup_iq4_xs_q8_1_cm1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, درست, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_mxfp4_q8_1",  matmul_id_subgroup_mxfp4_q8_1_cm1_len,  matmul_id_subgroup_mxfp4_q8_1_cm1_داده,  حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, درست, خطأ}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q3_k_q8_1",   matmul_id_subgroup_q3_k_q8_1_cm1_len,   matmul_id_subgroup_q3_k_q8_1_cm1_داده,   حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            cm1_create_mmq({GGML_TYPE_Q4_K,   GGML_TYPE_Q8_1, درست, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_k_q8_1",   matmul_id_subgroup_q4_k_q8_1_cm1_len,   matmul_id_subgroup_q4_k_q8_1_cm1_داده,   حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            cm1_create_mmq({GGML_TYPE_Q5_K,   GGML_TYPE_Q8_1, درست, خطأ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_k_q8_1",   matmul_id_subgroup_q5_k_q8_1_cm1_len,   matmul_id_subgroup_q5_k_q8_1_cm1_داده,   حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, درست, خطأ}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q6_k_q8_1",   matmul_id_subgroup_q6_k_q8_1_cm1_len,   matmul_id_subgroup_q6_k_q8_1_cm1_داده,   حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
            إذا (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmq_cm1_int_k, "matmul_id_subgroup_nvfp4_q8_1",  matmul_id_subgroup_nvfp4_q8_1_cm1_len,  matmul_id_subgroup_nvfp4_q8_1_cm1_داده,  حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار); }
        }
    } وإلا
#پایان_اگر  // defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    {
        // Helper for subgroup path with dot2 selection and filtering
        خودکار sg_create = [&](vk_matmul_pipeline_کلید کلید, ثابت std::vector<vk_tile_config>& tc_پایه,
                             ثابت std::string& name, size_t len, ثابت فراغ* داده, uint32_t pc_اندازه, uint32_t pc,
                             uint32_t rsgs = 0) {
            خودکار tc = filter_tc(tc_پایه, کلید.type_a, کلید.mul_mat_id);
            إذا (!tc.خالی()) create_mm_pipelines(کلید, tc, name, len, داده, pc_اندازه, pc,
                [&](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_spec(wt, a); },
                خطأ, rsgs > 0, rsgs);
        };
        خودکار sg_create_quant = [&](vk_matmul_pipeline_کلید کلید, ثابت std::vector<vk_tile_config>& tc_پایه,
                                   ثابت std::string& name, size_t len, ثابت فراغ* داده, uint32_t pc_اندازه, uint32_t pc,
                                   uint32_t rsgs = 0) {
            خودکار tc = filter_tc(tc_پایه, کلید.type_a, کلید.mul_mat_id);
            إذا (!tc.خالی()) {
                spec_fn_t qs = [&, type_a=کلید.type_a](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
                create_mm_pipelines(کلید, tc, name, len, داده, pc_اندازه, pc, qs, خطأ, rsgs > 0, rsgs);
            }
        };
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        خودکار sg_create_mmq = [&](vk_matmul_pipeline_کلید کلید, ثابت std::vector<vk_tile_config>& tc_پایه,
                                 ثابت std::string& name, size_t len, ثابت فراغ* داده, uint32_t pc_اندازه, uint32_t pc,
                                 uint32_t rsgs = 0) {
            خودکار tc = filter_tc(tc_پایه, کلید.type_a, کلید.mul_mat_id, درست);
            إذا (!tc.خالی()) {
                spec_fn_t عنصر_خنثی = [](ثابت std::vector<uint32_t>& wt, منطقي) { أرجع wt; };
                create_mm_pipelines(کلید, tc, name, len, داده, pc_اندازه, pc, عنصر_خنثی, خطأ, rsgs > 0, rsgs, خطأ);
            }
        };
#پایان_اگر

        std::vector<vk_tile_config> tc_id = {{s_warptile_id, s_wg_denoms, s_align}, {m_warptile_id, m_wg_denoms, m_align}, {l_warptile_id, l_wg_denoms, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid, l_mmq_wg_denoms, l_align}};

        إذا (device->fp16) {
            // FP16 subgroup path - with dot2 runtime selection
            #تعریف_کن SPV_DOT2(NAME) (device->dot2_f16 ? NAME ## _dot2_len : NAME ## _len), (device->dot2_f16 ? NAME ## dot2_داده : NAME ## داده_2)
            #تعریف_کن SPV_DOT2_F16ACC(NAME) (device->dot2_f16 ? NAME ## _dot2_f16acc_len : NAME ## _f16acc_len), (device->dot2_f16 ? NAME ## dot2_f16acc_داده : NAME ## f16acc_داده)

            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, خطأ, خطأ}, tc_mm, "matmul_f32_f32", SPV_DOT2(matmul_f32_f32), حجم(vk_mat_mat_افزودن_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f32_f16", SPV_DOT2(matmul_f32_f16), حجم(vk_mat_mat_افزودن_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, درست},  tc_mm, "matmul_f16_f16acc",     SPV_DOT2_F16ACC(matmul_f16),     حجم(vk_mat_mat_افزودن_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f16",            SPV_DOT2(matmul_f16),            حجم(vk_mat_mat_افزودن_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, خطأ, درست},  tc_mm, "matmul_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_f16_f32), حجم(vk_mat_mat_افزودن_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, خطأ, خطأ}, tc_mm, "matmul_f16_f32",        SPV_DOT2(matmul_f16_f32),        حجم(vk_mat_mat_افزودن_constants), 3);
            // BF16 - no dot2
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, خطأ, خطأ}, tc_mm, "matmul_bf16", matmul_bf16_len, matmul_bf16_داده, حجم(vk_mat_mat_افزودن_constants), 3);

            لكل (ثابت خودکار type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, خطأ, درست},  tc_mmq, "matmul_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_quant_f32), حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_quant({type, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_quant_f32",        SPV_DOT2(matmul_quant_f32),        حجم(vk_mat_mat_افزودن_constants), 3);
            }
    #تعریف_کن X_SG(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, خطأ, درست},  tc_mmq, "matmul_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_##tstr##_f32), حجم(vk_mat_mat_افزودن_constants), 3); \
            sg_create({TYPE, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_" #tstr "_f32",        SPV_DOT2(matmul_##tstr##_f32),        حجم(vk_mat_mat_افزودن_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG)
#لغو_تعریف X_SG

#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            إذا (device->integer_dot_حاصلضرب) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_len, matmul_q2_0_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_len, matmul_q4_0_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_len, matmul_q4_1_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_len, matmul_q5_0_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_len, matmul_q5_1_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_len, matmul_q8_0_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_mxfp4_q8_1", matmul_mxfp4_q8_1_len, matmul_mxfp4_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_len, matmul_iq4_xs_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_len, matmul_q2_k_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_len, matmul_q3_k_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_len, matmul_q4_k_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_len, matmul_q5_k_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_len, matmul_q6_k_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_len, matmul_iq3_s_q8_1_داده, حجم(vk_mat_mat_افزودن_constants), 3);
            }
#پایان_اگر

            إذا (device->subgroup_ballot && device->subgroup_require_پر_support && subgroup_min_اندازه_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, درست, خطأ}, tc_id, "matmul_id_subgroup_f32_f32", SPV_DOT2(matmul_id_subgroup_f32_f32), حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, درست, درست},  tc_id, "matmul_id_subgroup_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_subgroup_f16),     حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, درست, خطأ}, tc_id, "matmul_id_subgroup_f16",            SPV_DOT2(matmul_id_subgroup_f16),            حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, درست, درست},  tc_id, "matmul_id_subgroup_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_f16_f32), حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, درست, خطأ}, tc_id, "matmul_id_subgroup_f16_f32",        SPV_DOT2(matmul_id_subgroup_f16_f32),        حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, درست, خطأ}, tc_id, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_len, matmul_id_subgroup_bf16_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                لكل (ثابت خودکار type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, درست, درست},  tc_mmqid, "matmul_id_subgroup_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_quant_f32), حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه);
                    sg_create_quant({type, GGML_TYPE_F32, درست, خطأ}, tc_mmqid, "matmul_id_subgroup_quant_f32",        SPV_DOT2(matmul_id_subgroup_quant_f32),        حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه);
                }
        #تعریف_کن X_SG_ID_SUB(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, درست, درست},  tc_mmqid, "matmul_id_subgroup_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_##tstr##_f32), حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه); \
                sg_create({TYPE, GGML_TYPE_F32, درست, خطأ}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32",        SPV_DOT2(matmul_id_subgroup_##tstr##_f32),        حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB)
#لغو_تعریف X_SG_ID_SUB
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                إذا (device->integer_dot_حاصلضرب) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_subgroup_q2_0_q8_1", matmul_id_subgroup_q2_0_q8_1_len, matmul_id_subgroup_q2_0_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_subgroup_q4_0_q8_1", matmul_id_subgroup_q4_0_q8_1_len, matmul_id_subgroup_q4_0_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_subgroup_q4_1_q8_1", matmul_id_subgroup_q4_1_q8_1_len, matmul_id_subgroup_q4_1_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_subgroup_q5_0_q8_1", matmul_id_subgroup_q5_0_q8_1_len, matmul_id_subgroup_q5_0_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_subgroup_q5_1_q8_1", matmul_id_subgroup_q5_1_q8_1_len, matmul_id_subgroup_q5_1_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_subgroup_q8_0_q8_1", matmul_id_subgroup_q8_0_q8_1_len, matmul_id_subgroup_q8_0_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_subgroup_mxfp4_q8_1", matmul_id_subgroup_mxfp4_q8_1_len, matmul_id_subgroup_mxfp4_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_len, matmul_id_subgroup_iq4_xs_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int_k, "matmul_id_subgroup_q2_k_q8_1", matmul_id_subgroup_q2_k_q8_1_len, matmul_id_subgroup_q2_k_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int_k, "matmul_id_subgroup_q3_k_q8_1", matmul_id_subgroup_q3_k_q8_1_len, matmul_id_subgroup_q3_k_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int_k, "matmul_id_subgroup_q4_k_q8_1", matmul_id_subgroup_q4_k_q8_1_len, matmul_id_subgroup_q4_k_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int_k, "matmul_id_subgroup_q5_k_q8_1", matmul_id_subgroup_q5_k_q8_1_len, matmul_id_subgroup_q5_k_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int_k, "matmul_id_subgroup_q6_k_q8_1", matmul_id_subgroup_q6_k_q8_1_len, matmul_id_subgroup_q6_k_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int_k, "matmul_id_subgroup_iq3_s_q8_1", matmul_id_subgroup_iq3_s_q8_1_len, matmul_id_subgroup_iq3_s_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                }
#پایان_اگر
            } وإلا {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, درست, خطأ}, tc_mm, "matmul_id_f32_f32", SPV_DOT2(matmul_id_f32_f32), حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, درست, درست},  tc_mm, "matmul_id_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_f16),     حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, درست, خطأ}, tc_mm, "matmul_id_f16",            SPV_DOT2(matmul_id_f16),            حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, درست, درست},  tc_mm, "matmul_id_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_f16_f32), حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, درست, خطأ}, tc_mm, "matmul_id_f16_f32",        SPV_DOT2(matmul_id_f16_f32),        حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, درست, خطأ}, tc_mm, "matmul_id_bf16", matmul_id_bf16_len, matmul_id_bf16_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                لكل (ثابت خودکار type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, درست, درست},  tc_mmqid, "matmul_id_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_quant_f32), حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_quant({type, GGML_TYPE_F32, درست, خطأ}, tc_mmqid, "matmul_id_quant_f32",        SPV_DOT2(matmul_id_quant_f32),        حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                }
        #تعریف_کن X_SG_ID(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, درست, درست},  tc_mmqid, "matmul_id_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_##tstr##_f32), حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار); \
                sg_create({TYPE, GGML_TYPE_F32, درست, خطأ}, tc_mmqid, "matmul_id_" #tstr "_f32",        SPV_DOT2(matmul_id_##tstr##_f32),        حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                FOR_EACH_LUT_TYPE(X_SG_ID)
#لغو_تعریف X_SG_ID
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                إذا (device->integer_dot_حاصلضرب) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_q2_0_q8_1", matmul_id_q2_0_q8_1_len, matmul_id_q2_0_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_q4_0_q8_1", matmul_id_q4_0_q8_1_len, matmul_id_q4_0_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_q4_1_q8_1", matmul_id_q4_1_q8_1_len, matmul_id_q4_1_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_q5_0_q8_1", matmul_id_q5_0_q8_1_len, matmul_id_q5_0_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_q5_1_q8_1", matmul_id_q5_1_q8_1_len, matmul_id_q5_1_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_q8_0_q8_1", matmul_id_q8_0_q8_1_len, matmul_id_q8_0_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_mxfp4_q8_1", matmul_id_mxfp4_q8_1_len, matmul_id_mxfp4_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int, "matmul_id_iq4_xs_q8_1", matmul_id_iq4_xs_q8_1_len, matmul_id_iq4_xs_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int_k, "matmul_id_q2_k_q8_1", matmul_id_q2_k_q8_1_len, matmul_id_q2_k_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int_k, "matmul_id_q3_k_q8_1", matmul_id_q3_k_q8_1_len, matmul_id_q3_k_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int_k, "matmul_id_q4_k_q8_1", matmul_id_q4_k_q8_1_len, matmul_id_q4_k_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int_k, "matmul_id_q5_k_q8_1", matmul_id_q5_k_q8_1_len, matmul_id_q5_k_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int_k, "matmul_id_q6_k_q8_1", matmul_id_q6_k_q8_1_len, matmul_id_q6_k_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, درست, خطأ}, tc_mmqid_int_k, "matmul_id_iq3_s_q8_1", matmul_id_iq3_s_q8_1_len, matmul_id_iq3_s_q8_1_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                }
#پایان_اگر
            }
            #لغو_تعریف SPV_DOT2
            #لغو_تعریف SPV_DOT2_F16ACC
        } وإلا {
            // FP32-only fallback path
            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, خطأ, خطأ}, tc_mm, "matmul_f32_f32", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f32_f16", matmul_f32_f16_fp32_len, matmul_f32_f16_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, خطأ, خطأ}, tc_mm, "matmul_f16",     matmul_f16_fp32_len,     matmul_f16_fp32_داده,     حجم(vk_mat_mat_افزودن_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, خطأ, خطأ}, tc_mm, "matmul_f16_f32", matmul_f16_f32_fp32_len, matmul_f16_f32_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, خطأ, خطأ}, tc_mm, "matmul_bf16",  matmul_bf16_fp32_len,    matmul_bf16_fp32_داده,    حجم(vk_mat_mat_افزودن_constants), 3);

            لكل (ثابت خودکار type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_quant_f32", matmul_quant_f32_fp32_len, matmul_quant_f32_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
            }
    #تعریف_کن X_SG_FP32(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, خطأ, خطأ}, tc_mmq, "matmul_" #tstr "_f32", matmul_##tstr##_f32_fp32_len, matmul_##tstr##_f32_fp32_data, حجم(vk_mat_mat_افزودن_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG_FP32)
#لغو_تعریف X_SG_FP32

#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            إذا (device->integer_dot_حاصلضرب) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_fp32_len, matmul_q2_0_q8_1_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_fp32_len, matmul_q4_0_q8_1_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_fp32_len, matmul_q4_1_q8_1_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_fp32_len, matmul_q5_0_q8_1_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_fp32_len, matmul_q5_1_q8_1_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_fp32_len, matmul_q8_0_q8_1_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_fp32_len, matmul_iq4_xs_q8_1_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_fp32_len, matmul_q2_k_q8_1_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_fp32_len, matmul_q3_k_q8_1_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_fp32_len, matmul_q4_k_q8_1_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_fp32_len, matmul_q5_k_q8_1_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_fp32_len, matmul_q6_k_q8_1_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, خطأ, خطأ}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_fp32_len, matmul_iq3_s_q8_1_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3);
            }
#پایان_اگر

            إذا (device->subgroup_ballot && device->subgroup_require_پر_support && subgroup_min_اندازه_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, درست, خطأ}, tc_id, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_fp32_len, matmul_id_subgroup_f32_f32_fp32_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, درست, خطأ}, tc_id, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_fp32_len,     matmul_id_subgroup_f16_fp32_داده,     حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, درست, خطأ}, tc_id, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_fp32_len, matmul_id_subgroup_f16_f32_fp32_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, درست, خطأ}, tc_id, "matmul_id_subgroup_bf16",  matmul_id_subgroup_bf16_fp32_len,    matmul_id_subgroup_bf16_fp32_داده,    حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه_16);
                لكل (ثابت خودکار type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, درست, خطأ}, tc_mmqid, "matmul_id_subgroup_quant_f32", matmul_id_subgroup_quant_f32_fp32_len, matmul_id_subgroup_quant_f32_fp32_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه);
                }
        #تعریف_کن X_SG_ID_SUB_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, درست, خطأ}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32", matmul_id_subgroup_##tstr##_f32_fp32_len, matmul_id_subgroup_##tstr##_f32_fp32_data, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, mul_mat_subgroup_اندازه);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB_FP32)
#لغو_تعریف X_SG_ID_SUB_FP32
            } وإلا {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, درست, خطأ}, tc_mm, "matmul_id_f32_f32", matmul_id_f32_f32_fp32_len, matmul_id_f32_f32_fp32_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, درست, خطأ}, tc_mm, "matmul_id_f16",     matmul_id_f16_fp32_len,     matmul_id_f16_fp32_داده,     حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, درست, خطأ}, tc_mm, "matmul_id_f16_f32", matmul_id_f16_f32_fp32_len, matmul_id_f16_f32_fp32_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, درست, خطأ}, tc_mm, "matmul_id_bf16",  matmul_id_bf16_fp32_len,    matmul_id_bf16_fp32_داده,    حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                لكل (ثابت خودکار type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, درست, خطأ}, tc_mmqid, "matmul_id_quant_f32", matmul_id_quant_f32_fp32_len, matmul_id_quant_f32_fp32_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                }
        #تعریف_کن X_SG_ID_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, درست, خطأ}, tc_mmqid, "matmul_id_" #tstr "_f32", matmul_id_##tstr##_f32_fp32_len, matmul_id_##tstr##_f32_fp32_data, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار);
                FOR_EACH_LUT_TYPE(X_SG_ID_FP32)
#لغو_تعریف X_SG_ID_FP32
            }
        }
    }
#لغو_تعریف FOR_EACH_LUT_TYPE
#لغو_تعریف FOR_EACH_LUT_TYPE_NONFP4
#لغو_تعریف FOR_EACH_LUT_FP4_TYPE
    // BF16 fallback for coopmat devices without bf16 coopmat support
    إذا ((device->coopmat2 || device->coopmat_support)
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        && !device->coopmat_bf16_support
#پایان_اگر
        ) {
        ثابت uint32_t s_warptile_wm_bf16 = device->subgroup_اندازه == 8 ? 8 : 32;
        std::vector<vk_tile_config> tc_bf16_fb = {
            {{ subgroup_اندازه_32, 32, 32, 16, s_warptile_wm_bf16, 32, 2, 2, 2, 1, subgroup_اندازه_8 }, {32, 32, 1}, s_align},
            {{ 128, 64, 64, 16, mm_warp_8, 32, 2, 4, 2, 1, mm_warp_8 }, {64, 64, 1}, m_align},
            {{ 128, 128, 128, 16, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 }, {128, 128, 1}, l_align},
        };
        خودکار tc_bf16_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, خطأ);
        خودکار tc_bf16_id_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, درست);
        spec_fn_t bf16_spec = [&](ثابت std::vector<uint32_t>& wt, منطقي a) { أرجع ggml_vk_mul_mm_spec(wt, a); };
        إذا (!tc_bf16_filtered.خالی()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, خطأ, خطأ}, tc_bf16_filtered, "matmul_bf16", matmul_bf16_fp32_len, matmul_bf16_fp32_داده, حجم(vk_mat_mat_افزودن_constants), 3, bf16_spec);
        }
        إذا (!tc_bf16_id_filtered.خالی()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, درست, خطأ}, tc_bf16_id_filtered, "matmul_id_bf16", matmul_id_bf16_fp32_len, matmul_id_bf16_fp32_داده, حجم(vk_mat_mat_id_افزودن_constants), mul_mat_id_param_شمار, bf16_spec);
        }
    }

    // Set up tile selector functions
    إذا (device->coopmat2) {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t shader_core_شمار,
                                          ثابت std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            إذا (configs.size() <= 1) أرجع 0;
            uint32_t آخرین = (uint32_t)configs.size() - 1;
            إذا (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                أرجع (n > crossover) ? 1 : 0;
            }
            // 3+ configs: s=0, m=1, l=2
            ثابت uint32_t tiles_l = CEIL_DIV(m, configs[آخرین].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[آخرین].unaligned->wg_denoms[1]);
            ثابت uint32_t tiles_m = CEIL_DIV(m, configs[1].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[1].unaligned->wg_denoms[1]);
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            منطقي prefer_large = tiles_m > shader_core_شمار || tiles_l > shader_core_شمار ||
                                (tiles_l <= shader_core_شمار / 3 && tiles_m > shader_core_شمار / 2);
            إذا (n > crossover_large && prefer_large) أرجع آخرین;
            uint32_t crossover_medium_m = configs[0].unaligned->wg_denoms[0];
            uint32_t crossover_medium_n = configs[0].unaligned->wg_denoms[1];
            إذا (m > crossover_medium_m && n > crossover_medium_n) أرجع 1;
            أرجع 0;
        };
        device->matmul_id_tile_selector = [](uint32_t /*m*/, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                             ثابت std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            إذا (configs.size() <= 1) أرجع 0;
            uint32_t آخرین = (uint32_t)configs.size() - 1;
            إذا (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                أرجع (n > crossover) ? 1 : 0;
            }
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            إذا (n > crossover_large) أرجع آخرین;
            uint32_t crossover_medium = configs[0].unaligned->wg_denoms[1];
            إذا (n > crossover_medium) أرجع 1;
            أرجع 0;
        };
    } وإلا {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                          ثابت std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            إذا (configs.size() <= 1) أرجع 0;
            إذا (m <= 32 || n <= 32) أرجع 0;
            إذا (configs.size() == 2) أرجع 1;
            إذا (m <= 64 || n <= 64) أرجع 1;
            أرجع (uint32_t)configs.size() - 1;
        };
        device->matmul_id_tile_selector = device->matmul_tile_selector;
    }

    // mul mat vec

    // the number of rows computed per shader depends on GPU model and quant
    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;
    uint32_t rm_stdq_int = 1;
    uint32_t rm_kq_int = 1;
    خودکار ثابت &rm_iq_int = [](uint32_t i) { أرجع i == 0 ? 8u : 4u; };
    إذا (device->vendor_id == VK_VENDOR_ID_AMD) {
        إذا (device->architecture == AMD_GCN) {
            rm_stdq = 2;
            rm_kq = 4;
            rm_stdq_int = 4;
        }
    } وإلا إذا (device->vendor_id == VK_VENDOR_ID_INTEL) {
        rm_stdq = 2;
        rm_stdq_int = 2;
    }
    // RDNA3/4: above four columns, static 4 rows for all types bench faster than the default
    ثابت منطقي is_rdna3_or_4 = device->vendor_id == VK_VENDOR_ID_AMD && (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);
    خودکار ثابت &rm_int_n = [&](uint32_t ردیفها, uint32_t i) { أرجع (is_rdna3_or_4 && i >= 4) ? 4u : ردیفها; };
    // RDNA3/4 and NVIDIA except pre-Turing: use 4 rows for MUL_MAT_ID MMVQ.
    خودکار ثابت &rm_id = [&](uint32_t ردیفها) {
        إذا (device->vendor_id == VK_VENDOR_ID_NVIDIA &&
            device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) {
            أرجع 4u;
        }
        أرجع is_rdna3_or_4 ? 4u : ردیفها;
    };
    uint32_t rm_iq = 2 * rm_kq;

    ثابت منطقي use_subgroups = device->subgroup_arithmetic;
    // The Imagination proprietary compiler rejects the subgroup-only dequant mul_mat_vec
    // shaders that require a subgroup size >= 16; fall back to shared-memory reduction.
    ثابت منطقي is_imagination_proprietary =
        device->driver_id == vk::DriverId::eImaginationProprietary;
    // Ensure a subgroup size >= 16 is available
    ثابت منطقي use_subgroups16 = use_subgroups && subgroup_min_اندازه_16 && !is_imagination_proprietary;

    ثابت uint32_t subgroup_اندازه = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_اندازه_control && device->subgroup_min_اندازه <= 16 && device->subgroup_max_اندازه >= 16) ? 16 : device->subgroup_اندازه;
    ثابت uint32_t subgroup_size16 = std::max(subgroup_اندازه, 16u);

    ثابت uint32_t force_subgroup_اندازه = use_subgroups ? subgroup_اندازه : 0;
    ثابت uint32_t force_subgroup_size16 = use_subgroups16 ? subgroup_size16 : 0;
    ساكن constexpr uint32_t mul_mat_vec_رقم_bindings = 5;
    ساكن constexpr uint32_t mul_mat_vec_id_رقم_bindings = 6;

#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
#تعریف_کن OCP_DMMV_LEN(NAME, REDUC)  (device->ocp_fp4 ? NAME ## _ocp_len[REDUC]  : NAME ## _len[REDUC])
#تعریف_کن OCP_DMMV_داده(NAME, REDUC) (device->ocp_fp4 ? NAME ## ocp_داده[REDUC] : NAME ## داده_2[REDUC])
#وإلا
#تعریف_کن OCP_DMMV_LEN(NAME, REDUC)  NAME ## _len[REDUC]
#تعریف_کن OCP_DMMV_داده(NAME, REDUC) NAME ## داده_2[REDUC]
#پایان_اگر

    لكل (uint32_t w = 0; w < DMMV_WG_اندازه_شمار; ++w) {
        ثابت uint32_t wg_اندازه_subgroup   = (w == DMMV_WG_اندازه_SUBGROUP) ? subgroup_اندازه : (subgroup_اندازه * 4);
        ثابت uint32_t wg_اندازه_subgroup16 = (w == DMMV_WG_اندازه_SUBGROUP) ? subgroup_size16 : (subgroup_size16 * 4);

        ثابت shader_reduction_نما reduc = (use_subgroups && w == DMMV_WG_اندازه_SUBGROUP) ? SHADER_REDUCTION_نما_SUBGROUP :
                                            (use_subgroups && w == DMMV_WG_اندازه_LARGE) ? SHADER_REDUCTION_نما_HYBRID :
                                            SHADER_REDUCTION_نما_SHMEM;

        ثابت shader_reduction_نما reduc16 = (use_subgroups16 && w == DMMV_WG_اندازه_SUBGROUP) ? SHADER_REDUCTION_نما_SUBGROUP :
                                              (use_subgroups16 && w == DMMV_WG_اندازه_LARGE) ? SHADER_REDUCTION_نما_HYBRID :
                                              SHADER_REDUCTION_نما_SHMEM;

        لكل (uint32_t i = 0; i < mul_mat_vec_max_تعداد_ستون; ++i) {
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f32_f32",  arr_dmmv_f32_f32_f32_len[reduc],  arr_dmmv_f32_f32_f32_داده[reduc],  "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {1, 1, 1}, {wg_اندازه_subgroup, 1, i+1}, 1, خطأ, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32",  arr_dmmv_f16_f32_f32_len[reduc],  arr_dmmv_f16_f32_f32_داده[reduc],  "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2, 1, 1}, {wg_اندازه_subgroup, 2, i+1}, 1, خطأ, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f32_f32", arr_dmmv_bf16_f32_f32_len[reduc], arr_dmmv_bf16_f32_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2, 1, 1}, {wg_اندازه_subgroup, 2, i+1}, 1, خطأ, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f32_f32", arr_dmmv_q1_0_f32_f32_len[reduc], arr_dmmv_q1_0_f32_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f32_f32", arr_dmmv_q2_0_f32_f32_len[reduc], arr_dmmv_q2_0_f32_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f32_f32", arr_dmmv_q4_0_f32_f32_len[reduc], arr_dmmv_q4_0_f32_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f32_f32", arr_dmmv_q4_1_f32_f32_len[reduc], arr_dmmv_q4_1_f32_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f32_f32", arr_dmmv_q5_0_f32_f32_len[reduc], arr_dmmv_q5_0_f32_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f32_f32", arr_dmmv_q5_1_f32_f32_len[reduc], arr_dmmv_q5_1_f32_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f32_f32", arr_dmmv_q8_0_f32_f32_len[reduc], arr_dmmv_q8_0_f32_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {1*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 1*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f32_f32", arr_dmmv_q2_k_f32_f32_len[reduc16], arr_dmmv_q2_k_f32_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f32_f32", arr_dmmv_q3_k_f32_f32_len[reduc16], arr_dmmv_q3_k_f32_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f32_f32", arr_dmmv_q5_k_f32_f32_len[reduc16], arr_dmmv_q5_k_f32_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f32_f32", arr_dmmv_tq1_0_f32_f32_len[reduc16], arr_dmmv_tq1_0_f32_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f32_f32", arr_dmmv_tq2_0_f32_f32_len[reduc16], arr_dmmv_tq2_0_f32_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f32_f32",   arr_dmmv_iq1_s_f32_f32_len[reduc16],   arr_dmmv_iq1_s_f32_f32_داده[reduc16],   "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f32_f32",   arr_dmmv_iq1_m_f32_f32_len[reduc16],   arr_dmmv_iq1_m_f32_f32_داده[reduc16],   "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f32_f32", arr_dmmv_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_iq2_xxs_f32_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f32_f32",  arr_dmmv_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_iq2_xs_f32_f32_داده[reduc16],  "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f32_f32",   arr_dmmv_iq2_s_f32_f32_len[reduc16],   arr_dmmv_iq2_s_f32_f32_داده[reduc16],   "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f32_f32", arr_dmmv_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_iq3_xxs_f32_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f32_f32",   arr_dmmv_iq3_s_f32_f32_len[reduc16],   arr_dmmv_iq3_s_f32_f32_داده[reduc16],   "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f32_f32",  arr_dmmv_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_iq4_xs_f32_f32_داده[reduc16],  "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f32_f32",  arr_dmmv_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_iq4_nl_f32_f32_داده[reduc16],  "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f32_f32, reduc16), OCP_DMMV_داده(arr_dmmv_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f32_f32, reduc16), OCP_DMMV_داده(arr_dmmv_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f16_f32",  arr_dmmv_f32_f16_f32_len[reduc],  arr_dmmv_f32_f16_f32_داده[reduc],  "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {1, 1, 1}, {wg_اندازه_subgroup, 1, i+1}, 1, خطأ, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f16_f32",  arr_dmmv_f16_f16_f32_len[reduc],  arr_dmmv_f16_f16_f32_داده[reduc],  "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2, 1, 1}, {wg_اندازه_subgroup, 2, i+1}, 1, خطأ, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f16_f32", arr_dmmv_bf16_f16_f32_len[reduc], arr_dmmv_bf16_f16_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2, 1, 1}, {wg_اندازه_subgroup, 2, i+1}, 1, خطأ, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f16_f32", arr_dmmv_q1_0_f16_f32_len[reduc], arr_dmmv_q1_0_f16_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f16_f32", arr_dmmv_q2_0_f16_f32_len[reduc], arr_dmmv_q2_0_f16_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f16_f32", arr_dmmv_q4_0_f16_f32_len[reduc], arr_dmmv_q4_0_f16_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f16_f32", arr_dmmv_q4_1_f16_f32_len[reduc], arr_dmmv_q4_1_f16_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f16_f32", arr_dmmv_q5_0_f16_f32_len[reduc], arr_dmmv_q5_0_f16_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f16_f32", arr_dmmv_q5_1_f16_f32_len[reduc], arr_dmmv_q5_1_f16_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f16_f32", arr_dmmv_q8_0_f16_f32_len[reduc], arr_dmmv_q8_0_f16_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {1*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 1*rm_stdq, i+1}, 1, درست, use_subgroups, force_subgroup_اندازه);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f16_f32", arr_dmmv_q2_k_f16_f32_len[reduc16], arr_dmmv_q2_k_f16_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f16_f32", arr_dmmv_q3_k_f16_f32_len[reduc16], arr_dmmv_q3_k_f16_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f16_f32", arr_dmmv_q5_k_f16_f32_len[reduc16], arr_dmmv_q5_k_f16_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f16_f32", arr_dmmv_tq1_0_f16_f32_len[reduc16], arr_dmmv_tq1_0_f16_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f16_f32", arr_dmmv_tq2_0_f16_f32_len[reduc16], arr_dmmv_tq2_0_f16_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f16_f32",   arr_dmmv_iq1_s_f16_f32_len[reduc16],   arr_dmmv_iq1_s_f16_f32_داده[reduc16],   "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f16_f32",   arr_dmmv_iq1_m_f16_f32_len[reduc16],   arr_dmmv_iq1_m_f16_f32_داده[reduc16],   "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f16_f32", arr_dmmv_iq2_xxs_f16_f32_len[reduc16], arr_dmmv_iq2_xxs_f16_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f16_f32",  arr_dmmv_iq2_xs_f16_f32_len[reduc16],  arr_dmmv_iq2_xs_f16_f32_داده[reduc16],  "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f16_f32",   arr_dmmv_iq2_s_f16_f32_len[reduc16],   arr_dmmv_iq2_s_f16_f32_داده[reduc16],   "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f16_f32", arr_dmmv_iq3_xxs_f16_f32_len[reduc16], arr_dmmv_iq3_xxs_f16_f32_داده[reduc16], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f16_f32",   arr_dmmv_iq3_s_f16_f32_len[reduc16],   arr_dmmv_iq3_s_f16_f32_داده[reduc16],   "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f16_f32",  arr_dmmv_iq4_xs_f16_f32_len[reduc16],  arr_dmmv_iq4_xs_f16_f32_داده[reduc16],  "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f16_f32",  arr_dmmv_iq4_nl_f16_f32_len[reduc16],  arr_dmmv_iq4_nl_f16_f32_داده[reduc16],  "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f16_f32, reduc16), OCP_DMMV_داده(arr_dmmv_mxfp4_f16_f32, reduc16), "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f16_f32, reduc16), OCP_DMMV_داده(arr_dmmv_nvfp4_f16_f32, reduc16), "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq, i+1}, 1, درست, use_subgroups16, force_subgroup_size16);

#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            إذا (device->integer_dot_حاصلضرب) {
                ثابت uint32_t subgroup_اندازه_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_اندازه_control) ? device->subgroup_min_اندازه : device->subgroup_اندازه;
                ثابت uint32_t wg_اندازه_subgroup_int = (w == DMMV_WG_اندازه_SUBGROUP) ? subgroup_اندازه_int : (subgroup_اندازه_int * 4);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_q8_1_f32", arr_dmmv_q2_0_q8_1_f32_len[reduc], arr_dmmv_q2_0_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_اندازه_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_q8_1_f32", arr_dmmv_q4_0_q8_1_f32_len[reduc], arr_dmmv_q4_0_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_اندازه_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_q8_1_f32", arr_dmmv_q4_1_q8_1_f32_len[reduc], arr_dmmv_q4_1_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_اندازه_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_q8_1_f32", arr_dmmv_q5_0_q8_1_f32_len[reduc], arr_dmmv_q5_0_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_اندازه_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_q8_1_f32", arr_dmmv_q5_1_q8_1_f32_len[reduc], arr_dmmv_q5_1_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_اندازه_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_q8_1_f32", arr_dmmv_q8_0_q8_1_f32_len[reduc], arr_dmmv_q8_0_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_اندازه_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_MXFP4][i], "mul_mat_vec_mxfp4_q8_1_f32", arr_dmmv_mxfp4_q8_1_f32_len[reduc], arr_dmmv_mxfp4_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_int_n(2*rm_stdq_int, i), 1, 1}, {wg_اندازه_subgroup_int, rm_int_n(2*rm_stdq_int, i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_q8_1_f32", arr_dmmv_q2_k_q8_1_f32_len[reduc], arr_dmmv_q2_k_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_اندازه_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_q8_1_f32", arr_dmmv_q3_k_q8_1_f32_len[reduc], arr_dmmv_q3_k_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_اندازه_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_q8_1_f32", arr_dmmv_q4_k_q8_1_f32_len[reduc], arr_dmmv_q4_k_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_اندازه_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_q8_1_f32", arr_dmmv_q5_k_q8_1_f32_len[reduc], arr_dmmv_q5_k_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_اندازه_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_q8_1_f32", arr_dmmv_q6_k_q8_1_f32_len[reduc], arr_dmmv_q6_k_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_اندازه_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_S][i], "mul_mat_vec_iq1_s_q8_1_f32", arr_dmmv_iq1_s_q8_1_f32_len[reduc], arr_dmmv_iq1_s_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {1*rm_iq_int(i), 1, 1}, {wg_اندازه_subgroup_int, 1*rm_iq_int(i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_M][i], "mul_mat_vec_iq1_m_q8_1_f32", arr_dmmv_iq1_m_q8_1_f32_len[reduc], arr_dmmv_iq1_m_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {1*rm_iq_int(i), 1, 1}, {wg_اندازه_subgroup_int, 1*rm_iq_int(i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ4_XS][i], "mul_mat_vec_iq4_xs_q8_1_f32", arr_dmmv_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_iq4_xs_q8_1_f32_داده[reduc], "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_افزودن_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_اندازه_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, درست, use_subgroups, subgroup_اندازه_int);

            }
#پایان_اگر // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
        }

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F32 ], "mul_mat_vec_id_f32_f32",        arr_dmmv_id_f32_f32_f32_len[reduc],     arr_dmmv_id_f32_f32_f32_داده[reduc],     "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {1, 1, 1}, {wg_اندازه_subgroup, 1}, 1, خطأ, use_subgroups, force_subgroup_اندازه);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F16 ], "mul_mat_vec_id_f16_f32",        arr_dmmv_id_f16_f32_f32_len[reduc],     arr_dmmv_id_f16_f32_f32_داده[reduc],     "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {2, 1, 1}, {wg_اندازه_subgroup, 2}, 1, خطأ, use_subgroups, force_subgroup_اندازه);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_BF16], "mul_mat_vec_id_bf16_f32",       arr_dmmv_id_bf16_f32_f32_len[reduc],    arr_dmmv_id_bf16_f32_f32_داده[reduc],    "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {2, 1, 1}, {wg_اندازه_subgroup, 2}, 1, خطأ, use_subgroups, force_subgroup_اندازه);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q1_0], "mul_mat_vec_id_q1_0_f32",       arr_dmmv_id_q1_0_f32_f32_len[reduc],    arr_dmmv_id_q1_0_f32_f32_داده[reduc],    "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq}, 1, درست, use_subgroups, force_subgroup_اندازه);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_f32",       arr_dmmv_id_q2_0_f32_f32_len[reduc],    arr_dmmv_id_q2_0_f32_f32_داده[reduc],    "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq}, 1, درست, use_subgroups, force_subgroup_اندازه);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_f32",       arr_dmmv_id_q4_0_f32_f32_len[reduc],    arr_dmmv_id_q4_0_f32_f32_داده[reduc],    "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq}, 1, درست, use_subgroups, force_subgroup_اندازه);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_f32",       arr_dmmv_id_q4_1_f32_f32_len[reduc],    arr_dmmv_id_q4_1_f32_f32_داده[reduc],    "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq}, 1, درست, use_subgroups, force_subgroup_اندازه);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_f32",       arr_dmmv_id_q5_0_f32_f32_len[reduc],    arr_dmmv_id_q5_0_f32_f32_داده[reduc],    "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq}, 1, درست, use_subgroups, force_subgroup_اندازه);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_f32",       arr_dmmv_id_q5_1_f32_f32_len[reduc],    arr_dmmv_id_q5_1_f32_f32_داده[reduc],    "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {2*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 2*rm_stdq}, 1, درست, use_subgroups, force_subgroup_اندازه);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_f32",       arr_dmmv_id_q8_0_f32_f32_len[reduc],    arr_dmmv_id_q8_0_f32_f32_داده[reduc],    "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {1*rm_stdq, 1, 1}, {wg_اندازه_subgroup, 1*rm_stdq}, 1, درست, use_subgroups, force_subgroup_اندازه);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_f32",       arr_dmmv_id_q2_k_f32_f32_len[reduc16],    arr_dmmv_id_q2_k_f32_f32_داده[reduc16],    "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_f32",       arr_dmmv_id_q3_k_f32_f32_len[reduc16],    arr_dmmv_id_q3_k_f32_f32_داده[reduc16],    "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_f32",       arr_dmmv_id_q4_k_f32_f32_len[reduc16],    arr_dmmv_id_q4_k_f32_f32_داده[reduc16],    "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_f32",       arr_dmmv_id_q5_k_f32_f32_len[reduc16],    arr_dmmv_id_q5_k_f32_f32_داده[reduc16],    "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_f32",       arr_dmmv_id_q6_k_f32_f32_len[reduc16],    arr_dmmv_id_q6_k_f32_f32_داده[reduc16],    "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ1_0], "mul_mat_vec_id_tq1_0_f32",     arr_dmmv_id_tq1_0_f32_f32_len[reduc16],   arr_dmmv_id_tq1_0_f32_f32_داده[reduc16],   "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ2_0], "mul_mat_vec_id_tq2_0_f32",     arr_dmmv_id_tq2_0_f32_f32_len[reduc16],   arr_dmmv_id_tq2_0_f32_f32_داده[reduc16],   "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_kq, 1, 1}, {wg_اندازه_subgroup16, rm_kq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_S],   "mul_mat_vec_id_iq1_s_f32",   arr_dmmv_id_iq1_s_f32_f32_len[reduc16],   arr_dmmv_id_iq1_s_f32_f32_داده[reduc16],   "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_M],   "mul_mat_vec_id_iq1_m_f32",   arr_dmmv_id_iq1_m_f32_f32_len[reduc16],   arr_dmmv_id_iq1_m_f32_f32_داده[reduc16],   "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XXS], "mul_mat_vec_id_iq2_xxs_f32", arr_dmmv_id_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq2_xxs_f32_f32_داده[reduc16], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XS],  "mul_mat_vec_id_iq2_xs_f32",  arr_dmmv_id_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq2_xs_f32_f32_داده[reduc16],  "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_S],   "mul_mat_vec_id_iq2_s_f32",   arr_dmmv_id_iq2_s_f32_f32_len[reduc16],   arr_dmmv_id_iq2_s_f32_f32_داده[reduc16],   "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_XXS], "mul_mat_vec_id_iq3_xxs_f32", arr_dmmv_id_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq3_xxs_f32_f32_داده[reduc16], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_S],   "mul_mat_vec_id_iq3_s_f32",   arr_dmmv_id_iq3_s_f32_f32_len[reduc16],   arr_dmmv_id_iq3_s_f32_f32_داده[reduc16],   "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_XS],  "mul_mat_vec_id_iq4_xs_f32",  arr_dmmv_id_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq4_xs_f32_f32_داده[reduc16],  "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_NL],  "mul_mat_vec_id_iq4_nl_f32",  arr_dmmv_id_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_id_iq4_nl_f32_f32_داده[reduc16],  "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_MXFP4],   "mul_mat_vec_id_mxfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_mxfp4_f32_f32, reduc16), OCP_DMMV_داده(arr_dmmv_id_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq}, 1, درست, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_NVFP4],   "mul_mat_vec_id_nvfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_nvfp4_f32_f32, reduc16), OCP_DMMV_داده(arr_dmmv_id_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_iq, 1, 1}, {wg_اندازه_subgroup16, rm_iq}, 1, درست, use_subgroups16, force_subgroup_size16);

#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        إذا (device->integer_dot_حاصلضرب) {
            ثابت uint32_t subgroup_اندازه_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_اندازه_control) ? device->subgroup_min_اندازه : device->subgroup_اندازه;
            ثابت uint32_t wg_اندازه_subgroup_int = (w == DMMV_WG_اندازه_SUBGROUP) ? subgroup_اندازه_int : (subgroup_اندازه_int * 4);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_q8_1_f32", arr_dmmv_id_q2_0_q8_1_f32_len[reduc], arr_dmmv_id_q2_0_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_اندازه_subgroup_int, rm_id(2*rm_kq_int)}, 1, درست, use_subgroups, subgroup_اندازه_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_q8_1_f32", arr_dmmv_id_q4_0_q8_1_f32_len[reduc], arr_dmmv_id_q4_0_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_اندازه_subgroup_int, rm_id(1*rm_stdq_int)}, 1, درست, use_subgroups, subgroup_اندازه_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_q8_1_f32", arr_dmmv_id_q4_1_q8_1_f32_len[reduc], arr_dmmv_id_q4_1_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_اندازه_subgroup_int, rm_id(1*rm_stdq_int)}, 1, درست, use_subgroups, subgroup_اندازه_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_q8_1_f32", arr_dmmv_id_q5_0_q8_1_f32_len[reduc], arr_dmmv_id_q5_0_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_اندازه_subgroup_int, rm_id(1*rm_stdq_int)}, 1, درست, use_subgroups, subgroup_اندازه_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_q8_1_f32", arr_dmmv_id_q5_1_q8_1_f32_len[reduc], arr_dmmv_id_q5_1_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_اندازه_subgroup_int, rm_id(1*rm_stdq_int)}, 1, درست, use_subgroups, subgroup_اندازه_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_q8_1_f32", arr_dmmv_id_q8_0_q8_1_f32_len[reduc], arr_dmmv_id_q8_0_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_اندازه_subgroup_int, rm_id(1*rm_stdq_int)}, 1, درست, use_subgroups, subgroup_اندازه_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_MXFP4], "mul_mat_vec_id_mxfp4_q8_1_f32", arr_dmmv_id_mxfp4_q8_1_f32_len[reduc], arr_dmmv_id_mxfp4_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_id(2*rm_stdq_int), 1, 1}, {wg_اندازه_subgroup_int, rm_id(2*rm_stdq_int)}, 1, درست, use_subgroups, subgroup_اندازه_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_q8_1_f32", arr_dmmv_id_q2_k_q8_1_f32_len[reduc], arr_dmmv_id_q2_k_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_اندازه_subgroup_int, rm_id(2*rm_kq_int)}, 1, درست, use_subgroups, subgroup_اندازه_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_q8_1_f32", arr_dmmv_id_q3_k_q8_1_f32_len[reduc], arr_dmmv_id_q3_k_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_اندازه_subgroup_int, rm_id(1*rm_kq_int)}, 1, درست, use_subgroups, subgroup_اندازه_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_q8_1_f32", arr_dmmv_id_q4_k_q8_1_f32_len[reduc], arr_dmmv_id_q4_k_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_اندازه_subgroup_int, rm_id(1*rm_kq_int)}, 1, درست, use_subgroups, subgroup_اندازه_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_q8_1_f32", arr_dmmv_id_q5_k_q8_1_f32_len[reduc], arr_dmmv_id_q5_k_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_اندازه_subgroup_int, rm_id(1*rm_kq_int)}, 1, درست, use_subgroups, subgroup_اندازه_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_q8_1_f32", arr_dmmv_id_q6_k_q8_1_f32_len[reduc], arr_dmmv_id_q6_k_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_اندازه_subgroup_int, rm_id(1*rm_kq_int)}, 1, درست, use_subgroups, subgroup_اندازه_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_S], "mul_mat_vec_id_iq1_s_q8_1_f32", arr_dmmv_id_iq1_s_q8_1_f32_len[reduc], arr_dmmv_id_iq1_s_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {1*rm_iq_int(0), 1, 1}, {wg_اندازه_subgroup_int, 1*rm_iq_int(0)}, 1, درست, use_subgroups, subgroup_اندازه_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_M], "mul_mat_vec_id_iq1_m_q8_1_f32", arr_dmmv_id_iq1_m_q8_1_f32_len[reduc], arr_dmmv_id_iq1_m_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {1*rm_iq_int(0), 1, 1}, {wg_اندازه_subgroup_int, 1*rm_iq_int(0)}, 1, درست, use_subgroups, subgroup_اندازه_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ4_XS], "mul_mat_vec_id_iq4_xs_q8_1_f32", arr_dmmv_id_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_id_iq4_xs_q8_1_f32_داده[reduc], "main", mul_mat_vec_id_رقم_bindings, حجم(vk_mat_vec_id_افزودن_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_اندازه_subgroup_int, rm_id(1*rm_stdq_int)}, 1, درست, use_subgroups, subgroup_اندازه_int);
        }
#پایان_اگر // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
    }

#لغو_تعریف OCP_DMMV_داده
#لغو_تعریف OCP_DMMV_LEN

#إذا !defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    GGML_استفادهنشده(rm_stdq_int);
    GGML_استفادهنشده(rm_kq_int);
    GGML_استفادهنشده(is_rdna3_or_4);
    GGML_استفادهنشده(rm_int_n);
    GGML_استفادهنشده(rm_id);
    GGML_استفادهنشده(rm_iq_int);
#پایان_اگر

    // dequant shaders
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F32 ], "f32_to_f16",   dequant_f32_len,  dequant_f32_داده,  "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q1_0], "dequant_q1_0", dequant_q1_0_len, dequant_q1_0_داده, "main", 2, 5 * حجم(uint32_t), {256 * 8, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_0], "dequant_q2_0", dequant_q2_0_len, dequant_q2_0_داده, "main", 2, 5 * حجم(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_0], "dequant_q4_0", dequant_q4_0_len, dequant_q4_0_داده, "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_1], "dequant_q4_1", dequant_q4_1_len, dequant_q4_1_داده, "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_داده, "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_داده, "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_داده, "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_transpose[GGML_TYPE_Q8_0], "dequant_q8_0_transpose", dequant_q8_0_transpose_len, dequant_q8_0_transpose_داده, "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_داده, "main", 2, 5 * حجم(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_داده, "main", 2, 5 * حجم(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_داده, "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_داده, "main", 2, 5 * حجم(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_داده, "main", 2, 5 * حجم(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ1_0], "dequant_tq1_0", dequant_tq1_0_len, dequant_tq1_0_داده, "main", 2, 5 * حجم(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ2_0], "dequant_tq2_0", dequant_tq2_0_len, dequant_tq2_0_داده, "main", 2, 5 * حجم(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_S],   "dequant_iq1_s",   dequant_iq1_s_len,   dequant_iq1_s_داده,   "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_M],   "dequant_iq1_m",   dequant_iq1_m_len,   dequant_iq1_m_داده,   "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XXS], "dequant_iq2_xxs", dequant_iq2_xxs_len, dequant_iq2_xxs_داده, "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XS],  "dequant_iq2_xs",  dequant_iq2_xs_len,  dequant_iq2_xs_داده,  "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_S],   "dequant_iq2_s",   dequant_iq2_s_len,   dequant_iq2_s_داده,   "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_XXS], "dequant_iq3_xxs", dequant_iq3_xxs_len, dequant_iq3_xxs_داده, "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_S],   "dequant_iq3_s",   dequant_iq3_s_len,   dequant_iq3_s_داده,   "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_XS],  "dequant_iq4_xs",  dequant_iq4_xs_len,  dequant_iq4_xs_داده,  "main", 2, 5 * حجم(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_NL],  "dequant_iq4_nl",  dequant_iq4_nl_len,  dequant_iq4_nl_داده,  "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_MXFP4],   "dequant_mxfp4",   dequant_mxfp4_len,   dequant_mxfp4_داده,   "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_NVFP4],   "dequant_nvfp4",   dequant_nvfp4_len,   dequant_nvfp4_داده,   "main", 2, 5 * حجم(uint32_t), {256 * 16, 1, 1}, {}, 1);

    // get_rows
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_F32 ], "get_rows_f32",  get_ردیفها_f32_len,  get_ردیفها_f32_داده,  "main", 3, حجم(vk_op_دودویی_افزودن_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_F16 ], "get_rows_f16",  get_ردیفها_f16_len,  get_ردیفها_f16_داده,  "main", 3, حجم(vk_op_دودویی_افزودن_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_BF16], "get_rows_bf16", get_ردیفها_bf16_len, get_ردیفها_bf16_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_Q1_0], "get_rows_q1_0", get_ردیفها_q1_0_len, get_ردیفها_q1_0_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_Q2_0], "get_rows_q2_0", get_ردیفها_q2_0_len, get_ردیفها_q2_0_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_Q4_0], "get_rows_q4_0", get_ردیفها_q4_0_len, get_ردیفها_q4_0_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_Q4_1], "get_rows_q4_1", get_ردیفها_q4_1_len, get_ردیفها_q4_1_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_Q5_0], "get_rows_q5_0", get_ردیفها_q5_0_len, get_ردیفها_q5_0_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_Q5_1], "get_rows_q5_1", get_ردیفها_q5_1_len, get_ردیفها_q5_1_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_Q8_0], "get_rows_q8_0", get_ردیفها_q8_0_len, get_ردیفها_q8_0_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_Q2_K], "get_rows_q2_k", get_ردیفها_q2_k_len, get_ردیفها_q2_k_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_Q3_K], "get_rows_q3_k", get_ردیفها_q3_k_len, get_ردیفها_q3_k_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_Q4_K], "get_rows_q4_k", get_ردیفها_q4_k_len, get_ردیفها_q4_k_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_Q5_K], "get_rows_q5_k", get_ردیفها_q5_k_len, get_ردیفها_q5_k_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_Q6_K], "get_rows_q6_k", get_ردیفها_q6_k_len, get_ردیفها_q6_k_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_TQ1_0], "get_rows_tq1_0", get_ردیفها_tq1_0_len, get_ردیفها_tq1_0_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_TQ2_0], "get_rows_tq2_0", get_ردیفها_tq2_0_len, get_ردیفها_tq2_0_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_IQ1_S],   "get_rows_iq1_s",   get_ردیفها_iq1_s_len,   get_ردیفها_iq1_s_داده,   "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_IQ1_M],   "get_rows_iq1_m",   get_ردیفها_iq1_m_len,   get_ردیفها_iq1_m_داده,   "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs", get_ردیفها_iq2_xxs_len, get_ردیفها_iq2_xxs_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs",  get_ردیفها_iq2_xs_len,  get_ردیفها_iq2_xs_داده,  "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_IQ2_S],   "get_rows_iq2_s",   get_ردیفها_iq2_s_len,   get_ردیفها_iq2_s_داده,   "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs", get_ردیفها_iq3_xxs_len, get_ردیفها_iq3_xxs_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_IQ3_S],   "get_rows_iq3_s",   get_ردیفها_iq3_s_len,   get_ردیفها_iq3_s_داده,   "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs",  get_ردیفها_iq4_xs_len,  get_ردیفها_iq4_xs_داده,  "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl",  get_ردیفها_iq4_nl_len,  get_ردیفها_iq4_nl_داده,  "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_MXFP4],   "get_rows_mxfp4",   get_ردیفها_mxfp4_len,   get_ردیفها_mxfp4_داده,   "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_NVFP4],   "get_rows_nvfp4",   get_ردیفها_nvfp4_len,   get_ردیفها_nvfp4_داده,   "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها[GGML_TYPE_I32],     "get_rows_i32",     get_ردیفها_i32_len,     get_ردیفها_i32_داده,     "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_F32 ], "get_rows_f32_f32",  get_ردیفها_f32_f32_len,  get_ردیفها_f32_f32_داده,  "main", 3, حجم(vk_op_دودویی_افزودن_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_F16 ], "get_rows_f16_f32",  get_ردیفها_f16_f32_len,  get_ردیفها_f16_f32_داده,  "main", 3, حجم(vk_op_دودویی_افزودن_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_BF16], "get_rows_bf16_f32", get_ردیفها_bf16_f32_len, get_ردیفها_bf16_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_Q1_0], "get_rows_q1_0_f32", get_ردیفها_q1_0_f32_len, get_ردیفها_q1_0_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_Q2_0], "get_rows_q2_0_f32", get_ردیفها_q2_0_f32_len, get_ردیفها_q2_0_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_Q4_0], "get_rows_q4_0_f32", get_ردیفها_q4_0_f32_len, get_ردیفها_q4_0_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_Q4_1], "get_rows_q4_1_f32", get_ردیفها_q4_1_f32_len, get_ردیفها_q4_1_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_Q5_0], "get_rows_q5_0_f32", get_ردیفها_q5_0_f32_len, get_ردیفها_q5_0_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_ردیفها_q5_1_f32_len, get_ردیفها_q5_1_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_ردیفها_q8_0_f32_len, get_ردیفها_q8_0_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_Q2_K], "get_rows_q2_k_f32", get_ردیفها_q2_k_f32_len, get_ردیفها_q2_k_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_Q3_K], "get_rows_q3_k_f32", get_ردیفها_q3_k_f32_len, get_ردیفها_q3_k_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_Q4_K], "get_rows_q4_k_f32", get_ردیفها_q4_k_f32_len, get_ردیفها_q4_k_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_Q5_K], "get_rows_q5_k_f32", get_ردیفها_q5_k_f32_len, get_ردیفها_q5_k_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_Q6_K], "get_rows_q6_k_f32", get_ردیفها_q6_k_f32_len, get_ردیفها_q6_k_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_TQ1_0], "get_rows_tq1_0_f32", get_ردیفها_tq1_0_f32_len, get_ردیفها_tq1_0_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_TQ2_0], "get_rows_tq2_0_f32", get_ردیفها_tq2_0_f32_len, get_ردیفها_tq2_0_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_IQ1_S],   "get_rows_iq1_s_f32",   get_ردیفها_iq1_s_f32_len,   get_ردیفها_iq1_s_f32_داده,   "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_IQ1_M],   "get_rows_iq1_m_f32",   get_ردیفها_iq1_m_f32_len,   get_ردیفها_iq1_m_f32_داده,   "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs_f32", get_ردیفها_iq2_xxs_f32_len, get_ردیفها_iq2_xxs_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs_f32",  get_ردیفها_iq2_xs_f32_len,  get_ردیفها_iq2_xs_f32_داده,  "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_IQ2_S],   "get_rows_iq2_s_f32",   get_ردیفها_iq2_s_f32_len,   get_ردیفها_iq2_s_f32_داده,   "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs_f32", get_ردیفها_iq3_xxs_f32_len, get_ردیفها_iq3_xxs_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_IQ3_S],   "get_rows_iq3_s_f32",   get_ردیفها_iq3_s_f32_len,   get_ردیفها_iq3_s_f32_داده,   "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs_f32",  get_ردیفها_iq4_xs_f32_len,  get_ردیفها_iq4_xs_f32_داده,  "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl_f32",  get_ردیفها_iq4_nl_f32_len,  get_ردیفها_iq4_nl_f32_داده,  "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_MXFP4],   "get_rows_mxfp4_f32",   get_ردیفها_mxfp4_f32_len,   get_ردیفها_mxfp4_f32_داده,   "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_f32[GGML_TYPE_NVFP4],   "get_rows_nvfp4_f32",   get_ردیفها_nvfp4_f32_len,   get_ردیفها_nvfp4_f32_داده,   "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ردیفها_پشت_f32, "get_rows_back_f32", get_ردیفها_پشت_f32_len, get_ردیفها_پشت_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {256, 1, 1}, {}, 1, درست);

    ggml_vk_create_pipeline(device, device->pipeline_matmul_تقسیم_k_reduce, "split_k_reduce", تقسیم_k_reduce_len, تقسیم_k_reduce_داده, "main", 2, 2 * حجم(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_تقسیم_k_reduce, "fa_split_k_reduce", fa_تقسیم_k_reduce_len, fa_تقسیم_k_reduce_داده, "main", 3, حجم(vk_op_flash_attn_تقسیم_k_reduce_افزودن_constants), {1, device->subgroup_اندازه, 1}, {device->subgroup_اندازه}, 1, درست);

#إذا defined(VK_KHR_cooperative_ماتریس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    إذا (device->vendor_id == VK_VENDOR_ID_INTEL && (device->architecture == INTEL_XE2 || (device->architecture == INTEL_XE1 && device->coopmat_support && device->uma))) {
        خودکار فوقانی_power_از_2 = [&](uint32_t درون) {
            GGML_ASSERT(درون != 0);
            إذا (درون <= 1) أرجع 1u;
            uint32_t ret = درون - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            أرجع ret + 1;
            };

        uint32_t xe_native_sub_گروه_اندازه = 16;
        إذا (device->architecture == INTEL_XE1) {
            xe_native_sub_گروه_اندازه = 8;
        }

        لكل (خودکار& it : device->pipeline_xe_fa_decode_dual_phases) {
            ثابت uint32_t تقسیم_p_chunk = 32;
            خودکار HdQk = it.اولی;
            خودکار& pipelines = it.second;
            uint32_t سر_dim_qk = std::get<0>(HdQk);
            uint32_t سر_dim_pv = std::get<1>(HdQk);
            uint32_t gqa_نسبت = std::get<2>(HdQk);
            uint32_t q_len = std::get<3>(HdQk);
            ثابت uint32_t بیرون_dim_در_هر_wg = gqa_نسبت > 16 ? 8 : 16;
            uint32_t aligned_q_len = فوقانی_power_از_2(q_len);
            uint32_t گروه_sz_ph1 = std::min(std::max(aligned_q_len * xe_native_sub_گروه_اندازه, 64u), 256u);
            uint32_t بیرون_در_هر_wg_ph1 = std::min(q_len, 256u / xe_native_sub_گروه_اندازه);
            uint32_t aligned_gqa_نسبت = فوقانی_power_از_2(gqa_نسبت);
            uint32_t تقسیم_p_در_هر_iter_ph2 = 256;
            uint32_t تقسیم_p_در_هر_warp = 16;
            uint32_t گروه_sz_ph2 = (تقسیم_p_در_هر_iter_ph2 / تقسیم_p_در_هر_warp) * xe_native_sub_گروه_اندازه;
            uint32_t بیرون_در_هر_wg_ph2 = std::min(std::max(16u / aligned_gqa_نسبت, 1u), q_len);
            ggml_vk_create_pipeline(device, pipelines.اولی, "xe_fa_decode_ph1", fa_decode_ph1_cm1_len, fa_decode_ph1_cm1_داده, "main", 5, حجم(vk_fa_xe_opt_افزودن_constants), { 1, 32, 1 }, { گروه_sz_ph1, gqa_نسبت, سر_dim_qk, xe_native_sub_گروه_اندازه, تقسیم_p_chunk, بیرون_در_هر_wg_ph1 }, 1, خطأ, درست, xe_native_sub_گروه_اندازه);
            ggml_vk_create_pipeline(device, pipelines.second, "xe_fa_decode_ph2", fa_decode_ph2_cm1_len, fa_decode_ph2_cm1_داده, "main", 5, حجم(vk_fa_xe_opt_افزودن_constants), { 1, 1, 1 }, { گروه_sz_ph2, gqa_نسبت, سر_dim_pv, بیرون_در_هر_wg_ph2, xe_native_sub_گروه_اندازه, تقسیم_p_در_هر_iter_ph2, تقسیم_p_chunk, بیرون_dim_در_هر_wg }, 1, خطأ, درست, xe_native_sub_گروه_اندازه);
        }
    }
#پایان_اگر

    لكل (خودکار &it : device->pipeline_fa_ماسک_opt) {
        خودکار BrBc = it.اولی;
        ggml_vk_create_pipeline(device, it.second, "fa_mask_opt", fa_ماسک_opt_len, fa_ماسک_opt_داده, "main", 2, حجم(vk_op_flash_attn_ماسک_opt_افزودن_constants), {1, 1, 1}, {128, 128 / device->subgroup_اندازه, BrBc.اولی, BrBc.second}, 1, درست, درست, device->subgroup_اندازه);
    }

    {
        // Large workgroup so the per-row KV scan parallelizes; capped to device limits.
        ثابت uint32_t compact_max = std::min({1024u, device->properties.limits.max_Compute_Work_گروه_Invocations, device->properties.limits.max_Compute_Work_گروه_اندازه[0]});

        // Fast ballot prefix-sum path when the device supports full subgroups; otherwise
        // a shared-memory prefix-sum fallback. Both emit a deterministic ascending list.
        device->fa_sparse_compact_use_subgroups = device->subgroup_ballot && device->subgroup_require_پر_support;
        إذا (device->fa_sparse_compact_use_subgroups) {
            ثابت uint32_t compact_wg = std::max(device->subgroup_اندازه, (compact_max / device->subgroup_اندازه) * device->subgroup_اندازه);
            ثابت uint32_t compact_رقم_sg = compact_wg / device->subgroup_اندازه;
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact_subgroup, "fa_sparse_compact_subgroup", fa_sparse_compact_subgroup_len, fa_sparse_compact_subgroup_داده, "main", 2, حجم(vk_op_flash_attn_sparse_compact_افزودن_constants), {1, 1, 1}, {compact_wg, compact_رقم_sg}, 1, درست, درست, device->subgroup_اندازه);
        } وإلا {
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact, "fa_sparse_compact", fa_sparse_compact_len, fa_sparse_compact_داده, "main", 2, حجم(vk_op_flash_attn_sparse_compact_افزودن_constants), {1, 1, 1}, {compact_max}, 1, درست);
        }
    }

    إذا (device->subgroup_clustered && device->subgroup_require_پر_support) {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_subgroup_len, quantize_q8_1_x4_subgroup_داده, "main", 2, حجم(vk_quantize_q8_1_افزودن_constants), {32 * device->subgroup_اندازه / 8, 1, 1}, { device->subgroup_اندازه }, 1, درست, درست);
    } وإلا {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_len, quantize_q8_1_x4_داده, "main", 2, حجم(vk_quantize_q8_1_افزودن_constants), {32 * device->subgroup_اندازه / 8, 1, 1}, { device->subgroup_اندازه }, 1);
    }

    لكل (uint32_t i = 0; i < p021_max_gqa_نسبت; ++i) {
        إذا (device->subgroup_arithmetic && device->subgroup_require_پر_support) {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_subgroup_جمع_کن_len, mul_mat_vec_p021_f16_f32_subgroup_جمع_کن_داده, "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_p021_افزودن_constants), {1, 1, 1}, {device->subgroup_اندازه, i + 1}, 1, درست, درست);
        } وإلا {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_len,              mul_mat_vec_p021_f16_f32_داده,              "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_p021_افزودن_constants), {1, 1, 1}, {device->subgroup_اندازه, i + 1}, 1, درست);
        }
    }
    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", mul_mat_vec_nc_f16_f32_len, mul_mat_vec_nc_f16_f32_داده, "main", mul_mat_vec_رقم_bindings, حجم(vk_mat_vec_nc_افزودن_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_norm_f32, "norm_f32", norm_f32_len, norm_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_گروه_norm_f32, "group_norm_f32", گروه_norm_f32_len, گروه_norm_f32_داده, "main", 2, حجم(vk_op_افزودن_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32, "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_داده, "main", 4, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {0, 0}, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_f32, "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_داده, "main", 4, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {0, 1}, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_جمع_کن_f32, "rms_norm_mul_add_f32", rms_norm_mul_جمع_کن_f32_len, rms_norm_mul_جمع_کن_f32_داده, "main", 5, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {0, 1, 0}, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_جمع_کن_mul_f32, "rms_norm_mul_add_mul_f32", rms_norm_mul_جمع_کن_f32_len, rms_norm_mul_جمع_کن_f32_داده, "main", 5, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {0, 1, 1}, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_جمع_کن_partials_f32, "rms_norm_mul_add_partials_f32", rms_norm_mul_جمع_کن_partials_f32_len, rms_norm_mul_جمع_کن_partials_f32_داده, "main", 6, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {0, 1, 0}, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_جمع_کن_mul_partials_f32, "rms_norm_mul_add_mul_partials_f32", rms_norm_mul_جمع_کن_partials_f32_len, rms_norm_mul_جمع_کن_partials_f32_داده, "main", 6, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {0, 1, 1}, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_مجموعه_ردیفها_f32_f32, "rms_norm_set_rows_f32_f32", rms_norm_مجموعه_ردیفها_f32_f32_len, rms_norm_مجموعه_ردیفها_f32_f32_داده, "main", 4, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {0, 0}, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_مجموعه_ردیفها_f32_f16, "rms_norm_set_rows_f32_f16", rms_norm_مجموعه_ردیفها_f32_f16_len, rms_norm_مجموعه_ردیفها_f32_f16_داده, "main", 4, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {0, 0}, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_partials_f32, "rms_norm_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_داده, "main", 4, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {0, 0}, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_partials_f32, "rms_norm_mul_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_داده, "main", 4, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {0, 1}, 1, درست);

    إذا (حجم(vk_op_rms_norm_mul_rope_افزودن_constants) <= device->properties.limits.max_افزودن_Constants_اندازه) {
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f32, "rms_norm_mul_rope_f32_f32", rms_norm_mul_rope_f32_f32_len, rms_norm_mul_rope_f32_f32_داده, "main", 7, حجم(vk_op_rms_norm_mul_rope_افزودن_constants), {1, 1, 1}, {0, 1}, 1, درست);
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f16, "rms_norm_mul_rope_f32_f16", rms_norm_mul_rope_f32_f16_len, rms_norm_mul_rope_f32_f16_داده, "main", 7, حجم(vk_op_rms_norm_mul_rope_افزودن_constants), {1, 1, 1}, {0, 1}, 1, درست);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_پشت_f32, "rms_norm_back_f32", rms_norm_پشت_f32_len, rms_norm_پشت_f32_داده, "main", 3, حجم(vk_op_افزودن_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_l2_norm_f32, "l2_norm_f32", l2_norm_f32_len, l2_norm_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_32, "cpy_32", cpy_32_len, cpy_32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f16, "cpy_f32_f16", cpy_f32_f16_len, cpy_f32_f16_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_16, "cpy_16", cpy_16_len, cpy_16_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f32, "cpy_f16_f32", cpy_f16_f32_len, cpy_f16_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_bf16,"cpy_f32_bf16",cpy_f32_bf16_len,cpy_f32_bf16_داده,"main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f32,"cpy_bf16_f32",cpy_bf16_f32_len,cpy_bf16_f32_داده,"main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f16,"cpy_bf16_f16",cpy_bf16_f16_len,cpy_bf16_f16_داده,"main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_i32_f32, "cpy_i32_f32", cpy_i32_f32_len, cpy_i32_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_i32, "cpy_f32_i32", cpy_f32_i32_len, cpy_f32_i32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_32, "contig_cpy_32", contig_cpy_32_len, contig_cpy_32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f16, "contig_cpy_f32_f16", contig_cpy_f32_f16_len, contig_cpy_f32_f16_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_16, "contig_cpy_16", contig_cpy_16_len, contig_cpy_16_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f32, "contig_cpy_f16_f32", contig_cpy_f16_f32_len, contig_cpy_f16_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_bf16,"contig_cpy_f32_bf16",contig_cpy_f32_bf16_len,contig_cpy_f32_bf16_داده,"main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f32,"contig_cpy_bf16_f32",contig_cpy_bf16_f32_len,contig_cpy_bf16_f32_داده,"main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f16,"contig_cpy_bf16_f16",contig_cpy_bf16_f16_len,contig_cpy_bf16_f16_داده,"main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_i32_f32, "contig_cpy_i32_f32", contig_cpy_i32_f32_len, contig_cpy_i32_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_i32, "contig_cpy_f32_i32", contig_cpy_f32_i32_len, contig_cpy_f32_i32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_32, "cpy_transpose_32", cpy_transpose_32_len, cpy_transpose_32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_16, "cpy_transpose_16", cpy_transpose_16_len, cpy_transpose_16_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_32, "cpy_transpose_02_32", cpy_transpose_02_32_len, cpy_transpose_02_32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_16, "cpy_transpose_02_16", cpy_transpose_02_16_len, cpy_transpose_02_16_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q1_0], "cpy_f32_q1_0", cpy_f32_q1_0_len, cpy_f32_q1_0_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q2_0], "cpy_f32_q2_0", cpy_f32_q2_0_len, cpy_f32_q2_0_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_0], "cpy_f32_q4_0", cpy_f32_q4_0_len, cpy_f32_q4_0_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_1], "cpy_f32_q4_1", cpy_f32_q4_1_len, cpy_f32_q4_1_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_0], "cpy_f32_q5_0", cpy_f32_q5_0_len, cpy_f32_q5_0_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_1], "cpy_f32_q5_1", cpy_f32_q5_1_len, cpy_f32_q5_1_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q8_0], "cpy_f32_q8_0", cpy_f32_q8_0_len, cpy_f32_q8_0_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_IQ4_NL], "cpy_f32_iq4_nl", cpy_f32_iq4_nl_len, cpy_f32_iq4_nl_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {32, 1, 1}, {}, 1);

#تعریف_کن مجموعه_ردیفها_2(src_idx, src, itype) \
        ggml_vk_create_pipeline(device, device->pipeline_مجموعه_ردیفها ## itype [src_idx][GGML_TYPE_F32],  "set_rows_" #src "_f32" #itype,  مجموعه_ردیفها_3 ## src ## _f32 ## itype ## _len,  مجموعه_ردیفها_3 ## src ## _f32 ## itype ## داده_2,  "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {1}, 1, درست); \
        ggml_vk_create_pipeline(device, device->pipeline_مجموعه_ردیفها ## itype [src_idx][GGML_TYPE_F16],  "set_rows_" #src "_f16" #itype,  مجموعه_ردیفها_3 ## src ## _f16 ## itype ## _len,  مجموعه_ردیفها_3 ## src ## _f16 ## itype ## داده_2,  "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {1}, 1, درست); \
        ggml_vk_create_pipeline(device, device->pipeline_مجموعه_ردیفها ## itype [src_idx][GGML_TYPE_BF16], "set_rows_" #src "_bf16" #itype, مجموعه_ردیفها_3 ## src ## _bf16 ## itype ## _len, مجموعه_ردیفها_3 ## src ## _bf16 ## itype ## داده_2, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {1}, 1, درست); \
        ggml_vk_create_pipeline(device, device->pipeline_مجموعه_ردیفها ## itype [src_idx][GGML_TYPE_Q1_0], "set_rows_" #src "_q1_0" #itype, مجموعه_ردیفها_3 ## src ## _q1_0 ## itype ## _len, مجموعه_ردیفها_3 ## src ## _q1_0 ## itype ## داده_2, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {1}, 1, درست); \
        ggml_vk_create_pipeline(device, device->pipeline_مجموعه_ردیفها ## itype [src_idx][GGML_TYPE_Q2_0], "set_rows_" #src "_q2_0" #itype, مجموعه_ردیفها_3 ## src ## _q2_0 ## itype ## _len, مجموعه_ردیفها_3 ## src ## _q2_0 ## itype ## داده_2, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {1}, 1, درست); \
        ggml_vk_create_pipeline(device, device->pipeline_مجموعه_ردیفها ## itype [src_idx][GGML_TYPE_Q4_0], "set_rows_" #src "_q4_0" #itype, مجموعه_ردیفها_3 ## src ## _q4_0 ## itype ## _len, مجموعه_ردیفها_3 ## src ## _q4_0 ## itype ## داده_2, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {1}, 1, درست); \
        ggml_vk_create_pipeline(device, device->pipeline_مجموعه_ردیفها ## itype [src_idx][GGML_TYPE_Q4_1], "set_rows_" #src "_q4_1" #itype, مجموعه_ردیفها_3 ## src ## _q4_1 ## itype ## _len, مجموعه_ردیفها_3 ## src ## _q4_1 ## itype ## داده_2, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {1}, 1, درست); \
        ggml_vk_create_pipeline(device, device->pipeline_مجموعه_ردیفها ## itype [src_idx][GGML_TYPE_Q5_0], "set_rows_" #src "_q5_0" #itype, مجموعه_ردیفها_3 ## src ## _q5_0 ## itype ## _len, مجموعه_ردیفها_3 ## src ## _q5_0 ## itype ## داده_2, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {1}, 1, درست); \
        ggml_vk_create_pipeline(device, device->pipeline_مجموعه_ردیفها ## itype [src_idx][GGML_TYPE_Q5_1], "set_rows_" #src "_q5_1" #itype, مجموعه_ردیفها_3 ## src ## _q5_1 ## itype ## _len, مجموعه_ردیفها_3 ## src ## _q5_1 ## itype ## داده_2, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {1}, 1, درست); \
        ggml_vk_create_pipeline(device, device->pipeline_مجموعه_ردیفها ## itype [src_idx][GGML_TYPE_Q8_0], "set_rows_" #src "_q8_0" #itype, مجموعه_ردیفها_3 ## src ## _q8_0 ## itype ## _len, مجموعه_ردیفها_3 ## src ## _q8_0 ## itype ## داده_2, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {1}, 1, درست); \
        ggml_vk_create_pipeline(device, device->pipeline_مجموعه_ردیفها ## itype [src_idx][GGML_TYPE_IQ4_NL], "set_rows_" #src "_iq4_nl" #itype, مجموعه_ردیفها_3 ## src ## _iq4_nl ## itype ## _len, مجموعه_ردیفها_3 ## src ## _iq4_nl ## itype ## داده_2, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, {1}, 1, درست);

    مجموعه_ردیفها_2(0, f32, _i32)
    مجموعه_ردیفها_2(0, f32, _i64)
    مجموعه_ردیفها_2(1, f16, _i32)
    مجموعه_ردیفها_2(1, f16, _i64)
#لغو_تعریف مجموعه_ردیفها_2


    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q1_0], "cpy_q1_0_f32", cpy_q1_0_f32_len, cpy_q1_0_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {(uint32_t)ggml_blck_اندازه(GGML_TYPE_Q1_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q2_0], "cpy_q2_0_f32", cpy_q2_0_f32_len, cpy_q2_0_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {(uint32_t)ggml_blck_اندازه(GGML_TYPE_Q2_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_0], "cpy_q4_0_f32", cpy_q4_0_f32_len, cpy_q4_0_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {(uint32_t)ggml_blck_اندازه(GGML_TYPE_Q4_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_1], "cpy_q4_1_f32", cpy_q4_1_f32_len, cpy_q4_1_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {(uint32_t)ggml_blck_اندازه(GGML_TYPE_Q4_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_0], "cpy_q5_0_f32", cpy_q5_0_f32_len, cpy_q5_0_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {(uint32_t)ggml_blck_اندازه(GGML_TYPE_Q5_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_1], "cpy_q5_1_f32", cpy_q5_1_f32_len, cpy_q5_1_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {(uint32_t)ggml_blck_اندازه(GGML_TYPE_Q5_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q8_0], "cpy_q8_0_f32", cpy_q8_0_f32_len, cpy_q8_0_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {(uint32_t)ggml_blck_اندازه(GGML_TYPE_Q8_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_IQ4_NL], "cpy_iq4_nl_f32", cpy_iq4_nl_f32_len, cpy_iq4_nl_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {(uint32_t)ggml_blck_اندازه(GGML_TYPE_IQ4_NL), 1, 1}, {}, 1);

    خودکار get_suffix = [](منطقي src0_f16, منطقي src1_f16, منطقي dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        أرجع s;
    };

#تعریف_کن CREATE_دودویی(name, namemod, spec, bindings) \
    لكل (صحيح s0 : {0,1}) لكل (صحيح s1 : {0,1}) لكل (صحيح d : {0,1}) \
        ggml_vk_create_pipeline2(device, device->pipeline_ ## name ## namemod[s0][s1][d], \
                                #name + get_suffix(s0, s1, d) + #namemod, name ## _len[s0][s1][d], name ## داده_2[s0][s1][d], \
                                "main", (bindings), حجم(vk_op_دودویی_افزودن_constants), {512, 1, 1}, spec, 1);

    CREATE_دودویی(جمع_کن_2, , {0}, 4)
    CREATE_دودویی(جمع_کن_2, _norepeat, {1}, 4)
    CREATE_دودویی(sub, , {0}, 3)
    CREATE_دودویی(sub, _norepeat, {1}, 3)
    CREATE_دودویی(mul, , {0}, 3)
    CREATE_دودویی(mul, _norepeat, {1}, 3)
    CREATE_دودویی(div, , {0}, 3)
    CREATE_دودویی(div, _norepeat, {1}, 3)
    CREATE_دودویی(جمع_کن_rms, , {0}, 4)
    CREATE_دودویی(جمع_کن_rms, _norepeat, {1}, 4)
#لغو_تعریف CREATE_دودویی

    إذا (device->multi_جمع_کن) {
        لكل (uint32_t i = 0; i < MAX_FUSED_ADDS; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_multi_جمع_کن[i],     "multi_add_f32_"     + std::to_string(i+1), multi_جمع_کن_f32_len,     multi_جمع_کن_f32_داده,     "main", MAX_PARAMETER_شمار, حجم(vk_op_multi_جمع_کن_افزودن_constants), {512, 1, 1}, {i+2}, 1);
            ggml_vk_create_pipeline2(device, device->pipeline_multi_جمع_کن_rms[i], "multi_add_rms_f32_" + std::to_string(i+1), multi_جمع_کن_rms_f32_len, multi_جمع_کن_rms_f32_داده, "main", MAX_PARAMETER_شمار, حجم(vk_op_multi_جمع_کن_افزودن_constants), {512, 1, 1}, {i+2}, 1);
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_جمع_کن_id_f32, "add_id_f32", جمع_کن_id_f32_len, جمع_کن_id_f32_داده, "main", 4, حجم(vk_op_جمع_کن_id_افزودن_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_acc_f32, "acc_f32", acc_f32_len, acc_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {512, 1, 1}, {0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_مجموعه_f32, "set_f32", acc_f32_len, acc_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {512, 1, 1}, {0, 0}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_concat_i8, "concat_i8", concat_i8_len, concat_i8_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i16, "concat_i16", concat_i16_len, concat_i16_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i32, "concat_i32", concat_i32_len, concat_i32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i64, "concat_i64", concat_i64_len, concat_i64_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_upscale_نزدیکترین_f32, "upscale_f32", upscale_f32_len, upscale_f32_داده, "main", 2, حجم(vk_op_upscale_افزودن_constants), {512, 1, 1}, {GGML_SCALE_نما_نزدیکترین}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_f32, "upscale_f32", upscale_f32_len, upscale_f32_داده, "main", 2, حجم(vk_op_upscale_افزودن_constants), {512, 1, 1}, {GGML_SCALE_نما_BILINEAR}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bicubic_f32, "upscale_f32", upscale_f32_len, upscale_f32_داده, "main", 2, حجم(vk_op_upscale_افزودن_constants), {512, 1, 1}, {GGML_SCALE_نما_BICUBIC}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_antialias_f32, "upscale_f32", upscale_f32_len, upscale_f32_داده, "main", 2, حجم(vk_op_upscale_افزودن_constants), {512, 1, 1}, {GGML_SCALE_نما_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_scale_f32, "scale_f32", scale_f32_len, scale_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_log[0], "log_f32", log_f32_len, log_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_log[1], "log_f16", log_f16_len, log_f16_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_tri[0], "tri_f32", tri_f32_len, tri_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_tri[1], "tri_f16", tri_f16_len, tri_f16_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag[0], "diag_f32", diag_f32_len, diag_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_diag[1], "diag_f16", diag_f16_len, diag_f16_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pad_f32, "pad_f32", pad_f32_len, pad_f32_داده, "main", 2, حجم(vk_op_pad_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pad_reflect_1d_f32, "pad_reflect_1d_f32", pad_reflect_1d_f32_len, pad_reflect_1d_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_roll_f32, "roll_f32", roll_f32_len, roll_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i32, "repeat_i32", repeat_i32_len, repeat_i32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_repeat_پشت_f32, "repeat_back_f32", repeat_پشت_f32_len, repeat_پشت_f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i16, "repeat_i16", repeat_i16_len, repeat_i16_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);

#تعریف_کن CREATE_UNARY(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_داده, "main", 2, حجم(vk_op_unary_افزودن_constants), {512, 1, 1}, {}, 1);

    CREATE_UNARY(elu)
    CREATE_UNARY(gelu)
    CREATE_UNARY(gelu_erf)
    CREATE_UNARY(gelu_سریع)
    CREATE_UNARY(silu)
    CREATE_UNARY(relu)
    CREATE_UNARY(sqr)
    CREATE_UNARY(جذر)
    CREATE_UNARY(سینوس)
    CREATE_UNARY(کسینوس)
    CREATE_UNARY(clamp)
    CREATE_UNARY(leaky_relu)
    CREATE_UNARY(xielu)
    CREATE_UNARY(neg)
    CREATE_UNARY(tanh)
    CREATE_UNARY(sigmoid)
    CREATE_UNARY(hardsigmoid)
    CREATE_UNARY(hardswish)
    CREATE_UNARY(قدر_مطلق)
    CREATE_UNARY(softplus)
    CREATE_UNARY(گام)
    CREATE_UNARY(round)
    CREATE_UNARY(ceil)
    CREATE_UNARY(floor)
    CREATE_UNARY(trunc)
    CREATE_UNARY(sgn)
    CREATE_UNARY(نمایی_تابع)
    CREATE_UNARY(expm1)
#لغو_تعریف CREATE_UNARY

// spec constants: {norepeat, op_on_b}
#تعریف_کن CREATE_UNARY_MUL(name, idx) \
    لكل (صحيح dt = 0; dt < 2; ++dt) { \
        ثابت size_t len_ = dt ? name ## _mul_f16_len : name ## _mul_f32_len; \
        ثابت بیعلامت حرف * داده_3 = dt ? name ## mul_f16_داده : name ## mul_f32_داده; \
        ثابت std::string dts_ = dt ? "f16" : "f32"; \
        لكل (صحيح ob = 0; ob < 2; ++ob) \
            لكل (صحيح nr = 0; nr < 2; ++nr) \
                ggml_vk_create_pipeline(device, device->pipeline_unary_mul[(idx)][dt][nr][ob], \
                    (#name "_mul" + std::string(ob ? "_b" : "") + "_" + dts_ + (nr ? "_norepeat" : "")).c_str(), \
                    len_, داده_3, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {512, 1, 1}, \
                    { (uint32_t) nr, (uint32_t) ob }, 1); \
    }

    CREATE_UNARY_MUL(gelu, 0)
    CREATE_UNARY_MUL(sigmoid, 1)
    CREATE_UNARY_MUL(silu, 2)
    CREATE_UNARY_MUL(softplus, 3)
#لغو_تعریف CREATE_UNARY_MUL

    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f16, "add1_f16_f16", add1_f16_f16_len, add1_f16_f16_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f32, "add1_f16_f32", add1_f16_f32_len, add1_f16_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f32_f32, "add1_f32_f32", add1_f32_f32_len, add1_f32_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_arange_f32, "arange_f32", arange_f32_len, arange_f32_داده, "main", 1, حجم(vk_op_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_fill_f32, "fill_f32", fill_f32_len, fill_f32_داده, "main", 1, حجم(vk_op_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_fill_f16, "fill_f16", fill_f16_len, fill_f16_داده, "main", 1, حجم(vk_op_افزودن_constants), {512, 1, 1}, {}, 1);

#تعریف_کن CREATE_GLU(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_داده, "main", 3, حجم(vk_op_glu_افزودن_constants), {512, 1, 1}, {}, 1, درست);   \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_داده, "main", 3, حجم(vk_op_glu_افزودن_constants), {512, 1, 1}, {}, 1, درست);

    CREATE_GLU(geglu)
    CREATE_GLU(reglu)
    CREATE_GLU(swiglu)
    CREATE_GLU(swiglu_oai)
    CREATE_GLU(swiglu_clamp)
    CREATE_GLU(geglu_erf)
    CREATE_GLU(geglu_سریع)
#لغو_تعریف CREATE_GLU

    ggml_vk_create_pipeline(device, device->pipeline_silu_پشت_f32, "silu_back_f32", silu_پشت_f32_len, silu_پشت_f32_داده, "main", 3, حجم(vk_op_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag_ماسک_inf_f32, "diag_mask_inf_f32", diag_ماسک_inf_f32_len, diag_ماسک_inf_f32_داده, "main", 2, حجم(vk_op_diag_ماسک_افزودن_constants), {1, 512, 1}, {}, 1, درست);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32, "soft_max_f32", soft_max_f32_len, soft_max_f32_داده, "main", 4, حجم(vk_op_soft_max_افزودن_constants), {1, 1, 1}, { device->subgroup_اندازه }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_wg512, "soft_max_f32_wg512", soft_max_f32_len, soft_max_f32_داده, "main", 4, حجم(vk_op_soft_max_افزودن_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16, "soft_max_f32_f16", soft_max_f32_f16_len, soft_max_f32_f16_داده, "main", 4, حجم(vk_op_soft_max_افزودن_constants), {1, 1, 1}, { device->subgroup_اندازه }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16_wg512, "soft_max_f32_f16_wg512", soft_max_f32_f16_len, soft_max_f32_f16_داده, "main", 4, حجم(vk_op_soft_max_افزودن_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_پشت_f32, "soft_max_back_f32", soft_max_پشت_f32_len, soft_max_پشت_f32_داده, "main", 3, حجم(vk_op_افزودن_constants), {1, 1, 1}, { device->subgroup_اندازه }, 1, درست);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32,     "soft_max_large1_f32",     soft_max_large1_f32_len,     soft_max_large1_f32_داده,     "main", 6, حجم(vk_op_soft_max_افزودن_constants), {1, 1, 1}, { 128, 4 }, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32,     "soft_max_large2_f32",     soft_max_large2_f32_len,     soft_max_large2_f32_داده,     "main", 6, حجم(vk_op_soft_max_افزودن_constants), {1, 1, 1}, { 128, 4 }, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32,     "soft_max_large3_f32",     soft_max_large3_f32_len,     soft_max_large3_f32_داده,     "main", 6, حجم(vk_op_soft_max_افزودن_constants), {1, 1, 1}, { 128, 4 }, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32_f16, "soft_max_large1_f32_f16", soft_max_large1_f32_f16_len, soft_max_large1_f32_f16_داده, "main", 6, حجم(vk_op_soft_max_افزودن_constants), {1, 1, 1}, { 128, 4 }, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32_f16, "soft_max_large2_f32_f16", soft_max_large2_f32_f16_len, soft_max_large2_f32_f16_داده, "main", 6, حجم(vk_op_soft_max_افزودن_constants), {1, 1, 1}, { 128, 4 }, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32_f16, "soft_max_large3_f32_f16", soft_max_large3_f32_f16_len, soft_max_large3_f32_f16_داده, "main", 6, حجم(vk_op_soft_max_افزودن_constants), {1, 1, 1}, { 128, 4 }, 1, درست);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32, "rope_norm_f32", rope_norm_f32_len, rope_norm_f32_داده, "main", 5, حجم(vk_op_rope_افزودن_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32, "rope_neox_f32", rope_neox_f32_len, rope_neox_f32_داده, "main", 5, حجم(vk_op_rope_افزودن_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32, "rope_multi_f32", rope_multi_f32_len, rope_multi_f32_داده, "main", 5, حجم(vk_op_rope_افزودن_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f32, "rope_vision_f32", rope_vision_f32_len, rope_vision_f32_داده, "main", 5, حجم(vk_op_rope_افزودن_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_len, rope_norm_f16_داده, "main", 5, حجم(vk_op_rope_افزودن_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_len, rope_neox_f16_داده, "main", 5, حجم(vk_op_rope_افزودن_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f16, "rope_multi_f16", rope_multi_f16_len, rope_multi_f16_داده, "main", 5, حجم(vk_op_rope_افزودن_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f16, "rope_vision_f16", rope_vision_f16_len, rope_vision_f16_داده, "main", 5, حجم(vk_op_rope_افزودن_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32_f16, "rope_norm_f32_f16", rope_norm_f32_f16_len, rope_norm_f32_f16_داده, "main", 5, حجم(vk_op_rope_افزودن_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32_f16, "rope_neox_f32_f16", rope_neox_f32_f16_len, rope_neox_f32_f16_داده, "main", 5, حجم(vk_op_rope_افزودن_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32_f16, "rope_multi_f32_f16", rope_multi_f32_f16_len, rope_multi_f32_f16_داده, "main", 5, حجم(vk_op_rope_افزودن_constants), {1, 512, 1}, {}, 1);

    لكل (uint32_t i = 0; i < رقم_argsort_pipelines; ++i) {
        uint32_t بلوک_اندازه_2 = 1u << std::min(i, device->max_workgroup_اندازه_log2);
        إذا (i <= device->max_workgroup_اندازه_log2 &&
            2 * حجم(صحيح) * بلوک_اندازه_2 <= device->properties.limits.max_Compute_Shared_حافظه_اندازه) {
            ثابت uint32_t NCOLS_PADDED_LOG2 = i;
            ggml_vk_create_pipeline2(device, device->pipeline_argsort_f32[i], "argsort_f32_"+std::to_string(i), argsort_f32_len, argsort_f32_داده, "main", 3, حجم(vk_op_argsort_افزودن_constants), {بلوک_اندازه_2, 1, 1}, {بلوک_اندازه_2, NCOLS_PADDED_LOG2}, 1, درست);
        }
        ثابت uint32_t WG_UNROLL_ضریب = بلوک_اندازه_2 > 1 ? 2 : 1;
        بلوک_اندازه_2 /= WG_UNROLL_ضریب;
        ggml_vk_create_pipeline2(device, device->pipeline_argsort_large_f32[i], "argsort_large_f32_"+std::to_string(i), argsort_large_f32_len, argsort_large_f32_داده, "main", 3, حجم(vk_op_argsort_افزودن_constants), {بلوک_اندازه_2 * WG_UNROLL_ضریب, 1, 1}, {بلوک_اندازه_2, WG_UNROLL_ضریب}, 1, درست);
    }

    لكل (uint32_t i = 0; i < رقم_topk_pipelines; ++i) {
        ثابت uint32_t بلوک_اندازه_2 = 1u << i;
        ثابت uint32_t NCOLS_PADDED_LOG2 = i;
        إذا (i <= device->max_workgroup_اندازه_log2) {
            uint32_t nary_shmem = 2 * حجم(صحيح) * بلوک_اندازه_2 +
                                  حجم(صحيح) * device->subgroup_اندازه +
                                  2 * حجم(صحيح) +
                                  2 * (بلوک_اندازه_2 / device->subgroup_اندازه) * حجم(صحيح);
            إذا (device->subgroup_arithmetic && device->subgroup_require_پر_support && device->subgroup_shuffle && device->subgroup_ballot &&
                nary_shmem <= device->properties.limits.max_Compute_Shared_حافظه_اندازه) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_nary_جستجو_f32_len, topk_nary_جستجو_f32_داده, "main", 2, حجم(vk_op_topk_افزودن_constants), {بلوک_اندازه_2, 1, 1}, {بلوک_اندازه_2, device->subgroup_اندازه, device->subgroup_اندازه_log2}, 1, درست, درست, device->subgroup_اندازه);
            } وإلا إذا (2 * حجم(صحيح) * بلوک_اندازه_2 <= device->properties.limits.max_Compute_Shared_حافظه_اندازه) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_argsort_f32_len, topk_argsort_f32_داده, "main", 2, حجم(vk_op_topk_افزودن_constants), {بلوک_اندازه_2, 1, 1}, {بلوک_اندازه_2, NCOLS_PADDED_LOG2}, 1, درست);
            }
        }
    }

    // large-k fallback: one workgroup per row, radix-select instead of a full sort. The QSA
    // variant (spec constant 1) additionally gathers the qwen4 indexer input on the fly.
    {
        ثابت uint32_t بلوک_اندازه_2 = 1u << std::min(10u, device->max_workgroup_اندازه_log2);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_f32, "topk_radix_f32", topk_radix_select_f32_len, topk_radix_select_f32_داده, "main", 5, حجم(vk_op_topk_radix_افزودن_constants), {بلوک_اندازه_2, 1, 1}, {بلوک_اندازه_2, 0}, 1, درست);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_qsa, "topk_radix_qsa", topk_radix_select_f32_len, topk_radix_select_f32_داده, "main", 5, حجم(vk_op_topk_radix_افزودن_constants), {بلوک_اندازه_2, 1, 1}, {بلوک_اندازه_2, 1}, 1, درست);
    }

    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, "argmax_f32", argmax_f32_len, argmax_f32_داده, "main", 2, حجم(vk_op_افزودن_constants), {1, 1, 1}, { device->subgroup_اندازه }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_جمع_ردیفها_f32, "sum_rows_f32", جمع_ردیفها_f32_len, جمع_ردیفها_f32_داده, "main", 2, حجم(vk_op_جمع_ردیفها_افزودن_constants), {1, 1, 1}, { device->subgroup_اندازه }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32, "cross_entropy_loss_f32", cross_entropy_loss_f32_len, cross_entropy_loss_f32_داده, "main", 3, حجم(vk_op_افزودن_constants), {1, 1, 1}, { device->subgroup_اندازه }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32_wg512, "cross_entropy_loss_f32_wg512", cross_entropy_loss_f32_len, cross_entropy_loss_f32_داده, "main", 3, حجم(vk_op_افزودن_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_پشت_f32, "cross_entropy_loss_back_f32", cross_entropy_loss_پشت_f32_len, cross_entropy_loss_پشت_f32_داده, "main", 4, حجم(vk_op_افزودن_constants), {1, 1, 1}, { device->subgroup_اندازه }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_پشت_f32_wg512, "cross_entropy_loss_back_f32_wg512", cross_entropy_loss_پشت_f32_len, cross_entropy_loss_پشت_f32_داده, "main", 4, حجم(vk_op_افزودن_constants), {1, 1, 1}, { 512 }, 1);
    // Intel Windows driver in range [32.0.101.8509, 32.0.101.8860) will crash when using fwht kernels so we gate that here
    ثابت منطقي can_use_fwht = device->driver_id != vk::DriverId::eIntelProprietaryWindows ||
        !ggml_vk_intel_windows_driver_درون_range(device->properties.driver_نسخه_2, 101, 8509, 101, 8860);
    إذا (can_use_fwht && device->subgroup_basic && device->subgroup_shuffle) {
        صحيح idx = 0;
        لكل (uint32_t n : {64, 128, 256, 512}) {
            إذا (device->subgroup_اندازه <= n) {
                ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_داده, "main", 2, حجم(vk_op_fwht_افزودن_constants), {1, 1, 1}, { device->subgroup_اندازه, n }, 1, درست, درست, device->subgroup_اندازه);
            }
            ++idx;
        }
    } وإلا إذا (can_use_fwht) {
        صحيح idx = 0;
        لكل (uint32_t n : {64, 128, 256, 512}) {
            ثابت uint32_t بلوک_اندازه_3 = std::min(device->subgroup_اندازه, n);
            ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_داده, "main", 2, حجم(vk_op_fwht_افزودن_constants), {1, 1, 1}, { بلوک_اندازه_3, n }, 1);
            ++idx;
        }
    }

    ثابت uint32_t cumsum_elem_در_هر_thread = (device->vendor_id == VK_VENDOR_ID_AMD || device->vendor_id == VK_VENDOR_ID_INTEL) ? 2 : 4;
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_f32,       "cumsum_f32", cumsum_f32_len, cumsum_f32_داده, "main", 2, حجم(vk_op_جمع_ردیفها_افزودن_constants), {1, 1, 1}, { 256, device->subgroup_اندازه, cumsum_elem_در_هر_thread }, 1, درست, درست, device->subgroup_اندازه);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_small_f32, "cumsum_f32", cumsum_f32_len, cumsum_f32_داده, "main", 2, حجم(vk_op_جمع_ردیفها_افزودن_constants), {1, 1, 1}, { 128, device->subgroup_اندازه, 1 }, 1, درست, درست, device->subgroup_اندازه);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass1_f32, "cumsum_multipass1_f32", cumsum_multipass1_f32_len, cumsum_multipass1_f32_داده, "main", 3, حجم(vk_op_جمع_ردیفها_افزودن_constants), {256, 1, 1}, { 256, device->subgroup_اندازه }, 1, درست, درست, device->subgroup_اندازه);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass2_f32, "cumsum_multipass2_f32", cumsum_multipass2_f32_len, cumsum_multipass2_f32_داده, "main", 3, حجم(vk_op_جمع_ردیفها_افزودن_constants), {256, 1, 1}, { 256, device->subgroup_اندازه }, 1, درست, درست, device->subgroup_اندازه);

    ggml_vk_create_pipeline(device, device->pipeline_شمار_equal_i32, "count_equal_i32", شمار_equal_i32_len, شمار_equal_i32_داده, "main", 3, حجم(vk_op_افزودن_constants), {512, 1, 1}, { device->subgroup_اندازه }, 1);

    إذا (device->subgroup_arithmetic && device->subgroup_vote && device->subgroup_require_پر_support) {
        ggml_vk_create_pipeline(device, device->pipeline_شمار_experts, "count_experts", شمار_experts_subgroup_len, شمار_experts_subgroup_داده, "main", 2, حجم(vk_op_شمار_experts_افزودن_constants), {1, 1, 1}, {}, 1, درست, درست);
    } وإلا {
        ggml_vk_create_pipeline(device, device->pipeline_شمار_experts, "count_experts", شمار_experts_len, شمار_experts_داده, "main", 2, حجم(vk_op_شمار_experts_افزودن_constants), {1, 1, 1}, {}, 1, درست);
    }

    // comb holds a token's 4x4 matrix in one 16-lane slice of a subgroup, so it
    // needs at least 16 lanes, pinned to a known size.
    إذا (device->subgroup_basic && device->subgroup_shuffle && device->subgroup_require_پر_support && device->subgroup_اندازه >= 16) {
        ثابت uint32_t نشانهها_در_هر_workgroup = 4 * (device->subgroup_اندازه / 16);
        ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_comb_f32, "dsv4_hc_comb_f32", dsv4_hc_comb_f32_len, dsv4_hc_comb_f32_داده, "main", 4, حجم(vk_op_dsv4_hc_comb_افزودن_constants), {نشانهها_در_هر_workgroup, 1, 1}, { device->subgroup_اندازه }, 1, درست, درست, device->subgroup_اندازه);
    }

    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_f32,        "dsv4_hc_pre_f32",        dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_داده,  "main", 3, حجم(vk_op_dsv4_hc_pre_افزودن_constants),  {256, 1, 1}, { 256, 0 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_gated_f32,  "dsv4_hc_pre_gated_f32",  dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_داده,  "main", 3, حجم(vk_op_dsv4_hc_pre_افزودن_constants),  {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_f32,       "dsv4_hc_post_f32",       dsv4_hc_post_f32_len, dsv4_hc_post_f32_داده, "main", 5, حجم(vk_op_dsv4_hc_post_افزودن_constants), {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_nocomb_f32,"dsv4_hc_post_nocomb_f32",dsv4_hc_post_f32_len, dsv4_hc_post_f32_داده, "main", 5, حجم(vk_op_dsv4_hc_post_افزودن_constants), {256, 1, 1}, { 256, 0 }, 1);

    لكل (خودکار &s : device->pipeline_solve_tri_f32) {
        ثابت vk_solve_tri_pipeline_state &state = s.اولی;

        // Max number of rows to load at a time, limited by shared memory
        ثابت uint32_t batch_N = device->properties.limits.max_Compute_Shared_حافظه_اندازه / ((state.N + state.K) * حجم(عائم));
        // Need at least K invocations, and prefer a minimum of 128 to spread out loading shared memory
        ثابت uint32_t بلوک_اندازه_3 = std::max(128u, 1u << (uint32_t)ceilf(log2f(عائم(state.K))));

        ggml_vk_create_pipeline(
            device, s.second, "solve_tri_f32",
            solve_tri_f32_len, solve_tri_f32_داده, "main", 3,
            حجم(vk_op_دودویی_افزودن_constants), {1, 1, 1}, { 0, state.N, state.K, batch_N, بلوک_اندازه_3 }, 1, درست);
    }

#تعریف_کن IM2_ستون(bda) \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32 ## bda ## _len, im2col_f32 ## bda ## داده_2, "main", 2, حجم(vk_op_im2col_افزودن_constants), {512, 1, 1}, { device->subgroup_اندازه }, 1, درست);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32, "im2col_3d_f32", im2col_3d_f32 ## bda ## _len, im2col_3d_f32 ## bda ## داده_2, "main", 2, حجم(vk_op_im2col_3d_افزودن_constants), {512, 1, 1}, { 512 }, 1, درست);      \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16 ## bda ## _len, im2col_f32_f16 ## bda ## داده_2, "main", 2, حجم(vk_op_im2col_افزودن_constants), {512, 1, 1}, { device->subgroup_اندازه }, 1, درست);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", im2col_3d_f32_f16 ## bda ## _len, im2col_3d_f32_f16 ## bda ## داده_2, "main", 2, حجم(vk_op_im2col_3d_افزودن_constants), {512, 1, 1}, { 512 }, 1, درست);
    إذا (device->shader_int64 && device->میانگیر_device_آدرس) {
        IM2_ستون(_bda)
    } وإلا {
        IM2_ستون()
    }

    ggml_vk_create_pipeline(device, device->pipeline_timestep_embedding_f32, "timestep_embedding_f32", timestep_embedding_f32_len, timestep_embedding_f32_داده, "main", 2, حجم(vk_op_timestep_embedding_افزودن_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_conv_transpose_1d_f32, "conv_transpose_1d_f32", conv_transpose_1d_f32_len, conv_transpose_1d_f32_داده, "main", 3, حجم(vk_op_conv_transpose_1d_افزودن_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f32,  "col2im_1d_f32",  col2im_1d_f32_len,  col2im_1d_f32_داده,  "main", 2, حجم(vk_op_col2im_1d_افزودن_constants), {256, 1, 1}, {}, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f16,  "col2im_1d_f16",  col2im_1d_f16_len,  col2im_1d_f16_داده,  "main", 2, حجم(vk_op_col2im_1d_افزودن_constants), {256, 1, 1}, {}, 1, درست);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_bf16, "col2im_1d_bf16", col2im_1d_bf16_len, col2im_1d_bf16_داده, "main", 2, حجم(vk_op_col2im_1d_افزودن_constants), {256, 1, 1}, {}, 1, درست);

    ggml_vk_create_pipeline(device, device->pipeline_بیرون_prod_f32, "out_prod_f32", بیرون_prod_f32_len, بیرون_prod_f32_داده, "main", 3, حجم(vk_op_دودویی_افزودن_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_snake_f32,  "snake_f32",  snake_f32_len,  snake_f32_داده,  "main", 4, حجم(vk_op_snake_افزودن_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_f16,  "snake_f16",  snake_f16_len,  snake_f16_داده,  "main", 4, حجم(vk_op_snake_افزودن_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_bf16, "snake_bf16", snake_bf16_len, snake_bf16_داده, "main", 4, حجم(vk_op_snake_افزودن_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pool1d_f32, "pool1d_f32", pool1d_f32_len, pool1d_f32_داده, "main", 2, حجم(vk_op_pool1d_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pool2d_f32, "pool2d_f32", pool2d_f32_len, pool2d_f32_داده, "main", 2, حجم(vk_op_pool2d_افزودن_constants), {1, 1, 1}, {512}, 1);
    إذا (device->subgroup_arithmetic && device->subgroup_require_پر_support) {
        ggml_vk_create_pipeline(device, device->pipeline_pool2d_subgroup_f32, "pool2d_subgroup_f32", pool2d_subgroup_f32_len, pool2d_subgroup_f32_داده, "main", 2, حجم(vk_op_pool2d_افزودن_constants), {1, 1, 1}, {128}, 1, خطأ, درست, device->subgroup_اندازه);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv6_f32, "rwkv_wkv6_f32", rwkv_wkv6_f32_len, rwkv_wkv6_f32_داده, "main", 7, حجم(vk_op_rwkv_wkv6_افزودن_constants), {1, 1, 1}, {device->subgroup_اندازه}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv7_f32, "rwkv_wkv7_f32", rwkv_wkv7_f32_len, rwkv_wkv7_f32_داده, "main", 8, حجم(vk_op_rwkv_wkv7_افزودن_constants), {1, 1, 1}, {device->subgroup_اندازه}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_gated_خطی_attn_f32, "gated_linear_attn_f32", gated_خطی_attn_f32_len, gated_خطی_attn_f32_داده, "main", 6, حجم(vk_op_gated_خطی_attn_افزودن_constants), {1, 1, 1}, {}, 1);

    لكل (ggml_type k_type : lightning_indexer_k_types) {
        ثابت std::string name = "lightning_indexer_" + std::string(ggml_type_name(k_type)) + "_k_f32";
        ggml_vk_create_pipeline(device, device->pipeline_lightning_indexer_f32[k_type], name.c_str(), lightning_indexer_f32_len, lightning_indexer_f32_داده, "main", 5, حجم(vk_op_lightning_indexer_افزودن_constants), {1, 1, 1}, {(uint32_t)k_type, fa_بلوک_بایتها(k_type)}, 1, درست);
    }

    {
        ثابت uint32_t gdn_sizes[] = {16, 32, 64, 128};
        ثابت حرف * gdn_نامها[][2] = {
            {"gated_delta_net_f32_d16",     "gated_delta_net_f32_d16_kda"},
            {"gated_delta_net_f32_d32",     "gated_delta_net_f32_d32_kda"},
            {"gated_delta_net_f32_d64",     "gated_delta_net_f32_d64_kda"},
            {"gated_delta_net_f32_d128",    "gated_delta_net_f32_d128_kda"},
        };
        لكل (uint32_t si = 0; si < 4; si++) {
            ثابت uint32_t S_V = gdn_sizes[si];
            GGML_ASSERT(is_pow2(S_V));

            // Intel Xe regresses at SIMD32 for this scan; prefer a narrower subgroup.
            uint32_t gdn_subgroup_اندازه = device->subgroup_اندازه;
            إذا (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_اندازه_control &&
                device->subgroup_min_اندازه <= 16u && device->subgroup_max_اندازه >= 16u) {
                gdn_subgroup_اندازه = 16u;
            }
            uint32_t lanes_در_هر_ستونی;
            إذا (device->vendor_id == VK_VENDOR_ID_INTEL) {
                // Intel Xe: full-width reduction (min rows/lane) ~10x over the COLS_PER_WG=8 rule.
                lanes_در_هر_ستونی = std::min(gdn_subgroup_اندازه, S_V);
            } وإلا إذا (S_V >= 128u && device->subgroup_clustered) {
                // COLS_PER_WG=8: measured optimum (Ampere sg32->4, Vega20 sg64->8).
                lanes_در_هر_ستونی = std::max(1u, gdn_subgroup_اندازه / 8u);
            } وإلا {
                // Use largest power-of-two that divides both S_V and subgroup_size so that
                // (1) S_V % lanes_per_column == 0 and (2) S_V % (subgroup_size / lanes_per_column) == 0.
                // This means we don't need extra bounds checking logic in the shader.
                lanes_در_هر_ستونی = std::min(S_V, gdn_subgroup_اندازه);
            }

            // gated_delta_net.comp relies on S_V % COLS_PER_WG == 0 and
            // S_V % LANES_PER_COLUMN == 0 to avoid bounds checks.
            طالما (lanes_در_هر_ستونی > 1u) {
                ثابت منطقي valid_lanes = (gdn_subgroup_اندازه % lanes_در_هر_ستونی) == 0 &&
                                         (S_V % lanes_در_هر_ستونی) == 0;
                ثابت uint32_t تعداد_ستون_در_هر_wg = valid_lanes ? gdn_subgroup_اندازه / lanes_در_هر_ستونی : 0;
                إذا (valid_lanes && تعداد_ستون_در_هر_wg > 0 && (S_V % تعداد_ستون_در_هر_wg) == 0) {
                    اكسر;
                }
                lanes_در_هر_ستونی >>= 1u;
            }

            GGML_ASSERT((gdn_subgroup_اندازه % lanes_در_هر_ستونی) == 0);
            GGML_ASSERT((S_V % lanes_در_هر_ستونی) == 0);
            GGML_ASSERT((S_V % (gdn_subgroup_اندازه / lanes_در_هر_ستونی)) == 0);

            ثابت منطقي need_partial_subgroup_reduce = lanes_در_هر_ستونی != 1u && lanes_در_هر_ستونی < gdn_subgroup_اندازه;
            ثابت منطقي use_clustered_reduce = device->subgroup_arithmetic && device->subgroup_clustered && need_partial_subgroup_reduce;
            ثابت منطقي use_subgroup_reduce = device->subgroup_arithmetic && !need_partial_subgroup_reduce;
            ثابت منطقي use_subgroup_ops = use_clustered_reduce || use_subgroup_reduce;
            size_t gdn_len;
            ثابت فراغ * gdn_داده;
            إذا (use_clustered_reduce) {
                gdn_len = gated_delta_net_f32_len;
                gdn_داده = (ثابت فراغ *)gated_delta_net_f32_داده;
            } وإلا إذا (use_subgroup_reduce) {
                gdn_len = gated_delta_net_f32_nocluster_len;
                gdn_داده = (ثابت فراغ *)gated_delta_net_f32_nocluster_داده;
            } وإلا {
                gdn_len = gated_delta_net_f32_shmem_len;
                gdn_داده = (ثابت فراغ *)gated_delta_net_f32_shmem_داده;
            }

            ثابت uint32_t تعداد_ستون_در_هر_wg = gdn_subgroup_اندازه / lanes_در_هر_ستونی;
            ثابت std::array<uint32_t, 3> wg_denoms = {1u, 1u, تعداد_ستون_در_هر_wg};

            لكل (uint32_t kda = 0; kda < 2; kda++) {
                ggml_vk_create_pipeline(device, device->pipeline_gated_delta_net[si][kda],
                    gdn_نامها[si][kda], gdn_len, gdn_داده, "main", 7, حجم(vk_op_gated_delta_net_افزودن_constants),
                    wg_denoms, {S_V, kda, gdn_subgroup_اندازه, lanes_در_هر_ستونی}, 1, درست, use_subgroup_ops, gdn_subgroup_اندازه);
            }
        }
    }

    إذا (device->subgroup_arithmetic && device->subgroup_require_پر_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_پیمایش_f32_d128, "ssm_scan_128_f32", ssm_پیمایش_subgroup_f32_len, ssm_پیمایش_subgroup_f32_داده, "main", 8, حجم(vk_op_ssm_پیمایش_افزودن_constants), {1, 1, 1}, {128, device->subgroup_اندازه}, 1, درست, درست);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_پیمایش_f32_d256, "ssm_scan_256_f32", ssm_پیمایش_subgroup_f32_len, ssm_پیمایش_subgroup_f32_داده, "main", 8, حجم(vk_op_ssm_پیمایش_افزودن_constants), {1, 1, 1}, {256, device->subgroup_اندازه}, 1, درست, درست);
    } وإلا {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_پیمایش_f32_d128, "ssm_scan_128_f32", ssm_پیمایش_f32_len, ssm_پیمایش_f32_داده, "main", 8, حجم(vk_op_ssm_پیمایش_افزودن_constants), {1, 1, 1}, {128, device->subgroup_اندازه, 16}, 1, درست, درست);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_پیمایش_f32_d256, "ssm_scan_256_f32", ssm_پیمایش_f32_len, ssm_پیمایش_f32_داده, "main", 8, حجم(vk_op_ssm_پیمایش_افزودن_constants), {1, 1, 1}, {256, device->subgroup_اندازه, 16}, 1, درست, درست);
    }

    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_f32,           "ssm_conv_f32",           ssm_conv_f32_len, ssm_conv_f32_داده, "main", 4, حجم(vk_op_ssm_conv_افزودن_constants), {32, 16, 1}, {32, 16, 0, 0}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_silu_f32,      "ssm_conv_silu_f32",      ssm_conv_f32_len, ssm_conv_f32_داده, "main", 4, حجم(vk_op_ssm_conv_افزودن_constants), {32, 16, 1}, {32, 16, 0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_bias_silu_f32, "ssm_conv_bias_silu_f32", ssm_conv_f32_len, ssm_conv_f32_داده, "main", 4, حجم(vk_op_ssm_conv_افزودن_constants), {32, 16, 1}, {32, 16, 1, 1}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_گام_adamw_f32, "opt_step_adamw_f32", opt_گام_adamw_f32_len, opt_گام_adamw_f32_داده, "main", 5, حجم(vk_op_افزودن_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_گام_sgd_f32, "opt_step_sgd_f32", opt_گام_sgd_f32_len, opt_گام_sgd_f32_داده, "main", 3, حجم(vk_op_افزودن_constants), {512, 1, 1}, {}, 1);

    // conv2d, conv_transpose_2d, conv3d
    لكل (uint32_t s = 0; s < CONV_SHAPE_شمار; ++s) {
        // smaller WG for the small-tile fallback gives more concurrent WGs per SM
        uint32_t conv2d_WG_اندازه  = (s == CONV_SHAPE_64x32) ? 128 : 256;
        uint32_t use_collectives = 0;  // Enables subgroup ops for preventing the re-calculation of indices.
        uint32_t conv2d_TS_K     = (s == CONV_SHAPE_64x32) ? 4 : 8;
        uint32_t conv2d_SHMEM_PAD = 4;
        vk_conv_بلوک_اندازه conv2d_BS = vk_conv_بلوک_sizes[s];
        منطقي conv2d_UNROLL = درست;

#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        إذا (device->coopmat2) {
            conv2d_SHMEM_PAD = 8; // 8 float16_t
        }
#پایان_اگر

        إذا (device->vendor_id == VK_VENDOR_ID_INTEL) {
            conv2d_SHMEM_PAD = 0;
            conv2d_UNROLL = خطأ;
        } وإلا إذا (device->vendor_id == VK_VENDOR_ID_AMD) {
            conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
            إذا (s == CONV_SHAPE_128x128 && device->architecture != vk_device_architecture::AMD_GCN) {
                conv2d_UNROLL = خطأ;
            }
        }

        // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
        منطقي allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
                                    device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        منطقي allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
                                     device->architecture == vk_device_architecture::AMD_GCN;

        إذا (device->subgroup_shuffle &&
            device->vendor_id != VK_VENDOR_ID_INTEL &&   // Do not enable collectives on Intel, see PR 14316.
            allow_collectives_nv &&
            allow_collectives_amd) {
            use_collectives = 1;
            conv2d_BS.CRS   = std::min(
                device->subgroup_اندازه,
                conv2d_BS.CRS);  // CRS block size should be capped at subgroup size for correctness when shuffle is used.
        }

        // cm1 is used only when cm2 is unavailable; capped at 64x128 (due to shared memory size).
        // Requires 16x16x16 f16-acc since that's the fragment shape hard-coded in the shader.
        // Subgroup size must be 32 or 64 (to keep WG_SIZE sane) and we need
        // subgroup_size_control to force the driver to actually use it.
        منطقي conv2d_use_cm1 = خطأ;
#إذا defined(VK_KHR_cooperative_ماتریس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        conv2d_use_cm1 = !device->coopmat2 &&
                         device->coopmat_support && device->coopmat_support_16x16x16_f16acc &&
                         device->subgroup_اندازه_control &&
                         (device->subgroup_اندازه == 32 || device->subgroup_اندازه == 64) &&
                         s != CONV_SHAPE_128x128;
#پایان_اگر

        ثابت uint32_t conv2d_cm1_shmem_pad = 8;

        خودکار shmem_req = [&](uint32_t pad, منطقي csh_store, منطقي fp16_shmem) {
            ثابت uint32_t elem_اندازه = fp16_shmem ? (uint32_t)حجم(uint16_t) : (uint32_t)حجم(عائم);
            ثابت uint32_t csh_elems = csh_store ? conv2d_BS.K * conv2d_BS.NPQ : 0u;
            أرجع (conv2d_BS.K * (conv2d_BS.CRS + pad) + conv2d_BS.CRS * (conv2d_BS.NPQ + pad) + csh_elems) * elem_اندازه;
        };

        // 2D, transpose-2D, and 3D conv use the same KxCRS @ CRSxNPQ shmem
        // layout. cm1 needs Csh for output, so check before applying cm1 params.
        إذا (conv2d_use_cm1 && device->properties.limits.max_Compute_Shared_حافظه_اندازه < shmem_req(conv2d_cm1_shmem_pad, درست, درست)) {
            conv2d_use_cm1 = خطأ;
        }

        uint32_t conv2d_WM = 16, conv2d_WN = 16;  // cm1 subgroup tile, ignored otherwise
        إذا (conv2d_use_cm1) {
            conv2d_SHMEM_PAD = conv2d_cm1_shmem_pad;
            // 16x16x16 fragments; pick WM/WN to keep WG_SIZE at 256
            // (i.e. 8 subgroups for sg=32, 4 subgroups for sg=64).
            ثابت منطقي sg64 = (device->subgroup_اندازه == 64);
            بدل (s) {
                حالة CONV_SHAPE_64x32:   conv2d_WM = sg64 ? 32 : 16; conv2d_WN = 16; اكسر;
                حالة CONV_SHAPE_64x128:  conv2d_WM = 32; conv2d_WN = sg64 ? 64 : 32; اكسر;
                حالة CONV_SHAPE_32x256:  conv2d_WM = sg64 ? 16 : 32; conv2d_WN = sg64 ? 128 : 32; اكسر;
                افتراضي: اكسر;
            }
            ثابت uint32_t warps_M = conv2d_BS.K / conv2d_WM;
            ثابت uint32_t warps_N = conv2d_BS.NPQ / conv2d_WN;
            conv2d_WG_اندازه         = warps_M * warps_N * device->subgroup_اندازه;
        }

        // stage cm2 accumulator through shmem for coalesced global stores;
        // skipped on 128x128 where the extra Csh footprint hurts occupancy.
        // cm1 always uses the staged path.
        uint32_t conv2d_csh_store = (device->coopmat2 && s != CONV_SHAPE_128x128) ? 1u : 0u;
        إذا (conv2d_use_cm1) {
            conv2d_csh_store = 1;
        }

        // shmem is fp16 on cm2/cm1 (matches Csh), fp32 on scalar
        ثابت منطقي conv2d_use_fp16_shmem = device->coopmat2 || conv2d_use_cm1;

        // shrink CRS if the non-cm1 config still doesn't fit
        إذا (device->properties.limits.max_Compute_Shared_حافظه_اندازه < shmem_req(conv2d_SHMEM_PAD, conv2d_csh_store, conv2d_use_fp16_shmem)) {
            GGML_ASSERT(!conv2d_use_cm1);
            conv2d_BS.CRS = 8;
            إذا (use_collectives) {
                conv2d_BS.CRS = std::min(device->subgroup_اندازه, conv2d_BS.CRS);
            }
            conv2d_csh_store = 0;
        }

        std::array<uint32_t, 3> wg_denoms = { conv2d_BS.K, 1, 1 };
        std::vector<uint32_t> spec_constants = { conv2d_WG_اندازه, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, use_collectives, conv2d_SHMEM_PAD };

        // cm1 needs a fixed subgroup width to match the WG_SIZE we computed
        ثابت uint32_t conv2d_required_subgroup_اندازه = conv2d_use_cm1 ? device->subgroup_اندازه : 0;

#تعریف_کن CREATE_CONV(name, type_suffix, spv_suffix) \
        لكل (خودکار &c : device->pipeline_##name##type_suffix[s]) { \
            ثابت vk_conv2d_pipeline_state &state = c.اولی;  \
            std::vector<uint32_t> spec_constants_cpy = spec_constants; \
            spec_constants_cpy.push_back(state.s0); \
            spec_constants_cpy.push_back(state.s1); \
            spec_constants_cpy.push_back(state.p0); \
            spec_constants_cpy.push_back(state.p1); \
            spec_constants_cpy.push_back(state.d0); \
            spec_constants_cpy.push_back(state.d1); \
            spec_constants_cpy.push_back(state.KW); \
            spec_constants_cpy.push_back(state.KH); \
            spec_constants_cpy.push_back(state.aligned); \
            spec_constants_cpy.push_back(conv2d_csh_store); \
            spec_constants_cpy.push_back(conv2d_WM); \
            spec_constants_cpy.push_back(conv2d_WN); \
            ggml_vk_create_pipeline( \
                device, c.second, #name #type_suffix, \
                name##type_suffix##spv_suffix##_len, name##type_suffix##spv_suffix##_data, "main", 3, \
                حجم(vk_op_conv2d_افزودن_constants), wg_denoms, spec_constants_cpy, 1, درست, use_collectives || conv2d_required_subgroup_اندازه, conv2d_required_subgroup_اندازه);    \
        }
#تعریف_کن CREATE_CONVS(spv_suffix) \
        CREATE_CONV(conv2d, _f32, spv_suffix) \
        CREATE_CONV(conv2d, _f16_f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f16_f32, spv_suffix)
#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        إذا (device->coopmat2) {
            CREATE_CONVS(_cm2)
        } وإلا
#پایان_اگر
#إذا defined(VK_KHR_cooperative_ماتریس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        إذا (conv2d_use_cm1) {
            CREATE_CONVS(_cm1)
        } وإلا
#پایان_اگر
        إذا (conv2d_UNROLL) {
            CREATE_CONVS(_unroll)
        } وإلا {
            CREATE_CONVS( )
        }
#لغو_تعریف CREATE_CONV
#لغو_تعریف CREATE_CONVS

        std::vector<uint32_t> conv3d_spec_constants = { conv2d_WG_اندازه, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, conv2d_SHMEM_PAD };
#تعریف_کن CREATE_CONV3D(type_suffix, spv_suffix) \
        لكل (خودکار &c : device->pipeline_conv3d##type_suffix[s]) { \
            ثابت vk_conv3d_pipeline_state &state = c.اولی; \
            std::vector<uint32_t> spec_constants_cpy = conv3d_spec_constants; \
            spec_constants_cpy.push_back(state.s0); \
            spec_constants_cpy.push_back(state.s1); \
            spec_constants_cpy.push_back(state.s2); \
            spec_constants_cpy.push_back(state.p0); \
            spec_constants_cpy.push_back(state.p1); \
            spec_constants_cpy.push_back(state.p2); \
            spec_constants_cpy.push_back(state.d0); \
            spec_constants_cpy.push_back(state.d1); \
            spec_constants_cpy.push_back(state.d2); \
            spec_constants_cpy.push_back(state.KW); \
            spec_constants_cpy.push_back(state.KH); \
            spec_constants_cpy.push_back(state.KD); \
            spec_constants_cpy.push_back(state.aligned); \
            spec_constants_cpy.push_back(conv2d_csh_store); \
            spec_constants_cpy.push_back(conv2d_WM); \
            spec_constants_cpy.push_back(conv2d_WN); \
            ggml_vk_create_pipeline( \
                device, c.second, "conv3d" #type_suffix, \
                conv3d##type_suffix##spv_suffix##_len, conv3d##type_suffix##spv_suffix##_data, "main", 3, \
                حجم(vk_op_conv3d_افزودن_constants), wg_denoms, spec_constants_cpy, 1, درست, conv2d_required_subgroup_اندازه != 0, conv2d_required_subgroup_اندازه); \
        }
#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        إذا (device->coopmat2) {
            CREATE_CONV3D(_f32, _cm2)
            CREATE_CONV3D(_f16_f32, _cm2)
        } وإلا
#پایان_اگر
#إذا defined(VK_KHR_cooperative_ماتریس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        إذا (conv2d_use_cm1) {
            CREATE_CONV3D(_f32, _cm1)
            CREATE_CONV3D(_f16_f32, _cm1)
        } وإلا
#پایان_اگر
        إذا (conv2d_UNROLL) {
            CREATE_CONV3D(_f32, _unroll)
            CREATE_CONV3D(_f16_f32, _unroll)
        } وإلا {
            CREATE_CONV3D(_f32, )
            CREATE_CONV3D(_f16_f32, )
        }
#لغو_تعریف CREATE_CONV3D
    }

    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f32, "conv2d_dw_whcn_f32", conv2d_dw_whcn_f32_len, conv2d_dw_whcn_f32_داده, "main", 3, حجم(vk_op_conv2d_dw_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f32, "conv2d_dw_cwhn_f32", conv2d_dw_cwhn_f32_len, conv2d_dw_cwhn_f32_داده, "main", 3, حجم(vk_op_conv2d_dw_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f16_f32, "conv2d_dw_whcn_f16_f32", conv2d_dw_whcn_f16_f32_len, conv2d_dw_whcn_f16_f32_داده, "main", 3, حجم(vk_op_conv2d_dw_افزودن_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f16_f32, "conv2d_dw_cwhn_f16_f32", conv2d_dw_cwhn_f16_f32_len, conv2d_dw_cwhn_f16_f32_داده, "main", 3, حجم(vk_op_conv2d_dw_افزودن_constants), {512, 1, 1}, {}, 1);

    لكل (uint32_t use_افزودن = 0; use_افزودن < 2; ++use_افزودن) {
        لكل (uint32_t i = 0; i < رقم_topk_moe_pipelines; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_topk_moe[i][use_افزودن], "topk_moe_f32_"+std::to_string(i), topk_moe_f32_len, topk_moe_f32_داده, "main", 4, حجم(vk_op_topk_moe_افزودن_constants), {1, 1, 1}, {device->subgroup_اندازه, 1u<<i, use_افزودن}, 1, درست, درست, device->subgroup_اندازه);
        }
    }

    // Drop compile_mutex so other threads can walk while we compile.
    compile_قفل.unlock();

    // Compile what we claimed; create_pipeline_func reacquires compile_mutex
    // at the end to flip compile_pending/compiled and notify waiters.
    إذا (has_claimed_task) {
        خودکار & task = claimed_task;
        ggml_vk_create_pipeline_func(device, task.pipeline, task.spv_اندازه, task.spv_داده,
                                     task.entrypoint, task.parameter_شمار, task.wg_denoms,
                                     task.specialization_constants, task.disable_robustness,
                                     task.require_پر_subgroups, task.required_subgroup_اندازه);
    }

    // Another thread may be compiling the pipeline we need; block on it here.
    إذا (wait_pipeline) {
        std::unique_lock<std::mutex> wait_قفل(device->compile_mutex);
        device->compile_cv.wait(wait_قفل, [&] {
            أرجع wait_pipeline->compiled.load();
        });
    }
}

ساكن منطقي ggml_vk_external_حافظه_host_usable(vk::DriverId driver_id) {
    // Disable external_memory_host until https://github.com/KhronosGroup/MoltenVK/pull/2622
    // is available in the Vulkan SDK.
    أرجع driver_id != vk::DriverId::eMoltenvk;
}

ساكن منطقي ggml_vk_host_میانگیر_import_usable(uint32_t vendor_id) {
    // amdgpu re-faults all pages of every imported host allocation on each submission
    أرجع vendor_id != VK_VENDOR_ID_AMD;
}

vk_device ggml_vk_get_device(size_t idx) {
    VK_LOG_DEBUG("ggml_vk_get_device(" << idx << ")");

    ساكن std::mutex mutex;
    std::lock_guard<std::mutex> قفل(mutex);

    إذا (vk_instance.devices[idx] == اشاره_تهی) {
        VK_LOG_DEBUG("Initializing new vk_device");
        vk_device device = std::make_shared<vk_device_struct>();

        device->حافظه_logger = std::unique_ptr<vk_حافظه_logger>(جديد vk_حافظه_logger());

        size_t dev_رقم = vk_instance.device_indices[idx];

        std::vector<vk::PhysicalDevice> physical_devices = vk_instance.instance.enumeratePhysicalDevices();

        إذا (dev_رقم >= physical_devices.size()) {
            std::cerr << "ggml_vulkan: Device with index " << dev_رقم << " does not exist." << std::پایان_سطر;
            ارم std::runtime_error("Device not found");
        }

        device->physical_device = physical_devices[dev_رقم];
        ثابت std::vector<vk::ExtensionProperties> ext_props = device->physical_device.enumerateDeviceExtensionProperties();

        device->architecture = get_device_architecture(device->physical_device);

        ثابت حرف* GGML_VK_PREFER_HOST_حافظه = getenv("GGML_VK_PREFER_HOST_MEMORY");
        device->prefer_host_حافظه = GGML_VK_PREFER_HOST_حافظه != اشاره_تهی;

        ثابت حرف* GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM = getenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM");
        device->disable_host_visible_vidmem = GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM != اشاره_تهی;

        ثابت حرف* GGML_VK_ALLOW_SYSMEM_FALLBACK = getenv("GGML_VK_ALLOW_SYSMEM_FALLBACK");
        device->allow_sysmem_fallback = GGML_VK_ALLOW_SYSMEM_FALLBACK != اشاره_تهی;

        ثابت حرف* GGML_VK_DISABLE_گراف_OPTIMIZE = getenv("GGML_VK_DISABLE_GRAPH_OPTIMIZE");
        device->disable_گراف_optimize = GGML_VK_DISABLE_گراف_OPTIMIZE != اشاره_تهی;

        منطقي fp16_storage = خطأ;
        منطقي fp16_compute = خطأ;
        منطقي maintenance4_support = خطأ;
        منطقي sm_builtins = خطأ;
        منطقي amd_shader_core_properties2 = خطأ;
        منطقي pipeline_robustness = خطأ;
        منطقي coopmat2_support = خطأ;
        منطقي coopmat2_decode_وکتور_support = خطأ;
        منطقي pipeline_executable_properties_support = خطأ;
        منطقي internally_sync_support = خطأ;
        device->coopmat_support = خطأ;
        device->integer_dot_حاصلضرب = خطأ;
        device->shader_64b_indexing = خطأ;
        منطقي bfloat16_support = خطأ;
        منطقي dot2_f16_support = خطأ;
        منطقي ocp_microscaling_extension = خطأ;
        منطقي shader_float8_extension = خطأ;

        لكل (ثابت خودکار& properties : ext_props) {
            إذا (strcmp("VK_KHR_maintenance4", properties.extensionName) == 0) {
                maintenance4_support = درست;
            } وإلا إذا (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
                fp16_storage = درست;
            } وإلا إذا (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
                fp16_compute = درست;
            } وإلا إذا (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = درست;
            } وإلا إذا (strcmp("VK_AMD_shader_core_properties2", properties.extensionName) == 0) {
                amd_shader_core_properties2 = درست;
            } وإلا إذا (strcmp("VK_EXT_pipeline_robustness", properties.extensionName) == 0) {
                pipeline_robustness = درست;
            } وإلا إذا (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                device->subgroup_اندازه_control = درست;
#إذا defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            } وإلا إذا (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT")) {
                device->coopmat_support = درست;
                device->coopmat_m = 0;
                device->coopmat_n = 0;
                device->coopmat_k = 0;
#پایان_اگر
#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            } وإلا إذا (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2")) {
                coopmat2_support = درست;
#پایان_اگر
            } وإلا إذا (strcmp(VK_NV_COOPERATIVE_ماتریس_DECODE_وکتور_EXTENSION_NAME, properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
                coopmat2_decode_وکتور_support = درست;
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            } وإلا إذا (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
                device->integer_dot_حاصلضرب = درست;
#پایان_اگر
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            } وإلا إذا (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_BFLOAT16")) {
                bfloat16_support = درست;
#پایان_اگر
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
            } وإلا إذا (strcmp(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
                ocp_microscaling_extension = درست;
#پایان_اگر
#إذا defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            } وإلا إذا (strcmp(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
                shader_float8_extension = درست;
#پایان_اگر
            } وإلا إذا (strcmp("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_DOT2")) {
                dot2_f16_support = درست;
            } وإلا إذا (strcmp("VK_KHR_pipeline_executable_properties", properties.extensionName) == 0) {
                pipeline_executable_properties_support = درست;
            } وإلا إذا (strcmp("VK_EXT_memory_priority", properties.extensionName) == 0 &&
                       getenv("GGML_VK_ENABLE_MEMORY_PRIORITY")) {
                device->حافظه_اولویت = درست;
            } وإلا إذا (strcmp("VK_EXT_external_memory_host", properties.extensionName) == 0) {
                device->external_حافظه_host = درست;
#إذا defined(VK_EXT_shader_64bit_indexing)
            } وإلا إذا (strcmp("VK_EXT_shader_64bit_indexing", properties.extensionName) == 0) {
                device->shader_64b_indexing = درست;
#پایان_اگر
            } وإلا إذا (strcmp(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME, properties.extensionName) == 0) {
                internally_sync_support = درست;
            } وإلا إذا (strcmp("VK_EXT_device_fault", properties.extensionName) == 0) {
                device->device_fault = درست;
            }
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceMaintenance3Properties props3;
        vk::PhysicalDeviceMaintenance4Properties props4;
        vk::PhysicalDeviceSubgroupProperties subgroup_props;
        vk::PhysicalDeviceDriverProperties driver_props;
        vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;
        vk::PhysicalDeviceShaderCoreProperties2AMD amd_shader_core_properties2_props;
        vk::PhysicalDeviceVulkan11Properties vk11_props;
        vk::PhysicalDeviceVulkan12Properties vk12_props;
        vk::Physical_Device_Subgroup_اندازه_Control_Properties_EXT subgroup_اندازه_control_props;
        vk::Physical_Device_Shader_Integer_Dot_حاصلضرب_Properties_KHR shader_integer_dot_حاصلضرب_props;
        vk::Physical_Device_External_حافظه_Host_Properties_EXT external_حافظه_host_props;

        props2.p_بعدی = &props3;
        props3.p_بعدی = &subgroup_props;
        subgroup_props.p_بعدی = &driver_props;
        driver_props.p_بعدی = &vk11_props;
        vk11_props.p_بعدی = &vk12_props;

        Vk_پایه_بیرون_Structure * آخرین_struct = (Vk_پایه_بیرون_Structure *)&vk12_props;

        إذا (maintenance4_support) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&props4;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&props4;
        }
        إذا (sm_builtins) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&sm_props;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&sm_props;
        }
        إذا (amd_shader_core_properties2) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&amd_shader_core_properties2_props;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&amd_shader_core_properties2_props;
        }
        إذا (device->subgroup_اندازه_control) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&subgroup_اندازه_control_props;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&subgroup_اندازه_control_props;
        }

#إذا defined(VK_NV_cooperative_matrix2)
        vk::PhysicalDeviceCooperativeMatrix2PropertiesNV coopmat2_props;
        إذا (coopmat2_support) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&coopmat2_props;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&coopmat2_props;
        }
#پایان_اگر

        إذا (device->integer_dot_حاصلضرب) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&shader_integer_dot_حاصلضرب_props;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&shader_integer_dot_حاصلضرب_props;
        }

        إذا (device->external_حافظه_host) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&external_حافظه_host_props;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&external_حافظه_host_props;
        }

        device->physical_device.getProperties2(&props2);
        device->properties = props2.properties;
        device->vendor_id = device->properties.vendorID;
        device->driver_id = driver_props.driverID;

        device->external_حافظه_host = device->external_حافظه_host && ggml_vk_external_حافظه_host_usable(device->driver_id);

        // Implementing the async backend interfaces seems broken on older Intel HW,
        // see https://github.com/ggml-org/llama.cpp/issues/17302.
        device->support_async = (device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 std::string(device->properties.deviceName.داده()).find("(DG1)") == std::string::npos) &&
                                getenv("GGML_VK_DISABLE_ASYNC") == اشاره_تهی;

        إذا (!device->support_async) {
            GGML_LOG_DEBUG("ggml_vulkan: WARNING: Async execution disabled on certain Intel devices.\n");
        }

        ثابت حرف* GGML_VK_FORCE_MAX_تخصیص_اندازه = getenv("GGML_VK_FORCE_MAX_ALLOCATION_SIZE");

        إذا (GGML_VK_FORCE_MAX_تخصیص_اندازه != اشاره_تهی) {
            device->max_حافظه_تخصیص_اندازه = std::stoull(GGML_VK_FORCE_MAX_تخصیص_اندازه);
        } وإلا إذا (maintenance4_support) {
            device->max_حافظه_تخصیص_اندازه = std::min(props3.max_حافظه_تخصیص_اندازه_2, props4.max_میانگیر_اندازه_2);
        } وإلا {
            device->max_حافظه_تخصیص_اندازه = props3.max_حافظه_تخصیص_اندازه_2;
        }

        ثابت حرف* GGML_VK_FORCE_MAX_میانگیر_اندازه = getenv("GGML_VK_FORCE_MAX_BUFFER_SIZE");

        إذا (GGML_VK_FORCE_MAX_میانگیر_اندازه != اشاره_تهی) {
            device->max_میانگیر_اندازه = std::stoull(GGML_VK_FORCE_MAX_میانگیر_اندازه);
        } وإلا إذا (maintenance4_support) {
            device->max_میانگیر_اندازه = props4.max_میانگیر_اندازه_2;
        } وإلا {
            device->max_میانگیر_اندازه = device->max_حافظه_تخصیص_اندازه;
        }

        ثابت حرف* GGML_VK_SUBALLOCATION_بلوک_اندازه = getenv("GGML_VK_SUBALLOCATION_BLOCK_SIZE");

        إذا (GGML_VK_SUBALLOCATION_بلوک_اندازه != اشاره_تهی) {
            device->suballocation_بلوک_اندازه = std::stoull(GGML_VK_SUBALLOCATION_بلوک_اندازه);
        } وإلا {
            // Limit batching of allocations to 1GB by default to avoid fragmentation issues
            device->suballocation_بلوک_اندازه = 1024*1024*1024;
        }
        device->suballocation_بلوک_اندازه = std::min(device->suballocation_بلوک_اندازه, device->max_حافظه_تخصیص_اندازه);

        device->subgroup_اندازه = subgroup_props.subgroup_اندازه_2;
        device->subgroup_اندازه_log2 = uint32_t(log2f(عائم(device->subgroup_اندازه)));
        device->uma = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
        إذا (sm_builtins) {
            device->shader_core_شمار = sm_props.shaderSMCount;
        } وإلا إذا (amd_shader_core_properties2) {
            device->shader_core_شمار = amd_shader_core_properties2_props.فعال_Compute_Unit_شمار;
        } وإلا إذا (device->vendor_id == VK_VENDOR_ID_INTEL) {
            device->shader_core_شمار = ggml_vk_intel_shader_core_شمار(device->physical_device);
        } وإلا {
            device->shader_core_شمار = 0;
        }
        device->float_controls_rte_fp16 = vk12_props.shader_Rounding_نما_RTEFloat16;
        device->float_controls_denorm_preserve_fp16 = vk12_props.shaderDenormPreserveFloat16;

        device->subgroup_basic = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_بیتها::eCompute) &&
                                 (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_بیتها::eBasic);
        device->subgroup_arithmetic = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_بیتها::eCompute) &&
                                      (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_بیتها::eArithmetic);
#اگر_تعریف_شده __APPLE__
        // Workaround for subgroup arithmetic failing on MoltenVK with AMD GPUs (issue 15846)
        إذا (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_arithmetic = خطأ;
        }
#پایان_اگر
        device->subgroup_shuffle = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_بیتها::eCompute) &&
                                   (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_بیتها::eShuffle);
#اگر_تعریف_شده __APPLE__
        إذا (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_shuffle = خطأ;
        }
#پایان_اگر
        device->subgroup_clustered = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_بیتها::eCompute) &&
                                     (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_بیتها::eClustered);

        device->subgroup_ballot = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_بیتها::eCompute) &&
                                  (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_بیتها::eBallot);

        device->subgroup_vote = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_بیتها::eCompute) &&
                                (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_بیتها::eVote);

        // Submit at least every 100 nodes, in case there are workloads without as much matmul.
        device->max_nodes_در_هر_submit = 100;
        ثابت حرف* GGML_VK_MAX_NODES_در_هر_SUBMIT = getenv("GGML_VK_MAX_NODES_PER_SUBMIT");
        إذا (GGML_VK_MAX_NODES_در_هر_SUBMIT != اشاره_تهی) {
            uint32_t max_nodes_در_هر_submit = std::stoul(GGML_VK_MAX_NODES_در_هر_SUBMIT);
            device->max_nodes_در_هر_submit = std::max(max_nodes_در_هر_submit, 1u);
        }

        ثابت منطقي force_disable_f16 = getenv("GGML_VK_DISABLE_F16") != اشاره_تهی;

        device->fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

        إذا (!ggml_vk_khr_cooperative_ماتریس_support(device->properties, driver_props, device->architecture)) {
            device->coopmat_support = خطأ;
        }

        device->integer_dot_حاصلضرب = device->integer_dot_حاصلضرب && shader_integer_dot_حاصلضرب_props.integer_Dot_Product4x8_بیت_Packed_Signed_Accelerated;

        device->min_imported_host_اشاره_گر_alignment = external_حافظه_host_props.min_Imported_Host_اشاره_گر_Alignment;

        device->max_workgroup_اندازه_log2 = uint32_t(log2f(عائم(device->properties.limits.max_Compute_Work_گروه_Invocations)));

        std::vector<vk::صف_Family_Properties> صف_family_props = device->physical_device.get_صف_Family_Properties();

        // Try to find a non-graphics compute queue and transfer-focused queues
        // Allow overriding avoiding the graphics queue because it can increase performance on RADV
        ثابت منطقي allow_graphics_صف = (getenv("GGML_VK_ALLOW_GRAPHICS_QUEUE") != اشاره_تهی);
        ثابت vk::صف_Flag_بیتها graphics_flag = allow_graphics_صف ? (vk::صف_Flag_بیتها)0 : vk::صف_Flag_بیتها::eGraphics;
        ثابت uint32_t compute_صف_family_شاخص = ggml_vk_پیدا_کن_صف_family_شاخص(صف_family_props, vk::صف_Flag_بیتها::eCompute, graphics_flag, -1, 1);
        ثابت uint32_t transfer_صف_family_شاخص = ggml_vk_پیدا_کن_صف_family_شاخص(صف_family_props, vk::صف_Flag_بیتها::eTransfer, vk::صف_Flag_بیتها::eCompute | graphics_flag, compute_صف_family_شاخص, 1);

        ثابت عائم priorities[] = { 1.0f, 1.0f };
        device->single_صف = compute_صف_family_شاخص == transfer_صف_family_شاخص && صف_family_props[compute_صف_family_شاخص].صف_شمار == 1;

        std::vector<vk::Device_صف_Create_Info> device_صف_create_infos;
        vk::DeviceCreateInfo device_create_info{};
        std::vector<ثابت حرف *> device_extensions;
        vk::PhysicalDeviceFeatures device_features = device->physical_device.getFeatures();

        VkPhysicalDeviceFeatures2 device_features2;
        device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        device_features2.p_بعدی = اشاره_تهی;
        device_features2.features = (VkPhysicalDeviceFeatures)device_features;

        VkPhysicalDeviceVulkan11Features vk11_features;
        vk11_features.p_بعدی = اشاره_تهی;
        vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        device_features2.p_بعدی = &vk11_features;

        VkPhysicalDeviceVulkan12Features vk12_features;
        vk12_features.p_بعدی = اشاره_تهی;
        vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk11_features.p_بعدی = &vk12_features;

        آخرین_struct = (Vk_پایه_بیرون_Structure *)&vk12_features;

        VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR internally_synchronized_queues_features{};
        internally_synchronized_queues_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR;
        internally_synchronized_queues_features.p_بعدی = اشاره_تهی;
        internally_synchronized_queues_features.internallySynchronizedQueues = VK_FALSE;

        إذا (internally_sync_support) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&internally_synchronized_queues_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&internally_synchronized_queues_features;
            device_extensions.push_back(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME);
        }

        VkPhysicalDevicePipelineRobustnessFeaturesEXT pl_robustness_features;
        pl_robustness_features.p_بعدی = اشاره_تهی;
        pl_robustness_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
        pl_robustness_features.pipelineRobustness = VK_FALSE;

        إذا (pipeline_robustness) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&pl_robustness_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&pl_robustness_features;
            device_extensions.push_back("VK_EXT_pipeline_robustness");
        }

        Vk_Physical_Device_حافظه_اولویت_Features_EXT حافظه_اولویت_features;
        حافظه_اولویت_features.p_بعدی = اشاره_تهی;
        حافظه_اولویت_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_حافظه_اولویت_FEATURES_EXT;
        حافظه_اولویت_features.حافظه_اولویت_2 = VK_FALSE;
        إذا (device->حافظه_اولویت) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&حافظه_اولویت_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&حافظه_اولویت_features;
            device_extensions.push_back("VK_EXT_memory_priority");
        }

        Vk_Physical_Device_Subgroup_اندازه_Control_Features_EXT subgroup_اندازه_control_features;
        subgroup_اندازه_control_features.p_بعدی = اشاره_تهی;
        subgroup_اندازه_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_اندازه_CONTROL_FEATURES_EXT;
        subgroup_اندازه_control_features.compute_پر_Subgroups = خطأ;
        subgroup_اندازه_control_features.subgroup_اندازه_Control = خطأ;

        إذا (device->subgroup_اندازه_control) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&subgroup_اندازه_control_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&subgroup_اندازه_control_features;
        }

#إذا defined(VK_KHR_cooperative_ماتریس)
        Vk_Physical_Device_Cooperative_ماتریس_Features_KHR coopmat_features;
        coopmat_features.p_بعدی = اشاره_تهی;
        coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ماتریس_FEATURES_KHR;
        coopmat_features.cooperative_ماتریس_2 = VK_FALSE;

        إذا (device->coopmat_support) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&coopmat_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&coopmat_features;
        }
#پایان_اگر

#إذا defined(VK_NV_cooperative_matrix2)
        VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
        coopmat2_features.p_بعدی = اشاره_تهی;
        coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ماتریس_2_FEATURES_NV;
        إذا (coopmat2_support) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&coopmat2_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&coopmat2_features;
            device_extensions.push_back("VK_NV_cooperative_matrix2");
        }
#پایان_اگر

        Vk_Physical_Device_Cooperative_ماتریس_Decode_وکتور_Features_NV coopmat2_decode_وکتور_features {};
        coopmat2_decode_وکتور_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ماتریس_DECODE_وکتور_FEATURES_NV;
        إذا (coopmat2_decode_وکتور_support) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&coopmat2_decode_وکتور_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&coopmat2_decode_وکتور_features;
            device_extensions.push_back(VK_NV_COOPERATIVE_ماتریس_DECODE_وکتور_EXTENSION_NAME);
        }

#إذا defined(VK_KHR_shader_bfloat16)
        VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
        bfloat16_features.p_بعدی = اشاره_تهی;
        bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
        إذا (bfloat16_support) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&bfloat16_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&bfloat16_features;
            device_extensions.push_back("VK_KHR_shader_bfloat16");
        }
#پایان_اگر

        VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
        ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
        إذا (ocp_microscaling_extension) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&ocp_microscaling_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&ocp_microscaling_features;
            device_extensions.push_back(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME);
        }

        VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
        shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
        إذا (shader_float8_extension) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&shader_float8_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&shader_float8_features;
            device_extensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        }

        VkPhysicalDeviceMaintenance4Features maint4_features {};
        maint4_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
        إذا (maintenance4_support) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&maint4_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&maint4_features;
            device_extensions.push_back("VK_KHR_maintenance4");
        }

        Vk_Physical_Device_Shader_Integer_Dot_حاصلضرب_Features_KHR shader_integer_dot_حاصلضرب_features {};
        shader_integer_dot_حاصلضرب_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_حاصلضرب_FEATURES_KHR;
        إذا (device->integer_dot_حاصلضرب) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&shader_integer_dot_حاصلضرب_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&shader_integer_dot_حاصلضرب_features;
            device_extensions.push_back("VK_KHR_shader_integer_dot_product");
        }

        Vk_Physical_Device_Shader_Mixed_Float_Dot_حاصلضرب_Features_VALVE dot2_features {};
        dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_حاصلضرب_FEATURES_VALVE;
        إذا (dot2_f16_support) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&dot2_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&dot2_features;
            device_extensions.push_back("VK_VALVE_shader_mixed_float_dot_product");
        }

        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pep_features {};
        pep_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        إذا (pipeline_executable_properties_support) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&pep_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&pep_features;
            device_extensions.push_back("VK_KHR_pipeline_executable_properties");
        }

        إذا (device->external_حافظه_host) {
            device_extensions.push_back("VK_EXT_external_memory_host");
        }

#إذا defined(VK_EXT_shader_64bit_indexing)
        Vk_Physical_Device_Shader64_بیت_Indexing_Features_EXT shader_64bit_indexing_features {};
        shader_64bit_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_64_بیت_INDEXING_FEATURES_EXT;
        إذا (device->shader_64b_indexing) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&shader_64bit_indexing_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&shader_64bit_indexing_features;
            device_extensions.push_back("VK_EXT_shader_64bit_indexing");
        }
#پایان_اگر

        VkPhysicalDeviceFaultFeaturesEXT fault_features {};
        fault_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
        إذا (device->device_fault) {
            آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&fault_features;
            آخرین_struct = (Vk_پایه_بیرون_Structure *)&fault_features;
            device_extensions.push_back("VK_EXT_device_fault");
        }

        vkGetPhysicalDeviceFeatures2(device->physical_device, &device_features2);

        device->device_fault = device->device_fault && fault_features.deviceFault;

        device->has_internally_synchronized_queues = internally_synchronized_queues_features.internallySynchronizedQueues;

        // Build queue create infos only after querying whether internally synchronized queues are enabled.
        // getQueue2() later uses the same flag, so creation/retrieval must stay consistent.
        vk::Device_صف_Create_Flags صف_flags = device->has_internally_synchronized_queues ?
                                                eInternallySynchronizedKHR :
                                                vk::Device_صف_Create_Flags();

        إذا (compute_صف_family_شاخص != transfer_صف_family_شاخص) {
            device_صف_create_infos.push_back({صف_flags, compute_صف_family_شاخص, 1, priorities});
            device_صف_create_infos.push_back({صف_flags, transfer_صف_family_شاخص, 1, priorities + 1});
        } وإلا إذا(!device->single_صف) {
            device_صف_create_infos.push_back({صف_flags, compute_صف_family_شاخص, 2, priorities});
        } وإلا {
            device_صف_create_infos.push_back({صف_flags, compute_صف_family_شاخص, 1, priorities});
        }

        device->pipeline_executable_properties_support = pipeline_executable_properties_support;

        device->fp16 = device->fp16 && vk12_features.shaderFloat16;

#إذا defined(VK_KHR_shader_bfloat16)
        device->bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#وإلا
        device->bf16 = خطأ;
#پایان_اگر

        device->dot2_f16 = dot2_f16_support && dot2_features.shader_Mixed_Float_Dot_حاصلضرب_Float16_Acc_Float32;
        device->ocp_fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                          shader_float8_extension && shader_float8_features.shaderFloat8 &&
                          !getenv("GGML_VK_DISABLE_OCP_FP4");

        device->pipeline_robustness = pl_robustness_features.pipelineRobustness;

        device->multi_جمع_کن = vk12_props.shader_Rounding_نما_RTEFloat16 &&
                            device->properties.limits.max_افزودن_Constants_اندازه >= حجم(vk_op_multi_جمع_کن_افزودن_constants) &&
                            getenv("GGML_VK_DISABLE_MULTI_ADD") == اشاره_تهی;

        device->shader_int64 = device_features2.features.shaderInt64;
        device->میانگیر_device_آدرس = vk12_features.میانگیر_Device_آدرس;
        device->vulkan_حافظه_model = vk12_features.vulkan_حافظه_Model;

        إذا (device->subgroup_اندازه_control) {
            device->subgroup_min_اندازه = subgroup_اندازه_control_props.min_Subgroup_اندازه;
            device->subgroup_max_اندازه = subgroup_اندازه_control_props.max_Subgroup_اندازه;
            device_extensions.push_back("VK_EXT_subgroup_size_control");
        }

        device->subgroup_اندازه_control = device->subgroup_اندازه_control &&
                (subgroup_اندازه_control_props.required_Subgroup_اندازه_Stages & vk::Shader_Stage_Flag_بیتها::eCompute) &&
                subgroup_اندازه_control_features.subgroup_اندازه_Control;

        device->subgroup_require_پر_support = subgroup_اندازه_control_features.compute_پر_Subgroups;

#إذا defined(VK_KHR_cooperative_ماتریس)
        device->coopmat_support = device->coopmat_support && coopmat_features.cooperative_ماتریس_2;
        device->coopmat1_fa_support = device->coopmat_support && device->subgroup_require_پر_support;
#پایان_اگر

        إذا (coopmat2_support) {
#إذا defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            إذا (coopmat2_features.cooperative_ماتریس_Workgroup_Scope &&
                coopmat2_features.cooperative_ماتریس_Flexible_Dimensions &&
                coopmat2_features.cooperative_ماتریس_Reductions &&
                coopmat2_features.cooperative_ماتریس_Conversions &&
                coopmat2_features.cooperative_ماتریس_در_هر_Element_Operations &&
                coopmat2_features.cooperative_ماتریس_Tensor_Addressing &&
                coopmat2_features.cooperative_ماتریس_بلوک_Loads &&
                vk12_features.میانگیر_Device_آدرس) {

                std::vector<Vk_Cooperative_ماتریس_Flexible_Dimensions_Properties_NV> flexible_dimensions;
                uint32_t count = 0;

                PFN_vk_Get_Physical_Device_Cooperative_ماتریس_Flexible_Dimensions_Properties_NV
                    vk_Get_Physical_Device_Cooperative_ماتریس_Flexible_Dimensions_Properties_NV =
                        (PFN_vk_Get_Physical_Device_Cooperative_ماتریس_Flexible_Dimensions_Properties_NV)
                        vk_instance.instance.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");

                vk_Get_Physical_Device_Cooperative_ماتریس_Flexible_Dimensions_Properties_NV(device->physical_device, &count, اشاره_تهی);

                Vk_Cooperative_ماتریس_Flexible_Dimensions_Properties_NV خالی_prop {};
                خالی_prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_ماتریس_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
                flexible_dimensions.resize(count, خالی_prop);

                vk_Get_Physical_Device_Cooperative_ماتریس_Flexible_Dimensions_Properties_NV(device->physical_device, &count, flexible_dimensions.داده());

                منطقي یافت_fp16_128 = خطأ,
                     یافت_fp16_256 = خطأ,
                     یافت_fp32_128 = خطأ,
                     یافت_fp32_256 = خطأ;
                منطقي یافت_bf16_128 = خطأ,
                     یافت_bf16_256 = خطأ;
                // need to support fp16*fp16 with fp16/fp32 accumulator, for workgroupsize 128
                // with 32x16x16 and 256 with 32x32x16.
                لكل (خودکار &prop : flexible_dimensions) {
                    إذا (prop.saturatingAccumulation == VK_FALSE &&
                        prop.scope == VK_SCOPE_WORKGROUP_KHR) {

                        إذا (prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR) {

                            إذا (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                إذا (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.نتیجه_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    یافت_fp16_128 = درست;
                                }
                                إذا (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.نتیجه_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    یافت_fp32_128 = درست;
                                }
                            }
                            إذا (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                إذا (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.نتیجه_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    یافت_fp16_256 = درست;
                                }
                                إذا (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.نتیجه_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    یافت_fp32_256 = درست;
                                }
                            }
                        }

#إذا defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                        إذا (bfloat16_support &&
                            prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            prop.نتیجه_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {

                            إذا (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                یافت_bf16_128 = درست;
                            }
                            إذا (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                یافت_bf16_256 = درست;
                            }
                        }
#پایان_اگر
                    }
                }
                إذا (یافت_fp16_128 && یافت_fp16_256 &&
                    یافت_fp32_128 && یافت_fp32_256 &&
                    coopmat2_props.cooperative_ماتریس_Flexible_Dimensions_Max_Dimension >= 512) {
                    device->coopmat2 = درست;
                    device->coopmat2_bf16_support = یافت_bf16_128 && یافت_bf16_256;
                    device->coopmat2_decode_وکتور = coopmat2_decode_وکتور_support && coopmat2_decode_وکتور_features.cooperative_ماتریس_Decode_وکتور;
                }
            }
#پایان_اگر
        }

        إذا (!vk11_features.storage_Buffer16_بیت_Access) {
            std::cerr << "ggml_vulkan: device " << GGML_VK_NAME << idx << " does not support 16-bit storage." << std::پایان_سطر;
            ارم std::runtime_error("Unsupported device");
        }

        device_extensions.push_back("VK_KHR_16bit_storage");

#اگر_تعریف_شده GGML_VULKAN_VALIDATE
        device_extensions.push_back("VK_KHR_shader_non_semantic_info");
#پایان_اگر

        إذا (device->fp16) {
            device_extensions.push_back("VK_KHR_shader_float16_int8");
        }

#إذا defined(VK_KHR_cooperative_ماتریس)
        إذا (device->coopmat_support) {
            // Query supported shapes
            std::vector<Vk_Cooperative_ماتریس_Properties_KHR> cm_props;

            PFN_vk_Get_Physical_Device_Cooperative_ماتریس_Properties_KHR pfn_vk_Get_Physical_Device_Cooperative_ماتریس_Properties_KHR =
                (PFN_vk_Get_Physical_Device_Cooperative_ماتریس_Properties_KHR)vkGetInstanceProcAddr(vk_instance.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

            uint32_t cm_props_رقم;

            pfn_vk_Get_Physical_Device_Cooperative_ماتریس_Properties_KHR(device->physical_device, &cm_props_رقم, اشاره_تهی);

            cm_props.resize(cm_props_رقم);

            لكل (خودکار& prop : cm_props) {
                prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_ماتریس_PROPERTIES_KHR;
            }

            pfn_vk_Get_Physical_Device_Cooperative_ماتریس_Properties_KHR(device->physical_device, &cm_props_رقم, cm_props.داده());

            VK_LOG_DEBUG("ggml_vulkan: Cooperative Matrix Shapes: " << cm_props.size());

            لكل (خودکار& prop : cm_props) {
                VK_LOG_DEBUG("ggml_vulkan: M: " << prop.MSize << " N: " << prop.NSize << " K: " << prop.KSize << " A: " << vk::to_string((vk::ComponentTypeKHR)prop.AType) << " B: " << vk::to_string((vk::ComponentTypeKHR)prop.BType) << " C: " << vk::to_string((vk::ComponentTypeKHR)prop.CType) << " Result: " << vk::to_string((vk::ComponentTypeKHR)prop.نتیجه_Type) << " saturatingAccumulation: " << prop.saturatingAccumulation << " scope: " << vk::to_string((vk::ScopeKHR)prop.scope));

                إذا ((vk::ComponentTypeKHR)prop.AType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ComponentTypeKHR)prop.BType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    إذا ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat32 &&
                        (vk::ComponentTypeKHR)prop.نتیجه_Type == vk::ComponentTypeKHR::eFloat32) {
                        // coopmat sizes not set yet
                        إذا (device->coopmat_m == 0) {
                            device->coopmat_acc_f32_support = درست;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } وإلا إذا (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f32_support = درست;
                        }
                        إذا (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f32acc = درست;
                        }
                    } وإلا إذا ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat16 &&
                               (vk::ComponentTypeKHR)prop.نتیجه_Type == vk::ComponentTypeKHR::eFloat16) {
                        // coopmat sizes not set yet
                        إذا (device->coopmat_m == 0) {
                            device->coopmat_acc_f16_support = درست;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } وإلا إذا (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f16_support = درست;
                        }
                        إذا (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f16acc = درست;
                        }
                    }
                } وإلا إذا ((vk::ComponentTypeKHR)prop.AType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.BType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.CType      == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ComponentTypeKHR)prop.نتیجه_Type == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup &&
                           device->coopmat_int_m == 0
                ) {
                    device->coopmat_int_support = درست;
                    device->coopmat_int_m = prop.MSize;
                    device->coopmat_int_n = prop.NSize;
                    device->coopmat_int_k = prop.KSize;
                }
#إذا defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                إذا (bfloat16_support &&
                    prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    prop.نتیجه_Type == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    // coopmat sizes not set yet
                    إذا (device->coopmat_m == 0) {
                        device->coopmat_bf16_support = درست;
                        device->coopmat_m = prop.MSize;
                        device->coopmat_n = prop.NSize;
                        device->coopmat_k = prop.KSize;
                    } وإلا إذا (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                        // Only enable if shape is identical
                        device->coopmat_bf16_support = درست;
                    }
                }
#پایان_اگر
            }

            إذا (device->coopmat_m == 0 || !device->coopmat_acc_f32_support) {
                // No suitable matmul mode found
                GGML_LOG_DEBUG("ggml_vulkan: WARNING: No suitable matrix core mode found. Disabling matrix cores.\n");
                device->coopmat_support = خطأ;
            }
        }

        إذا (device->coopmat_support) {
            device_extensions.push_back("VK_KHR_cooperative_matrix");
        }
#پایان_اگر
        device->name = GGML_VK_NAME + std::to_string(idx);

        device_create_info
            .مجموعه_Flags(vk::DeviceCreateFlags())
            .مجموعه_صف_Create_Infos(device_صف_create_infos)
            .مجموعه_PEnabled_Extension_نامها(device_extensions);
        device_create_info.مجموعه_PNext(&device_features2);
        device->device = device->physical_device.createDevice(device_create_info);

        إذا (device->device_fault) {
            device->pfn_vkGetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)
                vkGetDeviceProcAddr(device->device, "vkGetDeviceFaultInfoEXT");
        }

        // Queues
        device->compute_صف = ggml_vk_create_صف(device, compute_صف_family_شاخص, 0, { vk::Pipeline_Stage_Flag_بیتها::eComputeShader | vk::Pipeline_Stage_Flag_بیتها::eTransfer }, خطأ);

        // Shaders
        // Disable matmul tile sizes early if performance low or not supported
        لكل (uint32_t i = 0; i < GGML_TYPE_شمار; ++i) {
            بدل (device->vendor_id) {
#اگر_تعریف_نشده GGML_VULKAN_پیاپی_TESTS
            حالة VK_VENDOR_ID_AMD:
                device->mul_mat_l[i]    = device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary;
                device->mul_mat_m[i]    = درست;
                device->mul_mat_s[i]    = درست;
                device->mul_mat_id_l[i] = خطأ;
                device->mul_mat_id_m[i] = درست;
                device->mul_mat_id_s[i] = درست;
                اكسر;
            حالة VK_VENDOR_ID_INTEL: {
                // Current Windows driver does not expose BF16 support.
                // We only want to use l_warptile if coopmat is available
                ثابت منطقي use_l_warptile = (i == GGML_TYPE_BF16) ? (device->coopmat_bf16_support && device->coopmat_support) : device->coopmat_support;
                device->mul_mat_l[i] = use_l_warptile;
                device->mul_mat_id_l[i] = use_l_warptile;
                device->mul_mat_m[i] = درست;
                device->mul_mat_s[i] = درست;
                device->mul_mat_id_m[i] = درست;
                device->mul_mat_id_s[i] = درست;
                اكسر;
            }
            حالة VK_VENDOR_ID_APPLE:
                device->mul_mat_l[i] = خطأ;
                device->mul_mat_m[i] = درست;
                device->mul_mat_s[i] = خطأ;
                device->mul_mat_id_l[i] = خطأ;
                device->mul_mat_id_m[i] = درست;
                device->mul_mat_id_s[i] = خطأ;
                اكسر;
            حالة VK_VENDOR_ID_QUALCOMM:
                device->mul_mat_l[i] = خطأ;
                device->mul_mat_m[i] = درست;
                device->mul_mat_s[i] = !device->coopmat_support;
                device->mul_mat_id_l[i] = خطأ;
                device->mul_mat_id_m[i] = درست;
                device->mul_mat_id_s[i] = !device->coopmat_support;
                اكسر;
            حالة VK_VENDOR_ID_SAMSUNG:
                device->mul_mat_l[i] = خطأ;
                device->mul_mat_m[i] = درست;
                device->mul_mat_s[i] = درست;
                device->mul_mat_id_l[i] = خطأ;
                device->mul_mat_id_m[i] = درست;
                device->mul_mat_id_s[i] = درست;
                اكسر;
#پایان_اگر
            افتراضي:
                device->mul_mat_l[i] = درست;
                device->mul_mat_m[i] = درست;
                device->mul_mat_s[i] = درست;
                device->mul_mat_id_l[i] = درست;
                device->mul_mat_id_m[i] = درست;
                device->mul_mat_id_s[i] = درست;
                اكسر;
            }

#إذا VK_سرآیند_نسخه >= 287
            // Honeykrisp driver for Asahi Linux doesn't report VK_VENDOR_ID_APPLE.
            // Check for Honeykrisp driver and force same configuration as the VK_VENDOR_ID_APPLE case.
            إذا (device->driver_id == vk::DriverId::eMesaHoneykrisp) {
                device->mul_mat_l[i] = خطأ;
                device->mul_mat_m[i] = درست;
                device->mul_mat_s[i] = خطأ;
                device->mul_mat_id_l[i] = خطأ;
                device->mul_mat_id_m[i] = درست;
                device->mul_mat_id_s[i] = خطأ;
            }
#پایان_اگر

            device->mul_mat_l_int[i]    = device->mul_mat_l[i];
            device->mul_mat_m_int[i]    = device->mul_mat_m[i];
            device->mul_mat_s_int[i]    = device->mul_mat_s[i];
            device->mul_mat_id_l_int[i] = device->mul_mat_id_l[i];
            device->mul_mat_id_m_int[i] = device->mul_mat_id_m[i];
            device->mul_mat_id_s_int[i] = device->mul_mat_id_s[i];
        }


        std::vector<vk::Descriptor_مجموعه_Layout_Binding> dsl_binding;
        std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
        لكل (uint32_t i = 0; i < MAX_PARAMETER_شمار; i++) {
            dsl_binding.push_back({i, vk::DescriptorType::e_Storage_میانگیر, 1, vk::Shader_Stage_Flag_بیتها::eCompute});
            dsl_binding_flags.push_back({});
        }

        vk::Descriptor_مجموعه_Layout_Binding_Flags_Create_Info dslbfci = { dsl_binding_flags };

        vk::Descriptor_مجموعه_Layout_Create_Info descriptor_مجموعه_layout_create_info(
            {},
            dsl_binding);
        descriptor_مجموعه_layout_create_info.مجموعه_PNext(&dslbfci);
        device->dsl = device->device.create_Descriptor_مجموعه_Layout(descriptor_مجموعه_layout_create_info);

        ggml_vk_بارگیری_shaders(device);

        // Prefer a dedicated transfer queue on AMD dGPUs (non-GCN) when graphics queue use is disabled.
        ثابت منطقي prefers_transfer_صف =
            device->vendor_id == VK_VENDOR_ID_AMD &&
            device->architecture != AMD_GCN &&
            !device->uma &&
            !allow_graphics_صف;

        إذا (!device->single_صف) {
            ثابت uint32_t transfer_صف_شاخص = compute_صف_family_شاخص == transfer_صف_family_شاخص ? 1 : 0;
            device->transfer_صف = ggml_vk_create_صف(device, transfer_صف_family_شاخص, transfer_صف_شاخص, { vk::Pipeline_Stage_Flag_بیتها::eTransfer }, درست);

            device->async_use_transfer_صف = prefers_transfer_صف || (getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") != اشاره_تهی);
        } وإلا {
            device->transfer_صف = ggml_vk_create_aliased_صف(device, device->compute_صف);

            device->async_use_transfer_صف = خطأ;
        }

        device->fence = device->device.createFence({});

        device->میانگیر_type = {
            /* .iface    = */ ggml_backend_vk_میانگیر_type_interface,
            /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), idx),
            /* .context  = */ جديد ggml_backend_vk_میانگیر_type_context{ device->name, device },
        };

        device->idx = idx;

        device->serialize_submissions = getenv("GGML_VK_SERIALIZE_SUBMISSIONS") != اشاره_تهی;

        device->disable_fusion = getenv("GGML_VK_DISABLE_FUSION") != اشاره_تهی;

        device->disable_descriptor_reuse = getenv("GGML_VK_DISABLE_DESCRIPTOR_REUSE") != اشاره_تهی;

        device->جمع_کن_rms_fusion = !device->disable_fusion &&
                                 device->subgroup_arithmetic &&
                                 device->vendor_id != VK_VENDOR_ID_INTEL;
        device->partials_binding_alignment =
            std::max(4u, (uint32_t)device->properties.limits.min_Storage_میانگیر_جابه_جایی_Alignment);

        device->mmvq_نما = 0;
        إذا (getenv("GGML_VK_DISABLE_MMVQ")) {
            device->mmvq_نما = -1;
        } وإلا إذا (getenv("GGML_VK_FORCE_MMVQ")) {
            device->mmvq_نما = 1;
        }

        // publish the device only when it is fully built
        vk_instance.devices[idx] = device;

        أرجع device;
    }

    أرجع vk_instance.devices[idx];
}

ساكن فراغ ggml_vk_print_gpu_info(size_t idx) {
    GGML_ASSERT(idx < vk_instance.device_indices.size());
    size_t dev_رقم = vk_instance.device_indices[idx];
    VK_LOG_DEBUG("ggml_vk_print_gpu_info(" << dev_رقم << ")");
    GGML_ASSERT(vk_instance_initialized);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    إذا (dev_رقم >= devices.size()) {
        std::cerr << "ggml_vulkan: Device with index " << dev_رقم << " does not exist." << std::پایان_سطر;
        ارم std::runtime_error("Device not found");
    }

    vk::PhysicalDevice physical_device = devices[dev_رقم];
    std::vector<vk::ExtensionProperties> ext_props = physical_device.enumerateDeviceExtensionProperties();

    منطقي fp16_storage = خطأ;
    منطقي fp16_compute = خطأ;
    منطقي coopmat_support = خطأ;
    منطقي coopmat2_support = خطأ;
    منطقي coopmat2_decode_وکتور_support = خطأ;
    منطقي integer_dot_حاصلضرب = خطأ;
    منطقي bfloat16_support = خطأ;
    منطقي dot2_f16_support = خطأ;
    منطقي ocp_microscaling_extension = خطأ;
    منطقي shader_float8_extension = خطأ;

    لكل (خودکار properties : ext_props) {
        إذا (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
            fp16_storage = درست;
        } وإلا إذا (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
            fp16_compute = درست;
#إذا defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
       } وإلا إذا (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT")) {
            coopmat_support = درست;
#پایان_اگر
#إذا defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        } وإلا إذا (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2")) {
            coopmat2_support = درست;
#پایان_اگر
        } وإلا إذا (strcmp(VK_NV_COOPERATIVE_ماتریس_DECODE_وکتور_EXTENSION_NAME, properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
            coopmat2_decode_وکتور_support = درست;
#إذا defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        } وإلا إذا (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
            integer_dot_حاصلضرب = درست;
#پایان_اگر
#إذا defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        } وإلا إذا (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_BFLOAT16")) {
            bfloat16_support = درست;
#پایان_اگر
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
        } وإلا إذا (strcmp(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
            ocp_microscaling_extension = درست;
#پایان_اگر
#إذا defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        } وإلا إذا (strcmp(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
            shader_float8_extension = درست;
#پایان_اگر
        } وإلا إذا (strcmp("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_DOT2")) {
            dot2_f16_support = درست;
        }
    }

    ثابت vk_device_architecture device_architecture = get_device_architecture(physical_device);

    ثابت حرف* GGML_VK_DISABLE_F16 = getenv("GGML_VK_DISABLE_F16");
    منطقي force_disable_f16 = GGML_VK_DISABLE_F16 != اشاره_تهی;

    منطقي fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceMaintenance3Properties props3;
    vk::PhysicalDeviceSubgroupProperties subgroup_props;
    vk::PhysicalDeviceDriverProperties driver_props;
    vk::Physical_Device_Shader_Integer_Dot_حاصلضرب_Properties_KHR shader_integer_dot_حاصلضرب_props;
    props2.p_بعدی = &props3;
    props3.p_بعدی = &subgroup_props;
    subgroup_props.p_بعدی = &driver_props;

    // Pointer to the last chain element
    Vk_پایه_بیرون_Structure * آخرین_struct = (Vk_پایه_بیرون_Structure *)&driver_props;

    إذا (integer_dot_حاصلضرب) {
        آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&shader_integer_dot_حاصلضرب_props;
        آخرین_struct = (Vk_پایه_بیرون_Structure *)&shader_integer_dot_حاصلضرب_props;
    }

    physical_device.getProperties2(&props2);

    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.p_بعدی = اشاره_تهی;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_بعدی = اشاره_تهی;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_بعدی = &vk11_features;

    VkPhysicalDeviceVulkan12Features vk12_features;
    vk12_features.p_بعدی = اشاره_تهی;
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11_features.p_بعدی = &vk12_features;

    // Pointer to the last chain element
    آخرین_struct = (Vk_پایه_بیرون_Structure *)&vk12_features;

#إذا defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    Vk_Physical_Device_Cooperative_ماتریس_Features_KHR coopmat_features;
    coopmat_features.p_بعدی = اشاره_تهی;
    coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ماتریس_FEATURES_KHR;
    coopmat_features.cooperative_ماتریس_2 = VK_FALSE;

    إذا (coopmat_support) {
        آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&coopmat_features;
        آخرین_struct = (Vk_پایه_بیرون_Structure *)&coopmat_features;
    }
#پایان_اگر

    Vk_Physical_Device_Shader_Integer_Dot_حاصلضرب_Features_KHR shader_integer_dot_حاصلضرب_features {};
    shader_integer_dot_حاصلضرب_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_حاصلضرب_FEATURES_KHR;
    إذا (integer_dot_حاصلضرب) {
        آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&shader_integer_dot_حاصلضرب_features;
        آخرین_struct = (Vk_پایه_بیرون_Structure *)&shader_integer_dot_حاصلضرب_features;
    }

#إذا defined(VK_KHR_shader_bfloat16)
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
    bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    إذا (bfloat16_support) {
        آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&bfloat16_features;
        آخرین_struct = (Vk_پایه_بیرون_Structure *)&bfloat16_features;
    }
#پایان_اگر

#إذا defined(VK_NV_cooperative_matrix2)
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
    coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ماتریس_2_FEATURES_NV;
    إذا (coopmat2_support) {
        آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&coopmat2_features;
        آخرین_struct = (Vk_پایه_بیرون_Structure *)&coopmat2_features;
    }
#پایان_اگر

    Vk_Physical_Device_Cooperative_ماتریس_Decode_وکتور_Features_NV coopmat2_decode_وکتور_features {};
    coopmat2_decode_وکتور_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ماتریس_DECODE_وکتور_FEATURES_NV;
    إذا (coopmat2_decode_وکتور_support) {
        آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&coopmat2_decode_وکتور_features;
        آخرین_struct = (Vk_پایه_بیرون_Structure *)&coopmat2_decode_وکتور_features;
    }

    Vk_Physical_Device_Shader_Mixed_Float_Dot_حاصلضرب_Features_VALVE dot2_features {};
    dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_حاصلضرب_FEATURES_VALVE;
    إذا (dot2_f16_support) {
        آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&dot2_features;
        آخرین_struct = (Vk_پایه_بیرون_Structure *)&dot2_features;
    }

#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
    ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
    VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
    shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    إذا (ocp_microscaling_extension) {
        آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&ocp_microscaling_features;
        آخرین_struct = (Vk_پایه_بیرون_Structure *)&ocp_microscaling_features;
    }
    إذا (shader_float8_extension) {
        آخرین_struct->p_بعدی = (Vk_پایه_بیرون_Structure *)&shader_float8_features;
        آخرین_struct = (Vk_پایه_بیرون_Structure *)&shader_float8_features;
    }
#پایان_اگر

    vkGetPhysicalDeviceFeatures2(physical_device, &device_features2);

    fp16 = fp16 && vk12_features.shaderFloat16;

#إذا defined(VK_KHR_shader_bfloat16)
    منطقي bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#وإلا
    منطقي bf16 = خطأ;
#پایان_اگر

    uint32_t default_subgroup_اندازه = get_subgroup_اندازه("", device_architecture);
    ثابت size_t subgroup_اندازه = (default_subgroup_اندازه != 0) ? default_subgroup_اندازه : subgroup_props.subgroup_اندازه_2;
    ثابت منطقي uma = props2.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    integer_dot_حاصلضرب = integer_dot_حاصلضرب
                       && shader_integer_dot_حاصلضرب_props.integer_Dot_Product4x8_بیت_Packed_Signed_Accelerated
                       && shader_integer_dot_حاصلضرب_features.shader_Integer_Dot_حاصلضرب;

    coopmat_support = coopmat_support
#إذا defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                   && coopmat_features.cooperative_ماتریس_2
#پایان_اگر
                   && ggml_vk_khr_cooperative_ماتریس_support(props2.properties, driver_props, device_architecture);

#إذا defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    coopmat2_support = coopmat2_support &&
                       coopmat2_features.cooperative_ماتریس_Workgroup_Scope &&
                       coopmat2_features.cooperative_ماتریس_Flexible_Dimensions &&
                       coopmat2_features.cooperative_ماتریس_Reductions &&
                       coopmat2_features.cooperative_ماتریس_Conversions &&
                       coopmat2_features.cooperative_ماتریس_در_هر_Element_Operations &&
                       coopmat2_features.cooperative_ماتریس_Tensor_Addressing &&
                       coopmat2_features.cooperative_ماتریس_بلوک_Loads;
#وإلا
    coopmat2_support = خطأ;
#پایان_اگر

    coopmat2_decode_وکتور_support = coopmat2_decode_وکتور_support && coopmat2_decode_وکتور_features.cooperative_ماتریس_Decode_وکتور;
#إذا !defined(GGML_VULKAN_COOPMAT2_DECODE_وکتور_GLSLC_SUPPORT)
    coopmat2_decode_وکتور_support = خطأ;
#پایان_اگر

    std::string ماتریس_cores = coopmat2_support ? (coopmat2_decode_وکتور_support ? "NV_coopmat2v" : "NV_coopmat2")
                             : coopmat_support  ? "KHR_coopmat"
                             : "none";

    منطقي dot2_f16 = dot2_f16_support && dot2_features.shader_Mixed_Float_Dot_حاصلضرب_Float16_Acc_Float32;
    ثابت حرف *fp16_str = fp16 ? (dot2_f16 ? "dot2" : "1") : "0";
#إذا defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    ثابت منطقي fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                     shader_float8_extension && shader_float8_features.shaderFloat8 &&
                     !getenv("GGML_VK_DISABLE_OCP_FP4");
#وإلا
    GGML_استفادهنشده(ocp_microscaling_extension);
    GGML_استفادهنشده(shader_float8_extension);
    ثابت منطقي fp4 = خطأ;
#پایان_اگر

    std::string device_name = props2.properties.deviceName.داده();
    GGML_LOG_DEBUG("ggml_vulkan: %zu = %s (%s) | uma: %d | fp16: %s | bf16: %d | fp4: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n",
              idx, device_name.c_str(), driver_props.driverName.داده(), uma, fp16_str, bf16, fp4, subgroup_اندازه,
              props2.properties.limits.max_Compute_Shared_حافظه_اندازه, integer_dot_حاصلضرب, ماتریس_cores.c_str());

    إذا (props2.properties.deviceType == vk::PhysicalDeviceType::eCpu) {
        GGML_LOG_DEBUG("ggml_vulkan: Warning: Device type is CPU. This is probably not the device you want.\n");
    }
}

ساكن DispatchLoaderDynamic ggml_vk_default_dispatcher_instance;

DispatchLoaderDynamic & ggml_vk_default_dispatcher() {
    أرجع ggml_vk_default_dispatcher_instance;
}

فراغ ggml_vk_instance_init() {
    إذا (vk_instance_initialized) {
        أرجع;
    }
    VK_LOG_DEBUG("ggml_vk_instance_init()");

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    ggml_vk_default_dispatcher_instance.init(vkGetInstanceProcAddr);

    // vkEnumerateInstanceVersion is Vulkan 1.1. A null value indicated Vulkan 1.0.
    إذا (ggml_vk_default_dispatcher_instance.vk_Enumerate_Instance_نسخه == اشاره_تهی) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        ارم vk::SystemError(vk::نتیجه_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    uint32_t api_نسخه = vk::enumerate_Instance_نسخه();

    إذا (api_نسخه < VK_API_نسخه_1_2) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        ارم vk::SystemError(vk::نتیجه_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, اشاره_تهی, 0, api_نسخه };

    ثابت std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();
    ثابت منطقي layer_settings = ggml_vk_instance_layer_settings_موجود();
#اگر_تعریف_شده __APPLE__
    ثابت منطقي portability_enumeration_ext = ggml_vk_instance_portability_enumeration_ext_موجود(instance_extensions);
#پایان_اگر
    ثابت منطقي debug_utils_ext = ggml_vk_instance_debug_utils_ext_موجود(instance_extensions) && getenv("GGML_VK_DEBUG_MARKERS") != اشاره_تهی;
    std::vector<ثابت حرف*> layers;

    إذا (layer_settings) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    std::vector<ثابت حرف*> extensions;
    إذا (layer_settings) {
        extensions.push_back("VK_EXT_layer_settings");
    }
#اگر_تعریف_شده __APPLE__
    إذا (portability_enumeration_ext) {
        extensions.push_back("VK_KHR_portability_enumeration");
    }
#پایان_اگر
    إذا (debug_utils_ext) {
        extensions.push_back("VK_EXT_debug_utils");
    }
    VkBool32 enable_بهترین_practice = layer_settings;
    std::vector<vk::LayerSettingEXT> settings = {
        {
            "VK_LAYER_KHRONOS_validation",
            "validate_best_practices",
            vk::LayerSettingTypeEXT::eBool32,
            1,
            &enable_بهترین_practice
        },
    };
    vk::LayerSettingsCreateInfoEXT layer_setting_info(settings);
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags{}, &app_info, layers, extensions, &layer_setting_info);
#اگر_تعریف_شده __APPLE__
    إذا (portability_enumeration_ext) {
        instance_create_info.flags |= vk::Instance_Create_Flag_بیتها::eEnumeratePortabilityKHR;
    }
#پایان_اگر

    vk_instance.instance = vk::createInstance(instance_create_info);
    vk_instance_initialized = درست;

    إذا (debug_utils_ext) {
        vk_instance.debug_utils_support              = درست;
        vk_instance.pfn_vk_مجموعه_Debug_Utils_Object_Name_EXT = (PFN_vk_مجموعه_Debug_Utils_Object_Name_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkSetDebugUtilsObjectNameEXT");
        vk_instance.pfn_vk_صف_Begin_Debug_Utils_Label_EXT = (PFN_vk_صف_Begin_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vk_صف_انتها_Debug_Utils_Label_EXT = (PFN_vk_صف_انتها_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueEndDebugUtilsLabelEXT");
        vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_انتها_Debug_Utils_Label_EXT =   (PFN_vk_Cmd_انتها_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdEndDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_درج_Debug_Utils_Label_EXT = (PFN_vk_Cmd_درج_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdInsertDebugUtilsLabelEXT");
    }

    vk_perf_logger_enabled = getenv("GGML_VK_PERF_LOGGER") != اشاره_تهی;
    vk_perf_logger_concurrent = getenv("GGML_VK_PERF_LOGGER_CONCURRENT") != اشاره_تهی;
    vk_enable_sync_logger = getenv("GGML_VK_SYNC_LOGGER") != اشاره_تهی;
    vk_حافظه_logger_enabled = getenv("GGML_VK_MEMORY_LOGGER") != اشاره_تهی;
    ثابت حرف* GGML_VK_PIPELINE_STATS = getenv("GGML_VK_PIPELINE_STATS");
    إذا (GGML_VK_PIPELINE_STATS != اشاره_تهی) {
        vk_pipeline_stats_filter = GGML_VK_PIPELINE_STATS;
    }
    ثابت حرف* GGML_VK_PERF_LOGGER_FREQUENCY = getenv("GGML_VK_PERF_LOGGER_FREQUENCY");

    إذا (GGML_VK_PERF_LOGGER_FREQUENCY != اشاره_تهی) {
        vk_perf_logger_frequency = std::stoul(GGML_VK_PERF_LOGGER_FREQUENCY);
    }

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance.instance);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    // Emulate behavior of CUDA_VISIBLE_DEVICES for Vulkan
    حرف * devices_env = getenv("GGML_VK_VISIBLE_DEVICES");
    إذا (devices_env != اشاره_تهی) {
        size_t رقم_موجود_devices = devices.size();

        std::string devices(devices_env);
        std::replace(devices.begin(), devices.end(), ',', ' ');

        std::stringstream ss(devices);
        size_t tmp;
        طالما (ss >> tmp) {
            إذا(tmp >= رقم_موجود_devices) {
                std::cerr << "ggml_vulkan: Invalid device index " << tmp << " in GGML_VK_VISIBLE_DEVICES." << std::پایان_سطر;
                ارم std::runtime_error("Invalid Vulkan device index");
            }
            vk_instance.device_indices.push_back(tmp);
        }
    } وإلا {
        // If no vulkan devices are found, return early
        إذا (devices.خالی()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            أرجع;
        }

        // Default to using all dedicated GPUs
        لكل (size_t i = 0; i < devices.size(); i++) {
            vk::PhysicalDeviceProperties2 new_props;
            vk::PhysicalDeviceDriverProperties new_driver;
            vk::PhysicalDeviceIDProperties new_id;
            new_props.p_بعدی = &new_driver;
            new_driver.p_بعدی = &new_id;
            devices[i].getProperties2(&new_props);

            إذا ((new_props.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu || new_props.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) && ggml_vk_device_is_supported(devices[i])) {
                // Check if there are two physical devices corresponding to the same GPU
                // This handles the case where the same GPU appears with different drivers (e.g., RADV + AMDVLK on Linux),
                // see https://github.com/ggml-org/llama.cpp/pull/7582 for original deduplication.
                // MoltenVK on macOS may report the same UUID for distinct GPUs on multi-GPU cards,
                // see https://github.com/KhronosGroup/MoltenVK/issues/2683. Skip when both old/new
                // driver is MoltenVK
                خودکار old_device = std::find_if(
                    vk_instance.device_indices.begin(),
                    vk_instance.device_indices.end(),
                    [&devices, &new_id, &new_driver](ثابت size_t k){
                        vk::PhysicalDeviceProperties2 old_props;
                        vk::PhysicalDeviceDriverProperties old_driver;
                        vk::PhysicalDeviceIDProperties old_id;
                        old_props.p_بعدی = &old_driver;
                        old_driver.p_بعدی = &old_id;
                        devices[k].getProperties2(&old_props);

                        منطقي same_uuid = std::equal(std::begin(old_id.deviceUUID), std::end(old_id.deviceUUID), std::begin(new_id.deviceUUID));
                        same_uuid = same_uuid || (
                            old_id.deviceLUIDValid && new_id.deviceLUIDValid &&
                            std::equal(std::begin(old_id.deviceLUID), std::end(old_id.deviceLUID), std::begin(new_id.deviceLUID))
                        );
                        منطقي both_molten_vk = (new_driver.driverID == vk::DriverId::eMoltenvk && old_driver.driverID == vk::DriverId::eMoltenvk);

                        أرجع same_uuid && !both_molten_vk;
                    }
                );
                إذا (old_device == vk_instance.device_indices.end()) {
                    vk_instance.device_indices.push_back(i);
                } وإلا {
                    // There can be two physical devices corresponding to the same GPU if there are 2 different drivers
                    // This can cause error when splitting layers aross the devices, need to keep only 1
                    VK_LOG_DEBUG("Device " << i << " and device " << *old_device << " have the same deviceUUID");

                    vk::PhysicalDeviceProperties2 old_props;
                    vk::PhysicalDeviceDriverProperties old_driver;
                    old_props.p_بعدی = &old_driver;
                    devices[*old_device].getProperties2(&old_props);

                    std::map<vk::DriverId, صحيح> driver_priorities {};
                    صحيح old_اولویت = std::numeric_limits<صحيح>::max();
                    صحيح new_اولویت = std::numeric_limits<صحيح>::max();

                    // Check https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDriverId.html for the list of driver id
                    // Smaller number -> higher priority
                    بدل (old_props.properties.vendorID) {
                        حالة VK_VENDOR_ID_AMD:
                            driver_priorities[vk::DriverId::eMesaRadv] = 1;
                            driver_priorities[vk::DriverId::e_Amd_Open_منبع] = 2;
                            driver_priorities[vk::DriverId::eAmdProprietary] = 3;
                            اكسر;
                        حالة VK_VENDOR_ID_INTEL:
                            driver_priorities[vk::DriverId::e_Intel_Open_منبع_MESA] = 1;
                            driver_priorities[vk::DriverId::eIntelProprietaryWindows] = 2;
                            اكسر;
                        حالة VK_VENDOR_ID_NVIDIA:
                            driver_priorities[vk::DriverId::eNvidiaProprietary] = 1;
#إذا defined(VK_API_نسخه_1_3) && VK_سرآیند_نسخه >= 235
                            driver_priorities[vk::DriverId::eMesaNvk] = 2;
#پایان_اگر
                            اكسر;
                        حالة VK_VENDOR_ID_QUALCOMM:
                            driver_priorities[vk::DriverId::eQualcommProprietary] = 1;
                            driver_priorities[vk::DriverId::eMesaTurnip] = 2;
                            اكسر;
                    }
                    driver_priorities[vk::DriverId::eMesaDozen] = 100;

                    إذا (driver_priorities.count(old_driver.driverID)) {
                        old_اولویت = driver_priorities[old_driver.driverID];
                    }
                    إذا (driver_priorities.count(new_driver.driverID)) {
                        new_اولویت = driver_priorities[new_driver.driverID];
                    }

                    إذا (new_اولویت < old_اولویت) {
                        خودکار r = std::remove(vk_instance.device_indices.begin(), vk_instance.device_indices.end(), *old_device);
                        vk_instance.device_indices.erase(r, vk_instance.device_indices.end());
                        vk_instance.device_indices.push_back(i);

                        VK_LOG_DEBUG("Prioritize device " << i << " driver " << new_driver.driverName << " over device " << *old_device << " driver " << old_driver.driverName);
                    }
                    وإلا {
                        VK_LOG_DEBUG("Prioritize device " << *old_device << " driver " << old_driver.driverName << " over device " << i << " driver " << new_driver.driverName << std::پایان_سطر);
                    }
                }
            }
        }

        // If no GPUs found, fall back to the first non-CPU device.
        // If only CPU devices are available, return without devices.
        إذا (vk_instance.device_indices.خالی()) {
            لكل (size_t i = 0; i < devices.size(); i++) {
                إذا (devices[i].getProperties().deviceType != vk::PhysicalDeviceType::eCpu) {
                    vk_instance.device_indices.push_back(i);
                    اكسر;
                }
            }
        }

        إذا (vk_instance.device_indices.خالی()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            أرجع;
        }
    }
    GGML_LOG_DEBUG("ggml_vulkan: Found %zu Vulkan devices:\n", vk_instance.device_indices.size());

    size_t host_میانگیر_import_شمار = 0;
    لكل (size_t i = 0; i < vk_instance.device_indices.size(); i++) {
        vk::PhysicalDevice vkdev = devices[vk_instance.device_indices[i]];
        std::vector<vk::ExtensionProperties> extensionprops = vkdev.enumerateDeviceExtensionProperties();

        منطقي membudget_supported = خطأ;
        منطقي external_حافظه_host_supported = خطأ;
        لكل (ثابت خودکار & ext : extensionprops) {
            إذا (strcmp(VK_EXT_حافظه_BUDGET_EXTENSION_NAME, ext.extensionName) == 0) {
                membudget_supported = درست;
            } وإلا إذا (strcmp("VK_EXT_external_memory_host", ext.extensionName) == 0) {
                external_حافظه_host_supported = درست;
            }
        }

        vk_instance.device_supports_membudget.push_back(membudget_supported);

        منطقي host_میانگیر_import = خطأ;
        إذا (external_حافظه_host_supported) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceDriverProperties driver_props;
            vk::Physical_Device_External_حافظه_Host_Properties_EXT ext_props;
            props2.p_بعدی = &driver_props;
            driver_props.p_بعدی = &ext_props;
            vkdev.getProperties2(&props2);
            host_میانگیر_import = ggml_vk_external_حافظه_host_usable(driver_props.driverID) &&
                                 ggml_vk_host_میانگیر_import_usable(props2.properties.vendorID);
            إذا (host_میانگیر_import) {
                vk_instance.host_import_alignment = std::max<size_t>(vk_instance.host_import_alignment, ext_props.min_Imported_Host_اشاره_گر_Alignment);
                host_میانگیر_import_شمار++;
            }
        }
        vk_instance.device_host_میانگیر_import.push_back(host_میانگیر_import);

        ggml_vk_print_gpu_info(i);
    }

    vk_instance.multi_device = host_میانگیر_import_شمار > 1 && vk_instance.device_host_میانگیر_import[0];
}

فراغ ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx) {
    VK_LOG_DEBUG("ggml_vk_init(" << ctx->name << ", " << idx << ")");
    ggml_vk_instance_init();
    GGML_ASSERT(idx < vk_instance.device_indices.size());

    ctx->name = GGML_VK_NAME + std::to_string(idx);

    ctx->device = ggml_vk_get_device(idx);

    ctx->semaphore_idx = 0;
    ctx->event_idx = 0;

    ctx->prealloc_اندازه_x = 0;
    ctx->prealloc_اندازه_y = 0;
    ctx->prealloc_اندازه_تقسیم_k = 0;
    // Fixed size of 1KB, for deterministic behavior
    ctx->prealloc_اندازه_جمع_کن_rms_partials = 1024;

    ctx->fence = ctx->device->device.createFence({});
    ctx->almost_ready_fence = ctx->device->device.createFence({});

    ctx->compute_cmd_pool.init(ctx->device, ctx->device->compute_صف.get());
    إذا (ctx->device->async_use_transfer_صف) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.مجموعه_PNext(&tci);
        ctx->transfer_semaphore.s = ctx->device->device.createSemaphore(ci);
        ctx->transfer_semaphore.value = 0;

        ctx->transfer_cmd_pool.init(ctx->device, ctx->device->transfer_صف.get());
    }

    إذا (vk_perf_logger_enabled) {
        ctx->perf_logger = std::unique_ptr<vk_perf_logger>(جديد vk_perf_logger());
    }

#اگر_تعریف_شده GGML_VULKAN_CHECK_RESULTS
    ثابت حرف* skip_checks = getenv("GGML_VULKAN_SKIP_CHECKS");
    vk_skip_checks = (skip_checks == عدم ? 0 : atoi(skip_checks));
    ثابت حرف* output_tensor = getenv("GGML_VULKAN_OUTPUT_TENSOR");
    vk_output_tensor = (output_tensor == عدم ? 0 : atoi(output_tensor));
#پایان_اگر
}

vk_pipeline ggml_vk_get_مقصد_fp16(ggml_backend_vk_context * ctx, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_get_to_fp16()");
    بدل (type) {
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ2_S:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_TQ1_0:
        حالة GGML_TYPE_TQ2_0:
            اكسر;
        افتراضي:
            أرجع اشاره_تهی;
    }

    أرجع ctx->device->pipeline_dequant[type];
}

ساكن vk_pipeline ggml_vk_get_dequantize_mul_mat_vec(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t رقم_تعداد_ستون, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_F16 || b_type == GGML_TYPE_Q8_1);
    GGML_ASSERT(رقم_تعداد_ستون >= 1 && رقم_تعداد_ستون <= mul_mat_vec_max_تعداد_ستون);

    إذا (b_type == GGML_TYPE_Q8_1) {
        بدل (a_type) {
            حالة GGML_TYPE_Q2_0:
            حالة GGML_TYPE_Q4_0:
            حالة GGML_TYPE_Q4_1:
            حالة GGML_TYPE_Q5_0:
            حالة GGML_TYPE_Q5_1:
            حالة GGML_TYPE_Q8_0:
            حالة GGML_TYPE_MXFP4:
            حالة GGML_TYPE_Q2_K:
            حالة GGML_TYPE_Q3_K:
            حالة GGML_TYPE_Q4_K:
            حالة GGML_TYPE_Q5_K:
            حالة GGML_TYPE_Q6_K:
            حالة GGML_TYPE_IQ1_S:
            حالة GGML_TYPE_IQ1_M:
            حالة GGML_TYPE_IQ4_XS:
                اكسر;
            افتراضي:
                أرجع اشاره_تهی;
        }
    }

    بدل (a_type) {
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_BF16:
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ2_S:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_TQ1_0:
        حالة GGML_TYPE_TQ2_0:
            اكسر;
        افتراضي:
            أرجع اشاره_تهی;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_اندازه_SUBGROUP;
    إذا ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        إذا (a_type == GGML_TYPE_Q6_K) {
            إذا (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_اندازه_LARGE;
            }
        } وإلا {
            إذا (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_اندازه_LARGE;
            }
        }
    }

    إذا (b_type == GGML_TYPE_Q8_1) {
        إذا (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_اندازه_SUBGROUP;
        }
        أرجع ctx->device->pipeline_dequant_mul_mat_vec_q8_1_f32[dmmv_wg][a_type][رقم_تعداد_ستون-1];
    }

    أرجع b_type == GGML_TYPE_F32 ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[dmmv_wg][a_type][رقم_تعداد_ستون-1] : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[dmmv_wg][a_type][رقم_تعداد_ستون-1];
}

ساكن vk_pipeline ggml_vk_get_dequantize_mul_mat_vec_id(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec_id()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_Q8_1);

    إذا (b_type == GGML_TYPE_Q8_1) {
        بدل (a_type) {
            حالة GGML_TYPE_Q2_0:
            حالة GGML_TYPE_Q4_0:
            حالة GGML_TYPE_Q4_1:
            حالة GGML_TYPE_Q5_0:
            حالة GGML_TYPE_Q5_1:
            حالة GGML_TYPE_Q8_0:
            حالة GGML_TYPE_MXFP4:
            حالة GGML_TYPE_Q2_K:
            حالة GGML_TYPE_Q3_K:
            حالة GGML_TYPE_Q4_K:
            حالة GGML_TYPE_Q5_K:
            حالة GGML_TYPE_Q6_K:
            حالة GGML_TYPE_IQ1_S:
            حالة GGML_TYPE_IQ1_M:
            حالة GGML_TYPE_IQ4_XS:
                اكسر;
            افتراضي:
                أرجع اشاره_تهی;
        }
    }

    بدل (a_type) {
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_BF16:
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ2_S:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_TQ1_0:
        حالة GGML_TYPE_TQ2_0:
            اكسر;
        افتراضي:
            أرجع اشاره_تهی;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_اندازه_SUBGROUP;
    إذا ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        إذا (a_type == GGML_TYPE_Q6_K) {
            إذا (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_اندازه_LARGE;
            }
        } وإلا {
            إذا (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_اندازه_LARGE;
            }
        }
    }

    إذا (b_type == GGML_TYPE_Q8_1) {
        إذا (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_اندازه_SUBGROUP;
        }
        أرجع ctx->device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[dmmv_wg][a_type];
    }

    أرجع ctx->device->pipeline_dequant_mul_mat_vec_id_f32[dmmv_wg][a_type];
}

vk_subbuffer ggml_vk_tensor_subbuffer(
    ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * tensor, منطقي allow_misalign) {

    vk_میانگیر میانگیر = اشاره_تهی;
    size_t جابه_جایی = 0;
    إذا (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, tensor->داده, میانگیر, جابه_جایی);
    }
    إذا (!میانگیر) {
        خودکار buf_ctx = (ggml_backend_vk_میانگیر_context *)tensor->میانگیر->context;
        میانگیر = buf_ctx->dev_میانگیر;
        جابه_جایی = vk_tensor_جابه_جایی(tensor) + tensor->view_offs;
    }
    GGML_ASSERT(میانگیر != اشاره_تهی);

    size_t size = ggml_nbytes(tensor);

    ثابت size_t descriptor_جابه_جایی = ggml_vk_descriptor_جابه_جایی(
        جابه_جایی, ctx->device->properties.limits.min_Storage_میانگیر_جابه_جایی_Alignment, ggml_type_اندازه(tensor->type));
    ثابت size_t misalign_بایتها = جابه_جایی - descriptor_جابه_جایی;
    // The shader must support misaligned offsets when indexing into the buffer
    GGML_ASSERT(allow_misalign || misalign_بایتها == 0);
    جابه_جایی = descriptor_جابه_جایی;
    size += misalign_بایتها;

    أرجع vk_subbuffer{میانگیر, جابه_جایی, size};
}

ساكن vk_command_میانگیر* ggml_vk_get_or_create_cmd_میانگیر(vk_device& device, vk_command_pool& pool) {
    لكل (خودکار& cmd_میانگیر : pool.cmd_buffers) {
        إذا (!cmd_میانگیر.درون_use) {
            cmd_میانگیر.use_counter++;
            cmd_میانگیر.درون_use = درست;
            أرجع &cmd_میانگیر;
        }
    }
    أرجع ggml_vk_create_cmd_میانگیر(device, pool);
}

ساكن vk_submission ggml_vk_begin_submission(vk_device& device, vk_command_pool& p, منطقي one_زمان = درست) {
    vk_submission s;
    s.میانگیر = ggml_vk_get_or_create_cmd_میانگیر(device, p);
    إذا (one_زمان) {
        s.میانگیر->buf.begin({ vk::Command_میانگیر_Usage_Flag_بیتها::e_One_زمان_Submit });
    } وإلا {
        s.میانگیر->buf.begin({ vk::Command_میانگیر_Usage_Flags{} });
    }

    أرجع s;
}

فراغ ggml_vk_cmd_label_begin(vk::Command_میانگیر buf, ثابت حرف * name) {
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<عائم, 4>{1.0f, 1.0f, 1.0f, 1.0f};
    vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT(buf, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

فراغ ggml_vk_ctx_انتها(vk_context& ctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_end(" << ctx << ", " << ctx->seqs.size() << ")");
    إذا (ctx->s == اشاره_تهی) {
        أرجع;
    }

    // close open labels so this buffer is balanced; reopened in ggml_vk_ctx_begin
    إذا (vk_instance.debug_utils_support) {
        لكل (size_t i = 0; i < ctx->debug_labels.size(); i++) {
            vk_instance.pfn_vk_Cmd_انتها_Debug_Utils_Label_EXT(ctx->s->میانگیر->buf);
        }
        // the enclosing per-command-buffer region
        vk_instance.pfn_vk_Cmd_انتها_Debug_Utils_Label_EXT(ctx->s->میانگیر->buf);
    }

    ctx->s->میانگیر->buf.end();
    ctx->s = اشاره_تهی;
}

فراغ ggml_vk_ctx_begin(vk_device& device, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_begin(" << device->name << ")");
    إذا (subctx->s != اشاره_تهی) {
        ggml_vk_ctx_انتها(subctx);
    }

    subctx->seqs.push_back({ ggml_vk_begin_submission(device, *subctx->p) });
    subctx->s = subctx->seqs[subctx->seqs.size() - 1].داده();

    إذا (vk_instance.debug_utils_support) {
        // outermost region, one per command buffer, so the gaps between submits stand out
        ثابت std::string name = "submit " + std::to_string(device->debug_cmdbuf_idx++);
        ggml_vk_cmd_label_begin(subctx->s->میانگیر->buf, name.c_str());

        // reopen labels left open when the previous command buffer was submitted
        لكل (ثابت std::string & label : subctx->debug_labels) {
            ggml_vk_cmd_label_begin(subctx->s->میانگیر->buf, label.c_str());
        }
    }
}

vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx) {
    vk_context نتیجه;
    إذا (!ctx->compute_ctx.expired()) {
        نتیجه = ctx->compute_ctx.قفل();
    } وإلا {
        نتیجه = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);

        ctx->compute_ctx = نتیجه;
        ggml_vk_ctx_begin(ctx->device, نتیجه);
    }

    إذا (ctx->device->async_use_transfer_صف && ctx->transfer_semaphore_آخرین_submitted < ctx->transfer_semaphore.value) {
        نتیجه->s->wait_semaphores.push_back(ctx->transfer_semaphore);
        ctx->transfer_semaphore_آخرین_submitted = ctx->transfer_semaphore.value;
    }

    أرجع نتیجه;
}

vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx) {
    vk_context نتیجه;
    إذا (!ctx->transfer_ctx.expired()) {
        نتیجه = ctx->transfer_ctx.قفل();
    } وإلا {
        نتیجه = ggml_vk_create_context(ctx, ctx->transfer_cmd_pool);

        ctx->transfer_ctx = نتیجه;
        ggml_vk_ctx_begin(ctx->device, نتیجه);
    }

    أرجع نتیجه;
}

منطقي ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx) {
    إذا (!ctx->device->async_use_transfer_صف || ctx->transfer_ctx.expired()) {
        أرجع خطأ;
    }

    vk_context cpy_ctx = ctx->transfer_ctx.قفل();
    ggml_vk_ctx_انتها(cpy_ctx);

    لكل (خودکار& cpy : cpy_ctx->درون_memcpys) {
        کپی_حافظه(cpy.dst, cpy.src, cpy.n);
    }

    ctx->transfer_semaphore.value++;
    cpy_ctx->seqs.back().back().signal_semaphores.push_back(ctx->transfer_semaphore);

    ggml_vk_submit(cpy_ctx, {});
    ctx->transfer_ctx.reset();
    أرجع درست;
}

size_t ggml_vk_align_اندازه(size_t عرض, size_t align) {
    VK_LOG_DEBUG("ggml_vk_align_size(" << عرض << ", " << align << ")");
    أرجع CEIL_DIV(عرض, align) * align;
}

فراغ deferred_memcpy(فراغ * dst, ثابت فراغ * src, size_t size, std::vector<vk_staging_memcpy>* memcpys) {
    إذا (memcpys == اشاره_تهی) {
        کپی_حافظه(dst, src, size);
    } وإلا {
        memcpys->emplace_back(dst, src, size);
    }
}

فراغ deferred_memset(فراغ * dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets) {
    إذا (memsets == اشاره_تهی) {
        memset(dst, val, size);
    } وإلا {
        memsets->emplace_back(dst, val, size);
    }
}

ساكن uint32_t ggml_vk_guess_تقسیم_k(ggml_backend_vk_context * ctx, uint32_t m, uint32_t n, uint32_t k, منطقي disable_تقسیم_k, ثابت vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_vk_guess_split_k(" << m << ", " << n << ", " << k << ", " << disable_تقسیم_k << ")");

    إذا (disable_تقسیم_k) {
        أرجع 1;
    }

    uint32_t تقسیم_k = 1;
    إذا (ctx->device->shader_core_شمار != 0 && n >= pipeline->wg_denoms[1]) {
        // If k is 'large' and the SMs will fill less than halfway, use split_k.
        uint32_t m_tiles = CEIL_DIV(m, pipeline->wg_denoms[0]);
        uint32_t n_tiles = CEIL_DIV(n, pipeline->wg_denoms[1]);

        إذا (k >= 2048) {
            إذا (m_tiles * n_tiles <= ctx->device->shader_core_شمار / 2) {
                تقسیم_k = ctx->device->shader_core_شمار / (m_tiles * n_tiles);
            } وإلا إذا (m_tiles * n_tiles <= ctx->device->shader_core_شمار * 2 / 3) {
                تقسیم_k = 3;
            }
            // Cap the split at 8x. Unless k is huge this is a lot of overhead.
            تقسیم_k = std::min(تقسیم_k, 8u);

            // ggml_vk_matmul will align the splits to be a multiple of 256.
            // If this rounded up size would cause the last split to be empty,
            // then reduce the split count.
            طالما (درست) {
                إذا (تقسیم_k == 1) {
                    اكسر;
                }
                uint32_t k_تقسیم = CEIL_DIV(k, تقسیم_k);
                k_تقسیم = ROUNDUP_POW2(k_تقسیم, 256);
                إذا (k_تقسیم * (تقسیم_k - 1) < k) {
                    اكسر;
                }
                تقسیم_k--;
            }
        }
    }

    أرجع تقسیم_k;
}

فراغ ggml_vk_matmul(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& تقسیم_k_میانگیر,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t تقسیم_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3,
        uint32_t padded_n, uint64_t b_ستون_بایتها, uint64_t d_ستون_بایتها) {
        VK_LOG_DEBUG("ggml_vk_matmul(a: (" << a.میانگیر->میانگیر << ", " << a.جابه_جایی << ", " << a.size << "), b: (" << b.میانگیر->میانگیر << ", " << b.جابه_جایی << ", " << b.size << "), d: (" << d.میانگیر->میانگیر << ", " << d.جابه_جایی << ", " << d.size << "), split_k: (" << (تقسیم_k_میانگیر.میانگیر != اشاره_تهی ? تقسیم_k_میانگیر.میانگیر->میانگیر : VK_NULL_HANDLE) << ", " << تقسیم_k_میانگیر.جابه_جایی << ", " << تقسیم_k_میانگیر.size << "), m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", split_k: " << تقسیم_k << ", batch: " << batch << ", ne02: " << ne02 << ", ne12: " << ne12 << ", broadcast2: " << broadcast2 << ", broadcast3: " << broadcast3 << ", padded_n: " << padded_n << ")");
    إذا (تقسیم_k == 1) {
        // a small tile can still produce more workgroups than the device allows on the n axis:
        // split the dispatch into n-chunks, shifting the B and D column windows
        ثابت uint64_t max_n = (uint64_t)ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1] * pipeline->wg_denoms[1];
        ثابت uint64_t n_chunks = CEIL_DIV(n, max_n);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, n_chunks * CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]));

        لكل (uint64_t n_پایه = 0; n_پایه < n; n_پایه += max_n) {
            ثابت uint32_t n_chunk = (uint32_t) std::min(n - n_پایه, max_n);
            GGML_ASSERT(n_پایه * b_ستون_بایتها < b.size && n_پایه * d_ستون_بایتها < d.size);
            GGML_ASSERT(b.size != VK_کامل_اندازه && d.size != VK_کامل_اندازه);
            vk_subbuffer b_chunk = b;
            vk_subbuffer d_chunk = d;
            b_chunk.جابه_جایی += n_پایه * b_ستون_بایتها;
            b_chunk.size   -= n_پایه * b_ستون_بایتها;
            d_chunk.جابه_جایی += n_پایه * d_ستون_بایتها;
            d_chunk.size   -= n_پایه * d_ستون_بایتها;

            uint32_t پایه_work_گروه_z = 0;
            طالما (پایه_work_گروه_z < batch) {
                uint32_t groups_z = std::min(batch - پایه_work_گروه_z, ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]);

                ثابت vk_mat_mat_افزودن_constants pc = { m, n_chunk, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, پایه_work_گروه_z, batch, k, ne02, ne12, broadcast2, broadcast3, uint32_t(padded_n - n_پایه) };
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b_chunk, d_chunk }, pc, { m, n_chunk, groups_z });
                پایه_work_گروه_z += groups_z;
            }
        }
        أرجع;
    }

    إذا (ctx->prealloc_تقسیم_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    GGML_ASSERT(batch_stride_d == m * n);

    // Round the split size up to a multiple of 256 (k-quant alignment)
    uint32_t k_تقسیم = CEIL_DIV(k, تقسیم_k);
    k_تقسیم = ROUNDUP_POW2(k_تقسیم, 256);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]));

    uint32_t پایه_work_گروه_z = 0;
    طالما (پایه_work_گروه_z < batch) {
        uint32_t groups_z = std::min(batch - پایه_work_گروه_z, ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]);

        ثابت vk_mat_mat_افزودن_constants pc1 = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, پایه_work_گروه_z, batch, k_تقسیم, ne02, ne12, broadcast2, broadcast3, padded_n };
        // Make sure enough workgroups get assigned for split k to work
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, تقسیم_k_میانگیر }, pc1, { (CEIL_DIV(m, pipeline->wg_denoms[0]) * pipeline->wg_denoms[0]) * تقسیم_k, n, groups_z });
        پایه_work_گروه_z += groups_z;
    }
    ggml_vk_sync_buffers(ctx, subctx);
    ثابت std::array<uint32_t, 2> pc2 = { (uint32_t)(m * n * batch), تقسیم_k };
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_matmul_تقسیم_k_reduce, { تقسیم_k_میانگیر, d }, pc2, { m * n * batch, 1, 1 });
    ctx->prealloc_تقسیم_k_need_sync = درست;
}

ساكن منطقي ggml_vk_get_mul_mat_mat_f16acc(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {
    إذا (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16) أرجع خطأ;
    إذا (src1_type == GGML_TYPE_Q8_1) أرجع خطأ;
    إذا (src0_type == GGML_TYPE_F16) {
        أرجع prec == GGML_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support);
    }
    // quant types
    إذا (ctx->device->coopmat2) {
        أرجع prec == GGML_PREC_DEFAULT;
    }
    إذا (ctx->device->coopmat_support) {
        أرجع ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT;
    }
    أرجع ctx->device->fp16 && prec == GGML_PREC_DEFAULT;
}

ساكن ثابت std::vector<vk_matmul_pipeline_pair>* ggml_vk_get_mul_mat_mat_pipeline_map(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec, منطقي mul_mat_id = خطأ) {
    منطقي f16acc = ggml_vk_get_mul_mat_mat_f16acc(ctx, src0_type, src1_type, prec);
    vk_matmul_pipeline_کلید کلید{src0_type, src1_type, mul_mat_id, f16acc};
    خودکار it = ctx->device->pipeline_matmul.find(کلید);
    إذا (it == ctx->device->pipeline_matmul.end() || it->second.خالی()) {
        // Try without f16acc
        إذا (f16acc) {
            کلید.f16acc = خطأ;
            it = ctx->device->pipeline_matmul.find(کلید);
            إذا (it != ctx->device->pipeline_matmul.end() && !it->second.خالی()) أرجع &it->second;
        }
        أرجع اشاره_تهی;
    }
    أرجع &it->second;
}

ساكن vk_pipeline ggml_vk_guess_matmul_pipeline_map(ggml_backend_vk_context * ctx,
        ثابت std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, منطقي aligned, منطقي mul_mat_id) {
    خودکار& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_شمار, configs);
    إذا (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    أرجع (aligned && configs[idx].aligned) ? configs[idx].aligned : configs[idx].unaligned;
}

ساكن uint32_t ggml_vk_guess_matmul_pipeline_align_map(ggml_backend_vk_context * ctx,
        ثابت std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, منطقي mul_mat_id) {
    خودکار& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_شمار, configs);
    إذا (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    أرجع configs[idx].align;
}

ساكن uint64_t ggml_vk_mul_mat_id_max_tiles(uint64_t n_experts, uint64_t n_ردیفها, uint32_t bn) {
    // Each nonempty expert can add at most one partial tile to the combined grid.
    أرجع std::min(n_ردیفها, CEIL_DIV(n_ردیفها, bn) + std::min(n_experts, n_ردیفها) - 1);
}

ساكن فراغ ggml_vk_matmul_id(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ids, ثابت vk_subbuffer & expert_شمار_buf,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t n_as, uint32_t nei0, uint32_t nei1, uint32_t nbi1, uint32_t ne11,
        uint32_t ردیف_ids_جابه_جایی, uint32_t max_tiles) {
    VK_LOG_DEBUG("ggml_vk_matmul_id(a: (" << a.میانگیر->میانگیر << ", " << a.جابه_جایی << ", " << a.size << "), b: (" << b.میانگیر->میانگیر << ", " << b.جابه_جایی << ", " << b.size << "), d: (" << d.میانگیر->میانگیر << ", " << d.جابه_جایی << ", " << d.size << "), ids: (" << ids.میانگیر->میانگیر << ", " << ids.جابه_جایی << ", " << ids.size << "), expert_count: (" << expert_شمار_buf.میانگیر->میانگیر << ", " << expert_شمار_buf.جابه_جایی << ", " << expert_شمار_buf.size << "), " <<
        "m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", " <<
        "batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", " <<
        "n_as: " << n_as << ", nei0: " << nei0 << ", nei1: " << nei1 << ", nbi1: " << nbi1 << ", ne11: " << ne11 << ")");
    ثابت vk_mat_mat_id_افزودن_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d,
                                              nei0, nei1, nbi1, ne11, n_as, ردیف_ids_جابه_جایی };
    ثابت uint32_t tiles_y = std::min(max_tiles, ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]);
    ثابت std::array<uint32_t, 3> elements = { m, tiles_y * pipeline->wg_denoms[1], uint32_t(CEIL_DIV(max_tiles, tiles_y)) };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d, ids, expert_شمار_buf }, pc, elements);
}

منطقي ggml_vk_dim01_contiguous(ثابت ggml_tensor * tensor) {
    أرجع
        tensor->nb[0] == ggml_type_اندازه(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_اندازه(tensor->type) &&
        (tensor->ne[3] == 1 || tensor->nb[3] == tensor->nb[2]*tensor->ne[2]);
}

// Batch stride in elements of a tensor read in place.
ساكن uint32_t ggml_vk_batch_stride(ثابت ggml_tensor * tensor) {
    أرجع (uint32_t)(tensor->nb[2] / ggml_type_اندازه(tensor->type) * ggml_blck_اندازه(tensor->type));
}

vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ثابت ggml_tensor * src, ثابت ggml_tensor * dst, ggml_type مقصد) {

    // Choose "contiguous copy" shader if src/dst are contiguous
    منطقي contig = ggml_is_contiguous(src) && (!dst || ggml_is_contiguous(dst));

    // Use optimized "transpose" shader if src dim1 is the innermost dimension.
    منطقي transpose = dst && src->nb[1] == ggml_type_اندازه(مقصد) && ggml_are_same_shape(dst, src);

    إذا (transpose && src->type == مقصد) {
        إذا (ggml_type_اندازه(مقصد) == 4) {
            أرجع ctx->device->pipeline_cpy_transpose_32;
        } وإلا إذا (ggml_type_اندازه(مقصد) == 2) {
            أرجع ctx->device->pipeline_cpy_transpose_16;
        }
    }

    // Same, for a 0<->2 swap: src dim2 is the innermost dimension.
    منطقي transpose02 = dst && !contig && src->nb[2] == ggml_type_اندازه(مقصد) &&
                       ggml_is_contiguous(dst) && ggml_are_same_shape(dst, src);

    إذا (transpose02 && src->type == مقصد) {
        إذا (ggml_type_اندازه(مقصد) == 4) {
            أرجع ctx->device->pipeline_cpy_transpose_02_32;
        } وإلا إذا (ggml_type_اندازه(مقصد) == 2) {
            أرجع ctx->device->pipeline_cpy_transpose_02_16;
        }
    }

    إذا (src->type == GGML_TYPE_F32 && مقصد == GGML_TYPE_F16) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_f32_f16;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_f32_f16;
        }
    }
    إذا (src->type == GGML_TYPE_F16 && مقصد == GGML_TYPE_F32) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_f16_f32;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_f16_f32;
        }
    }
    إذا (src->type == GGML_TYPE_F32 && مقصد == GGML_TYPE_BF16) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_f32_bf16;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_f32_bf16;
        }
    }
    إذا (src->type == GGML_TYPE_BF16 && مقصد == GGML_TYPE_F32) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_bf16_f32;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_bf16_f32;
        }
    }
    إذا (src->type == GGML_TYPE_BF16 && مقصد == GGML_TYPE_F16) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_bf16_f16;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_bf16_f16;
        }
    }
    إذا (src->type == GGML_TYPE_F32 && مقصد == GGML_TYPE_I32) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_f32_i32;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_f32_i32;
        }
    }
    إذا (src->type == GGML_TYPE_I32 && مقصد == GGML_TYPE_F32) {
        إذا (contig) {
            أرجع ctx->device->pipeline_contig_cpy_i32_f32;
        } وإلا {
            أرجع ctx->device->pipeline_cpy_i32_f32;
        }
    }
    إذا (src->type == GGML_TYPE_F32) {
        بدل (مقصد) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_IQ4_NL:
            أرجع ctx->device->pipeline_cpy_f32_quant[مقصد];
        افتراضي:
            اكسر;
        }
    }

    إذا (مقصد == GGML_TYPE_F32) {
        بدل (src->type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_IQ4_NL:
            أرجع ctx->device->pipeline_cpy_quant_f32[src->type];
        افتراضي:
            اكسر;
        }
    }

    إذا (src->type == مقصد) {
        // Copy two or four bytes at a time, depending on block size.
        // For quantized types, we scale by block size/type size. But
        // this path is also used for bf16->bf16 for example, where the
        // type size must be exactly 2 or 4.
        GGML_ASSERT(ggml_is_quantized(مقصد) || ggml_type_اندازه(src->type) == 2 || ggml_type_اندازه(src->type) == 4);
        إذا ((ggml_type_اندازه(src->type) % 4) == 0) {
            إذا (contig) {
                أرجع ctx->device->pipeline_contig_cpy_32;
            } وإلا {
                أرجع ctx->device->pipeline_cpy_32;
            }
        } وإلا {
            إذا (contig) {
                أرجع ctx->device->pipeline_contig_cpy_16;
            } وإلا {
                أرجع ctx->device->pipeline_cpy_16;
            }
        }
    }

    std::cerr << "Missing CPY op for types: " << ggml_type_name(src->type) << " " << ggml_type_name(مقصد) << std::پایان_سطر;
    GGML_ABORT("fatal error");
}

فراغ ggml_vk_cpy_مقصد_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ثابت ggml_tensor * tensor, ثابت vk_subbuffer & درون, ثابت vk_subbuffer & بیرون) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_contiguous((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "buffer in size=" << درون.میانگیر->size << ", buffer out size=" << بیرون.میانگیر->size << ")");

    ثابت uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    إذا (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } وإلا إذا (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } وإلا {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_افزودن_constants pc = vk_op_unary_افزودن_constants_init(tensor, tensor, ne);
    pc.nb10 = 1;
    pc.nb11 = (uint32_t)tensor->ne[0];
    pc.nb12 = (uint32_t)(tensor->ne[0] * tensor->ne[1]);
    pc.nb13 = (uint32_t)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]);
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { درون, بیرون }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

ساكن فراغ ggml_vk_cpy_مقصد_strided(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ثابت ggml_tensor * tensor,
        ثابت vk_subbuffer & درون, ثابت vk_subbuffer & بیرون,
        uint32_t nb10, uint32_t nb11, uint32_t nb12, uint32_t nb13) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_strided((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "dst_nb=(" << nb10 << ", " << nb11 << ", " << nb12 << ", " << nb13 << "), buffer in size=" << درون.میانگیر->size << ", buffer out size=" << بیرون.میانگیر->size << ")");

    ثابت uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    إذا (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } وإلا إذا (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } وإلا {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_افزودن_constants pc = vk_op_unary_افزودن_constants_init(tensor, tensor, ne);
    pc.nb10 = nb10;
    pc.nb11 = nb11;
    pc.nb12 = nb12;
    pc.nb13 = nb13;
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { درون, بیرون }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type) {
    بدل(type) {
        حالة GGML_TYPE_Q8_1:
            أرجع ctx->device->pipeline_quantize_q8_1_x4;
        افتراضي:
            std::cerr << "Missing quantize pipeline for type: " << ggml_type_name(type) << std::پایان_سطر;
            GGML_ABORT("fatal error");
    }
}

فراغ ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت vk_subbuffer & درون, ثابت vk_subbuffer & بیرون, uint32_t ne) {
    VK_LOG_DEBUG("ggml_vk_quantize_q8_1(" << "buffer in size=" << درون.میانگیر->size << ", buffer out size=" << بیرون.میانگیر->size << ", " << ne << ")");

    vk_pipeline pipeline = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);

    ثابت uint32_t رقم_بلوکها = CEIL_DIV(ne, pipeline->wg_denoms[0]);
    // clamp the number of elements to the max workgroup count. The shader will iterate over the total number of blocks.
    ثابت uint64_t max_elements = std::min<uint64_t>(uint64_t{ctx->device->properties.limits.max_Compute_Work_گروه_شمار[0]} * pipeline->wg_denoms[0], std::numeric_limits<uint32_t>::max());
    ثابت uint32_t elements = std::min(ne, static_cast<uint32_t>(max_elements));

    ثابت vk_quantize_q8_1_افزودن_constants pc = {
        ne,
        رقم_بلوکها,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { درون, بیرون }, pc, { elements, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
}

ساكن vk_pipeline ggml_vk_get_64b_indexing_pipeline(ggml_backend_vk_context * ctx, vk_pipeline &pipeline) {
    GGML_استفادهنشده(ctx);
#إذا defined(VK_EXT_shader_64bit_indexing)
    vk_pipeline *ptr = &pipeline;
    طالما (*ptr) {
        إذا ((*ptr)->is_64b_indexing) {
            أرجع *ptr;
        }
        ptr = &(*ptr)->بعدی;
    }
#پایان_اگر
    أرجع pipeline;
}

// perf-heuristic for choosing mul_mat b type, based on shader availability
ساكن ggml_type ggml_vk_mul_mat_b_type(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ثابت ggml_tensor * src1,
        ggml_type f16_type, ggml_prec prec, منطقي mul_mat_id, منطقي can_quantize_y, منطقي prefer_f16_b) {
    خودکار موجود = [&](ggml_type b) {
        أرجع ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0_type, b, prec, mul_mat_id) != اشاره_تهی;
    };
    إذا (can_quantize_y && موجود(GGML_TYPE_Q8_1))             أرجع GGML_TYPE_Q8_1;
    إذا (prefer_f16_b   && موجود(f16_type))                   أرجع f16_type;
    إذا (ggml_vk_dim01_contiguous(src1) && موجود(src1->type)) أرجع src1->type;
    إذا (src1->type == GGML_TYPE_BF16 && موجود(GGML_TYPE_F32)) أرجع GGML_TYPE_F32;
    أرجع f16_type;
}

ساكن فراغ ggml_vk_mul_mat_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst, منطقي disable_تقسیم_k) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_q_f16((" << src0 << ", name=" << src0->name << ", type=" << ggml_type_name(src0->type) << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << ggml_type_name(src1->type) << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << ggml_type_name(dst->type) << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    ثابت uint64_t ne02 = src0->ne[2];
    ثابت uint64_t ne03 = src0->ne[3];

    ثابت uint64_t ne10 = src1->ne[0];
    ثابت uint64_t ne11 = src1->ne[1];
    ثابت uint64_t ne12 = src1->ne[2];
    ثابت uint64_t ne13 = src1->ne[3];

    ثابت uint64_t ne21 = dst->ne[1];
    ثابت uint32_t stride_d = dst->nb[1] / ggml_type_اندازه(dst->type);
    ثابت uint32_t stride_batch_d = stride_d*ne21;

    ثابت uint64_t r2 = ne12 / ne02;
    ثابت uint64_t r3 = ne13 / ne03;

    ggml_backend_vk_میانگیر_context * dst_buf_ctx = (ggml_backend_vk_میانگیر_context *)dst->میانگیر->context;
    ggml_backend_vk_میانگیر_context * src0_buf_ctx = (ggml_backend_vk_میانگیر_context *)src0->میانگیر->context;
    ggml_backend_vk_میانگیر_context * src1_buf_ctx = (ggml_backend_vk_میانگیر_context *)src1->میانگیر->context;

    vk_میانگیر d_Qx = اشاره_تهی;
    size_t qx_buf_جابه_جایی = 0;
    vk_میانگیر d_Qy = اشاره_تهی;
    size_t qy_buf_جابه_جایی = 0;

    منطقي src0_uma = خطأ;
    منطقي src1_uma = خطأ;

    إذا (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->داده, d_Qx, qx_buf_جابه_جایی);
        ggml_vk_host_get(ctx->device, src1->داده, d_Qy, qy_buf_جابه_جایی);
        src0_uma = d_Qx != اشاره_تهی;
        src1_uma = d_Qy != اشاره_تهی;
    }

    ثابت منطقي x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;

    ثابت منطقي is_coopmat = ctx->device->coopmat_support || ctx->device->coopmat2;
    ثابت منطقي can_quantize_y = (ctx->device->integer_dot_حاصلضرب || ctx->device->coopmat_int_support) &&
                                src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // coopmat shaders are fast enough that memory bandwidth reduction from f16 activations helps
    ثابت منطقي prefer_f16_b = is_coopmat && ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    ثابت ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], خطأ, can_quantize_y, prefer_f16_b);

    ثابت منطقي quantize_y       = y_kernel_type == GGML_TYPE_Q8_1;
    ثابت منطقي y_f32_kernel     = y_kernel_type == GGML_TYPE_F32;
    ثابت منطقي y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type);

    ثابت std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0]);

    ثابت منطقي qx_needs_dequant = mmp_map == اشاره_تهی || x_non_contig;
    ثابت منطقي qy_needs_dequant = y_needs_reformat;

    إذا (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0]);
    }

    GGML_ASSERT(mmp_map != اشاره_تهی);

    ثابت uint32_t kpad = quantize_y ? 0 : ggml_vk_align_اندازه(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, ne11, خطأ));
    ثابت منطقي aligned = !quantize_y && ne10 == kpad && ne01 > 8 && ne11 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, ne11, aligned, خطأ);

    إذا (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_میانگیر_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
    uint32_t padded_n = qy_needs_dequant ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1]) : ne11;
    ثابت uint64_t x_ne = ggml_nelements(src0);
    // 128 elements per Q8_1 x4 block
    ثابت uint64_t y_ne = padded_n * ne10 * ne12 * ne13;
    ثابت uint64_t d_ne = ggml_nelements(dst);

    ثابت uint32_t تقسیم_k = ggml_vk_guess_تقسیم_k(ctx, ne01, ne11, ne10, disable_تقسیم_k, pipeline);

    ثابت uint64_t qx_sz = ggml_type_اندازه(src0->type) * x_ne / ggml_blck_اندازه(src0->type);
    ثابت uint64_t qy_sz = ggml_type_اندازه(src1->type) * y_ne / ggml_blck_اندازه(src1->type);
    ثابت uint64_t x_sz = !qx_needs_dequant ? qx_sz : حجم(ggml_fp16_t) * x_ne;
    ثابت uint64_t y_sz = quantize_y ? (ggml_vk_align_اندازه(y_ne, 128) * ggml_type_اندازه(GGML_TYPE_Q8_1) / ggml_blck_اندازه(GGML_TYPE_Q8_1)) : (y_f32_kernel ? حجم(عائم) * y_ne : حجم(ggml_fp16_t) * y_ne);
    ثابت uint64_t d_sz = حجم(عائم) * d_ne;

    vk_pipeline مقصد_fp16_vk_0 = اشاره_تهی;
    vk_pipeline مقصد_fp16_vk_1 = اشاره_تهی;
    vk_pipeline مقصد_q8_1 = اشاره_تهی;

    إذا (x_non_contig) {
        مقصد_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, اشاره_تهی, f16_type);
    } وإلا {
        مقصد_fp16_vk_0 = ggml_vk_get_مقصد_fp16(ctx, src0->type);
    }
    إذا (y_needs_reformat) {
        مقصد_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, اشاره_تهی, y_kernel_type);
    } وإلا {
        مقصد_fp16_vk_1 = ggml_vk_get_مقصد_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || مقصد_fp16_vk_0 != اشاره_تهی);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || مقصد_fp16_vk_1 != اشاره_تهی);  // NOLINT

    إذا (quantize_y) {
        مقصد_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    {
        ثابت uint64_t تقسیم_k_اندازه = تقسیم_k > 1 ? d_sz * تقسیم_k : 0;
        إذا (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_میانگیر_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_میانگیر_Range) ||
                (تقسیم_k > 1 && تقسیم_k_اندازه > ctx->device->properties.limits.max_Storage_میانگیر_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        إذا (qx_needs_dequant && ctx->prealloc_اندازه_x < x_sz) {
            ctx->prealloc_اندازه_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا ((qy_needs_dequant || quantize_y) && ctx->prealloc_اندازه_y < y_sz) {
            ctx->prealloc_اندازه_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (تقسیم_k > 1 && ctx->prealloc_اندازه_تقسیم_k < تقسیم_k_اندازه) {
            ctx->prealloc_اندازه_تقسیم_k = تقسیم_k_اندازه;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        إذا (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, مقصد_fp16_vk_0, 1);
        }
        إذا (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, مقصد_fp16_vk_1, 1);
        }
        إذا (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, مقصد_q8_1, 1);
        }
        إذا (تقسیم_k > 1) {
            ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_تقسیم_k_reduce, 1);
        }
    }

    vk_میانگیر d_D = dst_buf_ctx->dev_میانگیر;
    ثابت uint64_t d_buf_جابه_جایی = vk_tensor_جابه_جایی(dst) + dst->view_offs;
    GGML_ASSERT(d_D != اشاره_تهی);
    GGML_ASSERT(d_D->size >= d_buf_جابه_جایی + d_sz);
    vk_میانگیر d_X;
    uint64_t x_buf_جابه_جایی = 0;
    vk_میانگیر d_Y;
    uint64_t y_buf_جابه_جایی = 0;
    إذا (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_میانگیر;
        qx_buf_جابه_جایی = vk_tensor_جابه_جایی(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != اشاره_تهی);
    }
    إذا (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_میانگیر;
        qy_buf_جابه_جایی = vk_tensor_جابه_جایی(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != اشاره_تهی);
    }
    إذا (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } وإلا {
        d_X = d_Qx;
        x_buf_جابه_جایی = qx_buf_جابه_جایی;
        GGML_ASSERT(qx_sz == x_sz);
    }
    إذا (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } وإلا إذا (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } وإلا {
        d_Y = d_Qy;
        y_buf_جابه_جایی = qy_buf_جابه_جایی;
        GGML_ASSERT(qy_sz == y_sz);
    }

    إذا (x_non_contig || qx_needs_dequant) {
        إذا (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    إذا (x_non_contig) {
        ggml_vk_cpy_مقصد_contiguous(ctx, subctx, مقصد_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_جابه_جایی), ggml_vk_subbuffer(ctx, d_X, 0));
    } وإلا إذا (qx_needs_dequant) {
        ثابت std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, مقصد_fp16_vk_0, { vk_subbuffer{ d_Qx, qx_buf_جابه_جایی, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)(x_ne), 1, 1});
        ggml_vk_sync_buffers(ctx, subctx);
    }
    إذا (y_needs_reformat) {
        إذا (ctx->prealloc_y_آخرین_pipeline_استفاده_شده != مقصد_fp16_vk_1.get() ||
            ctx->prealloc_y_آخرین_tensor_استفاده_شده != src1 ||
            ctx->prealloc_y_آخرین_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_مقصد_contiguous(ctx, subctx, مقصد_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_جابه_جایی), ggml_vk_subbuffer(ctx, d_Y, 0));
            ctx->prealloc_y_آخرین_pipeline_استفاده_شده = مقصد_fp16_vk_1.get();
            ctx->prealloc_y_آخرین_tensor_استفاده_شده = src1;
            ctx->prealloc_y_آخرین_k_padded = خطأ;
        }
    }
    إذا (quantize_y) {
        إذا (ctx->prealloc_y_آخرین_pipeline_استفاده_شده != مقصد_q8_1.get() ||
            ctx->prealloc_y_آخرین_tensor_استفاده_شده != src1 ||
            ctx->prealloc_y_آخرین_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_جابه_جایی), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_آخرین_pipeline_استفاده_شده = مقصد_q8_1.get();
            ctx->prealloc_y_آخرین_tensor_استفاده_شده = src1;
            ctx->prealloc_y_آخرین_k_padded = خطأ;
        }
    }

    // The partial tiles of a quantized A rely on the bound range to read zeros past the last row,
    // so the range stays exact: the strided extent of a tensor read in place, the staged size otherwise.
    ثابت uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    ثابت uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    إذا (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_اندازه(src0->type);
    }

    إذا (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_اندازه(src1->type);
    }

    // column stride in bytes of the B buffer: raw src1 keeps its own stride, staging buffers are tightly packed
    ثابت uint64_t b_ستون_بایتها = (y_needs_reformat || qy_needs_dequant || quantize_y)
        ? (quantize_y ? (uint64_t)(ne10 / ggml_blck_اندازه(GGML_TYPE_Q8_1)) * ggml_type_اندازه(GGML_TYPE_Q8_1) : (uint64_t)ne10 * حجم(ggml_fp16_t))
        : (uint64_t)src1->nb[1];

    // compute
    ggml_vk_matmul(
        ctx, subctx, pipeline,
        { d_X, x_buf_جابه_جایی, x_range }, { d_Y, y_buf_جابه_جایی, y_range },
        ggml_vk_subbuffer(ctx, d_D, d_buf_جابه_جایی), { ctx->prealloc_تقسیم_k, 0, d_sz * تقسیم_k },
        ne01, ne11, ne10,
        ne10, ne10, stride_d, stride_batch_x, stride_batch_y, stride_batch_d,
        تقسیم_k, ne12*ne13, ne02, ne12, r2, r3, padded_n,
        b_ستون_بایتها, (uint64_t)dst->nb[1]
    );  // NOLINT

    إذا (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = درست;
    }
    إذا (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = درست;
    }
}

ساكن منطقي ggml_vk_should_use_mmvq(ثابت vk_device& device, uint32_t m, uint32_t n, uint32_t k, ggml_type src0_type) {
    إذا (device->mmvq_نما == 1) {
        أرجع درست;
    } وإلا إذا (device->mmvq_نما == -1) {
        أرجع خطأ;
    }

    // q6_k only has 2-byte alignment which makes it somewhat problematic,
    // using MMVQ is only a win on Intel.
    منطقي mmvq_q6 = device->vendor_id == VK_VENDOR_ID_INTEL;
    إذا (src0_type == GGML_TYPE_Q6_K && !mmvq_q6) {
        أرجع خطأ;
    }

    // MMVQ is generally good for batches
    إذا (n > 1) {
        أرجع درست;
    }

    // Quantization overhead is not worth it for small k
    بدل (device->vendor_id) {
    حالة VK_VENDOR_ID_NVIDIA:
        إذا (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_IQ1_S || src0_type == GGML_TYPE_IQ1_M) {
            أرجع درست;
        }

        إذا (k <= 4096) {
            أرجع خطأ;
        }

        بدل (src0_type) {
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_Q8_0:
            أرجع device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        افتراضي:
            أرجع درست;
        }
    حالة VK_VENDOR_ID_AMD:
        إذا (k < 2048) {
            أرجع خطأ;
        }

        بدل (src0_type) {
        حالة GGML_TYPE_Q8_0:
            أرجع device->architecture == vk_device_architecture::AMD_GCN;
        افتراضي:
            أرجع درست;
        }
    حالة VK_VENDOR_ID_INTEL:
        إذا (device->architecture == vk_device_architecture::INTEL_XE2) {
            إذا (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_Q6_K) {
                أرجع درست;
            }
        }

        إذا (device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            // Intel Windows proprietary driver MMVQ performance for !Q2/Q3/Q6 is worse than fp16,
            // see https://github.com/ggml-org/llama.cpp/issues/17628 and
            // https://github.com/ggml-org/llama.cpp/pull/23056
            أرجع خطأ;
        }

        إذا (k < 2048) {
            أرجع خطأ;
        }

        بدل (src0_type) {
        // From tests on A770 Linux, may need more tuning
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_IQ4_XS:
            أرجع خطأ;
        افتراضي:
            أرجع درست;
        }
    حالة VK_VENDOR_ID_QUALCOMM:
        أرجع خطأ;
    افتراضي:
        أرجع درست;
    }

    GGML_استفادهنشده(m);
}

ساكن فراغ ggml_vk_mul_mat_vec_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx, منطقي جابجا_inputs = خطأ) {
    ggml_tensor * dst = cgraph->nodes[گره_idx];
    ثابت ggml_tensor * src0 = dst->src[جابجا_inputs ? 1 : 0];
    ثابت ggml_tensor * src1 = dst->src[جابجا_inputs ? 0 : 1];

    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << ")),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    ثابت uint64_t ne02 = src0->ne[2];
    ثابت uint64_t ne03 = src0->ne[3];

    ثابت uint64_t ne10 = src1->ne[0];
    ثابت uint64_t ne11 = src1->ne[1];
    ثابت uint64_t ne12 = src1->ne[2];
    ثابت uint64_t ne13 = src1->ne[3];

    ثابت uint64_t ne20 = dst->ne[جابجا_inputs ? 1 : 0];
    ثابت uint64_t ne21 = dst->ne[جابجا_inputs ? 0 : 1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ثابت uint64_t r2 = ne12 / ne02;
    ثابت uint64_t r3 = ne13 / ne03;

    // batch_n indicates that we need to compute a few vector results, and this assumes
    // ne12 and ne13 are 1. It overloads the batch_strides to hold the row strides.
    GGML_ASSERT(ne11 == 1 || ne12 * ne13 == 1);
    منطقي batch_n = ne11 > 1;

    ثابت منطقي x_non_contig = !ggml_vk_dim01_contiguous(src0);

    // BF16 src1 has no mul_mat_vec B-type; widen to F32
    ثابت منطقي widen_bf16 = src1->type == GGML_TYPE_BF16;
    ثابت منطقي y_non_contig = !ggml_vk_dim01_contiguous(src1) || widen_bf16;

    ثابت ggml_type y_kernel_type = widen_bf16 ? GGML_TYPE_F32 : src1->type;
    ثابت منطقي f16_f32_kernel = y_kernel_type == GGML_TYPE_F32;
    منطقي quantize_y = ctx->device->integer_dot_حاصلضرب && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);

    vk_pipeline مقصد_fp16_vk_0 = اشاره_تهی;
    vk_pipeline مقصد_fp16_vk_1 = اشاره_تهی;
    إذا (x_non_contig) {
        مقصد_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, اشاره_تهی, src0->type);
    }
    إذا (y_non_contig) {
        مقصد_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, اشاره_تهی, y_kernel_type);
    } وإلا {
        مقصد_fp16_vk_1 = ggml_vk_get_مقصد_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, GGML_TYPE_Q8_1, ne11, ne20, ne00) : اشاره_تهی;
    vk_pipeline مقصد_q8_1 = اشاره_تهی;

    إذا (dmmv == اشاره_تهی) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, y_kernel_type, ne11, ne20, ne00);
        quantize_y = خطأ;
    }

    إذا (quantize_y) {
        مقصد_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    إذا (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_میانگیر_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    ثابت منطقي qx_needs_dequant = x_non_contig;
    ثابت منطقي qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    GGML_ASSERT(!qx_needs_dequant || مقصد_fp16_vk_0 != اشاره_تهی);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || مقصد_fp16_vk_1 != اشاره_تهی);  // NOLINT
    GGML_ASSERT(dmmv != اشاره_تهی);

    ثابت uint64_t x_ne = ggml_nelements(src0);
    ثابت uint64_t y_ne = ggml_nelements(src1);

    ثابت uint64_t qx_sz = ggml_vk_align_اندازه(ggml_type_اندازه(src0->type) * x_ne / ggml_blck_اندازه(src0->type), ctx->device->properties.limits.min_Storage_میانگیر_جابه_جایی_Alignment);
    ثابت uint64_t x_sz = x_non_contig ? ggml_vk_align_اندازه(ggml_type_اندازه(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_میانگیر_جابه_جایی_Alignment) : qx_sz;
    ثابت uint64_t y_sz = quantize_y ? (ggml_vk_align_اندازه(y_ne, 128) * ggml_type_اندازه(GGML_TYPE_Q8_1) / ggml_blck_اندازه(GGML_TYPE_Q8_1)) :
                         (f16_f32_kernel ? حجم(عائم) * y_ne : حجم(ggml_fp16_t) * y_ne);

    {
        إذا (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_میانگیر_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_میانگیر_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        إذا (qx_needs_dequant && ctx->prealloc_اندازه_x < x_sz) {
            ctx->prealloc_اندازه_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا ((qy_needs_dequant || quantize_y) && ctx->prealloc_اندازه_y < y_sz) {
            ctx->prealloc_اندازه_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        إذا (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, مقصد_fp16_vk_0, 1);
        }
        إذا (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, مقصد_fp16_vk_1, 1);
        }
        إذا (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, مقصد_q8_1, 1);
        }
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[گره_idx + ctx->رقم_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_X, d_Y;

    إذا (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } وإلا {
        d_X = d_Qx;
        GGML_ASSERT(qx_sz == x_sz);
    }
    إذا (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } وإلا {
        d_Y = d_Qy;
    }

    إذا (x_non_contig) {
        إذا (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        GGML_ASSERT(x_sz == ggml_vk_align_اندازه(ggml_type_اندازه(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_میانگیر_جابه_جایی_Alignment));
        ggml_vk_cpy_مقصد_contiguous(ctx, subctx, مقصد_fp16_vk_0, src0, d_Qx, d_X);
    }
    إذا (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_اندازه(y_kernel_type) * y_ne);
        إذا (ctx->prealloc_y_آخرین_pipeline_استفاده_شده != مقصد_fp16_vk_1.get() ||
            ctx->prealloc_y_آخرین_tensor_استفاده_شده != src1 ||
            ctx->prealloc_y_آخرین_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_مقصد_contiguous(ctx, subctx, مقصد_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_آخرین_pipeline_استفاده_شده = مقصد_fp16_vk_1.get();
            ctx->prealloc_y_آخرین_tensor_استفاده_شده = src1;
            ctx->prealloc_y_آخرین_k_padded = خطأ;
        }
    }
    إذا (quantize_y) {
        إذا (ctx->prealloc_y_آخرین_pipeline_استفاده_شده != مقصد_q8_1.get() ||
            ctx->prealloc_y_آخرین_tensor_استفاده_شده != src1 ||
            ctx->prealloc_y_آخرین_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_آخرین_pipeline_استفاده_شده = مقصد_q8_1.get();
            ctx->prealloc_y_آخرین_tensor_استفاده_شده = src1;
            ctx->prealloc_y_آخرین_k_padded = خطأ;
        }
    }

    // For batch_n, the A matrix is the same for each batch, and B/D use the row stride as the batch stride
    uint32_t stride_batch_x = batch_n ? 0 : (qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0));
    uint32_t stride_batch_y = batch_n ? ne10 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));
    uint32_t stride_batch_d = batch_n ? ne20 : (ne20*ne21);

    إذا (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_اندازه(src0->type);
    }

    إذا (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_اندازه(src1->type);
    }

    ثابت uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_گروه_شمار[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    إذا (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    vk_subbuffer d_F0 = d_D;
    إذا (ctx->رقم_additional_fused_ops > 0) {
        ثابت ggml_tensor * جمع_کن_2 = cgraph->nodes[گره_idx + 1];
        ثابت ggml_tensor * bias = جمع_کن_2->src[0] == dst ? جمع_کن_2->src[1] : جمع_کن_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    إذا (ctx->رقم_additional_fused_ops == 2) {
        ثابت ggml_tensor * جمع_کن_2 = cgraph->nodes[گره_idx + 2];
        ثابت ggml_tensor * bias = جمع_کن_2->src[0] == cgraph->nodes[گره_idx + 1] ? جمع_کن_2->src[1] : جمع_کن_2->src[0];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    ggml_pipeline_request_descriptor_sets(ctx, dmmv, CEIL_DIV(ne12 * ne13, ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]));

    uint32_t پایه_work_گروه_y = 0;
    طالما (پایه_work_گروه_y < ne12 * ne13) {

        uint32_t groups_y = std::min((uint32_t)(ne12 * ne13) - پایه_work_گروه_y, ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]);
        ثابت vk_mat_vec_افزودن_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            stride_batch_x, stride_batch_y, stride_batch_d,
            fusion_flags, پایه_work_گروه_y,
            (uint32_t)ne02, (uint32_t)ne12, (uint32_t)r2, (uint32_t)r3,
        };
        ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
                                  {
                                    d_X,
                                    d_Y,
                                    d_D,
                                    d_F0,
                                    d_F1,
                                  },
                                  pc, { groups_x, groups_y, groups_z });
        پایه_work_گروه_y += groups_y;
    }

    إذا (x_non_contig) {
        ctx->prealloc_x_need_sync = درست;
    }
    إذا (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = درست;
    }
}

ساكن فراغ ggml_vk_mul_mat_vec_p021_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {
    ggml_tensor * dst = cgraph->nodes[گره_idx];
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_p021_f16_f32(" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]);  // NOLINT
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]);  // NOLINT
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    ثابت uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    //const uint64_t ne10 = src1->ne[0];
    ثابت uint64_t ne11 = src1->ne[1];
    ثابت uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);

    // With grouped query attention there are > 1 Q matrices per K, V matrix.
    uint32_t gqa_نسبت = (uint32_t)ne12 / (uint32_t)ne02;
    إذا (gqa_نسبت > 8 || gqa_نسبت == 0 || ne12 != ne02 * gqa_نسبت) {
        gqa_نسبت = 1;
    }

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_نسبت - 1];

    إذا (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_میانگیر_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[گره_idx + ctx->رقم_additional_fused_ops], درست);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, درست);

    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    إذا (ctx->رقم_additional_fused_ops > 0) {
        ثابت ggml_tensor * جمع_کن_2 = cgraph->nodes[گره_idx + 1];
        ثابت ggml_tensor * bias = جمع_کن_2->src[0] == dst ? جمع_کن_2->src[1] : جمع_کن_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    إذا (ctx->رقم_additional_fused_ops > 1) {
        ثابت ggml_tensor * bias = cgraph->nodes[گره_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute

    vk_mat_vec_p021_افزودن_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne12,
        0, 0, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, اشاره_تهی, اشاره_تهی, cgraph->nodes[گره_idx + ctx->رقم_additional_fused_ops]);

    uint32_t workgroups_z = (uint32_t)ne12;
    // When gqa_ratio > 1, each invocation does multiple rows and we can launch fewer workgroups
    إذا (gqa_نسبت > 1) {
        workgroups_z /= gqa_نسبت;
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { 1, (uint32_t)ne01, workgroups_z });
}

ساكن فراغ ggml_vk_mul_mat_vec_nc_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {
    ggml_tensor * dst = cgraph->nodes[گره_idx];
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_nc_f16_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    ثابت uint64_t ne02 = src0->ne[2];
    ثابت uint64_t ne03 = src0->ne[3];

    ثابت uint64_t nb01 = src0->nb[1];
    ثابت uint64_t nb02 = src0->nb[2];

    ثابت uint64_t nb12 = src1->nb[2];

    // const uint64_t ne10 = src1->ne[0];
    ثابت uint64_t ne11 = src1->ne[1];
    ثابت uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    ثابت uint32_t nb03 = (uint32_t)(src0->nb[3] / حجم(ggml_fp16_t));
    ثابت uint32_t nb13 = (uint32_t)(src1->nb[3] / حجم(عائم));
    ثابت uint32_t nb23 = (uint32_t)(dst->nb[3] / حجم(عائم));

    GGML_ASSERT(ne11 == 1);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]); // checked in supports_op

    ثابت uint32_t ردیف_stride_x = nb01 / حجم(ggml_fp16_t);
    ثابت uint32_t channel_stride_x = nb02 / حجم(ggml_fp16_t);
    ثابت uint32_t channel_stride_y = nb12 / حجم(عائم);

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_nc_f16_f32;
    إذا (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_میانگیر_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[گره_idx + ctx->رقم_additional_fused_ops], درست);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, درست);
    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    إذا (ctx->رقم_additional_fused_ops > 0) {
        ثابت ggml_tensor * جمع_کن_2 = cgraph->nodes[گره_idx + 1];
        ثابت ggml_tensor * bias = جمع_کن_2->src[0] == dst ? جمع_کن_2->src[1] : جمع_کن_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    إذا (ctx->رقم_additional_fused_ops > 1) {
        ثابت ggml_tensor * bias = cgraph->nodes[گره_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute
    vk_mat_vec_nc_افزودن_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01,
        ردیف_stride_x, channel_stride_x, channel_stride_y,
        (uint32_t)(ne12 / ne02), (uint32_t)ne12,
        0, 0,
        nb03, nb13, nb23, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, اشاره_تهی, اشاره_تهی, cgraph->nodes[گره_idx + ctx->رقم_additional_fused_ops]);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { (uint32_t)ne03, (uint32_t)ne01, (uint32_t)ne12 });
}

ساكن صحيح ggml_vk_fwht_pipeline_idx(int64_t n) {
    بدل (n) {
        حالة 64:  أرجع 0;
        حالة 128: أرجع 1;
        حالة 256: أرجع 2;
        حالة 512: أرجع 3;
        افتراضي:  أرجع -1;
    }
}

منطقي ggml_vk_can_use_fwht(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst) {
    إذا (ctx->رقم_additional_fused_ops != 0) {
        أرجع خطأ;
    }

    إذا (ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        أرجع خطأ;
    }

    ثابت صحيح idx = ggml_vk_fwht_pipeline_idx(src1->ne[0]);
    إذا (idx < 0 || ctx->device->pipeline_fwht_f32[idx] == اشاره_تهی) {
        أرجع خطأ;
    }

    إذا (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(src1)) {
        أرجع خطأ;
    }
    GGML_ASSERT(ggml_is_contiguous(dst));

    أرجع درست;
}

فراغ ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src, ggml_tensor * dst) {
    ثابت صحيح idx = ggml_vk_fwht_pipeline_idx(src->ne[0]);
    vk_pipeline pipeline = ctx->device->pipeline_fwht_f32[idx];

    ثابت uint32_t ردیفها_در_هر_workgroup = 4;
    ثابت uint32_t n_ردیفها = (uint32_t)ggml_nrows(src);
    ثابت uint32_t max_workgroups_x = ctx->device->properties.limits.max_Compute_Work_گروه_شمار[0];

    ثابت uint32_t مجموع_workgroups = CEIL_DIV(n_ردیفها, ردیفها_در_هر_workgroup);
    ثابت uint32_t workgroups_x = std::min(مجموع_workgroups, max_workgroups_x);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ثابت vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src, درست);
    ثابت vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, درست);

    vk_op_fwht_افزودن_constants pc = {
        n_ردیفها,
        0,
        0,
        1.0f / std::جذر((عائم)src->ne[0]),
    };
    init_pushconst_tensor_offsets(ctx, pc, src, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc, { workgroups_x, 1, 1 });
}

ساكن uint32_t ggml_vk_nb_elem(ثابت ggml_tensor * t, صحيح i) {
    أرجع (uint32_t)(t->nb[i] / ggml_type_اندازه(t->type));
}

فراغ ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * mixes, ثابت ggml_tensor * scale, ثابت ggml_tensor * پایه, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_comb(" << mixes << ", " << scale << ", " << پایه << ", " << dst << ")");

    vk_pipeline pipeline = ctx->device->pipeline_dsv4_hc_comb_f32;
    GGML_ASSERT(pipeline != اشاره_تهی);

    ثابت uint32_t n_نشانهها = (uint32_t)mixes->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ثابت vk_subbuffer mixes_buf = ggml_vk_tensor_subbuffer(ctx, mixes, درست);
    ثابت vk_subbuffer scale_buf = ggml_vk_tensor_subbuffer(ctx, scale, درست);
    ثابت vk_subbuffer پایه_buf  = ggml_vk_tensor_subbuffer(ctx, پایه,  درست);
    ثابت vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, dst,   درست);

    vk_op_dsv4_hc_comb_افزودن_constants pc = {
        n_نشانهها,
        ggml_vk_nb_elem(mixes, 0), ggml_vk_nb_elem(mixes, 1),
        ggml_vk_nb_elem(scale, 0),
        ggml_vk_nb_elem(پایه,  0),
        ggml_vk_nb_elem(dst,   0), ggml_vk_nb_elem(dst, 1), ggml_vk_nb_elem(dst, 2),
        0, 0, 0, 0,
        ggml_get_op_params_f32(dst, 0),
        (uint32_t)ggml_get_op_params_i32(dst, 1),
    };
    init_pushconst_tensor_offsets(ctx, pc, mixes, scale, پایه, اشاره_تهی, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { mixes_buf, scale_buf, پایه_buf, dst_buf }, pc, { n_نشانهها, 1, 1 });
}

فراغ ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * x, ثابت ggml_tensor * weights, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_pre(" << x << ", " << weights << ", " << dst << ")");

    ثابت عائم scale = ggml_get_op_params_f32(dst, 0);
    ثابت منطقي  gated = ggml_get_op_params_i32(dst, 1) != 0;

    vk_pipeline pipeline = gated ? ctx->device->pipeline_dsv4_hc_pre_gated_f32 : ctx->device->pipeline_dsv4_hc_pre_f32;
    GGML_ASSERT(pipeline != اشاره_تهی);

    ثابت uint32_t n_embd   = (uint32_t)x->ne[0];
    ثابت uint32_t n_نشانهها = (uint32_t)x->ne[2];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ثابت vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,       درست);
    ثابت vk_subbuffer w_buf = ggml_vk_tensor_subbuffer(ctx, weights, درست);
    ثابت vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,     درست);

    vk_op_dsv4_hc_pre_افزودن_constants pc = {
        n_embd, n_نشانهها,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1), ggml_vk_nb_elem(x, 2),
        ggml_vk_nb_elem(weights, 0), ggml_vk_nb_elem(weights, 1), ggml_vk_nb_elem(weights, 2),
        ggml_vk_nb_elem(dst, 0), ggml_vk_nb_elem(dst, 1),
        0, 0, 0,
        scale,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, weights, اشاره_تهی, اشاره_تهی, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, w_buf, d_buf }, pc, { n_embd, n_نشانهها, 1 });
}

فراغ ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * x, ثابت ggml_tensor * residual, ثابت ggml_tensor * post, ثابت ggml_tensor * comb, ggml_tensor * dst, ثابت ggml_tensor * gate_scale_درون) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_post(" << x << ", " << residual << ", " << post << ", " << comb << ", " << dst << ")");

    vk_pipeline pipeline = comb ? ctx->device->pipeline_dsv4_hc_post_f32 : ctx->device->pipeline_dsv4_hc_post_nocomb_f32;
    GGML_ASSERT(pipeline != اشاره_تهی);

    ثابت uint32_t n_embd   = (uint32_t)x->ne[0];
    ثابت uint32_t n_نشانهها = (uint32_t)x->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ثابت vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,        درست);
    ثابت vk_subbuffer r_buf = ggml_vk_tensor_subbuffer(ctx, residual, درست);
    // with a fused gate, post is scale(sigmoid(scale(p_src))) and the shader applies it to p_src
    ثابت ggml_tensor * p_src = gate_scale_درون ? gate_scale_درون->src[0] : post;
    ثابت vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p_src,    درست);
    ثابت vk_subbuffer c_buf = comb ? ggml_vk_tensor_subbuffer(ctx, comb, درست) : x_buf;
    ثابت vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,      درست);

    vk_op_dsv4_hc_post_افزودن_constants pc = {
        n_embd, n_نشانهها,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1),
        ggml_vk_nb_elem(residual, 0), ggml_vk_nb_elem(residual, 1), ggml_vk_nb_elem(residual, 2),
        ggml_vk_nb_elem(p_src, 0), ggml_vk_nb_elem(p_src, 1),
        comb ? ggml_vk_nb_elem(comb, 0) : 0, comb ? ggml_vk_nb_elem(comb, 1) : 0, comb ? ggml_vk_nb_elem(comb, 2) : 0,
        ggml_vk_nb_elem(dst,  0), ggml_vk_nb_elem(dst,  1), ggml_vk_nb_elem(dst,  2),
        0, 0, 0, 0, 0,
        gate_scale_درون ? 1u : 0u,
        gate_scale_درون ? ggml_get_op_params_f32(gate_scale_درون, 0) : 1.0f,
        gate_scale_درون ? ggml_get_op_params_f32(post, 0) : 1.0f,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, residual, p_src, comb, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, r_buf, p_buf, c_buf, d_buf }, pc, { n_embd, n_نشانهها, 1 });
}

فراغ ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {
    ggml_tensor * dst = cgraph->nodes[گره_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")");

    // Handle huge A matrix by splitting the M dimensions. This works well for convolution use cases
    // where the M dimension is very large.
    // Split_k doesn't work with M splitting.
    // This only supports batchsize == 1.
    ثابت size_t nbytes = ggml_nbytes(src0);
    ثابت منطقي needs_تقسیم = dst->ne[2] == 1 && dst->ne[3] == 1 && nbytes > ctx->device->properties.limits.max_Storage_میانگیر_Range;
    إذا (needs_تقسیم) {
        // Choose the number of rows that can fit (and divide by two, to allow for any additional offsets)
        ثابت uint32_t M_تقسیم = ctx->device->properties.limits.max_Storage_میانگیر_Range / (2 * src0->nb[1]);
        uint32_t m_جابه_جایی = 0;
        طالما (m_جابه_جایی < dst->ne[0]) {
            ثابت uint32_t cur_M_اندازه = std::min(M_تقسیم, (uint32_t)(dst->ne[0] - m_جابه_جایی));
            ggml_tensor dst2 = *dst;
            ggml_tensor src02 = *src0;

            dst2.view_src = dst->view_src ? dst->view_src : dst;
            src02.view_src = src0->view_src ? src0->view_src : src0;

            dst2.view_offs += m_جابه_جایی * dst->nb[0];
            src02.view_offs += m_جابه_جایی * src0->nb[1];
            dst2.ne[0] = cur_M_اندازه;
            src02.ne[1] = cur_M_اندازه;

            ggml_vk_mul_mat_q_f16(ctx, subctx, &src02, src1, &dst2, درست);

            m_جابه_جایی += cur_M_اندازه;
        }
    } وإلا إذا (ggml_vk_can_use_fwht(ctx, src1, dst)) {
        ggml_vk_fwht(ctx, subctx, src1, dst);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && dst->ne[1] == 1 &&
        // detect 0213 permutation, and batch size of 1
        src0->nb[0] <= src0->nb[2] &&
        src0->nb[2] <= src0->nb[1] &&
        src0->nb[1] <= src0->nb[3] &&
        src1->nb[0] <= src1->nb[2] &&
        src1->nb[2] <= src1->nb[1] &&
        src1->nb[1] <= src1->nb[3] &&
        src0->ne[3] == 1 &&
        src1->ne[3] == 1 &&
        src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1] &&
        src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]) {
        ggml_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, cgraph, گره_idx);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && dst->ne[1] == 1 &&
               !ggml_is_permuted(src0) && !ggml_is_permuted(src1) &&
               src0->ne[3] <= ctx->device->properties.limits.max_Compute_Work_گروه_شمار[0] &&
               src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1] &&
               src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]) {
        ggml_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, cgraph, گره_idx);
    // With one output row, B^T*A has the same flat output as A^T*B.
    } وإلا إذا (ctx->رقم_additional_fused_ops == 0 &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
               (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16 || ggml_is_quantized(src1->type)) &&
               dst->ne[0] == 1 && dst->ne[1] > mul_mat_vec_max_تعداد_ستون &&
               src0->ne[2] == 1 && src0->ne[3] == 1 &&
               src1->ne[2] == 1 && src1->ne[3] == 1 &&
               ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
               get_misalign_بایتها(ctx, src0) == 0 && get_misalign_بایتها(ctx, src1) == 0 && get_misalign_بایتها(ctx, dst) == 0) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, گره_idx, درست);
    // mul_mat_vec supports batching ne12*ne13 when ne11==1, or treating ne11 as the batch size (up to four)
    // when ne12 and ne13 are one.
    } وإلا إذا ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_تعداد_ستون && src1->ne[2] * src1->ne[3] == 1)) &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, گره_idx);
    } وإلا {
        ggml_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst, خطأ);
    }
}

ساكن فراغ ggml_vk_mul_mat_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * ids, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    ثابت uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    ثابت uint64_t ne10 = src1->ne[0];
    ثابت uint64_t ne11 = src1->ne[1];
    ثابت uint64_t ne12 = src1->ne[2];
    ثابت uint64_t ne13 = src1->ne[3];

    ثابت uint64_t nei0 = ids->ne[0];
    ثابت uint64_t nei1 = ids->ne[1];

    ثابت uint32_t nbi0 = ids->nb[0];
    ثابت uint32_t nbi1 = ids->nb[1];

    ثابت uint64_t ne20 = dst->ne[0];
    ثابت uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ثابت uint64_t n_as = ne02;

    ggml_backend_vk_میانگیر_context * dst_buf_ctx = (ggml_backend_vk_میانگیر_context *)dst->میانگیر->context;
    ggml_backend_vk_میانگیر_context * src0_buf_ctx = (ggml_backend_vk_میانگیر_context *)src0->میانگیر->context;
    ggml_backend_vk_میانگیر_context * src1_buf_ctx = (ggml_backend_vk_میانگیر_context *)src1->میانگیر->context;

    vk_میانگیر d_Qx = اشاره_تهی;
    size_t qx_buf_جابه_جایی = 0;
    vk_میانگیر d_Qy = اشاره_تهی;
    size_t qy_buf_جابه_جایی = 0;

    منطقي src0_uma = خطأ;
    منطقي src1_uma = خطأ;

    إذا (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->داده, d_Qx, qx_buf_جابه_جایی);
        ggml_vk_host_get(ctx->device, src1->داده, d_Qy, qy_buf_جابه_جایی);
        src0_uma = d_Qx != اشاره_تهی;
        src1_uma = d_Qy != اشاره_تهی;
    }

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    ثابت منطقي x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
#إذا defined(GGML_VULKAN_COOPMAT2_DECODE_وکتور_GLSLC_SUPPORT)
    // B must already be, or be convertible to, the matmul B type used by this path.
    ثابت منطقي y_decode_وکتور_supported = ctx->device->coopmat2_decode_وکتور &&
                                           (f16_type != GGML_TYPE_BF16 || ctx->device->coopmat2_bf16_support) &&
                                           (src1->type == GGML_TYPE_F32 || src1->type == f16_type);
    // If B is copied to prealloc_y, we can choose a 4-element-aligned row stride.
    ثابت منطقي y_decode_وکتور_استفاده_میکند_prealloc = !ggml_vk_dim01_contiguous(src1) || src1->type != f16_type;
    // Direct B reads are safe only if row starts and the original buffer offset are 4-element aligned.
    ثابت منطقي y_decode_وکتور_aligned =
        (ne10 % 4 == 0) &&
        (y_decode_وکتور_استفاده_میکند_prealloc || get_misalign_بایتها(ctx, src1) % (4 * ggml_type_اندازه(src1->type)) == 0);
    // Stage B only when decode-vector is available and direct B reads would be misaligned.
    ثابت منطقي y_decode_وکتور_staging = y_decode_وکتور_supported && !y_decode_وکتور_aligned;
#وإلا
    ثابت منطقي y_decode_وکتور_staging = خطأ;
#پایان_اگر
    ثابت منطقي can_quantize_y = ctx->device->integer_dot_حاصلضرب && src1->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // Intel coopmat1 matrix cores prefer f16 activations for quantized weights
    ثابت منطقي prefer_f16_b = ctx->device->coopmat_support && !ctx->device->coopmat2 &&
                              ctx->device->vendor_id == VK_VENDOR_ID_INTEL &&
                              ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    ثابت ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], درست, can_quantize_y, prefer_f16_b);

    ثابت منطقي quantize_y   = y_kernel_type == GGML_TYPE_Q8_1;
    ثابت منطقي y_f32_kernel = y_kernel_type == GGML_TYPE_F32;

    ثابت std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0], درست);

    ثابت منطقي qx_needs_dequant = mmp_map == اشاره_تهی || x_non_contig;

    إذا (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0], درست);
    }

    GGML_ASSERT(mmp_map != اشاره_تهی);

    // Coopmat2 MUL_MAT_ID BK specialization constants in ggml_vk_load_shaders are at most 64.
    ثابت uint32_t y_staged_ردیف_stride = ctx->device->coopmat2 && !quantize_y ? ggml_vk_align_اندازه(ne10, 64) : ne10;
    ثابت منطقي y_needs_k_padding = ne10 != y_staged_ردیف_stride;

    ثابت منطقي y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type ||
                                                  y_decode_وکتور_staging || y_needs_k_padding);
    ثابت منطقي qy_needs_dequant = y_needs_reformat;

    ثابت uint32_t kpad = quantize_y ? 0 : ggml_vk_align_اندازه(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, nei1, درست));
    ثابت منطقي aligned = !quantize_y && ne10 == kpad && ne01 > 8 && nei1 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, nei1, aligned, درست);

    ثابت uint32_t bn = pipeline->wg_denoms[1];
    ثابت uint64_t max_tiles = ggml_vk_mul_mat_id_max_tiles(n_as, nei0 * nei1, bn);
    ثابت uint64_t ردیف_ids_جابه_جایی = 1 + 2 * max_tiles;
    ثابت uint64_t ردیف_map_واژگان = ردیف_ids_جابه_جایی + max_tiles * bn;

    إذا (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_میانگیر_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }
    ثابت uint64_t x_ne = ggml_nelements(src0);
    ثابت uint64_t y_ne = (uint64_t)y_staged_ردیف_stride * ne11 * ne12 * ne13;
    ثابت uint64_t d_ne = ggml_nelements(dst);

    ثابت uint64_t qx_sz = ggml_type_اندازه(src0->type) * x_ne / ggml_blck_اندازه(src0->type);
    ثابت uint64_t qy_sz = ggml_type_اندازه(src1->type) * ggml_nelements(src1) / ggml_blck_اندازه(src1->type);
    ثابت uint64_t x_sz = !qx_needs_dequant ? qx_sz : حجم(ggml_fp16_t) * x_ne;
    ثابت uint64_t y_sz = quantize_y ? (ggml_vk_align_اندازه(y_ne, 128) * ggml_type_اندازه(GGML_TYPE_Q8_1) / ggml_blck_اندازه(GGML_TYPE_Q8_1)) : (y_f32_kernel ? حجم(عائم) * y_ne : حجم(ggml_fp16_t) * y_ne);
    ثابت uint64_t d_sz = حجم(عائم) * d_ne;

    vk_pipeline مقصد_fp16_vk_0 = اشاره_تهی;
    vk_pipeline مقصد_fp16_vk_1 = اشاره_تهی;
    vk_pipeline مقصد_q8_1 = اشاره_تهی;

    خودکار make_y_staged_dst = [&]() {
        ggml_tensor y_staged_dst = *src1;
        y_staged_dst.type = f16_type;
        y_staged_dst.nb[0] = ggml_type_اندازه(f16_type);
        y_staged_dst.nb[1] = y_staged_dst.nb[0] * y_staged_ردیف_stride;
        y_staged_dst.nb[2] = y_staged_dst.nb[1] * ne11;
        y_staged_dst.nb[3] = y_staged_dst.nb[2] * y_staged_dst.ne[2];
        أرجع y_staged_dst;
    };

    إذا (x_non_contig) {
        مقصد_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, اشاره_تهی, f16_type);
    } وإلا {
        مقصد_fp16_vk_0 = ggml_vk_get_مقصد_fp16(ctx, src0->type);
    }
    إذا (y_needs_reformat) {
        ggml_tensor y_staged_dst;
        ثابت ggml_tensor * y_staged_dst_ptr = اشاره_تهی;
        إذا (y_needs_k_padding) {
            y_staged_dst = make_y_staged_dst();
            y_staged_dst_ptr = &y_staged_dst;
        }

        مقصد_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, y_staged_dst_ptr, f16_type);
    } وإلا {
        مقصد_fp16_vk_1 = ggml_vk_get_مقصد_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || مقصد_fp16_vk_0 != اشاره_تهی);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || مقصد_fp16_vk_1 != اشاره_تهی);  // NOLINT

    إذا (quantize_y) {
        مقصد_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }
    vk_pipeline شمار_experts = ctx->device->pipeline_شمار_experts;

    ثابت size_t expert_داده_اندازه = حجم(uint32_t) * ردیف_map_واژگان;

    {
        إذا (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_میانگیر_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_میانگیر_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        إذا (qx_needs_dequant && ctx->prealloc_اندازه_x < x_sz) {
            ctx->prealloc_اندازه_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا ((qy_needs_dequant || quantize_y) && ctx->prealloc_اندازه_y < y_sz) {
            ctx->prealloc_اندازه_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (ctx->prealloc_اندازه_تقسیم_k < expert_داده_اندازه) {
            ctx->prealloc_اندازه_تقسیم_k = expert_داده_اندازه;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        إذا (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, مقصد_fp16_vk_0, 1);
        }
        إذا (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, مقصد_fp16_vk_1, 1);
        }
        إذا (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, مقصد_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, شمار_experts, 1);
    }

    vk_میانگیر d_D = dst_buf_ctx->dev_میانگیر;
    ثابت uint64_t d_buf_جابه_جایی = vk_tensor_جابه_جایی(dst) + dst->view_offs;
    GGML_ASSERT(d_D != اشاره_تهی);
    vk_میانگیر d_X;
    uint64_t x_buf_جابه_جایی = 0;
    vk_میانگیر d_Y;
    uint64_t y_buf_جابه_جایی = 0;
    إذا (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_میانگیر;
        qx_buf_جابه_جایی = vk_tensor_جابه_جایی(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != اشاره_تهی);
    }
    إذا (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_میانگیر;
        qy_buf_جابه_جایی = vk_tensor_جابه_جایی(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != اشاره_تهی);
    }
    إذا (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } وإلا {
        d_X = d_Qx;
        x_buf_جابه_جایی = qx_buf_جابه_جایی;
        GGML_ASSERT(qx_sz == x_sz);
    }
    إذا (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } وإلا إذا (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } وإلا {
        d_Y = d_Qy;
        y_buf_جابه_جایی = qy_buf_جابه_جایی;
        GGML_ASSERT(qy_sz == y_sz);
    }

    إذا (x_non_contig || qx_needs_dequant) {
        إذا (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids, درست);

    // Count how many times each expert is used
    vk_subbuffer expert_شمار_buf = { ctx->prealloc_تقسیم_k, 0, expert_داده_اندازه };
    إذا (ctx->prealloc_تقسیم_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }
    {
        vk_op_شمار_experts_افزودن_constants pc = { (uint32_t)nei0,
                                           (uint32_t)nei1,
                                           (uint32_t)(nbi0 / ggml_type_اندازه(ids->type)),
                                           (uint32_t)(nbi1 / ggml_type_اندازه(ids->type)),
                                           (uint32_t)(get_misalign_بایتها(ctx, ids) / ggml_type_اندازه(ids->type)),
                                           (uint32_t)n_as,
                                           0, 0, bn, (uint32_t)ردیف_ids_جابه_جایی };
        init_pushconst_fastdiv(pc);
        ggml_vk_dispatch_pipeline(ctx, subctx, شمار_experts,
            { d_ids, expert_شمار_buf }, pc,
            { 1, 1, 1 });
    }

    إذا (x_non_contig) {
        ggml_vk_cpy_مقصد_contiguous(ctx, subctx, مقصد_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_جابه_جایی), ggml_vk_subbuffer(ctx, d_X, 0));
    } وإلا إذا (qx_needs_dequant) {
        ثابت std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, مقصد_fp16_vk_0,
            { vk_subbuffer{ d_Qx, qx_buf_جابه_جایی, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)x_ne, 1, 1});
    }
    إذا (y_needs_reformat) {
        إذا (ctx->prealloc_y_آخرین_pipeline_استفاده_شده != مقصد_fp16_vk_1.get() ||
            ctx->prealloc_y_آخرین_tensor_استفاده_شده != src1 ||
            ctx->prealloc_y_آخرین_k_padded != y_needs_k_padding) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            إذا (y_needs_k_padding) {
                GGML_ASSERT(y_sz % 4 == 0);
                // Zero B padding because clamping only A can produce 0 * Inf or NaN.
                subctx->s->میانگیر->buf.fill_میانگیر(d_Y->میانگیر, 0, y_sz, 0);
                ggml_vk_sync_buffers(ctx, subctx);
                ثابت ggml_tensor y_staged_dst = make_y_staged_dst();
                ثابت uint32_t y_staged_dst_type_اندازه = ggml_type_اندازه(y_staged_dst.type);
                ggml_vk_cpy_مقصد_strided(
                    ctx, subctx, مقصد_fp16_vk_1, src1,
                    ggml_vk_subbuffer(ctx, d_Qy, qy_buf_جابه_جایی), ggml_vk_subbuffer(ctx, d_Y, 0),
                    (uint32_t)(y_staged_dst.nb[0] / y_staged_dst_type_اندازه),
                    (uint32_t)(y_staged_dst.nb[1] / y_staged_dst_type_اندازه),
                    (uint32_t)(y_staged_dst.nb[2] / y_staged_dst_type_اندازه),
                    (uint32_t)(y_staged_dst.nb[3] / y_staged_dst_type_اندازه));
            } وإلا {
                ggml_vk_cpy_مقصد_contiguous(ctx, subctx, مقصد_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_جابه_جایی), ggml_vk_subbuffer(ctx, d_Y, 0));
            }
            ctx->prealloc_y_آخرین_pipeline_استفاده_شده = مقصد_fp16_vk_1.get();
            ctx->prealloc_y_آخرین_tensor_استفاده_شده = src1;
            ctx->prealloc_y_آخرین_k_padded = y_needs_k_padding;
        }
    }
    إذا (quantize_y) {
        إذا (ctx->prealloc_y_آخرین_pipeline_استفاده_شده != مقصد_q8_1.get() ||
            ctx->prealloc_y_آخرین_tensor_استفاده_شده != src1 ||
            ctx->prealloc_y_آخرین_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_جابه_جایی), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_آخرین_pipeline_استفاده_شده = مقصد_q8_1.get();
            ctx->prealloc_y_آخرین_tensor_استفاده_شده = src1;
            ctx->prealloc_y_آخرین_k_padded = خطأ;
        }
    }
    ggml_vk_sync_buffers(ctx, subctx);

    ثابت uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    ثابت uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_b_y = y_needs_k_padding ? y_staged_ردیف_stride : ne10;
    uint32_t stride_batch_y = y_needs_k_padding ? y_staged_ردیف_stride * ne11 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));

    إذا (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_اندازه(src0->type);
    }

    إذا (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_اندازه(src1->type);
    }

    // compute
    ggml_vk_matmul_id(
        ctx, subctx, pipeline,
        { d_X, x_buf_جابه_جایی, x_range }, { d_Y, y_buf_جابه_جایی, y_range },
        { d_D, d_buf_جابه_جایی, d_sz }, std::move(d_ids), expert_شمار_buf,
        ne01, ne21, ne10, ne10, stride_b_y, ne01,
        stride_batch_x, stride_batch_y, ne20*ne21,
        n_as, nei0, nei1, nbi1 / ggml_type_اندازه(ids->type), ne11, ردیف_ids_جابه_جایی, max_tiles
    );  // NOLINT

    إذا (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = درست;
    }
    إذا (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = درست;
    }
    ctx->prealloc_تقسیم_k_need_sync = درست;
}

ساكن فراغ ggml_vk_mul_mat_vec_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {
    ggml_tensor * dst = cgraph->nodes[گره_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * ids = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    // const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    ثابت uint64_t ne10 = src1->ne[0];
    ثابت uint64_t ne11 = src1->ne[1];
    ثابت uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    ثابت uint64_t nei0 = ids->ne[0];
    ثابت uint64_t nei1 = ids->ne[1];
    ثابت uint32_t nbi1 = (uint32_t)(ids->nb[1] / حجم(صحيح));

    ثابت uint64_t ne20 = dst->ne[0];
    ثابت uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ثابت منطقي x_non_contig = !ggml_vk_dim01_contiguous(src0);
    ثابت منطقي y_non_contig = !ggml_vk_dim01_contiguous(src1);

    ثابت منطقي f16_f32_kernel = src1->type == GGML_TYPE_F32;
    منطقي quantize_y = ctx->device->integer_dot_حاصلضرب && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne12, ne10, src0->type);

    vk_pipeline مقصد_fp16_vk_0 = اشاره_تهی;
    vk_pipeline مقصد_fp16_vk_1 = اشاره_تهی;
    إذا (x_non_contig) {
        مقصد_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, اشاره_تهی, src0->type);
    }
    إذا (y_non_contig) {
        مقصد_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, اشاره_تهی, src1->type);
    } وإلا {
        مقصد_fp16_vk_1 = ggml_vk_get_مقصد_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, GGML_TYPE_Q8_1, ne20, ne00) : اشاره_تهی;
    vk_pipeline مقصد_q8_1 = اشاره_تهی;

    إذا (dmmv == اشاره_تهی) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, src1->type, ne20, ne00);
        quantize_y = خطأ;
    }

    إذا (quantize_y) {
        مقصد_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    ثابت منطقي qx_needs_dequant = x_non_contig;
    ثابت منطقي qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    إذا (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_میانگیر_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT
    GGML_ASSERT(!qx_needs_dequant || مقصد_fp16_vk_0 != اشاره_تهی);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || مقصد_fp16_vk_1 != اشاره_تهی);  // NOLINT
    GGML_ASSERT(dmmv != اشاره_تهی);

    ثابت uint64_t x_ne = ggml_nelements(src0);
    ثابت uint64_t y_ne = ggml_nelements(src1);

    ثابت uint64_t qx_sz = ggml_vk_align_اندازه(ggml_type_اندازه(src0->type) * x_ne / ggml_blck_اندازه(src0->type), ctx->device->properties.limits.min_Storage_میانگیر_جابه_جایی_Alignment);
    ثابت uint64_t x_sz = x_non_contig ? ggml_vk_align_اندازه(ggml_type_اندازه(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_میانگیر_جابه_جایی_Alignment) : qx_sz;
    ثابت uint64_t y_sz = quantize_y ? (ggml_vk_align_اندازه(y_ne, 128) * ggml_type_اندازه(GGML_TYPE_Q8_1) / ggml_blck_اندازه(GGML_TYPE_Q8_1)) :
                                       (f16_f32_kernel ? حجم(عائم) * y_ne : حجم(ggml_fp16_t) * y_ne);

    {
        إذا (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_میانگیر_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_میانگیر_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        إذا (qx_needs_dequant && ctx->prealloc_اندازه_x < x_sz) {
            ctx->prealloc_اندازه_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا ((qy_needs_dequant || quantize_y) && ctx->prealloc_اندازه_y < y_sz) {
            ctx->prealloc_اندازه_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        إذا (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, مقصد_fp16_vk_0, 1);
        }
        إذا (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, مقصد_fp16_vk_1, 1);
        }
        إذا (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, مقصد_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, dmmv, nei1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[گره_idx + ctx->رقم_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids);
    vk_subbuffer d_F0 = d_D;
    vk_subbuffer d_X, d_Y;

    إذا (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } وإلا {
        d_X = d_Qx;
    }
    إذا (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } وإلا {
        d_Y = d_Qy;
    }

    إذا (x_non_contig) {
        إذا (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    إذا (x_non_contig) {
        GGML_ASSERT(x_sz == ggml_vk_align_اندازه(ggml_type_اندازه(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_میانگیر_جابه_جایی_Alignment));
        ggml_vk_cpy_مقصد_contiguous(ctx, subctx, مقصد_fp16_vk_0, src0, d_Qx, d_X);
    }
    إذا (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_اندازه(src1->type) * y_ne);
        إذا (ctx->prealloc_y_آخرین_pipeline_استفاده_شده != مقصد_fp16_vk_1.get() ||
            ctx->prealloc_y_آخرین_tensor_استفاده_شده != src1 ||
            ctx->prealloc_y_آخرین_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_مقصد_contiguous(ctx, subctx, مقصد_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_آخرین_pipeline_استفاده_شده = مقصد_fp16_vk_1.get();
            ctx->prealloc_y_آخرین_tensor_استفاده_شده = src1;
            ctx->prealloc_y_آخرین_k_padded = خطأ;
        }
    }
    إذا (quantize_y) {
        إذا (ctx->prealloc_y_آخرین_pipeline_استفاده_شده != مقصد_q8_1.get() ||
            ctx->prealloc_y_آخرین_tensor_استفاده_شده != src1 ||
            ctx->prealloc_y_آخرین_k_padded) {
            إذا (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_آخرین_pipeline_استفاده_شده = مقصد_q8_1.get();
            ctx->prealloc_y_آخرین_tensor_استفاده_شده = src1;
            ctx->prealloc_y_آخرین_k_padded = خطأ;
        }
    }

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    إذا (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[2] / ggml_type_اندازه(src1->type);
    }

    ثابت uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_گروه_شمار[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    إذا (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    إذا (ctx->رقم_additional_fused_ops > 0) {
        ثابت ggml_tensor * bias = cgraph->nodes[گره_idx + 1]->src[1];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);

        إذا (cgraph->nodes[گره_idx + 1]->op == GGML_OP_MUL) {
            fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE0;
        } وإلا {
            GGML_ASSERT(cgraph->nodes[گره_idx + 1]->op == GGML_OP_جمع_کن_ID);
            fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
        }
    }

    vk_subbuffer d_F1 = d_D;
    إذا (ctx->رقم_additional_fused_ops > 1) {
        ثابت ggml_tensor * scale = cgraph->nodes[گره_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, scale);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE1;
    }

    // Loop over the batch dimension
    لكل (uint32_t expert_i1 = 0; expert_i1 < nei1; ++expert_i1) {
        ثابت vk_mat_vec_id_افزودن_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            stride_batch_x, stride_batch_y, (uint32_t)(ne20 * ne21),
            fusion_flags,
            (uint32_t)nei0, (uint32_t)ne11, expert_i1, nbi1
        };
        ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
            {
                d_X,
                d_Y,
                d_D,
                d_F0,
                d_F1,
                d_ids,
            },
            pc, { groups_x, (uint32_t)nei0, groups_z });
    }

    إذا (x_non_contig) {
        ctx->prealloc_x_need_sync = درست;
    }
    إذا (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = درست;
    }
}

ساكن منطقي ggml_vk_use_mul_mat_vec_id(ثابت ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * ids = dst->src[2];
    أرجع ids->ne[1] <= 8 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));
}

منطقي ggml_vk_use_mul_mat_vec_id(ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {
    أرجع ggml_vk_use_mul_mat_vec_id(cgraph->nodes[گره_idx]);
}

فراغ ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {
    ggml_tensor * dst = cgraph->nodes[گره_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * src2 = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_id(" << src0 << ", " << src1 << ", " << src2 << ", " << dst << ")");
    إذا (ggml_vk_use_mul_mat_vec_id(cgraph, گره_idx)) {
        ggml_vk_mul_mat_vec_id_q_f16(ctx, subctx, cgraph, گره_idx);
    } وإلا {
        ggml_vk_mul_mat_id_q_f16(ctx, subctx, src0, src1, src2, dst);
    }
}

منطقي ggml_vk_flash_attn_scalar_shmem_support(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, منطقي f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_استفادهنشده(f32acc);
    // Needs to be kept up to date on shader changes
    ثابت uint32_t wg_اندازه = params.workgroup_اندازه;
    ثابت uint32_t Br = params.بلوک_ردیفها;
    ثابت uint32_t Bc = params.بلوک_تعداد_ستون;

    // BF16 uses the fp32 shader (FLOAT_TYPE=float)
    ثابت uint32_t float_type_اندازه = (device->fp16 && k_type != GGML_TYPE_BF16) ? حجم(ggml_fp16_t) : حجم(عائم);

    ثابت منطقي mmq = ggml_vk_fa_scalar_استفاده_میکند_mmq(device, k_type, v_type);

    // tmpsh is overestimated slightly
    ثابت uint32_t tmpsh = wg_اندازه * حجم(عائم);
    ثابت uint32_t tmpshv4 = wg_اندازه * 4 * float_type_اندازه;

    ثابت uint32_t masksh = Bc * (Br + 1) * float_type_اندازه;
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    ثابت uint32_t iq_shmem = 16 * float_type_اندازه;

    uint32_t Qf, kvsh, kblocksh_اندازه;
    إذا (mmq) {
        // block_b_cache: int32_t qs[8] + FLOAT_TYPEV2 ds
        ثابت uint32_t بلوک_b_اندازه = 8 * حجم(int32_t) + 2 * float_type_اندازه;
        Qf = Br * (hsk / 32) * بلوک_b_اندازه;

        // kvsh uses D = HSV (K goes through kblocksh instead)
        kvsh = params.shmem_staging ? Bc * (hsv / 4 + 1) * 4 * float_type_اندازه : 4 * float_type_اندازه;

        // The mixed MMQ shader uses a superset block_a_cache that fits every
        // FA-supported quant: int32_t qs[8] + uint32_t qh + FLOAT_TYPEV2 dm.
        // Single-scale types leave dm.y unused; non-Q5_* leave qh unused.
        ثابت uint32_t بلوک_a_اندازه = 8 * حجم(int32_t) + حجم(uint32_t) + 2 * float_type_اندازه;
        kblocksh_اندازه = params.shmem_staging ? Bc * (hsk / 32) * بلوک_a_اندازه : بلوک_a_اندازه;
    } وإلا {
        Qf = Br * (hsk / 4 + 1) * 4 * float_type_اندازه;

        ثابت uint32_t D = std::max(hsk, hsv);
        kvsh = params.shmem_staging ? Bc * (D / 4 + 1) * 4 * float_type_اندازه : 4 * float_type_اندازه;

        kblocksh_اندازه = 0;
    }

    ثابت uint32_t مجموع_اندازه = tmpsh + tmpshv4 + masksh + iq_shmem + Qf + kvsh + kblocksh_اندازه;
    ثابت منطقي supported = مجموع_اندازه <= device->properties.limits.max_Compute_Shared_حافظه_اندازه;

    VK_LOG_DEBUG("ggml_vk_flash_attn_scalar_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", mmq=" << mmq << ", total_size=" << مجموع_اندازه << ", supported=" << supported);

    أرجع supported;
}

منطقي ggml_vk_flash_attn_coopmat_shmem_support(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, منطقي f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_استفادهنشده(v_type);
    // Needs to be kept up to date on shader changes
    ثابت uint32_t Br = params.بلوک_ردیفها;
    ثابت uint32_t Bc = params.بلوک_تعداد_ستون;

    ثابت uint32_t MatBr = 16, MatBc = 16;

    ثابت uint32_t ردیف_تقسیم = Bc / MatBc;

    ثابت uint32_t hsk_pad = ROUNDUP_POW2(hsk, 16);
    ثابت uint32_t hsv_pad = ROUNDUP_POW2(hsv, 16);

    ثابت uint32_t acctype = f32acc ? 4 : 2;
    ثابت uint32_t f16vec4 = 8;

    ثابت uint32_t tmpsh = (Bc / MatBc) * حجم(عائم);
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    ثابت uint32_t iq_shmem = 16 * حجم(ggml_fp16_t);

    ثابت uint32_t qstride = hsk_pad / 4 + 2;
    ثابت uint32_t Qf = Br * qstride * f16vec4;

    ثابت uint32_t psh_stride = Br / 4 + 2;
    ثابت uint32_t Psh = Bc * psh_stride * f16vec4;

    ثابت uint32_t sfshstride = (hsk <= 128) ? (Br + 8) : Br;
    ثابت uint32_t sfsh = Bc * sfshstride * acctype;

    ثابت uint32_t kvshstride = (params.shmem_staging ? std::max(hsk_pad, hsv_pad) : MatBr) / 4 + 2;
    ثابت uint32_t vsh_stride = MatBc / 4 * ردیف_تقسیم;
    ثابت uint32_t ksh = ((kvshstride >= vsh_stride) ? (Bc * kvshstride) : (Bc * vsh_stride)) * f16vec4;

    // BF16 PVMat accumulator is f32 (no bf16 accumulator support), so pvsh is vec4 (16 bytes)
    ثابت uint32_t pvsh_elem_اندازه = (k_type == GGML_TYPE_BF16) ? 16u : f16vec4;
    ثابت uint32_t osh_stride = params.ردیف_تقسیم * MatBr / 4;
    ثابت uint32_t pvsh = MatBc * osh_stride * pvsh_elem_اندازه;

    ثابت uint32_t شیب = Br * acctype;

    ثابت uint32_t مجموع_اندازه = tmpsh + iq_shmem + Qf + Psh + sfsh + ksh + pvsh + شیب;
    ثابت منطقي supported = مجموع_اندازه <= device->properties.limits.max_Compute_Shared_حافظه_اندازه;

    VK_LOG_DEBUG("ggml_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", f32acc=" << f32acc << ", total_size=" << مجموع_اندازه << ", supported=" << supported);

    أرجع supported;
}

فراغ ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * q, ثابت ggml_tensor * k, ثابت ggml_tensor * v, ثابت ggml_tensor * ماسک, ثابت ggml_tensor * sinks, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_flash_attn((" << q << ", name=" << q->name << ", type=" << q->type << ", ne0=" << q->ne[0] << ", ne1=" << q->ne[1] << ", ne2=" << q->ne[2] << ", ne3=" << q->ne[3] << ", nb0=" << q->nb[0] << ", nb1=" << q->nb[1] << ", nb2=" << q->nb[2] << ", nb3=" << q->nb[3];
    std::cerr << "), (" << k << ", name=" << k->name << ", type=" << k->type << ", ne0=" << k->ne[0] << ", ne1=" << k->ne[1] << ", ne2=" << k->ne[2] << ", ne3=" << k->ne[3] << ", nb0=" << k->nb[0] << ", nb1=" << k->nb[1] << ", nb2=" << k->nb[2] << ", nb3=" << k->nb[3];
    std::cerr << "), (" << v << ", name=" << v->name << ", type=" << v->type << ", ne0=" << v->ne[0] << ", ne1=" << v->ne[1] << ", ne2=" << v->ne[2] << ", ne3=" << v->ne[3] << ", nb0=" << v->nb[0] << ", nb1=" << v->nb[1] << ", nb2=" << v->nb[2] << ", nb3=" << v->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    إذا (sinks) {
        std::cerr << "), (" << sinks << ", name=" << sinks->name << ", type=" << sinks->type << ", ne0=" << sinks->ne[0] << ", ne1=" << sinks->ne[1] << ", ne2=" << sinks->ne[2] << ", ne3=" << sinks->ne[3] << ", nb0=" << sinks->nb[0] << ", nb1=" << sinks->nb[1] << ", nb2=" << sinks->nb[2] << ", nb3=" << sinks->nb[3];
    }
    std::cerr << "))");

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ثابت uint32_t nem0 = ماسک ? ماسک->ne[0] : 0;
    ثابت uint32_t nem1 = ماسک ? ماسک->ne[1] : 0;
    ثابت uint32_t nem2 = ماسک ? ماسک->ne[2] : 0;
    ثابت uint32_t nem3 = ماسک ? ماسک->ne[3] : 0;

    ثابت uint32_t HSK = nek0;
    ثابت uint32_t HSV = nev0;
    uint32_t N = neq1;
    ثابت uint32_t KV = nek1;

    GGML_ASSERT(ne0 == HSV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_اندازه(q->type));
    GGML_ASSERT(nbk0 == ggml_type_اندازه(k->type));
    GGML_ASSERT(nbv0 == ggml_type_اندازه(v->type));

    GGML_ASSERT(neq0 == HSK);

    GGML_ASSERT(neq1 == N);

    GGML_ASSERT(nev1 == nek1);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    أكد(dst->type == GGML_TYPE_F32);
    أكد(q->type == GGML_TYPE_F32);
    uint32_t gqa_نسبت = 1;
    uint32_t qk_نسبت = neq2 / nek2;
    uint32_t workgroups_x = (uint32_t)neq1;
    uint32_t workgroups_y = (uint32_t)neq2;
    uint32_t workgroups_z = (uint32_t)neq3;

    ثابت منطقي f32acc = !ctx->device->fp16 || dst->op_params[3] == GGML_PREC_F32 || k->type == GGML_TYPE_BF16;

    // dequant K/V once into an f16 scratch, reordered KV layout so FA can read without a stride
    خودکار is_dense_kv_cache = [](ثابت ggml_tensor * t) {
        أرجع t->nb[0] == ggml_type_اندازه(t->type) &&
               t->nb[2] == ggml_ردیف_اندازه(t->type, t->ne[0]) &&
               t->nb[1] == t->nb[2] * t->ne[2] &&
               (t->ne[3] == 1 || t->nb[3] == t->nb[1] * t->ne[1]);
    };
    ثابت منطقي k_quant = k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16 && k->type != GGML_TYPE_F32;
    ثابت منطقي v_quant = v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_BF16 && v->type != GGML_TYPE_F32;
    ثابت منطقي kv_raw = (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) && k->type == v->type;
    // gated to AMD, except RDNA4, which regressed in benchmarks
    ثابت منطقي kv_raw_strided = ctx->device->vendor_id == VK_VENDOR_ID_AMD &&
                                ctx->device->architecture != vk_device_architecture::AMD_RDNA4 &&
                                neq1 >= 256 &&
                                kv_raw &&
                                (k->nb[1] != (uint64_t)HSK * ggml_type_اندازه(k->type) ||
                                 v->nb[1] != (uint64_t)HSV * ggml_type_اندازه(v->type)) &&
                                (HSK % 8) == 0 && (HSV % 8) == 0;
    ثابت size_t kv_scratch_ts = kv_raw_strided ? ggml_type_اندازه(k->type) : حجم(ggml_fp16_t);
    ثابت منطقي use_dequant_kv = ((k_quant && v_quant) || kv_raw_strided) && neq1 >= 64 &&
                                is_dense_kv_cache(k) && is_dense_kv_cache(v) &&
                                (uint64_t)ggml_nelements(k) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_میانگیر_Range &&
                                (uint64_t)ggml_nelements(v) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_میانگیر_Range &&
                                (kv_raw_strided ||
                                 (ctx->device->pipeline_dequant_transpose[k->type] != اشاره_تهی &&
                                  ctx->device->pipeline_dequant_transpose[v->type] != اشاره_تهی)) &&
                                // coopmat2 path does not benefit from the f16 scratch
                                !ctx->device->coopmat2 &&
                                // Intel Xe1 regresses, see PR 25494
                                (ctx->device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 (ctx->device->coopmat_support && ctx->device->architecture != vk_device_architecture::INTEL_XE1));
    ثابت ggml_type k_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : k->type;
    ثابت ggml_type v_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : v->type;

    // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
    // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
    vk_fa_tuning_params tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, 512, KV, k_type_eff, v_type_eff, f32acc);
    ثابت uint32_t max_gqa = std::min(tuning_params.بلوک_ردیفها, 32u);

    إذا (N <= 8 && qk_نسبت > 1 && qk_نسبت <= max_gqa &&
        qk_نسبت * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
        // grouped query attention - make the N dimension equal to gqa_ratio, reduce
        // workgroups proportionally in y dimension. The shader will detect gqa_ratio > 1
        // and change addressing calculations to index Q's dimension 2.
        gqa_نسبت = qk_نسبت;
        N = gqa_نسبت;
        workgroups_y /= gqa_نسبت;
    }

    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);

    عائم scale         = 1.0f;
    عائم max_bias      = 0.0f;
    عائم logit_softcap = 0.0f;

    کپی_حافظه(&scale,         (ثابت عائم *) dst->op_params + 0, حجم(عائم));
    کپی_حافظه(&max_bias,      (ثابت عائم *) dst->op_params + 1, حجم(عائم));
    کپی_حافظه(&logit_softcap, (ثابت عائم *) dst->op_params + 2, حجم(عائم));

    إذا (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // Sparse mask hint (op_params[4]): compact the <= n_kv_max finite positions and gather only those.
    ثابت int32_t n_kv_max = ماسک ? ggml_get_op_params_i32(dst, 4) : 0;
    ساكن ثابت منطقي disable_sparse = getenv("GGML_VK_FA_SPARSE_DISABLE") != اشاره_تهی;
    ثابت منطقي kv_f16 = k_type_eff == GGML_TYPE_F16 && v_type_eff == GGML_TYPE_F16;
    // cm2 dense is fast, so it needs a larger reduction to win.
    // With quantized K/V, sparse only breaks even around 16x (measured on RDNA3/RDNA4).
    ثابت int64_t min_نسبت = tuning_params.path == FA_COOPMAT2 ? 4 : (kv_f16 ? 2 : 16);
    // coopmat2 vector decode requires 8B strides.
    خودکار sparse_gather_aligned = [](ثابت ggml_tensor * t) {
        أرجع (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) ||
               (t->nb[1] | t->nb[2] | t->nb[3]) % (4 * حجم(ggml_fp16_t)) == 0;
    };
    ثابت منطقي use_sparse = !disable_sparse && n_kv_max > 0 && ماسک &&
                            max_bias == 0.0f && logit_softcap == 0.0f &&
                            (tuning_params.path != FA_COOPMAT2 || (sparse_gather_aligned(k) && sparse_gather_aligned(v))) &&
                            nem0 == KV &&
                            (int64_t)KV >= std::max<int64_t>(4096, min_نسبت * (int64_t)n_kv_max) &&
                            (gqa_نسبت > 1 || (tuning_params.path == FA_SCALAR && N == 1));

    ثابت uint32_t q_stride = (uint32_t)(nbq1 / ggml_type_اندازه(q->type));
    uint32_t k_stride = (uint32_t)(nbk1 / ggml_type_اندازه(k->type));
    uint32_t v_stride = (uint32_t)(nbv1 / ggml_type_اندازه(v->type));

    // For F32, the shader treats it as a block of size 4 (for vec4 loads)
    إذا (k->type == GGML_TYPE_F32) {
        k_stride /= 4;
    }
    إذا (v->type == GGML_TYPE_F32) {
        v_stride /= 4;
    }

    uint32_t nbk2_eff = (uint32_t)nbk2, nbk3_eff = (uint32_t)nbk3;
    uint32_t nbv2_eff = (uint32_t)nbv2, nbv3_eff = (uint32_t)nbv3;
    إذا (use_dequant_kv) {
        k_stride = HSK;
        v_stride = HSV;
        nbk2_eff = (uint32_t)((uint64_t)HSK * KV * حجم(ggml_fp16_t));
        nbk3_eff = (uint32_t)((uint64_t)HSK * KV * nek2 * حجم(ggml_fp16_t));
        nbv2_eff = (uint32_t)((uint64_t)HSV * KV * حجم(ggml_fp16_t));
        nbv3_eff = (uint32_t)((uint64_t)HSV * KV * nev2 * حجم(ggml_fp16_t));
    }
    ثابت uint32_t alignment = tuning_params.بلوک_تعداد_ستون;
    منطقي aligned = (KV % alignment) == 0 &&
                   // the "aligned" shader variant will forcibly align strides, for performance
                   (q_stride & 7) == 0 && (k_stride & 7) == 0 && (v_stride & 7) == 0;

    // Need to use the coopmat2 variant that clamps loads when HSK/HSV aren't sufficiently aligned.
    إذا (((HSK | HSV) % 16) != 0 && tuning_params.path == FA_COOPMAT2) {
        aligned = خطأ;
    }

    // Only use mask opt when the mask is fairly large. This hasn't been tuned extensively.
    منطقي use_ماسک_opt = ماسک && !use_sparse && nem1 >= 32 && nem0 * nem1 > 32768 && nem0 >= tuning_params.بلوک_تعداد_ستون * 16
                        && (ctx->device->architecture != vk_device_architecture::AMD_GCN || HSK > 256 || HSV > 256);
    vk_fa_pipeline_state fa_pipeline_state = get_fa_pipeline_state(ctx->device, tuning_params, HSK, HSV, aligned, f32acc,
                                                                   ماسک != اشاره_تهی, use_ماسک_opt, logit_softcap != 0, use_sparse, k_type_eff, v_type_eff);

    vk_pipeline pipeline = اشاره_تهی;

    منطقي xe_fa_opt = خطأ;
    منطقي fa_copy_qstate = خطأ;
    منطقي xe_fa_supported_platform =
        (ctx->device.get()->architecture == INTEL_XE2 && ctx->device.get()->properties.deviceID != 0xFD80 && ctx->device.get()->properties.deviceID != 0xFD81) ||
        (ctx->device.get()->architecture == INTEL_XE1 && ctx->device.get()->coopmat_support && ctx->device.get()->uma);
    منطقي xe_fa_supported_usage = neq0 % 32 == 0 && nev0 % 16 == 0 && q->nb[1] > q->nb[2] && k->nb[1] > k->nb[2] && v->nb[1] > v->nb[2] && ماسک != اشاره_تهی;
    منطقي xe_fa_supported_dtype = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && (ماسک != اشاره_تهی && ماسک->type == GGML_TYPE_F16);
    std::pair<vk_pipeline, vk_pipeline> xe_fa_pipeline_dual_phases = { اشاره_تهی , اشاره_تهی };
    vk_pipeline xe_fa_pipeline = اشاره_تهی;
    size_t اندازه_p = 0;
    size_t اندازه_گروه_max = 0;

    {
        std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
        خودکار &pipelines = ctx->device->pipeline_flash_attn_f32_f16;
        خودکار it = pipelines.find(fa_pipeline_state);
        إذا (it != pipelines.end()) {
            pipeline = it->second;
        } وإلا {
            pipelines[fa_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
        }
    }

    أكد(pipeline);
    // Compile early to initialize wg_denoms.
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    uint32_t تقسیم_kv = KV;
    uint32_t تقسیم_k = 1;

    // Intel Alchemist prefers more workgroups
    ثابت uint32_t shader_core_شمار_multiplier = (ctx->device->vendor_id == VK_VENDOR_ID_INTEL && ctx->device->architecture != INTEL_XE2) ? 2 : 1;

    // Use a placeholder core count if one isn't available. split_k is a big help for perf.
    ثابت uint32_t shader_core_شمار = ctx->device->shader_core_شمار ? ctx->device->shader_core_شمار * shader_core_شمار_multiplier : 16;

    ثابت uint32_t Br = fa_pipeline_state.Br;
    ثابت uint32_t Bc = fa_pipeline_state.Bc;

    GGML_ASSERT(Br == pipeline->wg_denoms[0]);
    ثابت uint32_t Tr = CEIL_DIV(N, Br);

    // Try to use split_k when KV is large enough to be worth the overhead.
    // Sparse: split_kv carries n_kv_max, split_k partitions its blocks for occupancy.
    إذا (use_sparse) {
        تقسیم_kv = (uint32_t)n_kv_max;
        ثابت uint32_t مجموع_بلوکها = CEIL_DIV((uint32_t)n_kv_max, Bc);
        ثابت uint32_t پایه_wgs = (gqa_نسبت > 1 ? workgroups_x : Tr) * workgroups_y * workgroups_z;
        إذا (پایه_wgs < shader_core_شمار * 2) {
            تقسیم_k = shader_core_شمار * 2 / پایه_wgs;
        }
        تقسیم_k = std::max(1u, std::min(تقسیم_k, مجموع_بلوکها));
        // Match the shader's per-split block count so no split is empty.
        ثابت uint32_t در_هر_بلوکها = CEIL_DIV(مجموع_بلوکها, تقسیم_k);
        تقسیم_k = CEIL_DIV(مجموع_بلوکها, در_هر_بلوکها);
    } وإلا إذا (gqa_نسبت > 1 && workgroups_x <= Br) {
        تقسیم_k = shader_core_شمار * 2 / (workgroups_x * workgroups_y * workgroups_z);
    } وإلا إذا (gqa_نسبت <= 1) {
        uint32_t مجموع_wgs_no_تقسیم = Tr * workgroups_y * workgroups_z;
        إذا (مجموع_wgs_no_تقسیم < shader_core_شمار * 2) {
            تقسیم_k = shader_core_شمار * 2 / مجموع_wgs_no_تقسیم;
        }
    }

    إذا (!use_sparse && تقسیم_k > 1) {
        // Try to evenly split KV into split_k chunks, but it needs to be a multiple
        // of "align", so recompute split_k based on that.
        تقسیم_kv = ROUNDUP_POW2(std::max(1u, KV / تقسیم_k), alignment);
        تقسیم_k = CEIL_DIV(KV, تقسیم_kv);
        xe_fa_opt = xe_fa_supported_platform && xe_fa_supported_usage && xe_fa_supported_dtype;
        إذا (xe_fa_opt) {
            ثابت uint32_t تقسیم_p_اندازه = 32;
            ثابت size_t max_dim = (nek1 + تقسیم_p_اندازه - 1) / تقسیم_p_اندازه;
            ثابت size_t p_dim = max_dim * تقسیم_p_اندازه;
#إذا defined(VK_KHR_cooperative_ماتریس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            خودکار& pipelines = ctx->device->pipeline_xe_fa_decode_dual_phases;
            خودکار it = pipelines.find({ (uint32_t)neq0, (uint32_t)nev0, qk_نسبت, (uint32_t)neq1 });
            إذا (it != pipelines.end()) {
                xe_fa_pipeline_dual_phases = it->second;
            } وإلا {
                pipelines[{(uint32_t)neq0, (uint32_t)nev0, qk_نسبت, (uint32_t)neq1}] = xe_fa_pipeline_dual_phases = std::make_pair(std::make_shared<vk_pipeline_struct>(), std::make_shared<vk_pipeline_struct>());
            }
#پایان_اگر
            إذا (xe_fa_pipeline_dual_phases.اولی == اشاره_تهی || xe_fa_pipeline_dual_phases.second == اشاره_تهی) {
                xe_fa_opt = خطأ;
                fa_copy_qstate = خطأ;
            } وإلا {
                اندازه_p = neq1 * neq2 * p_dim * neq3 * حجم(ggml_fp16_t);
                اندازه_گروه_max = neq1 * neq2 * max_dim * neq3 * حجم(عائم);
                size_t موقت_اندازه = ggml_nelements(q) * حجم(ggml_fp16_t) + اندازه_p + اندازه_گروه_max;
                fa_copy_qstate = درست;
                إذا (ctx->prealloc_اندازه_x < موقت_اندازه) {
                    ctx->prealloc_اندازه_x = موقت_اندازه;
                    ggml_vk_preallocate_buffers(ctx, subctx);
                }

                إذا (ctx->prealloc_x_need_sync) {
                    ggml_vk_sync_buffers(ctx, subctx);
                }
            }
        }
    }

    إذا (xe_fa_opt == درست) {
        use_ماسک_opt = خطأ;
    }

    // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)
    // and the per-row m and L values (ne1 rows). We store all the matrices first, followed by the rows.
    // For matrices, the order is (inner to outer) [HSV, ne1, k, ne2, ne3].
    // For L/M, the order is (inner to outer) [ne1, k, ne2, ne3].
    ثابت uint64_t تقسیم_k_اندازه = تقسیم_k > 1 ? (HSV * ne1 * حجم(عائم) + ne1 * حجم(عائم) * 2) * تقسیم_k * ne2 * ne3 : 0;
    إذا (تقسیم_k_اندازه > ctx->device->properties.limits.max_Storage_میانگیر_Range) {
        GGML_ABORT("Requested preallocation size is too large");
    }
    إذا (ctx->prealloc_اندازه_تقسیم_k < تقسیم_k_اندازه) {
        ctx->prealloc_اندازه_تقسیم_k = تقسیم_k_اندازه;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    ثابت uint32_t ماسک_opt_رقم_dwords = CEIL_DIV(nem0, 16 * Bc);
    ثابت uint64_t ماسک_opt_اندازه = حجم(uint32_t) * ماسک_opt_رقم_dwords * CEIL_DIV(nem1, Br) * nem2 * nem3;

    vk_pipeline pipeline_fa_ماسک_opt = اشاره_تهی;
    إذا (use_ماسک_opt) {
        {
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            خودکار &pipelines = ctx->device->pipeline_fa_ماسک_opt;
            خودکار it = pipelines.find({Br, Bc});
            إذا (it != pipelines.end()) {
                pipeline_fa_ماسک_opt = it->second;
            } وإلا {
                pipelines[{Br, Bc}] = pipeline_fa_ماسک_opt = std::make_shared<vk_pipeline_struct>();
            }
        }
        أكد(pipeline_fa_ماسک_opt);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_fa_ماسک_opt, 1);

        إذا (ctx->prealloc_اندازه_y < ماسک_opt_اندازه) {
            ctx->prealloc_اندازه_y = ماسک_opt_اندازه;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    // Sparse index scratch reuses prealloc_y (mutually exclusive with mask opt).
    ثابت uint64_t sparse_idx_اندازه = use_sparse
        ? حجم(int32_t) * (uint64_t)n_kv_max * nem1 * nem2 * nem3
        : 0;
    vk_pipeline sparse_compact_pipeline = ctx->device->fa_sparse_compact_use_subgroups
        ? ctx->device->pipeline_fa_sparse_compact_subgroup
        : ctx->device->pipeline_fa_sparse_compact;
    إذا (use_sparse) {
        ggml_pipeline_request_descriptor_sets(ctx, sparse_compact_pipeline, 1);
        إذا (ctx->prealloc_اندازه_y < sparse_idx_اندازه) {
            ctx->prealloc_اندازه_y = sparse_idx_اندازه;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ثابت uint32_t n_سر_kv   = neq2;
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floorf(log2f((عائم) n_سر_kv));
    ثابت عائم m0 = powf(2.0f, -(max_bias       ) / n_سر_log2);
    ثابت عائم m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    vk_subbuffer q_buf = ggml_vk_tensor_subbuffer(ctx, q);
    vk_subbuffer k_buf = ggml_vk_tensor_subbuffer(ctx, k);
    vk_subbuffer v_buf = ggml_vk_tensor_subbuffer(ctx, v);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer ماسک_buf = ماسک ? ggml_vk_tensor_subbuffer(ctx, ماسک) : q_buf;
    vk_subbuffer sinks_buf = sinks ? ggml_vk_tensor_subbuffer(ctx, sinks) : q_buf;
    vk_subbuffer ماسک_opt_buf = use_ماسک_opt ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    vk_subbuffer sparse_buf = use_sparse ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    إذا (use_ماسک_opt || use_sparse) {
        // the mask opt bits and the sparse index list overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_آخرین_pipeline_استفاده_شده = اشاره_تهی;
        ctx->prealloc_y_آخرین_tensor_استفاده_شده = اشاره_تهی;
    }

    إذا (use_dequant_kv) {
        ثابت uint64_t k_sz = (uint64_t)ggml_nelements(k) * kv_scratch_ts;
        ثابت uint64_t v_sz = (uint64_t)ggml_nelements(v) * kv_scratch_ts;
        إذا (ctx->prealloc_اندازه_x < k_sz + v_sz) {
            ctx->prealloc_اندازه_x = k_sz + v_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        vk_subbuffer k_dst = vk_subbuffer{ ctx->prealloc_x, 0,    k_sz };
        vk_subbuffer v_dst = vk_subbuffer{ ctx->prealloc_x, k_sz, v_sz };
        ثابت uint32_t k_nel = (uint32_t)ggml_nelements(k);
        ثابت uint32_t v_nel = (uint32_t)ggml_nelements(v);
        إذا (kv_raw_strided) {
            vk_pipeline cp_k = ggml_vk_get_cpy_pipeline(ctx, k, اشاره_تهی, k->type);
            vk_pipeline cp_v = ggml_vk_get_cpy_pipeline(ctx, v, اشاره_تهی, v->type);
            ggml_pipeline_request_descriptor_sets(ctx, cp_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, cp_v, 1);
            إذا (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            خودکار make_pc = [](uint32_t hs, uint32_t nh, uint32_t kv, uint32_t nel) {
                vk_op_unary_افزودن_constants pc{};
                pc.ne = nel;
                pc.ne00 = hs; pc.ne01 = nh;      pc.ne02 = kv; pc.ne03 = nel / (hs * nh * kv);
                pc.nb00 = 1;  pc.nb01 = hs;      pc.nb02 = hs * nh; pc.nb03 = hs * nh * kv;
                pc.ne10 = hs; pc.ne11 = nh;      pc.ne12 = kv; pc.ne13 = pc.ne03;
                pc.nb10 = 1;  pc.nb11 = hs * kv; pc.nb12 = hs; pc.nb13 = hs * kv * nh;
                init_pushconst_fastdiv(pc);
                أرجع pc;
            };
            خودکار cpy_elems = [](uint32_t ne) -> std::array<uint32_t, 3> {
                إذا (ne > 262144) { أرجع { 512, 512, CEIL_DIV(ne, 262144) }; }
                إذا (ne > 512)    { أرجع { 512, CEIL_DIV(ne, 512), 1 }; }
                أرجع { ne, 1, 1 };
            };
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_k, { k_buf, k_dst },
                make_pc((uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, k_nel), cpy_elems(k_nel));
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_v, { v_buf, v_dst },
                make_pc((uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, v_nel), cpy_elems(v_nel));
        } وإلا {
            vk_pipeline tr_k = ctx->device->pipeline_dequant_transpose[k->type];
            vk_pipeline tr_v = ctx->device->pipeline_dequant_transpose[v->type];
            ggml_pipeline_request_descriptor_sets(ctx, tr_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, tr_v, 1);
            إذا (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            { ثابت std::vector<uint32_t> pc = { (uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, 0, k_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_k, { k_buf, k_dst }, pc, { k_nel, 1, 1 }); }
            { ثابت std::vector<uint32_t> pc = { (uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, 0, v_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_v, { v_buf, v_dst }, pc, { v_nel, 1, 1 }); }
        }
        ggml_vk_sync_buffers(ctx, subctx);
        k_buf = k_dst;
        v_buf = v_dst;
    }

    uint32_t ماسک_n_سر_log2 = ((sinks != اشاره_تهی) << 24) | n_سر_log2;

    إذا (use_ماسک_opt)
    {
        ثابت vk_op_flash_attn_ماسک_opt_افزودن_constants opt_pc = {
            nem0,
            nem1,
            nem2,
            (uint32_t)(ماسک->nb[1] / حجم(ggml_fp16_t)),
            (uint32_t)(ماسک->nb[2] / حجم(ggml_fp16_t)),
            (uint32_t)(ماسک->nb[3] / حجم(ggml_fp16_t)),
            ماسک_opt_رقم_dwords,
            ماسک_opt_رقم_dwords * CEIL_DIV(nem1, Br),
            ماسک_opt_رقم_dwords * CEIL_DIV(nem1, Br) * nem2,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline_fa_ماسک_opt,
                                  { ماسک_buf, ماسک_opt_buf }, opt_pc,
                                  { ماسک_opt_رقم_dwords, CEIL_DIV(nem1, Br), nem2 * nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    إذا (use_sparse)
    {
        ثابت vk_op_flash_attn_sparse_compact_افزودن_constants sc_pc = {
            KV,
            nem1,
            nem2,
            (uint32_t)(ماسک->nb[1] / حجم(ggml_fp16_t)),
            (uint32_t)(ماسک->nb[2] / حجم(ggml_fp16_t)),
            (uint32_t)(ماسک->nb[3] / حجم(ggml_fp16_t)),
            (uint32_t)n_kv_max,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, sparse_compact_pipeline,
                                  { ماسک_buf, sparse_buf }, sc_pc,
                                  { nem1, nem2, nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ثابت vk_flash_attn_افزودن_constants pc = { N, KV,
                                              (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                                              (uint32_t)neq2, (uint32_t)neq3,
                                              (uint32_t)nek2, (uint32_t)nek3,
                                              (uint32_t)nev2, (uint32_t)nev3,
                                              nem1, nem2, nem3,
                                              q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
                                              k_stride, nbk2_eff, nbk3_eff,
                                              v_stride, nbv2_eff, nbv3_eff,
                                              scale, max_bias, logit_softcap,
                                              ماسک_n_سر_log2, m0, m1,
                                              gqa_نسبت, تقسیم_kv, تقسیم_k };

    إذا (xe_fa_opt && تقسیم_k > 1) {
        خودکار فوقانی_power_از_2 = [&](uint32_t درون) {
            GGML_ASSERT(درون != 0);
            إذا (درون <= 1) أرجع 1u;
            uint32_t ret = درون - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            أرجع ret + 1;
        };
        خودکار مقصد_fp16_vk_0 = ggml_vk_get_مقصد_fp16(ctx, q->type);
        ثابت uint32_t بیرون_dim_در_هر_wg = qk_نسبت > 16 ? 8 : 16;
        size_t x_ne = ggml_nelements(q);
        size_t موقت_buf_جابه_جایی = 0;
        uint32_t سر_stride_k = uint32_t(nbk2 / ggml_type_اندازه(k->type));
        uint32_t سر_stride_v = uint32_t(nbv2 / ggml_type_اندازه(v->type));
        uint32_t batch_stride_q = uint32_t(nbq3 / ggml_type_اندازه(q->type));
        uint32_t batch_stride_k = uint32_t(nbk3 / ggml_type_اندازه(k->type));
        uint32_t batch_stride_v = uint32_t(nbv3 / ggml_type_اندازه(v->type));
        uint32_t batch_stride_m = ماسک ? uint32_t(ماسک->nb[3] / ggml_type_اندازه(ماسک->type)) : 0u;
        uint32_t batch_stride_o = uint32_t(nb3 / ggml_type_اندازه(dst->type));
        vk_fa_xe_opt_افزودن_constants pc_ph1 = { (uint32_t)nek1, (uint32_t)neq1, (uint32_t)neq2, (uint32_t)nek2, qk_نسبت, 1, (sinks != اشاره_تهی) ? 1u : 0u, (uint32_t)k_stride, سر_stride_k,
            batch_stride_q, batch_stride_k, batch_stride_v, batch_stride_m, batch_stride_o, scale };
        vk_fa_xe_opt_افزودن_constants pc_ph2 = pc_ph1;
        pc_ph2.nbkv_tok = v_stride;
        pc_ph2.nbkv_سر = سر_stride_v;
        vk_subbuffer q_موقت_buf = fa_copy_qstate ? ggml_vk_subbuffer(ctx, ctx->prealloc_x, موقت_buf_جابه_جایی) : q_buf;
        موقت_buf_جابه_جایی += fa_copy_qstate ? x_ne * حجم(ggml_fp16_t) : 0;
        vk_subbuffer p_موقت_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, موقت_buf_جابه_جایی);
        موقت_buf_جابه_جایی += اندازه_p;
        vk_subbuffer max_موقت_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, موقت_buf_جابه_جایی);
        موقت_buf_جابه_جایی += اندازه_گروه_max;
        uint32_t xe_native_sub_گروه_اندازه = ctx->device.get()->architecture == INTEL_XE1 ? 8 : 16;
        uint32_t aligned_gqa_نسبت = فوقانی_power_از_2(qk_نسبت);
        uint32_t بیرون_در_هر_wg_ph1 = std::min(256u / xe_native_sub_گروه_اندازه, (uint32_t)neq1);
        uint32_t بیرون_در_هر_wg_ph2 = std::min(std::max(16u / aligned_gqa_نسبت, 1u), (uint32_t)neq1);
        uint32_t ph1_wg = ((neq1 + بیرون_در_هر_wg_ph1 - 1) / بیرون_در_هر_wg_ph1) * nek2;
        uint32_t ph2_wg = ((neq1 + بیرون_در_هر_wg_ph2 - 1) / بیرون_در_هر_wg_ph2) * ne0 / بیرون_dim_در_هر_wg;
        إذا (fa_copy_qstate) {
            ثابت std::vector<uint32_t> pc_cpy_fp16 =
            { (uint32_t)q->ne[0], (uint32_t)q->ne[1], (uint32_t)q->ne[2], (uint32_t)q->ne[3], (uint32_t)(x_ne) };
            ggml_vk_sync_buffers(ctx, subctx);
            ggml_pipeline_request_descriptor_sets(ctx, مقصد_fp16_vk_0, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, مقصد_fp16_vk_0, { q_buf, q_موقت_buf }, pc_cpy_fp16, { (uint32_t)(x_ne), 1, 1 });
        }

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.اولی, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.اولی,
            { q_موقت_buf, k_buf, ماسک_buf, p_موقت_buf, max_موقت_buf },
            pc_ph1, { (uint32_t)ph1_wg, (uint32_t)nek1, (uint32_t)neq3 });

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.second, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.second,
            { p_موقت_buf, v_buf, max_موقت_buf, sinks_buf, dst_buf },
            pc_ph2, { (uint32_t)ph2_wg, (uint32_t)nev2, (uint32_t)neq3 });

        ctx->prealloc_x_need_sync = درست;
    } وإلا إذا (تقسیم_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_تقسیم_k_reduce, 1);

        إذا (ctx->prealloc_تقسیم_k_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        // We reuse workgroups_x to mean the number of splits, so we need to
        // cancel out the divide by wg_denoms[0].
        uint32_t dispatch_x;
        إذا (gqa_نسبت > 1) {
            workgroups_x *= pipeline->wg_denoms[0];
            dispatch_x = تقسیم_k * workgroups_x;
        } وإلا {
            dispatch_x = Tr * تقسیم_k * pipeline->wg_denoms[0];
        }

        vk_subbuffer تقسیم_k_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_تقسیم_k, 0);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, ماسک_buf, sinks_buf, تقسیم_k_buf, ماسک_opt_buf, sparse_buf},
                                    pc, { dispatch_x, workgroups_y, workgroups_z });

        ggml_vk_sync_buffers(ctx, subctx);
        ثابت vk_op_flash_attn_تقسیم_k_reduce_افزودن_constants pc2 = { HSV, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3, تقسیم_k, (sinks != اشاره_تهی) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_flash_attn_تقسیم_k_reduce,
                                    {تقسیم_k_buf, sinks_buf, dst_buf},
                                    pc2, { (uint32_t)ne1, HSV, (uint32_t)(ne2 * ne3) });
        ctx->prealloc_تقسیم_k_need_sync = درست;
    } وإلا {
        إذا (gqa_نسبت > 1) {
            // When using gqa, we want one actual workgroup per batch, so cancel out wg_denoms
            workgroups_x *= pipeline->wg_denoms[0];
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, ماسک_buf, sinks_buf, dst_buf, ماسک_opt_buf, sparse_buf},
                                    pc, { workgroups_x, workgroups_y, workgroups_z });
    }

    إذا (use_dequant_kv) {
        ctx->prealloc_x_need_sync = درست;
    }
    إذا (use_ماسک_opt || use_sparse) {
        ctx->prealloc_y_need_sync = درست;
    }
}

ساكن vk_conv_shapes ggml_vk_conv_select_shape(ggml_backend_vk_context * ctx, uint32_t K, uint32_t NPQ) {
    خودکار n_tiles = [&](vk_conv_shapes s) {
        أرجع CEIL_DIV(K, vk_conv_بلوک_sizes[s].K)
            * CEIL_DIV(NPQ, vk_conv_بلوک_sizes[s].NPQ);
    };

    // We can't query number of shader cores on Intel, use 32 as a placeholder
    // so small convolutions will still choose a smaller tile.
    ثابت uint32_t shader_core_شمار = ctx->device->shader_core_شمار > 0 ? ctx->device->shader_core_شمار : 32;

    // 128x128 isn't used with cm1 due to shared memory size; fall through to a smaller tile.
    منطقي allow_128x128 = درست;
#إذا defined(VK_KHR_cooperative_ماتریس) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    إذا (!ctx->device->coopmat2 && ctx->device->coopmat_support && ctx->device->coopmat_support_16x16x16_f16acc) {
        allow_128x128 = خطأ;
    }
#پایان_اگر

    إذا (allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_128x128) >= shader_core_شمار * 2) {
        أرجع CONV_SHAPE_128x128;
    } وإلا إذا (K <= 32 && n_tiles(CONV_SHAPE_32x256) >= shader_core_شمار * 2) {
        أرجع CONV_SHAPE_32x256;
    } وإلا إذا (K <= 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_شمار * 2) {
        أرجع CONV_SHAPE_64x128;
    } وإلا إذا (!allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_شمار * 2) {
        // cm1 fallback for large K when 128x128 isn't available
        أرجع CONV_SHAPE_64x128;
    } وإلا {
        أرجع CONV_SHAPE_64x32;
    }
}

ساكن vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * dst, ggml_op op) {
    بدل (op) {
    حالة GGML_OP_GET_ردیفها:
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        إذا (src0->type == GGML_TYPE_I32) {
            // i32 src only supports i32 result
            GGML_ASSERT(dst->type == GGML_TYPE_I32);
            أرجع ctx->device->pipeline_get_ردیفها[src0->type];
        }
        إذا (dst->type == GGML_TYPE_F16) {
            أرجع ctx->device->pipeline_get_ردیفها[src0->type];
        }
        إذا (dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_get_ردیفها_f32[src0->type];
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_GET_ردیفها_پشت:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_get_ردیفها_پشت_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_ACC:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_acc_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_مجموعه:
        إذا (src0->type == src1->type && src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32)) {
            أرجع ctx->device->pipeline_مجموعه_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_جمع_کن:
    حالة GGML_OP_SUB:
    حالة GGML_OP_MUL:
    حالة GGML_OP_DIV:
        إذا ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)) {
            أرجع اشاره_تهی;
        }
        بدل (op) {
        حالة GGML_OP_جمع_کن:
        {
            إذا (ctx->رقم_additional_fused_ops > 0) {
                إذا (ctx->do_جمع_کن_rms_partials) {
                    أرجع ctx->device->pipeline_multi_جمع_کن_rms[ctx->رقم_additional_fused_ops];
                } وإلا {
                    أرجع ctx->device->pipeline_multi_جمع_کن[ctx->رقم_additional_fused_ops];
                }
            }
            إذا (ctx->do_جمع_کن_rms_partials) {
                خودکار pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_جمع_کن_rms_norepeat : ctx->device->pipeline_جمع_کن_rms;
                أرجع pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            } وإلا {
                خودکار pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_جمع_کن_norepeat : ctx->device->pipeline_جمع_کن;
                أرجع pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            }
        }
        حالة GGML_OP_SUB:
        {
            خودکار pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_sub_norepeat : ctx->device->pipeline_sub;
            أرجع pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        حالة GGML_OP_MUL:
        {
            خودکار pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_mul_norepeat : ctx->device->pipeline_mul;
            أرجع pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        حالة GGML_OP_DIV:
        {
            خودکار pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_div_norepeat : ctx->device->pipeline_div;
            أرجع pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        افتراضي:
            اكسر;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_جمع_کن_ID:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_جمع_کن_id_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_بیرون_PROD:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_بیرون_prod_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_CONCAT: {
        إذا (!ggml_vk_concat_supported(src0, src1, dst)) {
            أرجع اشاره_تهی;
        }
        بدل (ggml_vk_concat_unit_اندازه(src0->type)) {
        حالة 1:
            أرجع ctx->device->pipeline_concat_i8;
        حالة 2:
            أرجع ctx->device->pipeline_concat_i16;
        حالة 4:
            أرجع ctx->device->pipeline_concat_i32;
        حالة 8:
            أرجع ctx->device->pipeline_concat_i64;
        افتراضي:
            أرجع اشاره_تهی;
        }
    }
    حالة GGML_OP_UPSCALE:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t mode = (ggml_get_op_params_i32(dst, 0) & (0xFF | GGML_SCALE_FLAG_ANTIALIAS));
            بدل (mode) {
                حالة GGML_SCALE_نما_نزدیکترین:
                    أرجع ctx->device->pipeline_upscale_نزدیکترین_f32;
                حالة GGML_SCALE_نما_BILINEAR:
                    أرجع ctx->device->pipeline_upscale_bilinear_f32;
                حالة GGML_SCALE_نما_BICUBIC:
                    أرجع ctx->device->pipeline_upscale_bicubic_f32;
                حالة GGML_SCALE_نما_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS:
                    أرجع ctx->device->pipeline_upscale_bilinear_antialias_f32;
                افتراضي:
                    أرجع اشاره_تهی;
            }
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_SCALE:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_scale_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_SQR:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_sqr[dst->type == GGML_TYPE_F16];
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_SQRT:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_sqrt[dst->type == GGML_TYPE_F16];
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_SIN:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_sin[dst->type == GGML_TYPE_F16];
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_COS:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_cos[dst->type == GGML_TYPE_F16];
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_LOG:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_log[dst->type == GGML_TYPE_F16];
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_TRI:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_tri[dst->type == GGML_TYPE_F16];
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_DIAG:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_diag[dst->type == GGML_TYPE_F16];
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_CLAMP:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_clamp[dst->type == GGML_TYPE_F16];
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_PAD:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_pad_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_PAD_REFLECT_1D:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_pad_reflect_1d_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_ROLL:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_roll_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_REPEAT:
        إذا (ggml_type_اندازه(src0->type) == حجم(عائم) && ggml_type_اندازه(dst->type) == حجم(عائم)) {
            أرجع ctx->device->pipeline_repeat_i32;
        }
        إذا (ggml_type_اندازه(src0->type) == 2 && ggml_type_اندازه(dst->type) == 2) {
            أرجع ctx->device->pipeline_repeat_i16;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_REPEAT_پشت:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_repeat_پشت_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_CPY:
    حالة GGML_OP_CONT:
    حالة GGML_OP_DUP:
        أرجع ggml_vk_get_cpy_pipeline(ctx, src0, dst, dst->type);
    حالة GGML_OP_مجموعه_ردیفها:
        {
            إذا (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                أرجع اشاره_تهی;
            }
            ثابت صحيح src_idx = src0->type == GGML_TYPE_F16;
            إذا (src1->type == GGML_TYPE_I64) {
                أرجع ctx->device->pipeline_مجموعه_ردیفها_i64[src_idx][dst->type];
            } وإلا إذا (src1->type == GGML_TYPE_I32) {
                أرجع ctx->device->pipeline_مجموعه_ردیفها_i32[src_idx][dst->type];
            }
            أرجع اشاره_تهی;
        }
    حالة GGML_OP_SILU_پشت:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_silu_پشت_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_NORM:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_norm_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_گروه_NORM:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_گروه_norm_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_RMS_NORM:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            إذا (ctx->do_جمع_کن_rms_partials) {
                أرجع ctx->fused_rms_norm_نما == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_partials_f32 : ctx->device->pipeline_rms_norm_partials_f32;
            }
            أرجع ctx->fused_rms_norm_نما == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_f32 : ctx->device->pipeline_rms_norm_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_RMS_NORM_پشت:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_rms_norm_پشت_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_L2_NORM:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_l2_norm_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_UNARY:
        إذا ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            أرجع اشاره_تهی;
        }

        بدل (ggml_get_unary_op(dst)) {
            حالة GGML_UNARY_OP_EXP:
                أرجع ctx->device->pipeline_exp[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_EXPM1:
                أرجع ctx->device->pipeline_expm1[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_ELU:
                أرجع ctx->device->pipeline_elu[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_SILU:
                أرجع ctx->device->pipeline_silu[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_GELU:
                أرجع ctx->device->pipeline_gelu[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_GELU_ERF:
                أرجع ctx->device->pipeline_gelu_erf[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_GELU_سریع:
                أرجع ctx->device->pipeline_gelu_سریع[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_RELU:
                أرجع ctx->device->pipeline_relu[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_XIELU:
                أرجع ctx->device->pipeline_xielu[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_NEG:
                أرجع ctx->device->pipeline_neg[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_TANH:
                أرجع ctx->device->pipeline_tanh[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_SIGMOID:
                أرجع ctx->device->pipeline_sigmoid[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_HARDSIGMOID:
                أرجع ctx->device->pipeline_hardsigmoid[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_HARDSWISH:
                أرجع ctx->device->pipeline_hardswish[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_ABS:
                أرجع ctx->device->pipeline_abs[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_SOFTPLUS:
                أرجع ctx->device->pipeline_softplus[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_گام:
                أرجع ctx->device->pipeline_گام[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_دور:
                أرجع ctx->device->pipeline_دور[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_CEIL:
                أرجع ctx->device->pipeline_ceil[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_FLOOR:
                أرجع ctx->device->pipeline_floor[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_TRUNC:
                أرجع ctx->device->pipeline_trunc[dst->type == GGML_TYPE_F16];
            حالة GGML_UNARY_OP_SGN:
                أرجع ctx->device->pipeline_sgn[dst->type == GGML_TYPE_F16];
            افتراضي:
                اكسر;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_GLU:
        إذا ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            أرجع اشاره_تهی;
        }

        بدل (ggml_get_glu_op(dst)) {
            حالة GGML_GLU_OP_GEGLU:
                أرجع ctx->device->pipeline_geglu[dst->type == GGML_TYPE_F16];
            حالة GGML_GLU_OP_REGLU:
                أرجع ctx->device->pipeline_reglu[dst->type == GGML_TYPE_F16];
            حالة GGML_GLU_OP_SWIGLU:
                أرجع ctx->device->pipeline_swiglu[dst->type == GGML_TYPE_F16];
            حالة GGML_GLU_OP_SWIGLU_OAI:
                أرجع ctx->device->pipeline_swiglu_oai[dst->type == GGML_TYPE_F16];
            حالة GGML_GLU_OP_SWIGLU_CLAMP:
                أرجع ctx->device->pipeline_swiglu_clamp[dst->type == GGML_TYPE_F16];
            حالة GGML_GLU_OP_GEGLU_ERF:
                أرجع ctx->device->pipeline_geglu_erf[dst->type == GGML_TYPE_F16];
            حالة GGML_GLU_OP_GEGLU_سریع:
                أرجع ctx->device->pipeline_geglu_سریع[dst->type == GGML_TYPE_F16];
            افتراضي:
                اكسر;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_DIAG_ماسک_INF:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_diag_ماسک_inf_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_SOFT_MAX:
        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        GGML_ASSERT(!src2 || src2->type == GGML_TYPE_F32);

        إذا (ctx->رقم_additional_fused_ops) {
            uint32_t idx = (uint32_t)ceilf(log2f(عائم(dst->ne[0])));
            GGML_ASSERT(idx < رقم_topk_moe_pipelines);
            // use n_experts from push constant if it's not equal to the power of two spec constant
            منطقي use_افزودن = dst->ne[0] != (1u << idx);
            أرجع ctx->device->pipeline_topk_moe[idx][use_افزودن];
        }

        إذا (src0->type == GGML_TYPE_F32 && (src1 == اشاره_تهی || src1->type == GGML_TYPE_F32) && dst->type == GGML_TYPE_F32) {
            أرجع src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_wg512 : ctx->device->pipeline_soft_max_f32;
        }
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            أرجع src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_f16_wg512 : ctx->device->pipeline_soft_max_f32_f16;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_SOFT_MAX_پشت:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_soft_max_پشت_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_ROPE:
    حالة GGML_OP_ROPE_پشت:
        {
            ثابت ggml_tensor *rope = ctx->رقم_additional_fused_ops == 2 ? dst->src[0]->src[0] : dst;
            ثابت صحيح mode = ((ثابت int32_t *) rope->op_params)[2];
            ثابت منطقي is_neox = mode & GGML_ROPE_TYPE_NEOX;
            ثابت منطقي is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            ثابت منطقي is_vision = mode == GGML_ROPE_TYPE_VISION;

            إذا (is_neox) {
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    أرجع ctx->device->pipeline_rope_neox_f32;
                }
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_neox_f32_f16;
                }
                إذا (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_neox_f16;
                }
            } وإلا إذا (is_mrope && !is_vision) {
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    أرجع ctx->device->pipeline_rope_multi_f32;
                }
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_multi_f32_f16;
                }
                إذا (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_multi_f16;
                }
            } وإلا إذا (is_vision) {
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    أرجع ctx->device->pipeline_rope_vision_f32;
                }
                إذا (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_vision_f16;
                }
            } وإلا {
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    أرجع ctx->device->pipeline_rope_norm_f32;
                }
                إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_norm_f32_f16;
                }
                إذا (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    أرجع ctx->device->pipeline_rope_norm_f16;
                }
            }
            أرجع اشاره_تهی;
        }
    حالة GGML_OP_جمع:
    حالة GGML_OP_جمع_ردیفها:
    حالة GGML_OP_میانگین:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_جمع_ردیفها_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_CROSS_ENTROPY_LOSS:
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع src0->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_CROSS_ENTROPY_LOSS_پشت:
        // src0 is the scalar grad; src1 is logits
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2 && src2->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع src1->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_پشت_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_پشت_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_CUMSUM:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            إذا (src0->ne[0] <= 512) {
                أرجع ctx->device->pipeline_cumsum_small_f32;
            } وإلا {
                أرجع ctx->device->pipeline_cumsum_f32;
            }
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_SOLVE_TRI:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {

            vk_solve_tri_pipeline_state solve_tri_pipeline_state(src0->ne[0], src1->ne[0]);

            vk_pipeline pipeline = اشاره_تهی;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                خودکار it = ctx->device->pipeline_solve_tri_f32.find(solve_tri_pipeline_state);
                إذا (it != ctx->device->pipeline_solve_tri_f32.end()) {
                    pipeline = it->second;
                } وإلا {
                    ctx->device->pipeline_solve_tri_f32[solve_tri_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            أرجع pipeline;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_ARGMAX:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
            أرجع ctx->device->pipeline_argmax_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_شمار_EQUAL:
        إذا (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I64) {
            أرجع ctx->device->pipeline_شمار_equal_i32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_IM2_ستون:
        إذا (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_im2col_f32;
        }
        إذا (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            أرجع ctx->device->pipeline_im2col_f32_f16;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_IM2_ستون_3_D:
        إذا (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_im2col_3d_f32;
        }
        إذا (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            أرجع ctx->device->pipeline_im2col_3d_f32_f16;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_TIMESTEP_EMBEDDING:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_timestep_embedding_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_CONV_TRANSPOSE_1D:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_conv_transpose_1d_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_COL2IM_1D:
        بدل (src0->type) {
            حالة GGML_TYPE_F32:  أرجع ctx->device->pipeline_col2im_1d_f32;
            حالة GGML_TYPE_F16:  أرجع ctx->device->pipeline_col2im_1d_f16;
            حالة GGML_TYPE_BF16: أرجع ctx->device->pipeline_col2im_1d_bf16;
            افتراضي:             أرجع اشاره_تهی;
        }
    حالة GGML_OP_POOL_1D:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_pool1d_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_POOL_2D:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            // Wide windows use neighboring subgroup lanes to read neighboring inputs.
            إذا (ctx->device->pipeline_pool2d_subgroup_f32 && dst->op_params[1] >= int32_t(ctx->device->subgroup_اندازه)) {
                أرجع ctx->device->pipeline_pool2d_subgroup_f32;
            }
            أرجع ctx->device->pipeline_pool2d_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_RWKV_WKV6:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_rwkv_wkv6_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_RWKV_WKV7:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_rwkv_wkv7_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_GATED_خطی_ATTN:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_gated_خطی_attn_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_LIGHTNING_INDEXER:
        // only the k type selects a pipeline, the other types are fixed by ggml_lightning_indexer()
        إذا (ggml_vk_lightning_indexer_k_type_supported(src1->type)) {
            أرجع ctx->device->pipeline_lightning_indexer_f32[src1->type];
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_GATED_DELTA_NET:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ثابت uint32_t S_v = dst->src[2]->ne[0];
            ثابت uint32_t kda = (dst->src[3]->ne[0] == (int64_t)S_v) ? 1 : 0;
            uint32_t si;
            بدل (S_v) {
                حالة 16:  si = 0; اكسر;
                حالة 32:  si = 1; اكسر;
                حالة 64:  si = 2; اكسر;
                حالة 128: si = 3; اكسر;
                افتراضي: أرجع اشاره_تهی;
            }
            أرجع ctx->device->pipeline_gated_delta_net[si][kda];
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_SSM_پیمایش:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ثابت uint32_t d_state = src0->ne[0];
            إذا (d_state == 128) {
                أرجع ctx->device->pipeline_ssm_پیمایش_f32_d128;
            } وإلا إذا (d_state == 256) {
                أرجع ctx->device->pipeline_ssm_پیمایش_f32_d256;
            }
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_SSM_CONV:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            بدل (ctx->رقم_additional_fused_ops) {
                حالة 0:  أرجع ctx->device->pipeline_ssm_conv_f32;
                حالة 1:  أرجع ctx->device->pipeline_ssm_conv_silu_f32;
                حالة 2:  أرجع ctx->device->pipeline_ssm_conv_bias_silu_f32;
                افتراضي: أرجع اشاره_تهی;
            }
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_OPT_گام_ADAMW:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_opt_گام_adamw_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_OPT_گام_SGD:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_opt_گام_sgd_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_LEAKY_RELU:
        إذا (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            أرجع ctx->device->pipeline_leaky_relu[dst->type == GGML_TYPE_F16];
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_CONV_2D:
    حالة GGML_OP_CONV_TRANSPOSE_2D:
        إذا (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t K = dst->ne[2]; // Cout
            uint32_t NPQ = dst->ne[3] * dst->ne[1] * dst->ne[0]; // N * OH * OW
            vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, K, NPQ);

            منطقي transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;
            uint32_t KW = (uint32_t)src0->ne[0];
            uint32_t KH = (uint32_t)src0->ne[1];
            uint32_t s0 = (uint32_t)(ggml_get_op_params_i32(dst, 0));
            uint32_t s1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 1) : s0;
            uint32_t p0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 2) : 0;
            uint32_t p1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 3) : 0;
            uint32_t d0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 4) : 1;
            uint32_t d1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 5) : 1;

            // tile-aligned shapes let the shader skip bounds checks
            ثابت uint32_t Cin = (uint32_t)src1->ne[2];
            ثابت uint32_t CRS = Cin * KW * KH;
            ثابت uint32_t BS_K   = vk_conv_بلوک_sizes[shape].K;
            ثابت uint32_t BS_CRS = vk_conv_بلوک_sizes[shape].CRS;
            ثابت uint32_t BS_NPQ = vk_conv_بلوک_sizes[shape].NPQ;
            ثابت uint32_t aligned = ((K   % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv2d_pipeline_state conv2d_pipeline_state(s0, s1, p0, p1, d0, d1, KW, KH, aligned);

            std::map<vk_conv2d_pipeline_state, vk_pipeline> *pipelines = اشاره_تهی;
            إذا (op == GGML_OP_CONV_2D) {
                إذا (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv2d_f32[shape];
                } وإلا إذا (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv2d_f16_f32[shape];
                }
            } وإلا إذا (op == GGML_OP_CONV_TRANSPOSE_2D) {
                إذا (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f32[shape];
                } وإلا إذا (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f16_f32[shape];
                }
            }

            vk_pipeline pipeline = اشاره_تهی;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                خودکار it = pipelines->find(conv2d_pipeline_state);
                إذا (it != pipelines->end()) {
                    pipeline = it->second;
                } وإلا {
                    (*pipelines)[conv2d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            أرجع pipeline;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_CONV_2D_DW:
        إذا (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            إذا (ggml_is_contiguous(src1)) {
                أرجع ctx->device->pipeline_conv2d_dw_whcn_f32;
            } وإلا إذا (ggml_is_contiguous_channels(src1)) {
                أرجع ctx->device->pipeline_conv2d_dw_cwhn_f32;
            }
        } وإلا إذا (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            إذا (ggml_is_contiguous(src1)) {
                أرجع ctx->device->pipeline_conv2d_dw_whcn_f16_f32;
            } وإلا إذا (ggml_is_contiguous_channels(src1)) {
                أرجع ctx->device->pipeline_conv2d_dw_cwhn_f16_f32;
            }
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_CONV_3D:
        إذا (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ثابت uint32_t OC = (uint32_t)ggml_get_op_params_i32(dst, 11);
            ثابت uint32_t IC = (uint32_t)ggml_get_op_params_i32(dst, 9);
            ثابت uint32_t N  = (uint32_t)ggml_get_op_params_i32(dst, 10);
            ثابت uint32_t NPQ = N * dst->ne[2] * dst->ne[1] * dst->ne[0];
            ثابت vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, OC, NPQ);

            ثابت uint32_t KW = (uint32_t)src0->ne[0];
            ثابت uint32_t KH = (uint32_t)src0->ne[1];
            ثابت uint32_t KD = (uint32_t)src0->ne[2];
            ثابت uint32_t s0 = (uint32_t)ggml_get_op_params_i32(dst, 0);
            ثابت uint32_t s1 = (uint32_t)ggml_get_op_params_i32(dst, 1);
            ثابت uint32_t s2 = (uint32_t)ggml_get_op_params_i32(dst, 2);
            ثابت uint32_t p0 = (uint32_t)ggml_get_op_params_i32(dst, 3);
            ثابت uint32_t p1 = (uint32_t)ggml_get_op_params_i32(dst, 4);
            ثابت uint32_t p2 = (uint32_t)ggml_get_op_params_i32(dst, 5);
            ثابت uint32_t d0 = (uint32_t)ggml_get_op_params_i32(dst, 6);
            ثابت uint32_t d1 = (uint32_t)ggml_get_op_params_i32(dst, 7);
            ثابت uint32_t d2 = (uint32_t)ggml_get_op_params_i32(dst, 8);

            ثابت uint32_t CRS = IC * KW * KH * KD;
            ثابت uint32_t BS_K   = vk_conv_بلوک_sizes[shape].K;
            ثابت uint32_t BS_CRS = vk_conv_بلوک_sizes[shape].CRS;
            ثابت uint32_t BS_NPQ = vk_conv_بلوک_sizes[shape].NPQ;
            ثابت uint32_t aligned = ((OC  % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv3d_pipeline_state conv3d_pipeline_state(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned);

            std::map<vk_conv3d_pipeline_state, vk_pipeline> *pipelines = اشاره_تهی;
            إذا (src0->type == GGML_TYPE_F32) {
                pipelines = &ctx->device->pipeline_conv3d_f32[shape];
            } وإلا إذا (src0->type == GGML_TYPE_F16) {
                pipelines = &ctx->device->pipeline_conv3d_f16_f32[shape];
            } وإلا {
                أرجع اشاره_تهی;
            }

            vk_pipeline pipeline = اشاره_تهی;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                خودکار it = pipelines->find(conv3d_pipeline_state);
                إذا (it != pipelines->end()) {
                    pipeline = it->second;
                } وإلا {
                    (*pipelines)[conv3d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            أرجع pipeline;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_ADD1:
        إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            أرجع ctx->device->pipeline_add1_f16_f16;
        }
        إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            أرجع ctx->device->pipeline_add1_f16_f32;
        }
        إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_add1_f32_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_ARANGE:
        إذا (dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_arange_f32;
        }
        أرجع اشاره_تهی;
    حالة GGML_OP_FILL:
        إذا (dst->type == GGML_TYPE_F32) {
            أرجع ctx->device->pipeline_fill_f32;
        }
        إذا (dst->type == GGML_TYPE_F16) {
            أرجع ctx->device->pipeline_fill_f16;
        }
        أرجع اشاره_تهی;
    افتراضي:
        أرجع اشاره_تهی;
    }

    GGML_استفادهنشده(src2);
}

قالب<نام_نوع PC>
ساكن فراغ ggml_vk_op_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ثابت ggml_tensor * src3, ggml_tensor * dst, ggml_op op, PC&& pc, vk_pipeline pipeline_override = اشاره_تهی) {
    VK_LOG_DEBUG("ggml_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    إذا (src1 != اشاره_تهی) {
        std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    إذا (src2 != اشاره_تهی) {
        std::cerr << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] << ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1=" << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    إذا (src3 != اشاره_تهی) {
        std::cerr << "), (" << src3 << ", name=" << src3->name << ", type=" << src3->type << ", ne0=" << src3->ne[0] << ", ne1=" << src3->ne[1] << ", ne2=" << src3->ne[2] << ", ne3=" << src3->ne[3] << ", nb0=" << src3->nb[0] << ", nb1=" << src3->nb[1] << ", nb2=" << src3->nb[2] << ", nb3=" << src3->nb[3];
    }
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << ggml_op_name(op) << ")");
    GGML_ASSERT(op == GGML_OP_GET_ردیفها || op == GGML_OP_CPY || op == GGML_OP_CONCAT || (!ggml_is_quantized(src0->type) && (src1 == اشاره_تهی || !ggml_is_quantized(src1->type))));  // NOLINT
    GGML_ASSERT(dst->میانگیر != اشاره_تهی);
    ثابت uint64_t ne00 = src0->ne[0];
    ثابت uint64_t ne01 = src0->ne[1];
    ثابت uint64_t ne02 = src0->ne[2];
    ثابت uint64_t ne03 = src0->ne[3];

    ثابت منطقي use_src1 = src1 != اشاره_تهی;
    ثابت uint64_t ne10 = use_src1 ? src1->ne[0] : 0;
    ثابت uint64_t ne11 = use_src1 ? src1->ne[1] : 0;
    ثابت uint64_t ne12 = use_src1 ? src1->ne[2] : 0;
    ثابت uint64_t ne13 = use_src1 ? src1->ne[3] : 0;

    ثابت منطقي use_src2 = src2 != اشاره_تهی;
    ثابت منطقي use_src3 = src3 != اشاره_تهی;

    init_pushconst_fastdiv(pc);

    vk_pipeline pipeline;
    إذا (pipeline_override) {
        pipeline = pipeline_override;
    } وإلا {
        pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);
    }

    إذا (pipeline == اشاره_تهی) {
        std::cerr << "ggml_vulkan: Error: Missing op: " << ggml_op_name(op) << " for " << ggml_type_name(src0->type);
        إذا (src1 != اشاره_تهی) {
            std::cerr << " and " << ggml_type_name(src1->type);
        }
        std::cerr << " to " << ggml_type_name(dst->type) << std::پایان_سطر;
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0, درست);
    vk_subbuffer src1_buf = use_src1 ? ggml_vk_tensor_subbuffer(ctx, src1, درست) : vk_subbuffer{};
    vk_subbuffer src2_buf = use_src2 ? ggml_vk_tensor_subbuffer(ctx, src2, درست) : vk_subbuffer{};
    vk_subbuffer src3_buf = use_src3 ? ggml_vk_tensor_subbuffer(ctx, src3, درست) : vk_subbuffer{};
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, درست);

    // Compute misalignment offset for descriptors and store it in in push constants.
    init_pushconst_tensor_offsets(ctx, pc, src0, src1, src2, src3, dst);

    std::array<uint32_t, 3> elements;

    بدل (op) {
    حالة GGML_OP_NORM:
    حالة GGML_OP_RMS_NORM_پشت:
    حالة GGML_OP_L2_NORM:
    حالة GGML_OP_SOFT_MAX:
    حالة GGML_OP_SOFT_MAX_پشت:
    حالة GGML_OP_جمع_ردیفها:
    حالة GGML_OP_CUMSUM:
    حالة GGML_OP_میانگین:
    حالة GGML_OP_ARGMAX:
        {
            ثابت uint32_t nr = ggml_nrows(src0);
            إذا (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } وإلا إذا (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } وإلا {
                elements = { nr, 1, 1 };
            }
        } اكسر;
    حالة GGML_OP_SOLVE_TRI:
        {
            uint32_t nr = (uint32_t)(ne02 * ne03);
            إذا (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } وإلا إذا (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } وإلا {
                elements = { nr, 1, 1 };
            }
        }
        اكسر;
    حالة GGML_OP_RMS_NORM:
        إذا (ctx->do_جمع_کن_rms_partials) {
            // Run one element per thread, 128 threads per workgroup
            elements = { (uint32_t)CEIL_DIV(ne00, 128), 1, 1 };
        } وإلا {
            elements = { (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne03 };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]);
        }
        اكسر;

    حالة GGML_OP_جمع:
        // We use GGML_OP_SUM_ROWS with 1 row.
        elements = { 1, 1, 1 };
        اكسر;
    حالة GGML_OP_گروه_NORM:
        {
            ثابت uint32_t رقم_groups = dst->op_params[0];
            elements = { رقم_groups * (uint32_t)src0->ne[3], 1, 1 };
        } اكسر;
    حالة GGML_OP_DIAG_ماسک_INF:
        elements = { (uint32_t)ggml_nrows(src0), (uint32_t)ne00, 1 };
        اكسر;
    حالة GGML_OP_ROPE:
    حالة GGML_OP_ROPE_پشت:
        {
            uint32_t nrows = (uint32_t)ggml_nrows(src0);
            uint32_t z = 1;
            إذا (nrows > ctx->device->properties.limits.max_Compute_Work_گروه_شمار[0]) {
                z = CEIL_DIV(nrows, 32768);
                nrows = 32768;
            }
            elements = { nrows, (uint32_t)ne00, z };

        } اكسر;
    حالة GGML_OP_GET_ردیفها:
        elements = { (uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12) };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]);
        اكسر;
    حالة GGML_OP_GET_ردیفها_پشت:
        elements = { (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], 1 };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]);
        اكسر;
    حالة GGML_OP_ARGSORT:
        GGML_ASSERT(0);
        اكسر;
    حالة GGML_OP_IM2_ستون:
        {
            ثابت منطقي is_2D = dst->op_params[6] == 1;

            ثابت uint32_t IC = src1->ne[is_2D ? 2 : 1];

            ثابت uint32_t KH = is_2D ? src0->ne[1] : 1;
            ثابت uint32_t KW =         src0->ne[0];

            ثابت uint32_t OH = is_2D ? dst->ne[2] : 1;
            ثابت uint32_t OW =         dst->ne[1];

            ثابت uint32_t batch = src1->ne[is_2D ? 3 : 2];

            ثابت uint32_t CHW = IC * KH * KW;
            // Cap X workgroups to limit concurrent IC channel reads.
            // The shader loops over X to cover the full CHW dimension.
            // AMD prefers a lower limit
            ثابت uint32_t min_cap = ctx->device->vendor_id == VK_VENDOR_ID_AMD ? 512u : 4096u;
            ثابت uint32_t x_elements = std::min(CHW, std::max(min_cap, OW * KH * KW));
            elements = { x_elements, OW, OH * batch };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]);
        } اكسر;
    حالة GGML_OP_IM2_ستون_3_D:
        {
            ثابت uint32_t IC = ((ثابت uint32_t *)(dst->op_params))[9];

            ثابت uint32_t N  = ne13 / IC;

            ثابت uint32_t KD = ne02;
            ثابت uint32_t KH = ne01;
            ثابت uint32_t KW = ne00;

            ثابت uint32_t OD = dst->ne[3] / N;
            ثابت uint32_t OH = dst->ne[2];
            ثابت uint32_t OW = dst->ne[1];

            ثابت uint32_t IC_KD_KH_KW = IC*KD*KH*KW;
            ثابت uint32_t N_OD_OH = N*OD*OH;

            elements = { IC_KD_KH_KW, OW, N_OD_OH };
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]);
        } اكسر;
    حالة GGML_OP_TIMESTEP_EMBEDDING:
        {
            ثابت uint32_t dim = dst->op_params[0];
            uint32_t half_ceil = (dim + 1) / 2;
            elements = { half_ceil, (uint32_t)src0->ne[0], 1 };
        } اكسر;
    حالة GGML_OP_CONV_TRANSPOSE_1D:
        {
            elements = {uint32_t(src0->ne[1]), 1, 1}; // parallelize in {Cout, 1, 1}
        } اكسر;
    حالة GGML_OP_COL2IM_1D:
        {
            elements = { uint32_t(dst->ne[0]), uint32_t(dst->ne[1]), 1 };
        } اكسر;
    حالة GGML_OP_POOL_1D:
        {
            ثابت uint32_t N = dst->ne[3] * dst->ne[2];
            ثابت uint32_t OC = dst->ne[1];
            ثابت uint32_t OL = dst->ne[0];
            elements = { N * OC * OL, 1, 1};
        } اكسر;
    حالة GGML_OP_POOL_2D:
        {
            ثابت uint32_t outputs = ggml_nelements(dst);
            ثابت uint32_t outputs_در_هر_workgroup = pipeline == ctx->device->pipeline_pool2d_subgroup_f32 ? 128 / ctx->device->subgroup_اندازه : 512;
            ثابت uint32_t workgroups = CEIL_DIV(outputs, outputs_در_هر_workgroup);
            ثابت uint32_t groups_y = CEIL_DIV(workgroups, ctx->device->properties.limits.max_Compute_Work_گروه_شمار[0]);
            ثابت uint32_t groups_x = CEIL_DIV(workgroups, groups_y);
            elements = { groups_x, groups_y, 1 };
        } اكسر;
    حالة GGML_OP_CONV_2D:
    حالة GGML_OP_CONV_TRANSPOSE_2D:
        إذا constexpr (std::is_same_v<PC, vk_op_conv2d_افزودن_constants>) {
            ثابت uint32_t NPQ = pc.N * pc.OH * pc.OW;
            ثابت vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.Cout, NPQ);
            ثابت uint32_t NPQ_بلوکها = CEIL_DIV(NPQ, vk_conv_بلوک_sizes[shape].NPQ);

            elements = { pc.Cout, NPQ_بلوکها, 1 };
            إذا (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } وإلا {
            GGML_ABORT("invalid push constant type for CONV_2D");
        }
        اكسر;
    حالة GGML_OP_CONV_3D:
        إذا constexpr (std::is_same_v<PC, vk_op_conv3d_افزودن_constants>) {
            ثابت uint32_t NPQ = pc.N * pc.OD * pc.OH * pc.OW;
            ثابت vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.OC, NPQ);
            ثابت uint32_t NPQ_بلوکها = CEIL_DIV(NPQ, vk_conv_بلوک_sizes[shape].NPQ);

            elements = { pc.OC, NPQ_بلوکها, 1 };
            إذا (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } وإلا {
            GGML_ABORT("invalid push constant type for CONV_3D");
        }
        اكسر;
    حالة GGML_OP_جمع_کن:
    حالة GGML_OP_SUB:
    حالة GGML_OP_DIV:
    حالة GGML_OP_MUL:
    حالة GGML_OP_ADD1:
    حالة GGML_OP_بیرون_PROD:
    حالة GGML_OP_ARANGE:
    حالة GGML_OP_FILL:
    حالة GGML_OP_SCALE:
    حالة GGML_OP_SQR:
    حالة GGML_OP_SQRT:
    حالة GGML_OP_SIN:
    حالة GGML_OP_COS:
    حالة GGML_OP_LOG:
    حالة GGML_OP_TRI:
    حالة GGML_OP_DIAG:
    حالة GGML_OP_CLAMP:
    حالة GGML_OP_LEAKY_RELU:
    حالة GGML_OP_PAD:
    حالة GGML_OP_PAD_REFLECT_1D:
    حالة GGML_OP_ROLL:
    حالة GGML_OP_REPEAT:
    حالة GGML_OP_REPEAT_پشت:
    حالة GGML_OP_CPY:
    حالة GGML_OP_CONCAT:
    حالة GGML_OP_UPSCALE:
    حالة GGML_OP_UNARY:
    حالة GGML_OP_GLU:
    حالة GGML_OP_CONV_2D_DW:
        {
            uint32_t ne = ggml_nelements(dst);
            إذا (op == GGML_OP_CPY && ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                // Convert from number of logical elements to 2- or 4-byte units.
                ne /= ggml_blck_اندازه(src0->type);
                إذا ((ggml_type_اندازه(src0->type) % 4) == 0) {
                    ne *= ggml_type_اندازه(src0->type) / 4;
                } وإلا {
                    ne *= ggml_type_اندازه(src0->type) / 2;
                }
            }
            إذا (op == GGML_OP_CONCAT && ggml_is_quantized(dst->type)) {
                ne = ne / ggml_blck_اندازه(dst->type) * ggml_type_اندازه(dst->type) / ggml_vk_concat_unit_اندازه(dst->type);
            }
            // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
            // Splitting into 512x512xZ wouldn't work well since each workgroup does 1024 elements.
            // So divide by block size here before splitting into 512x512 groups.
            إذا (op == GGML_OP_CPY && !ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                ne = CEIL_DIV(ne, ggml_blck_اندازه(dst->type));
            }
            إذا (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } وإلا إذا (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } وإلا {
                elements = { ne, 1, 1 };
            }

            إذا (pipeline == ctx->device->pipeline_cpy_transpose_02_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_02_16) {
                // 32x32 tiles over dims 0 and 2; dim1 and dim3 are the batch
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[2], 32);
                elements[2] = (uint32_t)(dst->ne[1]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]);
            } وإلا إذا (pipeline == ctx->device->pipeline_cpy_transpose_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_16) {
                // 32x32 tiles
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[1], 32);
                elements[2] = (uint32_t)(dst->ne[2]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]);
            }
        } اكسر;
    حالة GGML_OP_جمع_کن_ID:
        {
            elements = { (uint32_t)ne01, (uint32_t)ne02, 1 };
        } اكسر;
    حالة GGML_OP_مجموعه_ردیفها:
        {
            uint32_t ne = ggml_nelements(src0);
            إذا (ggml_is_quantized(dst->type)) {
                // quants run 32 threads each doing QUANT_K elements
                ne = CEIL_DIV(ne, 32 * ggml_blck_اندازه(dst->type));
            } وإلا {
                // scalar types do one element per thread, running 512 threads
                ne = CEIL_DIV(ne, 512);
            }
            إذا (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } وإلا إذا (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } وإلا {
                elements = { ne, 1, 1 };
            }
        }
        اكسر;
    حالة GGML_OP_SSM_CONV:
        {
            ثابت uint32_t nr  = src0->ne[1];
            ثابت uint32_t n_t = dst->ne[1];
            ثابت uint32_t n_s = dst->ne[2];
            elements = { nr, n_t, n_s };
        }
        اكسر;
    افتراضي:
        elements = { (uint32_t)ggml_nelements(src0), 1, 1 };
        اكسر;
    }

    إذا (op == GGML_OP_جمع_کن || op == GGML_OP_RMS_NORM) {
        vk_subbuffer a_buf = src0_buf;
        إذا (ctx->do_جمع_کن_rms_partials) {
            a_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_جمع_کن_rms_partials, ctx->prealloc_اندازه_جمع_کن_rms_partials_جابه_جایی);
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, src1_buf, dst_buf, a_buf }, pc, elements);
    } وإلا إذا (op == GGML_OP_GLU) {
        // Empty src1 is possible in glu, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc, elements);
    } وإلا إذا (op == GGML_OP_SOFT_MAX) {
        // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, subbuf2, dst_buf }, pc, elements);
    } وإلا إذا (op == GGML_OP_ROPE || op == GGML_OP_ROPE_پشت) {
        // Empty src2 and src3 is possible in rope, but the shader needs a buffer
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        vk_subbuffer subbuf3 = use_src3 ? src3_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, subbuf2, dst_buf, subbuf3 }, pc, elements);
    } وإلا إذا (op == GGML_OP_IM2_ستون || op == GGML_OP_IM2_ستون_3_D) {
        إذا (ctx->device->shader_int64 && ctx->device->میانگیر_device_آدرس) {
            // buffer device address path doesn't use dst buffer
            dst_buf.size = 1;
        }
        // im2col uses only src1 and dst buffers
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src1_buf, dst_buf }, pc, elements);
    } وإلا إذا (op == GGML_OP_شمار_EQUAL) {
        // count_equal assumes that destination buffer is initialized with zeroes
        ggml_vk_میانگیر_memset_async(subctx, dst_buf.میانگیر, dst_buf.جابه_جایی, 0, dst_buf.size);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } وإلا إذا (op == GGML_OP_OPT_گام_SGD) {
        // OPT_STEP_SGD works on src0, it does not need dst
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf }, pc, elements);
    } وإلا إذا (use_src3) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, src3_buf, dst_buf }, pc, elements);
    } وإلا إذا (use_src2) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, dst_buf }, pc, elements);
    } وإلا إذا (use_src1) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } وإلا {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, dst_buf }, pc, elements);
    }
}

فراغ ggml_vk_get_ردیفها(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(dst->type);

    ggml_vk_op_f32<vk_op_دودویی_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_GET_ردیفها, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_اندازه, (uint32_t)src0->nb[1] / src0_type_اندازه, (uint32_t)src0->nb[2] / src0_type_اندازه, (uint32_t)src0->nb[3] / src0_type_اندازه,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_اندازه, (uint32_t)src1->nb[1] / src1_type_اندازه, (uint32_t)src1->nb[2] / src1_type_اندازه, (uint32_t)src1->nb[3] / src1_type_اندازه,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_اندازه, (uint32_t) dst->nb[1] /  dst_type_اندازه, (uint32_t) dst->nb[2] /  dst_type_اندازه, (uint32_t) dst->nb[3] /  dst_type_اندازه,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_get_ردیفها_پشت(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(dst->type);

    ggml_vk_op_f32<vk_op_دودویی_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_GET_ردیفها_پشت, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_اندازه, (uint32_t)src0->nb[1] / src0_type_اندازه, (uint32_t)src0->nb[2] / src0_type_اندازه, (uint32_t)src0->nb[3] / src0_type_اندازه,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2], (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_اندازه, (uint32_t)src1->nb[1] / src1_type_اندازه, (uint32_t)src1->nb[2] / src1_type_اندازه, (uint32_t)src1->nb[3] / src1_type_اندازه,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_اندازه, (uint32_t) dst->nb[1] /  dst_type_اندازه, (uint32_t) dst->nb[2] /  dst_type_اندازه, (uint32_t) dst->nb[3] /  dst_type_اندازه,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(dst->type);

    صحيح nb1 = dst->op_params[0] / src0_type_اندازه; // 4 bytes of float32
    صحيح nb2 = dst->op_params[1] / src0_type_اندازه; // 4 bytes of float32
    صحيح nb3 = dst->op_params[2] / src0_type_اندازه; // 4 bytes of float32
    صحيح جابه_جایی = dst->op_params[3] / src0_type_اندازه; // offset in bytes

    ggml_vk_op_f32<vk_op_دودویی_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, dst->op, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_اندازه, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_اندازه, (uint32_t)src1->nb[1] / src1_type_اندازه, (uint32_t)src1->nb[2] / src1_type_اندازه, (uint32_t)src1->nb[3] / src1_type_اندازه,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_اندازه, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        0,
        0.0f, 0.0f, جابه_جایی,
    });
}

فراغ ggml_vk_multi_جمع_کن(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح گره_idx) {
    ثابت ggml_tensor *اولی_گره = cgraph->nodes[گره_idx];
    ثابت ggml_tensor *dst = cgraph->nodes[گره_idx + ctx->رقم_additional_fused_ops];

    // Make a list of all the tensors used by the op.
    // Last element of the list is the dest tensor.
    ثابت ggml_tensor *tensors[MAX_PARAMETER_شمار];
    uint32_t رقم_srcs = ctx->رقم_additional_fused_ops + 2;
    uint32_t رقم_tensors = رقم_srcs + 1;
    GGML_ASSERT(رقم_tensors + ctx->do_جمع_کن_rms_partials <= MAX_PARAMETER_شمار);

    tensors[0] = اولی_گره->src[0];
    tensors[1] = اولی_گره->src[1];
    لكل (int32_t i = 0; i < ctx->رقم_additional_fused_ops; ++i) {
        // check whether the previous result is src[0] or src[1]
        إذا (cgraph->nodes[گره_idx + i] == cgraph->nodes[گره_idx + i + 1]->src[0]) {
            tensors[i+2] = cgraph->nodes[گره_idx + i + 1]->src[1];
        } وإلا {
            tensors[i+2] = cgraph->nodes[گره_idx + i + 1]->src[0];
        }
    }
    tensors[رقم_srcs] = dst;

    vk_op_multi_جمع_کن_افزودن_constants pc;
    pc.ne20 = (uint32_t)dst->ne[0];
    pc.ne21 = (uint32_t)dst->ne[1];
    pc.ne22 = (uint32_t)dst->ne[2];
    pc.ne23 = (uint32_t)dst->ne[3];

    لكل (uint32_t i = 0; i < رقم_tensors; ++i) {
        ثابت ggml_tensor *t = tensors[i];
        pc.nb[i][0] = (uint32_t)t->nb[0] / حجم(عائم);
        pc.nb[i][1] = (uint32_t)t->nb[1] / حجم(عائم);
        pc.nb[i][2] = (uint32_t)t->nb[2] / حجم(عائم);
        pc.nb[i][3] = (uint32_t)t->nb[3] / حجم(عائم);
    }
    pc.rms_partials = ctx->do_جمع_کن_rms_partials;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, tensors[0], tensors[1], اشاره_تهی, dst, dst->op);

    إذا (pipeline == اشاره_تهی) {
        std::cerr << "ggml_vulkan: Error: Missing multi_add";
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ggml_backend_vk_میانگیر_context * buf_ctx[MAX_PARAMETER_شمار];
    vk_میانگیر buf[MAX_PARAMETER_شمار];
    size_t جابه_جایی[MAX_PARAMETER_شمار];
    منطقي uma[MAX_PARAMETER_شمار];

    لكل (uint32_t i = 0; i < رقم_tensors; ++i) {
        buf_ctx[i] = (ggml_backend_vk_میانگیر_context *)tensors[i]->میانگیر->context;
        buf[i] = اشاره_تهی;
        جابه_جایی[i] = 0;
        uma[i] = خطأ;

        إذا (ctx->device->uma) {
            ggml_vk_host_get(ctx->device, tensors[i]->داده, buf[i], جابه_جایی[i]);
            uma[i] = buf[i] != اشاره_تهی;
        }
        إذا (!uma[i]) {
            buf[i] = buf_ctx[i]->dev_میانگیر;
            جابه_جایی[i] = vk_tensor_جابه_جایی(tensors[i]) + tensors[i]->view_offs;
        }
        GGML_ASSERT(buf[i] != اشاره_تهی);
    }
    // If any remaining descriptors are unused, just point them at src[0]
    لكل (uint32_t i = رقم_tensors; i < MAX_PARAMETER_شمار; ++i) {
        buf[i] = buf[0];
        جابه_جایی[i] = 0;
    }
    إذا (ctx->do_جمع_کن_rms_partials) {
        buf[رقم_tensors] = ctx->prealloc_جمع_کن_rms_partials;
        جابه_جایی[رقم_tensors] = ctx->prealloc_اندازه_جمع_کن_rms_partials_جابه_جایی;
    }

    std::array<uint32_t, 3> elements;

    uint32_t ne = ggml_nelements(dst);
    إذا (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } وإلا إذا (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } وإلا {
        elements = { ne, 1, 1 };
    }

    static_assert(MAX_PARAMETER_شمار == 12);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            ggml_vk_subbuffer(ctx, buf[0], جابه_جایی[0]),
            ggml_vk_subbuffer(ctx, buf[1], جابه_جایی[1]),
            ggml_vk_subbuffer(ctx, buf[2], جابه_جایی[2]),
            ggml_vk_subbuffer(ctx, buf[3], جابه_جایی[3]),
            ggml_vk_subbuffer(ctx, buf[4], جابه_جایی[4]),
            ggml_vk_subbuffer(ctx, buf[5], جابه_جایی[5]),
            ggml_vk_subbuffer(ctx, buf[6], جابه_جایی[6]),
            ggml_vk_subbuffer(ctx, buf[7], جابه_جایی[7]),
            ggml_vk_subbuffer(ctx, buf[8], جابه_جایی[8]),
            ggml_vk_subbuffer(ctx, buf[9], جابه_جایی[9]),
            ggml_vk_subbuffer(ctx, buf[10], جابه_جایی[10]),
            ggml_vk_subbuffer(ctx, buf[11], جابه_جایی[11]),
        }, pc, elements);
}

فراغ ggml_vk_جمع_کن(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(dst->type);

    ggml_vk_op_f32<vk_op_دودویی_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_جمع_کن, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_اندازه, (uint32_t)src0->nb[1] / src0_type_اندازه, (uint32_t)src0->nb[2] / src0_type_اندازه, (uint32_t)src0->nb[3] / src0_type_اندازه,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_اندازه, (uint32_t)src1->nb[1] / src1_type_اندازه, (uint32_t)src1->nb[2] / src1_type_اندازه, (uint32_t)src1->nb[3] / src1_type_اندازه,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_اندازه, (uint32_t) dst->nb[1] /  dst_type_اندازه, (uint32_t) dst->nb[2] /  dst_type_اندازه, (uint32_t) dst->nb[3] /  dst_type_اندازه,
        0,
        0.0f, 0.0f, ctx->do_جمع_کن_rms_partials,
    });
}

فراغ ggml_vk_بیرون_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(dst->type);

    ggml_vk_op_f32<vk_op_دودویی_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_بیرون_PROD, {
        (uint32_t)ggml_nelements(dst),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3],
        (uint32_t)src0->nb[0] / src0_type_اندازه, (uint32_t)src0->nb[1] / src0_type_اندازه, (uint32_t)src0->nb[2] / src0_type_اندازه, (uint32_t)src0->nb[3] / src0_type_اندازه,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3],
        (uint32_t)src1->nb[0] / src1_type_اندازه, (uint32_t)src1->nb[1] / src1_type_اندازه, (uint32_t)src1->nb[2] / src1_type_اندازه, (uint32_t)src1->nb[3] / src1_type_اندازه,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3],
        (uint32_t) dst->nb[0] /  dst_type_اندازه, (uint32_t) dst->nb[1] /  dst_type_اندازه, (uint32_t) dst->nb[2] /  dst_type_اندازه, (uint32_t) dst->nb[3] /  dst_type_اندازه,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(dst->type);

    ggml_vk_op_f32<vk_op_دودویی_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_SUB, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_اندازه, (uint32_t)src0->nb[1] / src0_type_اندازه, (uint32_t)src0->nb[2] / src0_type_اندازه, (uint32_t)src0->nb[3] / src0_type_اندازه,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_اندازه, (uint32_t)src1->nb[1] / src1_type_اندازه, (uint32_t)src1->nb[2] / src1_type_اندازه, (uint32_t)src1->nb[3] / src1_type_اندازه,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_اندازه, (uint32_t) dst->nb[1] /  dst_type_اندازه, (uint32_t) dst->nb[2] /  dst_type_اندازه, (uint32_t) dst->nb[3] /  dst_type_اندازه,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(dst->type);

    ggml_vk_op_f32<vk_op_دودویی_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_MUL, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_اندازه, (uint32_t)src0->nb[1] / src0_type_اندازه, (uint32_t)src0->nb[2] / src0_type_اندازه, (uint32_t)src0->nb[3] / src0_type_اندازه,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_اندازه, (uint32_t)src1->nb[1] / src1_type_اندازه, (uint32_t)src1->nb[2] / src1_type_اندازه, (uint32_t)src1->nb[3] / src1_type_اندازه,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_اندازه, (uint32_t) dst->nb[1] /  dst_type_اندازه, (uint32_t) dst->nb[2] /  dst_type_اندازه, (uint32_t) dst->nb[3] /  dst_type_اندازه,
        0,
        0.0f, 0.0f, 0,
    });
}

صحيح ggml_vk_unary_mul_op_شاخص(ggml_unary_op op) {
    بدل (op) {
        حالة GGML_UNARY_OP_GELU:     أرجع 0;
        حالة GGML_UNARY_OP_SIGMOID:  أرجع 1;
        حالة GGML_UNARY_OP_SILU:     أرجع 2;
        حالة GGML_UNARY_OP_SOFTPLUS: أرجع 3;
        افتراضي:                     أرجع -1;
    }
}

فراغ ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {
    ثابت ggml_tensor * unary = cgraph->nodes[گره_idx];
    ggml_tensor * mul = cgraph->nodes[گره_idx + 1];

    // unary on src1 that tiles into src0
    ثابت منطقي op_on_b = mul->src[1] == unary &&
                         !ggml_are_same_shape(unary->src[0], mul->src[0]) &&
                         ggml_can_repeat(unary, mul->src[0]);

    ثابت ggml_tensor * src0 = op_on_b ? mul->src[0] : unary->src[0];
    ثابت ggml_tensor * src1 = op_on_b ? unary->src[0] :
        ((mul->src[0] == unary) ? mul->src[1] : mul->src[0]);

    ثابت منطقي f16 = src0->type == GGML_TYPE_F16;
    ثابت منطقي norepeat = ggml_are_same_shape(src0, src1);
    ثابت صحيح oi = ggml_vk_unary_mul_op_شاخص(ggml_get_unary_op(unary));
    إذا (oi < 0) {
        GGML_ABORT("fatal error");
    }
    vk_pipeline pipeline = ctx->device->pipeline_unary_mul[oi][f16][norepeat][op_on_b];

    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(mul->type);

    ggml_vk_op_f32<vk_op_دودویی_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, mul, GGML_OP_UNARY, {
        (uint32_t)ggml_nelements(op_on_b ? mul : src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_اندازه, (uint32_t)src0->nb[1] / src0_type_اندازه, (uint32_t)src0->nb[2] / src0_type_اندازه, (uint32_t)src0->nb[3] / src0_type_اندازه,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_اندازه, (uint32_t)src1->nb[1] / src1_type_اندازه, (uint32_t)src1->nb[2] / src1_type_اندازه, (uint32_t)src1->nb[3] / src1_type_اندازه,
        (uint32_t) mul->ne[0], (uint32_t) mul->ne[1], (uint32_t) mul->ne[2],(uint32_t) mul->ne[3], (uint32_t) mul->nb[0] /  dst_type_اندازه, (uint32_t) mul->nb[1] /  dst_type_اندازه, (uint32_t) mul->nb[2] /  dst_type_اندازه, (uint32_t) mul->nb[3] /  dst_type_اندازه,
        0,
        0.0f, 0.0f, 0,
    }, pipeline);
}

فراغ ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(dst->type);

    ggml_vk_op_f32<vk_op_دودویی_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_DIV, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_اندازه, (uint32_t)src0->nb[1] / src0_type_اندازه, (uint32_t)src0->nb[2] / src0_type_اندازه, (uint32_t)src0->nb[3] / src0_type_اندازه,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_اندازه, (uint32_t)src1->nb[1] / src1_type_اندازه, (uint32_t)src1->nb[2] / src1_type_اندازه, (uint32_t)src1->nb[3] / src1_type_اندازه,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_اندازه, (uint32_t) dst->nb[1] /  dst_type_اندازه, (uint32_t) dst->nb[2] /  dst_type_اندازه, (uint32_t) dst->nb[3] /  dst_type_اندازه,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_جمع_کن_id(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t src2_type_اندازه = ggml_type_اندازه(src2->type);

    ggml_vk_op_f32<vk_op_جمع_کن_id_افزودن_constants>(ctx, subctx, src0, src1, src2, اشاره_تهی, dst, GGML_OP_جمع_کن_ID, {
        (uint32_t)dst->ne[0],
        (uint32_t)dst->ne[1],
        (uint32_t)src0->nb[1] / src0_type_اندازه,
        (uint32_t)src0->nb[2] / src0_type_اندازه,
        (uint32_t)src1->nb[1] / src1_type_اندازه,
        (uint32_t)src2->nb[1] / src2_type_اندازه,
    });
}

ساكن فراغ ggml_vk_op_f32_wkv(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, ثابت vk_op_rwkv_wkv6_افزودن_constants&& pc, صحيح نسخه_2) {
    GGML_ASSERT(نسخه_2 == 6 || نسخه_2 == 7);
    صحيح رقم_srcs = نسخه_2 == 6 ? 6 : 7;

    لكل (صحيح i = 0; i < رقم_srcs; i++) {
        GGML_ASSERT(!ggml_is_quantized(dst->src[i]->type));
    }

    GGML_ASSERT(dst->میانگیر != اشاره_تهی);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != اشاره_تهی);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    لكل (صحيح i = 0; i < رقم_srcs; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements = {
        (uint32_t)(pc.B * pc.H),
        1,
        1
    };

    إذا (نسخه_2 == 6) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
            pc, elements);
    } وإلا إذا (نسخه_2 == 7) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
            pc, elements);
    } وإلا {
        // shouldn't happen
        GGML_ASSERT(خطأ);
    }
}

فراغ ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت size_t seq_درازی = dst->src[0]->ne[2];
    ثابت size_t n_embed = dst->ne[0];
    ثابت size_t n_heads = dst->src[0]->ne[1];
    ثابت size_t n_seqs = dst->src[5]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_درازی,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        6
    );
}

فراغ ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت size_t seq_درازی = dst->src[0]->ne[2];
    ثابت size_t n_embed = dst->ne[0];
    ثابت size_t n_heads = dst->src[0]->ne[1];
    ثابت size_t n_seqs = dst->src[6]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_درازی,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        7
    );
}

فراغ ggml_vk_gated_خطی_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت size_t seq_درازی = dst->src[0]->ne[2];
    ثابت size_t n_embed    = dst->ne[0];
    ثابت size_t n_heads    = dst->src[0]->ne[1];
    ثابت size_t n_seqs     = dst->src[4]->ne[1];

    عائم scale;
    کپی_حافظه(&scale, dst->op_params, حجم(عائم));

    GGML_ASSERT(dst->میانگیر != اشاره_تهی);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != اشاره_تهی);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[5] = {};
    لكل (صحيح i = 0; i < 5; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    ثابت vk_op_gated_خطی_attn_افزودن_constants pc = {
        (uint32_t)n_seqs,
        (uint32_t)seq_درازی,
        (uint32_t)n_embed,
        (uint32_t)n_heads,
        scale,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], dst_buf},
        pc, { (uint32_t)(n_seqs * n_heads), 1, 1 });
}

فراغ ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت ggml_tensor * q = dst->src[0];
    ثابت ggml_tensor * k = dst->src[1];
    ثابت ggml_tensor * w = dst->src[2];
    ثابت ggml_tensor * m = dst->src[3];

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, q, k, w, dst, dst->op);
    GGML_ASSERT(pipeline != اشاره_تهی);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ثابت uint32_t n_kv      = k->ne[2];
    ثابت uint32_t n_heads   = q->ne[1];
    ثابت uint32_t n_نشانهها  = q->ne[2];
    ثابت uint32_t n_streams = q->ne[3];
    ثابت uint32_t n_masks   = m->ne[3];

    // one workgroup per tile of 64 keys and 8 tokens, see lightning_indexer.comp
    ثابت uint32_t n_tiles_kv = CEIL_DIV(n_kv, 64);
    ثابت uint32_t n_tiles_t  = CEIL_DIV(n_نشانهها, 8);

    // q, w and dst are f32 and m is f16, so their strides are passed in elements;
    // k may be quantized, so its strides stay in bytes
    ثابت uint32_t q_nb1 = q->nb[1] / حجم(عائم);
    ثابت uint32_t q_nb2 = q->nb[2] / حجم(عائم);
    ثابت uint32_t q_nb3 = q->nb[3] / حجم(عائم);
    ثابت uint32_t k_nb2 = k->nb[2];
    ثابت uint32_t k_nb3 = k->nb[3];
    ثابت uint32_t w_nb1 = w->nb[1] / حجم(عائم);
    ثابت uint32_t w_nb3 = w->nb[3] / حجم(عائم);
    ثابت uint32_t m_nb1 = m->nb[1] / حجم(ggml_fp16_t);
    ثابت uint32_t m_nb3 = m->nb[3] / حجم(ggml_fp16_t);
    ثابت uint32_t d_nb1 = dst->nb[1] / حجم(عائم);
    ثابت uint32_t d_nb3 = dst->nb[3] / حجم(عائم);

    ثابت vk_op_lightning_indexer_افزودن_constants pc = {
        n_kv, n_heads, n_نشانهها, n_masks,
        q_nb1, q_nb2, q_nb3,
        k_nb2, k_nb3,
        w_nb1, w_nb3,
        m_nb1, m_nb3,
        d_nb1, d_nb3,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {ggml_vk_tensor_subbuffer(ctx, q), ggml_vk_tensor_subbuffer(ctx, k), ggml_vk_tensor_subbuffer(ctx, w), ggml_vk_tensor_subbuffer(ctx, m), ggml_vk_tensor_subbuffer(ctx, dst)},
        pc, {n_tiles_kv, n_tiles_t, n_streams});
}

فراغ ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src_q     = dst->src[0];
    ثابت ggml_tensor * src_v     = dst->src[2];
    ثابت ggml_tensor * src_beta  = dst->src[4];

    GGML_ASSERT(dst->میانگیر != اشاره_تهی);

    ثابت uint32_t S_v      = (uint32_t)src_v->ne[0];
    ثابت uint32_t H        = (uint32_t)src_v->ne[1];
    ثابت uint32_t n_نشانهها = (uint32_t)src_v->ne[2];
    ثابت uint32_t n_seqs   = (uint32_t)src_v->ne[3];

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ثابت uint32_t K = (uint32_t)ggml_get_op_params_i32(dst, 0);

    ثابت uint32_t s_off = S_v * H * n_نشانهها * n_seqs;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != اشاره_تهی);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[6] = {};
    لكل (صحيح i = 0; i < 6; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    ثابت uint32_t sq1 = (uint32_t)(src_q->nb[1] / حجم(عائم));
    ثابت uint32_t sq2 = (uint32_t)(src_q->nb[2] / حجم(عائم));
    ثابت uint32_t sq3 = (uint32_t)(src_q->nb[3] / حجم(عائم));
    ثابت uint32_t sv1 = (uint32_t)(src_v->nb[1] / حجم(عائم));
    ثابت uint32_t sv2 = (uint32_t)(src_v->nb[2] / حجم(عائم));
    ثابت uint32_t sv3 = (uint32_t)(src_v->nb[3] / حجم(عائم));
    ثابت uint32_t sb1 = (uint32_t)(src_beta->nb[1] / حجم(عائم));
    ثابت uint32_t sb2 = (uint32_t)(src_beta->nb[2] / حجم(عائم));
    ثابت uint32_t sb3 = (uint32_t)(src_beta->nb[3] / حجم(عائم));

    ثابت uint32_t neq1 = (uint32_t)src_q->ne[1];
    ثابت uint32_t rq3  = (uint32_t)(src_v->ne[3] / src_q->ne[3]);

    ثابت عائم scale = 1.0f / sqrtf((عائم)S_v);
    ثابت vk_op_gated_delta_net_افزودن_constants pc = {
        H, n_نشانهها, n_seqs, s_off,
        sq1, sq2, sq3,
        sv1, sv2, sv3,
        sb1, sb2, sb3,
        neq1, rq3,
        scale,
        K
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
        pc, { H, n_seqs, S_v });
}

فراغ ggml_vk_ssm_پیمایش(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];
    ثابت ggml_tensor * src3 = dst->src[3];
    ثابت ggml_tensor * src4 = dst->src[4];
    ثابت ggml_tensor * src5 = dst->src[5];

    GGML_ASSERT(dst->میانگیر != اشاره_تهی);

    ثابت uint32_t سر_dim = src0->ne[1];
    ثابت uint32_t n_سر = src1->ne[1];
    ثابت uint32_t n_گروه = src4->ne[1];
    ثابت uint32_t n_tok = src1->ne[2];
    ثابت uint32_t n_seq = src1->ne[3];

    منطقي is_mamba2 = (src3->nb[1] == حجم(عائم));
    GGML_ASSERT(is_mamba2);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, dst->op);
    GGML_ASSERT(pipeline != اشاره_تهی);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ثابت int64_t s_off = ggml_nelements(src1) * حجم(عائم);

    ثابت vk_op_ssm_پیمایش_افزودن_constants pc = {
        (uint32_t)src0->nb[2], (uint32_t)src0->nb[3],
        (uint32_t)src1->nb[2], (uint32_t)src1->nb[3],
        (uint32_t)src2->nb[1], (uint32_t)src2->nb[2],
        (uint32_t)src3->nb[1],
        (uint32_t)src4->nb[2], (uint32_t)src4->nb[3],
        (uint32_t)src5->nb[2], (uint32_t)src5->nb[3],
        (uint32_t)s_off,
        n_سر, سر_dim, n_گروه, n_tok,
        n_seq, (uint32_t) ggml_get_op_params_i32(dst, 0)
    };

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    لكل (صحيح i = 0; i < 7 && dst->src[i] != اشاره_تهی; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements;

    ثابت uint32_t d_state = src0->ne[0];
    uint32_t رقم_subgroups = d_state / ctx->device->subgroup_اندازه;
    ثابت uint32_t رقم_workgroups_x = CEIL_DIV(n_سر * سر_dim, رقم_subgroups);
    ثابت uint32_t رقم_workgroups_y = n_seq;
    elements = { رقم_workgroups_x, رقم_workgroups_y, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
        pc, elements);
}

فراغ ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {
    ggml_tensor * conv = cgraph->nodes[گره_idx];
    ثابت ggml_tensor * src0 = conv->src[0];
    ثابت ggml_tensor * src1 = conv->src[1];

    // Pick the destination tensor (last node in the fused chain) and the optional bias.
    // Fusion modes: 0 = ssm_conv, 1 = ssm_conv+silu, 2 = ssm_conv+add(bias)+silu.
    ggml_tensor * dst = conv;
    ثابت ggml_tensor * bias = اشاره_تهی;

    إذا (ctx->رقم_additional_fused_ops == 1) {
        dst = cgraph->nodes[گره_idx + 1]; // silu
    } وإلا إذا (ctx->رقم_additional_fused_ops == 2) {
        ggml_tensor * جمع_کن_2 = cgraph->nodes[گره_idx + 1];
        bias = (جمع_کن_2->src[0] == conv) ? جمع_کن_2->src[1] : جمع_کن_2->src[0];
        dst = cgraph->nodes[گره_idx + 2]; // silu
    }

    // The shader always declares 4 bindings; bind src0 as a dummy when bias isn't fused.
    ثابت ggml_tensor * src2 = bias ? bias : src0;

    ggml_vk_op_f32<vk_op_ssm_conv_افزودن_constants>(ctx, subctx, src0, src1, src2, اشاره_تهی, dst, GGML_OP_SSM_CONV, {
        (uint32_t)src0->nb[1], (uint32_t)src0->nb[2],
        (uint32_t)src1->nb[1],
        (uint32_t)dst->nb[0], (uint32_t)dst->nb[1], (uint32_t)dst->nb[2],
        (uint32_t)src1->ne[0],
        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)dst->ne[1],
        (uint32_t)dst->ne[2],
    });
}

ساكن فراغ ggml_vk_op_f32_opt_گام_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, ثابت vk_op_افزودن_constants&& pc) {
    ثابت ggml_tensor * x = dst->src[0];
    ثابت ggml_tensor * g = dst->src[1];
    ثابت ggml_tensor * gm = dst->src[2];
    ثابت ggml_tensor * gv = dst->src[3];
    ثابت ggml_tensor * p = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(gm->type == GGML_TYPE_F32);
    GGML_ASSERT(gv->type == GGML_TYPE_F32);
    GGML_ASSERT(p->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->میانگیر != اشاره_تهی);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(gm));
    GGML_ASSERT(ggml_is_contiguous(gv));
    GGML_ASSERT(ggml_is_contiguous(p));
    GGML_ASSERT(ggml_are_same_shape(x, g));
    GGML_ASSERT(ggml_are_same_shape(x, gm));
    GGML_ASSERT(ggml_are_same_shape(x, gv));
    GGML_ASSERT(ggml_nelements(p) == 7);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, g, gm, gv, dst, GGML_OP_OPT_گام_ADAMW);
    GGML_ASSERT(pipeline != اشاره_تهی);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer g_buf = ggml_vk_tensor_subbuffer(ctx, g);
    vk_subbuffer gm_buf = ggml_vk_tensor_subbuffer(ctx, gm);
    vk_subbuffer gv_buf = ggml_vk_tensor_subbuffer(ctx, gv);
    vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(x), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {x_buf, g_buf, gm_buf, gv_buf, p_buf},
        pc, elements);
}

فراغ ggml_vk_opt_گام_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32_opt_گام_adamw(
        ctx, subctx, dst,
        { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f }
    );
}

فراغ ggml_vk_opt_گام_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst) {
    ثابت size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32<vk_op_افزودن_constants>(ctx, subctx, src0, src1, src2, اشاره_تهی, dst, GGML_OP_OPT_گام_SGD, { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

فراغ ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    صحيح * op_params = (صحيح *)dst->op_params;

    ثابت uint32_t unit_اندازه = ggml_vk_concat_unit_اندازه(dst->type);
    ثابت uint32_t units_در_هر_بلوک = ggml_type_اندازه(dst->type) / unit_اندازه;
    ثابت uint32_t بلوک_اندازه_3 = ggml_blck_اندازه(dst->type);
    ثابت منطقي quantized = ggml_is_quantized(dst->type);

    // Address dimension 0 in packed storage units; higher strides may be noncontiguous.
    ثابت uint32_t ne00 = src0->ne[0] / بلوک_اندازه_3 * units_در_هر_بلوک;
    ثابت uint32_t ne10 = src1->ne[0] / بلوک_اندازه_3 * units_در_هر_بلوک;
    ثابت uint32_t ne20 =  dst->ne[0] / بلوک_اندازه_3 * units_در_هر_بلوک;
    ثابت uint32_t nb00 = quantized ? 1 : src0->nb[0] / unit_اندازه;
    ثابت uint32_t nb10 = quantized ? 1 : src1->nb[0] / unit_اندازه;
    ثابت uint32_t nb20 = quantized ? 1 :  dst->nb[0] / unit_اندازه;

    vk_op_concat_افزودن_constants pc {{
        ne20 * (uint32_t)dst->ne[1] * (uint32_t)dst->ne[2] * (uint32_t)dst->ne[3],
        ne00, (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], nb00, (uint32_t)src0->nb[1] / unit_اندازه, (uint32_t)src0->nb[2] / unit_اندازه, (uint32_t)src0->nb[3] / unit_اندازه,
        ne10, (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], nb10, (uint32_t)src1->nb[1] / unit_اندازه, (uint32_t)src1->nb[2] / unit_اندازه, (uint32_t)src1->nb[3] / unit_اندازه,
        ne20, (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], nb20, (uint32_t) dst->nb[1] / unit_اندازه, (uint32_t) dst->nb[2] / unit_اندازه, (uint32_t) dst->nb[3] / unit_اندازه,
        0,
        0.0f, 0.0f, op_params[0],
    }};
    ggml_vk_op_f32<vk_op_concat_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_CONCAT, std::move(pc));
}

فراغ ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t mode = (uint32_t)ggml_get_op_params_i32(dst, 0);

    GGML_TENSOR_UNARY_OP_LOCALS

    عائم sf0 = (عائم)ne0 / ne00;
    عائم sf1 = (عائم)ne1 / ne01;
    عائم sf2 = (عائم)ne2 / ne02;
    عائم sf3 = (عائم)ne3 / ne03;
    عائم pixel_جابه_جایی = 0.5f;

    إذا (mode & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = ne0 > 1 && ne00 > 1 ? (عائم)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (عائم)(ne1 - 1) / (ne01 - 1) : sf1;
        pixel_جابه_جایی = 0.0f;
    }

    ggml_vk_op_f32<vk_op_upscale_افزودن_constants>(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_UPSCALE, {
        (uint32_t)ggml_nelements(dst), 0, 0,
        (uint32_t)ne00, (uint32_t)ne01,
        (uint32_t)nb00 / src0_type_اندازه, (uint32_t)nb01 / src0_type_اندازه, (uint32_t)nb02 / src0_type_اندازه, (uint32_t)nb03 / src0_type_اندازه,
        (uint32_t)ne0, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
        sf0, sf1, sf2, sf3, pixel_جابه_جایی
    });
}

فراغ ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_افزودن_constants p = vk_op_unary_افزودن_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_SCALE, std::move(p));
}

فراغ ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_SQR, vk_op_unary_افزودن_constants_init(src0, dst));
}

فراغ ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_SQRT, vk_op_unary_افزودن_constants_init(src0, dst));
}

فراغ ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(dst->type);

    ggml_vk_op_f32<vk_op_دودویی_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_ADD1, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_اندازه, (uint32_t)src0->nb[1] / src0_type_اندازه, (uint32_t)src0->nb[2] / src0_type_اندازه, (uint32_t)src0->nb[3] / src0_type_اندازه,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_اندازه, (uint32_t)src1->nb[1] / src1_type_اندازه, (uint32_t)src1->nb[2] / src1_type_اندازه, (uint32_t)src1->nb[3] / src1_type_اندازه,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_اندازه, (uint32_t) dst->nb[1] /  dst_type_اندازه, (uint32_t) dst->nb[2] /  dst_type_اندازه, (uint32_t) dst->nb[3] /  dst_type_اندازه,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_arange(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");

    vk_op_افزودن_constants pc = {
        (uint32_t)ggml_nelements(dst),
        1,
        ggml_get_op_params_f32(dst, 0),
        ggml_get_op_params_f32(dst, 2),
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_ARANGE);
    GGML_ASSERT(pipeline != اشاره_تهی);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, خطأ);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(dst), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

فراغ ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_fill(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");
    ثابت uint64_t n = ggml_nelements(dst);
    GGML_ASSERT(n > 0);

    vk_op_افزودن_constants pc = {
        (uint32_t)n,
        1,
        ggml_get_op_params_f32(dst, 0),
        0.0f,
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_FILL);
    GGML_ASSERT(pipeline != اشاره_تهی);

    // Split the task distribution to 2D to avoid exceeding maxComputeWorkGroupCount
    ثابت uint32_t مجموع_wg = CEIL_DIV(n, pipeline->wg_denoms[0]);
    ثابت uint32_t wg_x = std::min(مجموع_wg, ctx->device->properties.limits.max_Compute_Work_گروه_شمار[0]);
    ثابت uint32_t wg_y = CEIL_DIV(مجموع_wg, wg_x);
    GGML_ASSERT(wg_y <= ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, خطأ);

    std::array<uint32_t, 3> elements = { wg_x * pipeline->wg_denoms[0], wg_y * pipeline->wg_denoms[1], 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

فراغ ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_SIN, vk_op_unary_افزودن_constants_init(src0, dst));
}

فراغ ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_COS, vk_op_unary_افزودن_constants_init(src0, dst));
}

فراغ ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_LOG, vk_op_unary_افزودن_constants_init(src0, dst));
}

فراغ ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_افزودن_constants p = vk_op_unary_افزودن_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);

    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_TRI, std::move(p));
}

فراغ ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_افزودن_constants p = vk_op_unary_افزودن_constants_init(src0, dst, ggml_nelements(dst));

    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_DIAG, std::move(p));
}

فراغ ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_افزودن_constants p = vk_op_unary_افزودن_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_CLAMP, std::move(p));
}

فراغ ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_pad_افزودن_constants p = vk_op_pad_افزودن_constants_init(src0, dst);
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_PAD, std::move(p));
}

فراغ ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت uint32_t p0 = (uint32_t)dst->op_params[0];
    ثابت uint32_t p1 = (uint32_t)dst->op_params[1];

    vk_op_unary_افزودن_constants p = vk_op_unary_افزودن_constants_init(src0, dst, ggml_nelements(dst));
    کپی_حافظه(&p.param1, &p0, حجم(عائم));
    کپی_حافظه(&p.param2, &p1, حجم(عائم));

    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_PAD_REFLECT_1D, std::move(p));
}

فراغ ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ثابت int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ثابت int32_t s2 = ggml_get_op_params_i32(dst, 2);
    ثابت int32_t s3 = ggml_get_op_params_i32(dst, 3);
    ثابت uint32_t s01_packed = ((s0 + 0x8000) << 16) | (s1 + 0x8000);
    ثابت uint32_t s23_packed = ((s2 + 0x8000) << 16) | (s3 + 0x8000);

    vk_op_unary_افزودن_constants p = vk_op_unary_افزودن_constants_init(src0, dst);
    کپی_حافظه(&p.param1, &s01_packed, حجم(عائم));
    کپی_حافظه(&p.param2, &s23_packed, حجم(عائم));

    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_ROLL, std::move(p));
}

فراغ ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_افزودن_constants p = vk_op_unary_افزودن_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_REPEAT, std::move(p));
}

فراغ ggml_vk_repeat_پشت(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_افزودن_constants p = vk_op_unary_افزودن_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_REPEAT_پشت, std::move(p));
}

فراغ ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t)ggml_nelements(src0);
    إذا (ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
        // Convert from number of logical elements to 2- or 4-byte units.
        ne /= ggml_blck_اندازه(src0->type);
        إذا ((ggml_type_اندازه(src0->type) % 4) == 0) {
            ne *= ggml_type_اندازه(src0->type) / 4;
        } وإلا {
            ne *= ggml_type_اندازه(src0->type) / 2;
        }
    }

    vk_op_unary_افزودن_constants p = vk_op_unary_افزودن_constants_init(src0, dst, ne);
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_CPY, std::move(p));
}

فراغ ggml_vk_مجموعه_ردیفها(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(dst->type);

    // Skip empty skip_rows operations. For most ops the empty check at the start
    // of ggml_vk_build_graph is sufficient, but set_rows can have a nonempty dst
    // with empty srcs.
    إذا (ggml_is_خالی(src0) || ggml_is_خالی(src1)) {
        أرجع;
    }

    ggml_vk_op_f32<vk_op_دودویی_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_مجموعه_ردیفها, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_اندازه, (uint32_t)src0->nb[1] / src0_type_اندازه, (uint32_t)src0->nb[2] / src0_type_اندازه, (uint32_t)src0->nb[3] / src0_type_اندازه,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_اندازه, (uint32_t)src1->nb[1] / src1_type_اندازه, (uint32_t)src1->nb[2] / src1_type_اندازه, (uint32_t)src1->nb[3] / src1_type_اندازه,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_اندازه, (uint32_t) dst->nb[1] /  dst_type_اندازه, (uint32_t) dst->nb[2] /  dst_type_اندازه, (uint32_t) dst->nb[3] /  dst_type_اندازه,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_silu_پشت(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_SILU_پشت, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

فراغ ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    عائم * op_params = (عائم *)dst->op_params;
    vk_op_unary_افزودن_constants p = vk_op_unary_افزودن_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_NORM, std::move(p));
}

فراغ ggml_vk_گروه_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت صحيح * int_op_params = (ثابت صحيح *)dst->op_params;
    ثابت عائم * float_op_params = (ثابت عائم *)dst->op_params;

    ثابت uint32_t رقم_groups = int_op_params[0];
    ثابت عائم eps = float_op_params[1];
    ثابت uint32_t گروه_اندازه = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + رقم_groups - 1) / رقم_groups);

    ggml_vk_op_f32<vk_op_افزودن_constants>(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_گروه_NORM, { گروه_اندازه, 0, eps, 0.0f, 0.0f, 0.0f });
}

ساكن uint32_t ggml_vk_rms_رقم_partials(ggml_backend_vk_context * ctx, ثابت ggml_tensor *گره) {
    ثابت uint32_t ne = (uint32_t)گره->ne[0];
    ثابت uint32_t denom = ctx->device->pipeline_جمع_کن_rms[0][0][0]->wg_denoms[0];
    ثابت uint32_t رقم_partials = CEIL_DIV(ne, denom);
    أرجع رقم_partials;
}

uint32_t ggml_vk_rms_partials_اندازه(ggml_backend_vk_context * ctx, ثابت ggml_tensor *گره) {
    ثابت uint32_t رقم_partials = ggml_vk_rms_رقم_partials(ctx, گره);
    ثابت uint32_t رقم_بایتها = ROUNDUP_POW2(رقم_partials * حجم(uint32_t), ctx->device->partials_binding_alignment);
    أرجع رقم_بایتها;
}

ساكن vk_op_rope_افزودن_constants ggml_vk_make_rope_constants(ثابت ggml_tensor *dst, ثابت ggml_tensor *src0, ثابت منطقي has_ff, منطقي backprop, ثابت uint32_t مجموعه_ردیفها_stride) {
    ثابت صحيح n_dims        = ((ثابت int32_t *) dst->op_params)[1];
    ثابت صحيح mode          = ((ثابت int32_t *) dst->op_params)[2];
    ثابت صحيح n_offs        = ((ثابت int32_t *) dst->op_params)[15];
    // const int n_ctx         = ((const int32_t *) dst->op_params)[3];
    ثابت صحيح n_ctx_orig    = ((ثابت int32_t *) dst->op_params)[4];
    ثابت عائم freq_پایه   = ((ثابت عائم *)   dst->op_params)[5];
    ثابت عائم freq_scale  = ((ثابت عائم *)   dst->op_params)[6];
    ثابت عائم ext_ضریب  = ((ثابت عائم *)   dst->op_params)[7];
    ثابت عائم attn_ضریب = ((ثابت عائم *)   dst->op_params)[8];
    ثابت عائم beta_fast   = ((ثابت عائم *)   dst->op_params)[9];
    ثابت عائم beta_slow   = ((ثابت عائم *)   dst->op_params)[10];
    صحيح sections[4] {};
    إذا (mode & GGML_ROPE_TYPE_MROPE) {
        کپی_حافظه(sections, (ثابت int32_t *) dst->op_params + 11, حجم(صحيح)*4);
    }

    ثابت منطقي is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    عائم corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_پایه, beta_fast, beta_slow, corr_dims);

    ثابت عائم theta_scale = powf(freq_پایه, -2.0f/n_dims);

    uint32_t nb01 = src0->nb[1] / ggml_type_اندازه(src0->type);
    uint32_t nb02 = src0->nb[2] / ggml_type_اندازه(src0->type);
    uint32_t nb03 = src0->nb[3] / ggml_type_اندازه(src0->type);

    uint32_t nb11 = dst->nb[1] / ggml_type_اندازه(dst->type);
    uint32_t nb12 = dst->nb[2] / ggml_type_اندازه(dst->type);
    uint32_t nb13 = dst->nb[3] / ggml_type_اندازه(dst->type);

    vk_op_rope_افزودن_constants rope {
        (uint32_t)mode, (uint32_t)ggml_nrows(src0), (uint32_t)n_dims, (uint32_t)n_offs, freq_scale,
        freq_پایه, ext_ضریب, attn_ضریب, {corr_dims[0], corr_dims[1]}, theta_scale, has_ff,
        { sections[0], sections[1], sections[2], sections[3] }, is_imrope, backprop, مجموعه_ردیفها_stride,

        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)src0->ne[2],
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        0, 0, // a_offset, d_offset filled in by init_pushconst_tensor_offsets
    };

    أرجع rope;
}

ساكن فراغ ggml_vk_rms_norm_پایان(ggml_backend_vk_context * ctx, ثابت ggml_tensor * src0) {
    إذا (ctx->do_جمع_کن_rms_partials_جابه_جایی_calculation) {
        ctx->prealloc_اندازه_جمع_کن_rms_partials_جابه_جایی += ggml_vk_rms_partials_اندازه(ctx, src0);
        ctx->do_جمع_کن_rms_partials = خطأ;
        ctx->do_جمع_کن_rms_partials_جابه_جایی_calculation = خطأ;
    }
}

فراغ ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx, عائم * op_params) {
    ggml_tensor * rms = cgraph->nodes[گره_idx];
    ثابت ggml_tensor * src0 = rms->src[0];

    إذا (ctx->fused_rms_norm_نما == RMS_NORM_VIEW_مجموعه_ردیفها) {
        GGML_ASSERT(ctx->رقم_additional_fused_ops == 2);
        ggml_tensor * مجموعه_ردیفها = cgraph->nodes[گره_idx + 2];
        ثابت ggml_tensor * indices = مجموعه_ردیفها->src[1];
        vk_op_دودویی_افزودن_constants pc = ggml_vk_rms_norm_افزودن_constants(src0, src0, مجموعه_ردیفها, op_params[0], 0);
        init_pushconst_tensor_offsets(ctx, pc, src0, src0, اشاره_تهی, اشاره_تهی, مجموعه_ردیفها);

        vk_pipeline pipeline = مجموعه_ردیفها->type == GGML_TYPE_F16 ?
            ctx->device->pipeline_rms_norm_مجموعه_ردیفها_f32_f16 : ctx->device->pipeline_rms_norm_مجموعه_ردیفها_f32_f32;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_tensor_subbuffer(ctx, src0, درست),
                ggml_vk_tensor_subbuffer(ctx, src0, درست),
                ggml_vk_tensor_subbuffer(ctx, مجموعه_ردیفها, درست),
                ggml_vk_tensor_subbuffer(ctx, indices),
            }, pc, {
                (uint32_t)src0->ne[1],
                std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]),
                std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]),
            });
        ggml_vk_rms_norm_پایان(ctx, src0);
        أرجع;
    }

    إذا (ctx->fused_rms_norm_نما == RMS_NORM_MUL_جمع_کن || ctx->fused_rms_norm_نما == RMS_NORM_MUL_جمع_کن_MUL) {
        ggml_tensor * mul = cgraph->nodes[گره_idx + 1];
        ggml_tensor * جمع_کن_2 = cgraph->nodes[گره_idx + 2];
        ثابت ggml_tensor * وزن = mul->src[0] == rms ? mul->src[1] : mul->src[0];
        ثابت ggml_tensor * residual = جمع_کن_2->src[0] == mul ? جمع_کن_2->src[1] : جمع_کن_2->src[0];
        ثابت منطقي do_post_multiply = ctx->fused_rms_norm_نما == RMS_NORM_MUL_جمع_کن_MUL;
        GGML_ASSERT(ctx->رقم_additional_fused_ops == (do_post_multiply ? 3 : 2));
        ggml_tensor * dst = do_post_multiply ? cgraph->nodes[گره_idx + 3] : جمع_کن_2;
        ثابت ggml_tensor * post_scale = do_post_multiply ?
            (dst->src[0] == جمع_کن_2 ? dst->src[1] : dst->src[0]) : src0;

        ثابت uint32_t رقم_partials = ctx->do_جمع_کن_rms_partials ? ggml_vk_rms_رقم_partials(ctx, dst) : 0;
        vk_op_دودویی_افزودن_constants pc = ggml_vk_rms_norm_افزودن_constants(src0, وزن, dst, op_params[0], رقم_partials);
        init_pushconst_tensor_offsets(ctx, pc, src0, وزن, residual, post_scale, dst);

        vk_pipeline pipeline;
        إذا (ctx->do_جمع_کن_rms_partials) {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_جمع_کن_mul_partials_f32 : ctx->device->pipeline_rms_norm_mul_جمع_کن_partials_f32;
        } وإلا {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_جمع_کن_mul_f32 : ctx->device->pipeline_rms_norm_mul_جمع_کن_f32;
        }
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        إذا (ctx->do_جمع_کن_rms_partials) {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, درست),
                    ggml_vk_tensor_subbuffer(ctx, وزن, درست),
                    ggml_vk_tensor_subbuffer(ctx, dst, درست),
                    ggml_vk_subbuffer(ctx, ctx->prealloc_جمع_کن_rms_partials, ctx->prealloc_اندازه_جمع_کن_rms_partials_جابه_جایی),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, { (uint32_t)CEIL_DIV(src0->ne[0], 128), 1, 1 });
        } وإلا {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, درست),
                    ggml_vk_tensor_subbuffer(ctx, وزن, درست),
                    ggml_vk_tensor_subbuffer(ctx, dst, درست),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, {
                    (uint32_t)src0->ne[1],
                    std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]),
                    std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]),
                });
        }
        ggml_vk_rms_norm_پایان(ctx, src0);
        أرجع;
    }

    ggml_tensor * dst;
    ثابت ggml_tensor * src1;

    إذا (ctx->fused_rms_norm_نما != RMS_NORM_شمار) {
        ggml_tensor * mul = cgraph->nodes[گره_idx + 1];
        dst = mul;
        src1 = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    } وإلا {
        dst = rms;
        src1 = src0;
    }

    ثابت uint32_t رقم_partials = ctx->do_جمع_کن_rms_partials ? ggml_vk_rms_رقم_partials(ctx, dst) : 0;
    vk_op_دودویی_افزودن_constants bin = ggml_vk_rms_norm_افزودن_constants(src0, src1, dst, op_params[0], رقم_partials);

    إذا (ctx->fused_rms_norm_نما == RMS_NORM_MUL_ROPE ||
        ctx->fused_rms_norm_نما == RMS_NORM_MUL_ROPE_VIEW_مجموعه_ردیفها) {
        ساكن constexpr uint32_t max_tensors = 7;
        ثابت ggml_tensor *tensors[max_tensors] {};

        ggml_tensor *rms = cgraph->nodes[گره_idx + 0];
        ggml_tensor *mul = cgraph->nodes[گره_idx + 1];
        ggml_tensor *rope = cgraph->nodes[گره_idx + 2];

        ggml_tensor *other_src = mul->src[0] == rms ? mul->src[1] : mul->src[0];

        منطقي do_مجموعه_ردیفها = ctx->fused_rms_norm_نما == RMS_NORM_MUL_ROPE_VIEW_مجموعه_ردیفها;
        GGML_ASSERT(ctx->رقم_additional_fused_ops == (do_مجموعه_ردیفها ? 4 : 2));

        tensors[0] = rms->src[0];
        tensors[1] = other_src;
        tensors[2] = mul;
        tensors[3] = rope->src[1]; // pos
        tensors[4] = rope->src[2]; // ff
        tensors[5] = cgraph->nodes[گره_idx + ctx->رقم_additional_fused_ops]; // dst
        tensors[6] = do_مجموعه_ردیفها ? tensors[5]->src[1] : اشاره_تهی;
        ثابت uint32_t مجموعه_ردیفها_stride = do_مجموعه_ردیفها ? tensors[5]->nb[1] / ggml_type_اندازه(tensors[5]->type) : 0;

        vk_op_rms_norm_mul_rope_افزودن_constants pc;
        pc.bin = bin;
        pc.rope = ggml_vk_make_rope_constants(rope, rope->src[0], tensors[4] != اشاره_تهی, خطأ, مجموعه_ردیفها_stride);

        vk_pipeline pipeline = tensors[5]->type == GGML_TYPE_F16 ? ctx->device->pipeline_rms_norm_mul_rope_f32_f16 : ctx->device->pipeline_rms_norm_mul_rope_f32_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

        ggml_backend_vk_میانگیر_context * buf_ctx[max_tensors];
        vk_میانگیر buf[max_tensors];
        size_t جابه_جایی[max_tensors];
        منطقي uma[max_tensors];

        لكل (uint32_t i = 0; i < max_tensors; ++i) {
            إذا (!tensors[i]) {
                // If any remaining descriptors are unused, just point them at src[0]
                buf[i] = buf[0];
                جابه_جایی[i] = 0;
                استمر;
            }
            buf_ctx[i] = (ggml_backend_vk_میانگیر_context *)tensors[i]->میانگیر->context;
            buf[i] = اشاره_تهی;
            جابه_جایی[i] = 0;
            uma[i] = خطأ;

            إذا (ctx->device->uma) {
                ggml_vk_host_get(ctx->device, tensors[i]->داده, buf[i], جابه_جایی[i]);
                uma[i] = buf[i] != اشاره_تهی;
            }
            إذا (!uma[i]) {
                buf[i] = buf_ctx[i]->dev_میانگیر;
                جابه_جایی[i] = vk_tensor_جابه_جایی(tensors[i]) + tensors[i]->view_offs;
            }
            GGML_ASSERT(buf[i] != اشاره_تهی);
        }

        // a_offset is unused (the fused path reads from shared memory), but the rope/set_rows dst can be misaligned.
        // Round the binding offset down to the storage buffer alignment; the in-element shift goes in pc.rope.d_offset.
        pc.rope.d_جابه_جایی = get_misalign_بایتها(ctx, tensors[5]) / ggml_type_اندازه(tensors[5]->type);
        جابه_جایی[5] &= ~(size_t(ctx->device->properties.limits.min_Storage_میانگیر_جابه_جایی_Alignment) - 1);

        std::array<uint32_t, 3> elements;
        elements = { (uint32_t)rms->src[0]->ne[1], (uint32_t)rms->src[0]->ne[2], (uint32_t)rms->src[0]->ne[3] };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]);

        static_assert(max_tensors == 7);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_subbuffer(ctx, buf[0], جابه_جایی[0]),
                ggml_vk_subbuffer(ctx, buf[1], جابه_جایی[1]),
                ggml_vk_subbuffer(ctx, buf[2], جابه_جایی[2]),
                ggml_vk_subbuffer(ctx, buf[3], جابه_جایی[3]),
                ggml_vk_subbuffer(ctx, buf[4], جابه_جایی[4]),
                ggml_vk_subbuffer(ctx, buf[5], جابه_جایی[5]),
                ggml_vk_subbuffer(ctx, buf[6], جابه_جایی[6]),
            }, pc, elements);
    } وإلا {
        GGML_ASSERT(ctx->fused_rms_norm_نما == RMS_NORM_MUL || ctx->fused_rms_norm_نما == RMS_NORM_شمار);
        ggml_vk_op_f32<vk_op_دودویی_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_RMS_NORM, std::move(bin));
    }

    ggml_vk_rms_norm_پایان(ctx, src0);
}

فراغ ggml_vk_rms_norm_پشت(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    عائم * op_params = (عائم *)dst->op_params;
    ggml_vk_op_f32<vk_op_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_RMS_NORM_پشت, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f, 0.0f, 0.0f });
}

فراغ ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت عائم * op_params = (ثابت عائم *)dst->op_params;
    vk_op_unary_افزودن_constants p = vk_op_unary_افزودن_constants_init(src0, dst);
    p.param1 = op_params[0];
    ggml_vk_op_f32<vk_op_unary_افزودن_constants>(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_L2_NORM, std::move(p));
}

فراغ ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_UNARY, vk_op_unary_افزودن_constants_init(src0, dst));
}

فراغ ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    عائم * op_params = (عائم *)dst->op_params;
    vk_op_unary_افزودن_constants p = vk_op_unary_افزودن_constants_init(src0, dst);
    p.param1 = op_params[1];
    p.param2 = op_params[2];
    p.param3 = op_params[3];
    p.param4 = op_params[4];
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_UNARY, std::move(p));
}

فراغ ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت عائم * op_params_f = (ثابت عائم *)dst->op_params;

    ثابت منطقي swapped = (منطقي)dst->op_params[1];
    ثابت منطقي تقسیم = src1 != اشاره_تهی;
    ثابت عائم alpha = op_params_f[2];
    ثابت عائم حد = op_params_f[3];

    إذا (!تقسیم) {
        GGML_ASSERT(src0->ne[0] / 2 == dst->ne[0]);
    } وإلا {
        GGML_ASSERT(src0->ne[0] == src1->ne[0]);
        GGML_ASSERT(src0->ne[0] == dst->ne[0]);
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت uint32_t mode = تقسیم ? 2 : (swapped ? 1 : 0);
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = تقسیم ? ggml_type_اندازه(src1->type) : src0_type_اندازه;
    ثابت uint32_t dst_type_اندازه  = ggml_type_اندازه(dst->type);

    ggml_vk_op_f32<vk_op_glu_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_GLU,
        {
            (uint32_t)ggml_nelements(dst),
            (uint32_t)src0->ne[0],
            (uint32_t)dst->ne[0],
            mode,
            alpha,
            حد,
            (uint32_t)(src0->nb[0] / src0_type_اندازه),
            (uint32_t)(src0->nb[1] / src0_type_اندازه),
            (uint32_t)(src0->nb[2] / src0_type_اندازه),
            (uint32_t)(src0->nb[3] / src0_type_اندازه),
            (uint32_t)((تقسیم ? src1->nb[0] : src0->nb[0]) / src1_type_اندازه),
            (uint32_t)((تقسیم ? src1->nb[1] : src0->nb[1]) / src1_type_اندازه),
            (uint32_t)((تقسیم ? src1->nb[2] : src0->nb[2]) / src1_type_اندازه),
            (uint32_t)((تقسیم ? src1->nb[3] : src0->nb[3]) / src1_type_اندازه),
            (uint32_t)(dst->nb[0] / dst_type_اندازه),
            (uint32_t)(dst->nb[1] / dst_type_اندازه),
            (uint32_t)(dst->nb[2] / dst_type_اندازه),
            (uint32_t)(dst->nb[3] / dst_type_اندازه),
            (uint32_t)dst->ne[1],
            (uint32_t)dst->ne[2],
            0,
            0, 0, 0, 0, 0, 0,
        });
}

فراغ ggml_vk_diag_ماسک_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    int32_t * op_params = (int32_t *)dst->op_params;
    ggml_vk_op_f32<vk_op_diag_ماسک_افزودن_constants>(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_DIAG_ماسک_INF, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0] });
}

فراغ ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst) {
    عائم * op_params = (عائم *)dst->op_params;

    عائم scale = op_params[0];
    عائم max_bias = op_params[1];

    ثابت uint32_t ncols =   (uint32_t)src0->ne[0];
    ثابت uint32_t nrows_x = (uint32_t)ggml_nrows(src0);
    ثابت uint32_t nrows_y = (uint32_t)src0->ne[1];

    ثابت uint32_t ne12 = src1 ? (uint32_t)(src1->ne[2]) : 0u;
    ثابت uint32_t ne13 = src1 ? (uint32_t)(src1->ne[3]) : 0u;
    ثابت uint32_t nb11 = src1 ? (uint32_t)(src1->nb[1] / src1->nb[0]) : 0u;
    ثابت uint32_t nb12 = src1 ? (uint32_t)(src1->nb[2] / src1->nb[0]) : 0u;
    ثابت uint32_t nb13 = src1 ? (uint32_t)(src1->nb[3] / src1->nb[0]) : 0u;

    ثابت uint32_t n_سر_kv   = src0->ne[2];
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floorf(log2f((عائم) n_سر_kv));

    ثابت عائم m0 = powf(2.0f, -(max_bias       ) / n_سر_log2);
    ثابت عائم m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    vk_op_soft_max_افزودن_constants pc {
        ncols,
        src1 != اشاره_تهی ? nrows_y : (uint32_t)0,
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
        ne12, ne13,
        nb11, nb12, nb13,
        scale, max_bias,
        m0, m1,
        n_سر_log2,
        nrows_x,
        src2 != اشاره_تهی
    };

    إذا (ncols <= 16384) {
        ggml_vk_op_f32<vk_op_soft_max_افزودن_constants>(ctx, subctx, src0, src1, src2, اشاره_تهی, dst, GGML_OP_SOFT_MAX, std::move(pc));
    } وإلا {

        vk_subbuffer buf_a = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer buf_b = src1 ? ggml_vk_tensor_subbuffer(ctx, src1) : buf_a;
        vk_subbuffer buf_c = src2 ? ggml_vk_tensor_subbuffer(ctx, src2) : buf_a;
        vk_subbuffer buf_d = ggml_vk_tensor_subbuffer(ctx, dst);

        uint32_t elems_در_هر_wg = 128 * 4;
        uint32_t رقم_wgs = CEIL_DIV(ncols, elems_در_هر_wg);
        size_t tmp_اندازه = رقم_wgs * nrows_x * حجم(عائم);

        إذا (ctx->prealloc_اندازه_x < tmp_اندازه) {
            ctx->prealloc_اندازه_x = tmp_اندازه;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (ctx->prealloc_اندازه_y < tmp_اندازه) {
            ctx->prealloc_اندازه_y = tmp_اندازه;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (ctx->prealloc_x_need_sync || ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_subbuffer buf_x = { ctx->prealloc_x, 0, tmp_اندازه };
        vk_subbuffer buf_y = { ctx->prealloc_y, 0, tmp_اندازه };

        // the partial results overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_آخرین_pipeline_استفاده_شده = اشاره_تهی;
        ctx->prealloc_y_آخرین_tensor_استفاده_شده = اشاره_تهی;

        std::array<uint32_t, 3> elements = { رقم_wgs, nrows_x, 1 };

        vk_pipeline pipeline1 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large1_f32_f16 : ctx->device->pipeline_soft_max_large1_f32;
        vk_pipeline pipeline2 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large2_f32_f16 : ctx->device->pipeline_soft_max_large2_f32;
        vk_pipeline pipeline3 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large3_f32_f16 : ctx->device->pipeline_soft_max_large3_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline1, 1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline2, 1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline3, 1);

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline1, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline2, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline3, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);

        ctx->prealloc_x_need_sync = درست;
        ctx->prealloc_y_need_sync = درست;
    }
}

فراغ ggml_vk_soft_max_پشت(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    عائم * op_params = (عائم *)dst->op_params;
    ggml_vk_op_f32<vk_op_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_SOFT_MAX_پشت, { (uint32_t)src0->ne[0], (uint32_t)ggml_nrows(src0), op_params[0], op_params[1], 0.0f, 0.0f });
}

فراغ ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح گره_idx) {
    topk_moe_نما mode = ctx->fused_topk_moe_نما;
    ثابت منطقي has_bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS || mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
    ggml_tensor * logits = cgraph->nodes[گره_idx + 0]->src[0];
    ggml_tensor * bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[گره_idx + 2]->src[1] :
                         mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[گره_idx + 3]->src[1] :
                                                                   logits;
    ggml_tensor * weights = cgraph->nodes[گره_idx + ctx->رقم_additional_fused_ops];
    ggml_tensor * ids = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[گره_idx + 4] :
                        mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[گره_idx + 5] :
                        mode == TOPK_MOE_LATE_SOFTMAX             ? cgraph->nodes[گره_idx + 1] :
                                                                   cgraph->nodes[گره_idx + 3];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ثابت صحيح n_experts = logits->ne[0];
    ثابت صحيح n_ردیفها    = logits->ne[1];
    ثابت صحيح n_expert_استفاده_شده = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_اندازه(ids->type) == (size_t) n_experts);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, اشاره_تهی, اشاره_تهی, اشاره_تهی, cgraph->nodes[گره_idx], GGML_OP_SOFT_MAX);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer bias_buf = ggml_vk_tensor_subbuffer(ctx, bias);
    vk_subbuffer weights_buf = ggml_vk_tensor_subbuffer(ctx, weights);
    vk_subbuffer ids_buf = ggml_vk_tensor_subbuffer(ctx, ids);

    vk_op_topk_moe_افزودن_constants pc {};
    pc.n_ردیفها = n_ردیفها;
    pc.n_experts_افزودن = n_experts;
    pc.n_expert_استفاده_شده = n_expert_استفاده_شده;
    pc.clamp_min = -std::numeric_limits<عائم>::infinity();
    pc.clamp_max = std::numeric_limits<عائم>::infinity();
    إذا (mode == TOPK_MOE_EARLY_SOFTMAX_NORM) {
        ggml_tensor * clamp = cgraph->nodes[گره_idx + 7];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    إذا (mode == TOPK_MOE_SIGMOID_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[گره_idx + 8];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    إذا (mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[گره_idx + 9];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }

#تعریف_کن GATING_FUNC_SOFTMAX 0
#تعریف_کن GATING_FUNC_SIGMOID 1
#تعریف_کن GATING_FUNC_SOFTMAX_وزن 2
#تعریف_کن GATING_FUNC_SQRT_SOFTPLUS 3

    pc.gating_func = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? GATING_FUNC_SIGMOID :
                     mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? GATING_FUNC_SQRT_SOFTPLUS :
                     mode == TOPK_MOE_LATE_SOFTMAX             ? GATING_FUNC_SOFTMAX_وزن :
                                                                 GATING_FUNC_SOFTMAX;
    pc.has_bias = has_bias;
    pc.with_norm = mode == TOPK_MOE_EARLY_SOFTMAX_NORM || has_bias;
    إذا (ctx->fused_topk_moe_scale) {
        GGML_ASSERT(weights->op == GGML_OP_SCALE);
        pc.output_scale = ggml_get_op_params_f32(weights, 0);
        pc.output_bias = ggml_get_op_params_f32(weights, 1);
    } وإلا {
        pc.output_scale = 1.0f;
        pc.output_bias = 0.0f;
    }

    GGML_ASSERT(n_expert_استفاده_شده <= n_experts);

    ثابت uint32_t ردیفها_در_هر_بلوک = 4;
    std::array<uint32_t, 3> elements = { CEIL_DIV(n_ردیفها, ردیفها_در_هر_بلوک), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {logits_buf, bias_buf, weights_buf, ids_buf}, pc, elements);
}

فراغ ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_cgraph * cgraph, صحيح گره_idx, منطقي backprop) {
    ggml_tensor * dst = cgraph->nodes[گره_idx];
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];
    ثابت ggml_tensor * src3 = اشاره_تهی;
    ثابت صحيح n_dims        = ((int32_t *) dst->op_params)[1];
    ثابت صحيح mode          = ((int32_t *) dst->op_params)[2];
    // const int n_ctx         = ((int32_t *) dst->op_params)[3];
    ثابت صحيح n_ctx_orig    = ((int32_t *) dst->op_params)[4];
    ثابت عائم freq_پایه   = ((عائم *)   dst->op_params)[5];
    ثابت عائم beta_fast   = ((عائم *)   dst->op_params)[9];
    ثابت عائم beta_slow   = ((عائم *)   dst->op_params)[10];
    صحيح sections[4] {};
    إذا (mode & GGML_ROPE_TYPE_MROPE) {
        کپی_حافظه(sections, (int32_t *) dst->op_params + 11, حجم(صحيح)*4);
    }

    عائم corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_پایه, beta_fast, beta_slow, corr_dims);

    uint32_t مجموعه_ردیفها_stride = 0;
    // Fused rope + view + set_rows passes the set_rows destination stride in set_rows_stride
    // and overrides the dst and sets src3=row_indices
    إذا (ctx->رقم_additional_fused_ops > 0) {
        مجموعه_ردیفها_stride = cgraph->nodes[گره_idx + 2]->nb[1] / ggml_type_اندازه(cgraph->nodes[گره_idx + 2]->type);
        src3 = cgraph->nodes[گره_idx + 2]->src[1];
        dst = cgraph->nodes[گره_idx + 2];
    }

    ggml_vk_op_f32<vk_op_rope_افزودن_constants>(ctx, subctx, src0, src1, src2, src3, dst, GGML_OP_ROPE,
        ggml_vk_make_rope_constants(cgraph->nodes[گره_idx], src0, src2 != اشاره_تهی, backprop, مجموعه_ردیفها_stride));
}

فراغ ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت uint32_t * op_params = (ثابت uint32_t *)dst->op_params;

    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);

    uint32_t ncols_pad_log2 = (uint32_t)ceilf(log2f(عائم(ncols)));
    uint32_t ncolsp2 = 1 << ncols_pad_log2;

    vk_op_argsort_افزودن_constants pc { ncols, ncolsp2, ncols_pad_log2, nrows, op_params[0], 0, 0, 0, 0, };

    // Pick the largest workgroup size <= ncolsp2
    uint32_t pipeline_idx = std::min(ncols_pad_log2, رقم_argsort_pipelines - 1);

    uint32_t max_wg_log2 = std::min(ctx->device->max_workgroup_اندازه_log2, رقم_argsort_pipelines - 1);

    // Use the "small" argsort shader if the whole sort can be done by a single workgroup.
    منطقي use_small = ncols_pad_log2 <= max_wg_log2 &&
                     ctx->device->pipeline_argsort_f32[pipeline_idx] != اشاره_تهی;

    vk_pipeline pipeline = use_small ? ctx->device->pipeline_argsort_f32[pipeline_idx]
                                     : ctx->device->pipeline_argsort_large_f32[pipeline_idx];

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer subbuf1 = dst_buf;

    // Reserve space for ivec2 per element, with rows padded to a power of two
    إذا (!use_small) {
        ثابت size_t x_sz = size_t{ncolsp2} * nrows * 2 * حجم(صحيح);

        إذا (ctx->prealloc_اندازه_x < x_sz) {
            ctx->prealloc_اندازه_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        إذا (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        subbuf1 = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    }

    std::array<uint32_t, 3> elements;

    elements[0] = ncolsp2;
    elements[1] = std::min((uint32_t)ggml_nrows(src0), ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]);
    elements[2] = 1;

    // First dispatch initializes tmp_idx and does the first N passes where
    // there is only communication between threads in the same workgroup.
    {
        vk_op_argsort_افزودن_constants pc2 = pc;
        pc2.outer_شروع = 0;
        pc2.outer_انتها = std::min(ncols_pad_log2, max_wg_log2);
        pc2.inner_شروع = 0;
        pc2.inner_انتها = 100;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
    }
    إذا (!use_small) {
        ggml_vk_sync_buffers(ctx, subctx);
        // Loop over outer/inner passes, synchronizing between each pass.
        لكل (uint32_t outer = max_wg_log2; outer < ncols_pad_log2; ++outer) {
            لكل (uint32_t inner = 0; inner < outer + 1; ++inner) {
                vk_op_argsort_افزودن_constants pc2 = pc;
                pc2.outer_شروع = outer;
                pc2.outer_انتها = outer + 1;
                pc2.inner_شروع = inner;
                pc2.inner_انتها = inner + 1;
                // When the inner idx is large enough, there's only communication
                // within a workgroup. So the remaining inner iterations can all
                // run in the same dispatch.
                إذا (outer - inner < pipeline_idx) {
                    pc2.inner_انتها = 100;
                    inner = outer;
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx];
                } وإلا {
                    // Smaller workgroup empirically seems to perform better
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx - 2];
                }
                ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
                ggml_vk_sync_buffers(ctx, subctx);
            }
        }
        ctx->prealloc_x_need_sync = درست;
    }
}

فراغ ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);
    uint32_t k = dst->ne[0];

    // tournament path is faster where it fits; use radix-select only past its k limit
    ثابت uint32_t k_min_pipeline = std::max((uint32_t) log2f(عائم(k)) + 1, ctx->device->subgroup_اندازه_log2);
    إذا (k_min_pipeline >= رقم_topk_pipelines || ctx->device->pipeline_topk_f32[k_min_pipeline] == اشاره_تهی) {
        vk_pipeline pipeline = ctx->device->pipeline_topk_radix_f32;
        GGML_ASSERT(pipeline != اشاره_تهی);

        إذا (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_op_topk_radix_افزودن_constants pc { ncols, k, nrows, 0, 0, 0 };
        std::array<uint32_t, 3> elements {
            pipeline->wg_denoms[0],
            std::min(nrows, ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]),
            1,
        };
        // the non-QSA path only uses bindings 0/1; bind valid buffers for the unused QSA slots
        vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, dst_buf, src0_buf, src0_buf, src0_buf }, pc, elements);
        أرجع;
    }

    vk_op_topk_افزودن_constants pc { ncols, ncols, ncols, k, nrows, 0, 0 };

    إذا (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    std::array<uint32_t, 3> elements;
    elements[1] = std::min(nrows, ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]);
    elements[2] = 1;

    uint32_t رقم_elements = ncols;

    // Each iteration reduces a workgroup's worth of elements down to the K
    // largest elements. Repeat until we have the top K elements.
    // Need to do at least one iteration to write out the results.
    منطقي انجامشده_one_iter = خطأ;
    uint32_t dbl_buf_شاخص = 0;
    size_t dbl_buf_اندازه;
    طالما (رقم_elements > k || !انجامشده_one_iter) {

        // Prefer going as small as num_topk_pipelines - 3 for perf reasons.
        // But if K is larger, then we need a larger workgroup
        uint32_t max_pipeline = رقم_topk_pipelines - 1;
        uint32_t preferred_pipeline = std::max(رقم_topk_pipelines - 3, (uint32_t)log2f(عائم(k)) + 2);
        max_pipeline = std::min(preferred_pipeline, max_pipeline);
        uint32_t min_pipeline = (uint32_t)log2f(عائم(k)) + 1;
        // require full subgroup
        min_pipeline = std::max(min_pipeline, ctx->device->subgroup_اندازه_log2);

        uint32_t pipeline_idx = (uint32_t)ceilf(log2f(عائم(رقم_elements)));
        pipeline_idx = std::min(pipeline_idx, max_pipeline);
        pipeline_idx = std::max(pipeline_idx, min_pipeline);

        إذا (رقم_elements > (1u << pipeline_idx)) {
            // If we could finish on this loop iteration (i.e. a single workgroup)
            // then do so. It's better than the overhead of another pass.
            لكل (uint32_t i = pipeline_idx; i < رقم_topk_pipelines; ++i) {
                إذا (رقم_elements <= (1u << i)) {
                    pipeline_idx = i;
                    اكسر;
                }
            }
        }

        vk_pipeline pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        // If the device doesn't support a pipeline this large, use smaller
        طالما (!pipeline) {
            pipeline_idx--;
            GGML_ASSERT(pipeline_idx >= min_pipeline);
            pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        }

        vk_op_topk_افزودن_constants pc2 = pc;
        pc2.ncols_درون_داد = رقم_elements;

        // Number of elements remaining after this pass
        uint32_t رقم_dst_elements = (رقم_elements / pipeline->wg_denoms[0]) * k + std::min(k, رقم_elements % pipeline->wg_denoms[0]);

        pc2.ncols_output = رقم_dst_elements;

        إذا (!انجامشده_one_iter) {
            // Reserve space for ivec2 per element, double buffered
            // K per workgroup per row
            dbl_buf_اندازه = رقم_dst_elements * nrows * 2 * حجم(صحيح);
            dbl_buf_اندازه = ROUNDUP_POW2(dbl_buf_اندازه, ctx->device->properties.limits.min_Storage_میانگیر_جابه_جایی_Alignment);
            ثابت size_t x_sz = dbl_buf_اندازه * 2;

            إذا (ctx->prealloc_اندازه_x < x_sz) {
                ctx->prealloc_اندازه_x = x_sz;
                ggml_vk_preallocate_buffers(ctx, subctx);
            }
        }

        vk_subbuffer src_buf;
        vk_subbuffer dst_buf;

        إذا (رقم_elements == ncols) {
            pc2.اولی_pass = 1;
            src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        } وإلا {
            src_buf = { ctx->prealloc_x, dbl_buf_شاخص * dbl_buf_اندازه, dbl_buf_اندازه };
        }
        إذا (رقم_dst_elements == k) {
            pc2.آخرین_pass = 1;
            dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
        } وإلا {
            dst_buf = { ctx->prealloc_x, (dbl_buf_شاخص ^ 1) * dbl_buf_اندازه, dbl_buf_اندازه };
        }

        elements[0] = رقم_elements;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc2, elements);
        رقم_elements = رقم_dst_elements;
        dbl_buf_شاخص ^= 1;
        إذا (رقم_elements > k) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        انجامشده_one_iter = درست;
    }
    ctx->prealloc_x_need_sync = درست;
}

فراغ ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_cgraph * cgraph, صحيح گره_idx) {
    ثابت ggml_tensor * get_ردیفها = cgraph->nodes[گره_idx + 0];
    ثابت ggml_tensor * جمع_کن_2      = cgraph->nodes[گره_idx + ctx->رقم_additional_fused_ops - 1];
    ggml_tensor *       بالاترین_k    = cgraph->nodes[گره_idx + ctx->رقم_additional_fused_ops];

    ثابت ggml_tensor * scores   = get_ردیفها->src[0]; // [n_tps, n_blocks, n_stream]
    ثابت ggml_tensor * cell_blk = get_ردیفها->src[1]; // [n_kv, n_stream]

    // raw f16 mask: follow the reshape/cpy chain back to the materialized input
    ثابت ggml_tensor * ماسک = جمع_کن_2->src[1];
    طالما (ماسک->op == GGML_OP_RESHAPE || ماسک->op == GGML_OP_CPY) {
        ماسک = ماسک->src[0];
    }

    ثابت uint32_t n_tps    = scores->ne[0];
    ثابت uint32_t n_بلوکها = scores->ne[1];
    ثابت uint32_t n_جریان = scores->ne[2];
    ثابت uint32_t n_kv     = cell_blk->ne[0];
    ثابت uint32_t عرض    = بالاترین_k->ne[0];
    ثابت uint32_t nrows    = n_tps * n_جریان;

    vk_pipeline pipeline = ctx->device->pipeline_topk_radix_qsa;
    GGML_ASSERT(pipeline != اشاره_تهی);

    // scratch holds the gathered+masked input, materialized once and reused across passes
    ثابت size_t scratch_اندازه = size_t{ n_kv } * nrows * حجم(عائم);
    إذا (ctx->prealloc_اندازه_x < scratch_اندازه) {
        ctx->prealloc_اندازه_x = scratch_اندازه;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    إذا (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_op_topk_radix_افزودن_constants pc { n_kv, عرض, nrows, n_tps, n_بلوکها, n_جریان };
    std::array<uint32_t, 3> elements {
        pipeline->wg_denoms[0],
        std::min(nrows, ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1]),
        1,
    };
    vk_subbuffer scratch_buf { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ggml_vk_tensor_subbuffer(ctx, scores), ggml_vk_tensor_subbuffer(ctx, بالاترین_k),
          ggml_vk_tensor_subbuffer(ctx, cell_blk), ggml_vk_tensor_subbuffer(ctx, ماسک),
          scratch_buf }, pc, elements);
    ctx->prealloc_x_need_sync = درست;
}

فراغ ggml_vk_جمع(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_جمع_ردیفها_افزودن_constants p = vk_op_جمع_ردیفها_افزودن_constants_init(src0, dst, ggml_nelements(src0));
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_جمع, p);
}

فراغ ggml_vk_جمع_ردیفها(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_جمع_ردیفها_افزودن_constants p = vk_op_جمع_ردیفها_افزودن_constants_init(src0, dst, src0->ne[0]);
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_جمع_ردیفها, p);
}

فراغ ggml_vk_میانگین(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_جمع_ردیفها_افزودن_constants p = vk_op_جمع_ردیفها_افزودن_constants_init(src0, dst, src0->ne[0]);
    p.وزن = 1.0f / (عائم)src0->ne[0];
    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_میانگین, p);
}

فراغ ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_جمع_ردیفها_افزودن_constants pc = vk_op_جمع_ردیفها_افزودن_constants_init(src0, dst, src0->ne[0]);
    // Use the single pass shader when the rows are small or there are enough rows to fill the GPU.
    // For fewer, larger rows, use the multipass shader to spread each row across SMs.
    إذا (dst->ne[0] <= 4096 || ggml_nrows(dst) >= ctx->device->shader_core_شمار) {
        ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_CUMSUM, pc);
        أرجع;
    }

    // First pass computes partial sums within a block, and stores the last partial
    // to the temp buffer. Second pass sums the block partials from the temp buffer
    // and adds that to the result of the first pass.
    vk_pipeline pipeline1 = ctx->device->pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline2 = ctx->device->pipeline_cumsum_multipass2_f32;
    GGML_ASSERT(pipeline1 != اشاره_تهی && pipeline2 != اشاره_تهی);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline1, 1);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline2, 1);

    std::array<uint32_t, 3> elements;

    elements[0] = dst->ne[0];
    elements[1] = (uint32_t)ggml_nrows(dst);
    elements[2] = 1;

    size_t موقت_اندازه = حجم(عائم) * elements[0] * ggml_nrows(dst);

    إذا (ctx->prealloc_اندازه_تقسیم_k < موقت_اندازه) {
        ctx->prealloc_اندازه_تقسیم_k = موقت_اندازه;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer موقت_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_تقسیم_k, 0);

    إذا (ctx->prealloc_تقسیم_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline1, {src_buf, dst_buf, موقت_buf}, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline2, {src_buf, dst_buf, موقت_buf}, pc, elements);

    ctx->prealloc_تقسیم_k_need_sync = درست;
}

ساكن std::array<uint32_t, 3> ggml_vk_nrows_elements(uint32_t nr) {
    إذا (nr > 262144) {
        أرجع { 512, 512, CEIL_DIV(nr, 262144) };
    }
    إذا (nr > 512) {
        أرجع { 512, CEIL_DIV(nr, 512), 1 };
    }
    أرجع { nr, 1, 1 };
}

فراغ ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    ثابت uint32_t nclasses = (uint32_t)src0->ne[0];
    ثابت uint32_t nrows    = (uint32_t)ggml_nrows(src0);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, اشاره_تهی, dst, GGML_OP_CROSS_ENTROPY_LOSS);
    GGML_ASSERT(pipeline != اشاره_تهی);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_جمع_ردیفها_f32, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer src1_buf = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst, درست);

    ثابت vk_op_افزودن_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };

    ثابت size_t tmp_اندازه = (size_t)nrows * حجم(عائم);
    إذا (ctx->prealloc_اندازه_x < tmp_اندازه) {
        ctx->prealloc_اندازه_x = tmp_اندازه;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    إذا (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_subbuffer tmp_buf = { ctx->prealloc_x, 0, tmp_اندازه };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, tmp_buf }, pc, ggml_vk_nrows_elements(nrows));
    ggml_vk_sync_buffers(ctx, subctx);

    vk_op_جمع_ردیفها_افزودن_constants sp = {};
    sp.n_تعداد_ستون = nrows;
    sp.ne01 = 1;
    sp.ne02 = 1;
    sp.وزن = 1.0f;
    init_pushconst_fastdiv(sp);
    sp.misalign_offsets = get_misalign_بایتها(ctx, dst) / ggml_type_اندازه(dst->type);

    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_جمع_ردیفها_f32, { tmp_buf, dst_buf }, sp, { 1, 1, 1 });
    ctx->prealloc_x_need_sync = درست;
}

فراغ ggml_vk_cross_entropy_loss_پشت(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ثابت ggml_tensor * grad   = dst->src[0];
    ثابت ggml_tensor * logits = dst->src[1];
    ثابت ggml_tensor * labels = dst->src[2];

    GGML_ASSERT(grad->type   == GGML_TYPE_F32);
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(labels->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_is_contiguous(logits));
    GGML_ASSERT(ggml_is_contiguous(labels));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(logits, labels));
    GGML_ASSERT(ggml_are_same_shape(logits, dst));

    ثابت uint32_t nclasses = (uint32_t)logits->ne[0];
    ثابت uint32_t nrows    = (uint32_t)ggml_nrows(logits);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, grad, logits, labels, dst, GGML_OP_CROSS_ENTROPY_LOSS_پشت);
    GGML_ASSERT(pipeline != اشاره_تهی);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer grad_buf   = ggml_vk_tensor_subbuffer(ctx, grad);
    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer labels_buf = ggml_vk_tensor_subbuffer(ctx, labels);
    vk_subbuffer dst_buf    = ggml_vk_tensor_subbuffer(ctx, dst);

    ثابت vk_op_افزودن_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { grad_buf, logits_buf, labels_buf, dst_buf }, pc, ggml_vk_nrows_elements(nrows));
}

فراغ ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_افزودن_constants>(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_ARGMAX, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], 0.0f, 0.0f, 0.0f, 0.0f });
}

فراغ ggml_vk_شمار_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_شمار_EQUAL, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

فراغ ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت uint32_t src0_type_اندازه = ggml_type_اندازه(src0->type);
    ثابت uint32_t src1_type_اندازه = ggml_type_اندازه(src1->type);
    ثابت uint32_t dst_type_اندازه = ggml_type_اندازه(dst->type);

    ggml_vk_op_f32<vk_op_دودویی_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_SOLVE_TRI, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_اندازه, (uint32_t)src0->nb[1] / src0_type_اندازه, (uint32_t)src0->nb[2] / src0_type_اندازه, (uint32_t)src0->nb[3] / src0_type_اندازه,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_اندازه, (uint32_t)src1->nb[1] / src1_type_اندازه, (uint32_t)src1->nb[2] / src1_type_اندازه, (uint32_t)src1->nb[3] / src1_type_اندازه,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_اندازه, (uint32_t) dst->nb[1] /  dst_type_اندازه, (uint32_t) dst->nb[2] /  dst_type_اندازه, (uint32_t) dst->nb[3] /  dst_type_اندازه,
        0,
        0.0f, 0.0f, 0,
    });
}

فراغ ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت int32_t s0 = dst->op_params[0];
    ثابت int32_t s1 = dst->op_params[1];
    ثابت int32_t p0 = dst->op_params[2];
    ثابت int32_t p1 = dst->op_params[3];
    ثابت int32_t d0 = dst->op_params[4];
    ثابت int32_t d1 = dst->op_params[5];

    ثابت منطقي is_2D = dst->op_params[6] == 1;

    ثابت uint32_t IC = src1->ne[is_2D ? 2 : 1];
    ثابت uint32_t IH = is_2D ? src1->ne[1] : 1;
    ثابت uint32_t IW =         src1->ne[0];

    ثابت uint32_t KH = is_2D ? src0->ne[1] : 1;
    ثابت uint32_t KW =         src0->ne[0];

    ثابت uint32_t OH = is_2D ? dst->ne[2] : 1;
    ثابت uint32_t OW =         dst->ne[1];

    ثابت uint32_t جابه_جایی_delta = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    ثابت uint32_t batch_جابه_جایی = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    ثابت uint32_t batch = src1->ne[is_2D ? 3 : 2];

    ثابت ggml_backend_vk_میانگیر_context * d_buf_ctx = (ggml_backend_vk_میانگیر_context *)dst->میانگیر->context;
    ثابت vk_میانگیر d_buf = d_buf_ctx->dev_میانگیر;

    ثابت vk::Device_آدرس dst_addr = d_buf->bda_addr + vk_tensor_جابه_جایی(dst) + dst->view_offs;

    ggml_vk_op_f32<vk_op_im2col_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_IM2_ستون, {
        dst_addr,
        batch_جابه_جایی, جابه_جایی_delta,
        IC, IW, IH, OW, OH, KW, KH,
        OH * batch,
        IC * KH * KW,
        s0, s1, p0, p1, d0, d1, batch * IC
    });
}

فراغ ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_دودویی_OP_LOCALS

    ثابت int32_t s0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t s1 = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t s2 = ((ثابت int32_t *)(dst->op_params))[2];
    ثابت int32_t p0 = ((ثابت int32_t *)(dst->op_params))[3];
    ثابت int32_t p1 = ((ثابت int32_t *)(dst->op_params))[4];
    ثابت int32_t p2 = ((ثابت int32_t *)(dst->op_params))[5];
    ثابت int32_t d0 = ((ثابت int32_t *)(dst->op_params))[6];
    ثابت int32_t d1 = ((ثابت int32_t *)(dst->op_params))[7];
    ثابت int32_t d2 = ((ثابت int32_t *)(dst->op_params))[8];
    ثابت int32_t IC = ((ثابت int32_t *)(dst->op_params))[9];

    ثابت int64_t N  = ne13 / IC;
    ثابت int64_t ID = ne12;
    ثابت int64_t IH = ne11;
    ثابت int64_t IW = ne10;

    ثابت int64_t KD = ne02;
    ثابت int64_t KH = ne01;
    ثابت int64_t KW = ne00;

    ثابت int64_t OD = ne3 / N;
    ثابت int64_t OH = ne2;
    ثابت int64_t OW = ne1;

    ثابت ggml_backend_vk_میانگیر_context * d_buf_ctx = (ggml_backend_vk_میانگیر_context *)dst->میانگیر->context;
    ثابت vk_میانگیر d_buf = d_buf_ctx->dev_میانگیر;

    ثابت vk::Device_آدرس dst_addr = d_buf->bda_addr + vk_tensor_جابه_جایی(dst) + dst->view_offs;

    vk_op_im2col_3d_افزودن_constants pc {};

    pc.dst_addr = dst_addr;
    pc.nb10 = nb10 / ggml_type_اندازه(src1->type);
    pc.nb11 = nb11 / ggml_type_اندازه(src1->type);
    pc.nb12 = nb12 / ggml_type_اندازه(src1->type);
    pc.nb13 = nb13 / ggml_type_اندازه(src1->type);
    pc.s0 = s0;
    pc.s1 = s1;
    pc.s2 = s2;
    pc.p0 = p0;
    pc.p1 = p1;
    pc.p2 = p2;
    pc.d0 = d0;
    pc.d1 = d1;
    pc.d2 = d2;
    pc.IW = IW;
    pc.IH = IH;
    pc.ID = ID;
    pc.IC = IC;
    pc.KW = KW;
    pc.OH = OH;
    pc.KD_KH_KW = KD*KH*KW;
    pc.KH_KW = KH*KW;
    pc.IC_KD_KH_KW = IC*KD*KH*KW;
    pc.N_OD_OH = N*OD*OH;
    pc.OD_OH = OD*OH;
    pc.OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    pc.OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    pc.OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;

    ggml_vk_op_f32<vk_op_im2col_3d_افزودن_constants>(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_IM2_ستون_3_D, std::move(pc));
}

فراغ ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت uint32_t dim = dst->op_params[0];
    ثابت uint32_t max_period = dst->op_params[1];
    ثابت uint32_t nb1 = dst->nb[1] / ggml_type_اندازه(dst->type);

    ggml_vk_op_f32<vk_op_timestep_embedding_افزودن_constants>(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_TIMESTEP_EMBEDDING, {
        nb1, dim, max_period,
    });
}

فراغ ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    // src0: (K, Cout, Cin, 1) -- kernel
    // src1: (L, Cin, 1, 1) -- input
    // dst: (*, Cout, 1, 1)

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_دودویی_OP_LOCALS

    GGML_ASSERT(nb00 == حجم(عائم));
    GGML_ASSERT(nb10 == حجم(عائم));

    ثابت int32_t s0 = dst->op_params[0];

    vk_op_conv_transpose_1d_افزودن_constants p{};
    p.Cout = static_cast<uint32_t>(ne01);
    p.Cin = static_cast<uint32_t>(ne02);
    p.K = static_cast<uint32_t>(ne00);
    p.L = static_cast<uint32_t>(ne10);
    p.KL = static_cast<uint32_t>(ne0);
    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.s0 = static_cast<uint32_t>(s0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_CONV_TRANSPOSE_1D, std::move(p));
}

فراغ ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    // src0: [K_OC, T_in] columns from matmul
    // dst:  [T_out, OC]

    ثابت int32_t stride = dst->op_params[0];
    ثابت int32_t oc     = dst->op_params[1];
    ثابت int32_t p0     = dst->op_params[2];

    ثابت uint32_t K_OC  = static_cast<uint32_t>(src0->ne[0]);
    ثابت uint32_t T_درون  = static_cast<uint32_t>(src0->ne[1]);
    ثابت uint32_t T_بیرون = static_cast<uint32_t>(dst->ne[0]);
    ثابت uint32_t OC    = static_cast<uint32_t>(oc);
    ثابت uint32_t K     = K_OC / OC;

    vk_op_col2im_1d_افزودن_constants p{};
    p.T_بیرون  = T_بیرون;
    p.OC     = OC;
    p.K_OC   = K_OC;
    p.T_درون   = T_درون;
    p.K      = K;
    p.stride = stride;
    p.p0     = p0;

    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_COL2IM_1D, std::move(p));
}

فراغ ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح گره_idx) {
    ثابت ggml_tensor * mul0 = cgraph->nodes[گره_idx + 0];
    ثابت ggml_tensor * sqr  = cgraph->nodes[گره_idx + 2];
    ثابت ggml_tensor * mul1 = cgraph->nodes[گره_idx + 3];
    ggml_tensor *       جمع_کن_2  = cgraph->nodes[گره_idx + 4];

    // x carries the full activation shape, a is the broadcast operand
    ثابت ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ثابت ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ثابت ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    vk_pipeline pipeline = اشاره_تهی;
    بدل (x->type) {
        حالة GGML_TYPE_F32:  pipeline = ctx->device->pipeline_snake_f32;  اكسر;
        حالة GGML_TYPE_F16:  pipeline = ctx->device->pipeline_snake_f16;  اكسر;
        حالة GGML_TYPE_BF16: pipeline = ctx->device->pipeline_snake_bf16; اكسر;
        افتراضي:             GGML_ABORT("unsupported type");
    }
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf     = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer a_buf     = ggml_vk_tensor_subbuffer(ctx, a);
    vk_subbuffer inv_b_buf = ggml_vk_tensor_subbuffer(ctx, inv_b);
    vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, جمع_کن_2);

    vk_op_snake_افزودن_constants pc{};
    pc.ne0 = static_cast<uint32_t>(x->ne[0]);
    pc.ne1 = static_cast<uint32_t>(x->ne[1]);

    std::array<uint32_t, 3> elements = { pc.ne0, pc.ne1, 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, a_buf, inv_b_buf, dst_buf }, pc, elements);
}

فراغ ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    ثابت int32_t k0 = dst->op_params[1];
    ثابت int32_t s0 = dst->op_params[2];
    ثابت int32_t p0 = dst->op_params[3];

    ثابت uint32_t IL = src0->ne[0];

    ثابت uint32_t N = dst->ne[3] * dst->ne[2];

    ثابت uint32_t OC = dst->ne[1];
    ثابت uint32_t OL = dst->ne[0];

    ثابت uint32_t parallel_elements = N * OC * OL;

    ggml_vk_op_f32<vk_op_pool1d_افزودن_constants>(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_POOL_1D, {
        IL, OL, OC,
        parallel_elements,
        op,
        k0, s0, p0,
    });
}

فراغ ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    ثابت int32_t k1 = dst->op_params[1];
    ثابت int32_t k0 = dst->op_params[2];
    ثابت int32_t s1 = dst->op_params[3];
    ثابت int32_t s0 = dst->op_params[4];
    ثابت int32_t p1 = dst->op_params[5];
    ثابت int32_t p0 = dst->op_params[6];

    ثابت uint32_t IH = src0->ne[1];
    ثابت uint32_t IW = src0->ne[0];

    ثابت uint32_t N = dst->ne[3];

    ثابت uint32_t OC = dst->ne[2];
    ثابت uint32_t OH = dst->ne[1];
    ثابت uint32_t OW = dst->ne[0];

    ثابت uint32_t parallel_elements = N * OC * OH * OW;

    ggml_vk_op_f32<vk_op_pool2d_افزودن_constants>(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_POOL_2D, {
        IW, IH, OW, OH, OC,
        parallel_elements,
        op,
        k0, k1, s0, s1, p0, p1,
    });
}

فراغ ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ثابت ggml_tensor * src0,
                            ثابت ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_دودویی_OP_LOCALS
    GGML_ASSERT(nb00 == حجم(عائم) || nb00 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb10 == حجم(عائم));
    GGML_ASSERT(nb0 == حجم(عائم));

    منطقي transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;

    vk_op_conv2d_افزودن_constants p{};
    p.Cout = static_cast<uint32_t>(!transpose ? ne03 : ne02);
    p.Cin  = static_cast<uint32_t>(!transpose ? ne02 : ne03);
    p.N    = static_cast<uint32_t>(ne13);
    GGML_ASSERT(p.Cout == ne2);
    GGML_ASSERT(p.Cin == ne12);

    p.W  = static_cast<uint32_t>(ne10);
    p.H  = static_cast<uint32_t>(ne11);
    p.OW = static_cast<uint32_t>(ne0);
    p.OH = static_cast<uint32_t>(ne1);

    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb03 = static_cast<uint32_t>(nb03 / nb00);

    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb12 = static_cast<uint32_t>(nb12 / nb10);
    p.nb13 = static_cast<uint32_t>(nb13 / nb10);

    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.nb2 = static_cast<uint32_t>(nb2 / nb0);
    p.nb3 = static_cast<uint32_t>(nb3 / nb0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, dst->op, std::move(p));
}

فراغ ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ثابت ggml_tensor * src0,
                            ثابت ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_دودویی_OP_LOCALS
    GGML_ASSERT(nb00 == حجم(عائم) || nb00 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb10 == حجم(عائم));
    GGML_ASSERT(nb0 == حجم(عائم));

    vk_op_conv3d_افزودن_constants p{};
    p.IC = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 9));
    p.N  = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 10));
    p.OC = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 11));
    GGML_ASSERT(src0->ne[3] == (int64_t)p.IC * p.OC);
    GGML_ASSERT(src1->ne[3] == (int64_t)p.IC * p.N);
    GGML_ASSERT(dst->ne[3] == (int64_t)p.OC * p.N);

    p.IW = static_cast<uint32_t>(ne10);
    p.IH = static_cast<uint32_t>(ne11);
    p.ID = static_cast<uint32_t>(ne12);
    p.OW = static_cast<uint32_t>(ne0);
    p.OH = static_cast<uint32_t>(ne1);
    p.OD = static_cast<uint32_t>(ne2);

    // the shader clamps src addresses to p.IC * p.N * p.IW * p.IH * p.ID - 1 in uint32, so the
    // total input element count must fit in a uint32.
    GGML_ASSERT((uint64_t)p.IC * p.N * p.IW * p.IH * p.ID <= 0xFFFFFFFFull);

    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb03 = static_cast<uint32_t>(nb03 / nb00);

    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb12 = static_cast<uint32_t>(nb12 / nb10);
    p.nb13 = static_cast<uint32_t>(nb13 / nb10);

    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.nb2 = static_cast<uint32_t>(nb2 / nb0);
    p.nb3 = static_cast<uint32_t>(nb3 / nb0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_CONV_3D, std::move(p));
}

فراغ ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    vk_op_conv2d_dw_افزودن_constants p{};
    p.ne = ggml_nelements(dst);
    p.channels = dst->ne[2];
    p.batches = dst->ne[3];
    p.dst_w = dst->ne[0];
    p.dst_h = dst->ne[1];
    p.src_w = src1->ne[0];
    p.src_h = src1->ne[1];
    p.knl_w = src0->ne[0];
    p.knl_h = src0->ne[1];
    p.stride_x = dst->op_params[0];
    p.stride_y = dst->op_params[1];
    p.pad_x = dst->op_params[2];
    p.pad_y = dst->op_params[3];
    p.dilation_x = dst->op_params[4];
    p.dilation_y = dst->op_params[5];

    GGML_ASSERT(src0->ne[3] == p.channels);
    GGML_ASSERT(src1->ne[3] == p.batches);

    ggml_vk_op_f32(ctx, subctx, src0, src1, اشاره_تهی, اشاره_تهی, dst, GGML_OP_CONV_2D_DW, std::move(p));
}

فراغ ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst) {
    ثابت عائم * op_params = (ثابت عائم *)dst->op_params;
    vk_op_unary_افزودن_constants p = vk_op_unary_افزودن_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, اشاره_تهی, اشاره_تهی, اشاره_تهی, dst, GGML_OP_LEAKY_RELU, std::move(p));
}

فراغ ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {
#إذا defined(GGML_VULKAN_پیاپی_TESTS)
    ثابت std::vector<size_t> vals {
        512, 512, 128,
        128, 512, 512,
        4096, 512, 4096,
        11008, 512, 4096,
        4096, 512, 11008,
        32000, 512, 4096,
        8, 8, 8,
        100, 46, 576,
        623, 111, 128,
        100, 46, 558,
        512, 1, 256,
        128, 110, 622,
        511, 511, 127,
        511, 511, 7,
        511, 511, 17,
        49, 49, 128,
        128, 49, 49,
        4096, 49, 4096,
    };
    ثابت size_t رقم_it = 100;

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, رقم_it, 1, 0, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, رقم_it, 1, 1, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, رقم_it, 1, 2, GGML_TYPE_Q4_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, رقم_it, 1, 0, GGML_TYPE_Q4_0, درست);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, رقم_it, 1, 1, GGML_TYPE_Q4_0, درست);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, رقم_it, 1, 2, GGML_TYPE_Q4_0, درست);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, رقم_it, 1, 0, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, رقم_it, 1, 1, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, رقم_it, 1, 2, GGML_TYPE_Q8_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, رقم_it, 1, 0, GGML_TYPE_Q8_0, درست);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, رقم_it, 1, 1, GGML_TYPE_Q8_0, درست);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, رقم_it, 1, 2, GGML_TYPE_Q8_0, درست);

    abort();

    لكل (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 1, 0);
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 1, 1);
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 1, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 2, 0);
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 2, 1);
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 2, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 4, 0);
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 4, 1);
        ggml_vk_test_matmul<ggml_fp16_t, عائم>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 4, 2);
        std::cerr << '\n' << std::پایان_سطر;

        إذا (vals[i + 2] % 32 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 1, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 1, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 1, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 2, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 2, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 2, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 4, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 4, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 4, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n' << std::پایان_سطر;
        }

        إذا (vals[i + 2] % 256 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 1, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 1, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 1, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 2, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 2, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 2, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 4, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 4, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, رقم_it, 4, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n' << std::پایان_سطر;
        }
    }

    GGML_ABORT("fatal error");
#پایان_اگر

    إذا (subctx) {
        // Submit and wait for any pending work before reallocating the buffers
        ggml_vk_ctx_انتها(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = درست;
        ggml_vk_synchronize(ctx);
        GGML_ASSERT(ctx->compute_ctx.expired());
        ggml_vk_ctx_begin(ctx->device, subctx);
        ctx->compute_ctx = subctx;
    }

    إذا (ctx->prealloc_x == اشاره_تهی || (ctx->prealloc_اندازه_x > 0 && ctx->prealloc_x->size < ctx->prealloc_اندازه_x)) {
        VK_LOG_حافظه("ggml_vk_preallocate_buffers(x_size: " << ctx->prealloc_اندازه_x << ")");
        // Resize buffer
        إذا (ctx->prealloc_x != اشاره_تهی) {
            ggml_vk_destroy_میانگیر(ctx->prealloc_x);
        }
        ctx->prealloc_x = ggml_vk_create_میانگیر_device(ctx->device, ctx->prealloc_اندازه_x);
    }
    إذا (ctx->prealloc_y == اشاره_تهی || (ctx->prealloc_اندازه_y > 0 && ctx->prealloc_y->size < ctx->prealloc_اندازه_y)) {
        VK_LOG_حافظه("ggml_vk_preallocate_buffers(y_size: " << ctx->prealloc_اندازه_y << ")");
        // Resize buffer
        إذا (ctx->prealloc_y != اشاره_تهی) {
            ggml_vk_destroy_میانگیر(ctx->prealloc_y);
        }
        ctx->prealloc_y = ggml_vk_create_میانگیر_device(ctx->device, ctx->prealloc_اندازه_y);
        ctx->prealloc_y_آخرین_pipeline_استفاده_شده = اشاره_تهی;
        ctx->prealloc_y_آخرین_tensor_استفاده_شده = اشاره_تهی;
        ctx->prealloc_y_آخرین_k_padded = خطأ;
    }
    إذا (ctx->prealloc_تقسیم_k == اشاره_تهی || (ctx->prealloc_اندازه_تقسیم_k > 0 && ctx->prealloc_تقسیم_k->size < ctx->prealloc_اندازه_تقسیم_k)) {
        VK_LOG_حافظه("ggml_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_اندازه_تقسیم_k << ")");
        // Resize buffer
        إذا (ctx->prealloc_تقسیم_k != اشاره_تهی) {
            ggml_vk_destroy_میانگیر(ctx->prealloc_تقسیم_k);
        }
        ctx->prealloc_تقسیم_k = ggml_vk_create_میانگیر_device(ctx->device, ctx->prealloc_اندازه_تقسیم_k);
    }
    إذا (ctx->prealloc_جمع_کن_rms_partials == اشاره_تهی || (ctx->prealloc_اندازه_جمع_کن_rms_partials > 0 && ctx->prealloc_جمع_کن_rms_partials->size < ctx->prealloc_اندازه_جمع_کن_rms_partials)) {
        VK_LOG_حافظه("ggml_vk_preallocate_buffers(add_partials_size: " << ctx->prealloc_جمع_کن_rms_partials << ")");
        // Resize buffer
        إذا (ctx->prealloc_جمع_کن_rms_partials != اشاره_تهی) {
            ggml_vk_destroy_میانگیر(ctx->prealloc_جمع_کن_rms_partials);
        }
        ctx->prealloc_جمع_کن_rms_partials = ggml_vk_create_میانگیر_device(ctx->device, ctx->prealloc_اندازه_جمع_کن_rms_partials);
    }
}

منطقي ggml_vk_build_گراف(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحيح گره_idx, ggml_tensor *گره_begin, صحيح گره_idx_begin, منطقي آخرین_گره, منطقي almost_ready, منطقي submit){
    ggml_tensor * گره = cgraph->nodes[گره_idx];
    إذا (ggml_is_خالی(گره) || ggml_op_is_خالی(گره->op) || !گره->میانگیر) {
        أرجع خطأ;
    }
    إذا ((گره->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        أرجع خطأ;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << گره << ", " << ggml_op_name(گره->op) << ")");
    ctx->semaphore_idx = 0;

    ggml_tensor * src0 = گره->src[0];
    ggml_tensor * src1 = گره->src[1];
    ggml_tensor * src2 = گره->src[2];
    ggml_tensor * src3 = گره->src[3];

    إذا (گره->op == GGML_OP_جمع_کن) {
        صحيح بعدی_گره_idx = گره_idx + 1 + ctx->رقم_additional_fused_ops;
        إذا (بعدی_گره_idx < cgraph->n_nodes &&
            cgraph->nodes[بعدی_گره_idx]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[بعدی_گره_idx]->src[0] == cgraph->nodes[بعدی_گره_idx - 1] &&
            ggml_nrows(cgraph->nodes[بعدی_گره_idx]) == 1 &&
            ctx->device->جمع_کن_rms_fusion) {
            uint32_t size = ggml_vk_rms_partials_اندازه(ctx, cgraph->nodes[گره_idx]);
            ctx->do_جمع_کن_rms_partials_جابه_جایی_calculation = درست;
            إذا (ctx->prealloc_اندازه_جمع_کن_rms_partials_جابه_جایی + size <= ctx->prealloc_اندازه_جمع_کن_rms_partials) {
                ctx->do_جمع_کن_rms_partials = درست;
            }
        }
    }

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    {
        // This logic detects dependencies between modes in the graph and calls ggml_vk_sync_buffers
        // to synchronize them. This handles most "normal" synchronization when computing the graph, and when
        // there is no auxiliary memory use, it shouldn't be necessary to call ggml_vk_sync_buffers
        // outside of this logic. When a node uses one of the prealloc buffers for something like
        // dequantization or split_k, additional synchronization is needed between those passes.
        منطقي need_sync = خطأ;

        // Check whether "node" requires synchronization. The node requires synchronization if it
        // overlaps in memory with another unsynchronized node and at least one of them is a write.
        // Destination nodes are checked against both the written/read lists. Source nodes are only
        // checked against the written list. Two nodes overlap in memory if they come from the same
        // buffer and the tensor or view ranges overlap.
        خودکار ثابت &overlaps_unsynced = [&](ثابت ggml_tensor *گره, ثابت std::vector<ثابت ggml_tensor *> &unsynced_nodes) -> منطقي {
            إذا (unsynced_nodes.size() == 0) {
                أرجع خطأ;
            }
            خودکار n_پایه = vk_tensor_جابه_جایی(گره) + گره->view_offs;
            خودکار n_اندازه = ggml_nbytes(گره);
            ggml_backend_vk_میانگیر_context * a_buf_ctx = (ggml_backend_vk_میانگیر_context *)گره->میانگیر->context;
            vk_میانگیر a_buf = a_buf_ctx->dev_میانگیر;
            لكل (خودکار &other : unsynced_nodes) {
                ggml_backend_vk_میانگیر_context * o_buf_ctx = (ggml_backend_vk_میانگیر_context *)other->میانگیر->context;
                vk_میانگیر o_buf = o_buf_ctx->dev_میانگیر;
                إذا (a_buf == o_buf) {
                    خودکار o_پایه = vk_tensor_جابه_جایی(other) + other->view_offs;
                    خودکار o_اندازه = ggml_nbytes(other);

                    إذا ((o_پایه <= n_پایه && n_پایه < o_پایه + o_اندازه) ||
                        (n_پایه <= o_پایه && o_پایه < n_پایه + n_اندازه)) {
                        أرجع درست;
                    }
                }
            }
            أرجع خطأ;
        };

        // For all fused ops, check if the destination node or any of the source
        // nodes require synchronization.
        لكل (int32_t i = 0; i < ctx->رقم_additional_fused_ops + 1 && !need_sync; ++i) {
            ثابت ggml_tensor *cur_گره = cgraph->nodes[گره_idx + i];
            // If the node actually writes to memory, then check if it needs to sync
            إذا (ctx->fused_ops_write_ماسک & (1 << i)) {
                إذا (overlaps_unsynced(cur_گره, ctx->unsynced_nodes_read) || overlaps_unsynced(cur_گره, ctx->unsynced_nodes_written)) {
                    need_sync = درست;
                    اكسر;
                }
            }
            لكل (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                إذا (!cur_گره->src[j]) {
                    استمر;
                }
                إذا (overlaps_unsynced(cur_گره->src[j], ctx->unsynced_nodes_written)) {
                    need_sync = درست;
                    اكسر;
                }
            }
        }

        إذا (need_sync) {
            إذا (vk_enable_sync_logger) {
                std::cerr <<  "sync" << std::پایان_سطر;
            }
            ctx->unsynced_nodes_written.clear();
            ctx->unsynced_nodes_read.clear();
            ggml_vk_sync_buffers(ctx, compute_ctx);

            إذا (vk_perf_logger_enabled && vk_perf_logger_concurrent) {
                ctx->query_گره_idx[ctx->query_idx] = گره_idx;
                compute_ctx->s->میانگیر->buf.writeTimestamp(vk::Pipeline_Stage_Flag_بیتها::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            }
        }
        // Add all fused nodes to the unsynchronized lists.
        لكل (int32_t i = 0; i < ctx->رقم_additional_fused_ops + 1; ++i) {
            ثابت ggml_tensor *cur_گره = cgraph->nodes[گره_idx + i];
            // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
            إذا (ctx->fused_ops_write_ماسک & (1 << i)) {
                ctx->unsynced_nodes_written.push_back(cur_گره);
            }
            لكل (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                إذا (!cur_گره->src[j]) {
                    استمر;
                }
                ctx->unsynced_nodes_read.push_back(cur_گره->src[j]);
            }
        }
    }
    إذا (vk_enable_sync_logger) {
        لكل (صحيح i = 0; i < ctx->رقم_additional_fused_ops + 1; ++i) {
            خودکار *n = cgraph->nodes[گره_idx + i];
            std::cerr << گره_idx + i << " " << ggml_op_name(n->op) << " " <<  n->name;
            إذا (n->op == GGML_OP_GLU) {
                std::cerr << " " << ggml_glu_op_name(ggml_get_glu_op(n)) << " " << (n->src[1] ? "split" : "single") << " ";
            }
            إذا (n->op == GGML_OP_ROPE) {
                ثابت صحيح mode = ((ثابت int32_t *) n->op_params)[2];
                std::cerr << " rope mode: " << mode;
            }
            std::cerr << std::پایان_سطر;
        }
    }

    // closed explicitly below, and by the destructor on the early returns
    ggml_vk_debug_label dbg(compute_ctx, cgraph, گره_idx, ctx->رقم_additional_fused_ops);

    بدل (گره->op) {
    حالة GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_REPEAT_پشت:
        ggml_vk_repeat_پشت(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_ACC:
    حالة GGML_OP_مجموعه:
        ggml_vk_acc(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_GET_ردیفها:
        إذا (ctx->fused_topk_qsa) {
            ggml_vk_topk_qsa(ctx, compute_ctx, cgraph, گره_idx);
        } وإلا {
            ggml_vk_get_ردیفها(ctx, compute_ctx, src0, src1, گره);
        }

        اكسر;
    حالة GGML_OP_GET_ردیفها_پشت:
        ggml_vk_get_ردیفها_پشت(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_جمع_کن:
        إذا (ctx->رقم_additional_fused_ops) {
            ggml_vk_multi_جمع_کن(ctx, compute_ctx, cgraph, گره_idx);
        } وإلا {
            ggml_vk_جمع_کن(ctx, compute_ctx, src0, src1, گره);
        }
        اكسر;
    حالة GGML_OP_بیرون_PROD:
        ggml_vk_بیرون_prod(ctx, compute_ctx, src0, src1, گره);
        اكسر;
    حالة GGML_OP_SUB:
        ggml_vk_sub(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_MUL:
        إذا (ctx->رقم_additional_fused_ops) {
            ggml_vk_snake_dispatch_fused(ctx, compute_ctx, cgraph, گره_idx);
        } وإلا {
            ggml_vk_mul(ctx, compute_ctx, src0, src1, گره);
        }

        اكسر;
    حالة GGML_OP_DIV:
        ggml_vk_div(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_جمع_کن_ID:
        ggml_vk_جمع_کن_id(ctx, compute_ctx, src0, src1, src2, گره);

        اكسر;
    حالة GGML_OP_CONCAT:
        ggml_vk_concat(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_UPSCALE:
        ggml_vk_upscale(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_ADD1:
        ggml_vk_add1(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_ARANGE:
        ggml_vk_arange(ctx, compute_ctx, گره);

        اكسر;
    حالة GGML_OP_FILL:
        ggml_vk_fill(ctx, compute_ctx, گره);

        اكسر;
    حالة GGML_OP_SCALE:
        إذا (ctx->fused_hc_post_gate) {
            ggml_tensor * hc_post = cgraph->nodes[گره_idx + ctx->رقم_additional_fused_ops];
            ggml_vk_dsv4_hc_post(ctx, compute_ctx, hc_post->src[0], hc_post->src[1], hc_post->src[2], hc_post->src[3], hc_post, گره);
        } وإلا {
            ggml_vk_scale(ctx, compute_ctx, src0, گره);
        }

        اكسر;
    حالة GGML_OP_SQR:
        ggml_vk_sqr(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_SQRT:
        ggml_vk_sqrt(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_SIN:
        ggml_vk_sin(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_COS:
        ggml_vk_cos(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_LOG:
        ggml_vk_log(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_TRI:
        ggml_vk_tri(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_DIAG:
        ggml_vk_diag(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_CLAMP:
        ggml_vk_clamp(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_PAD:
        ggml_vk_pad(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_PAD_REFLECT_1D:
        ggml_vk_pad_reflect_1d(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_ROLL:
        ggml_vk_roll(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_CPY:
    حالة GGML_OP_CONT:
    حالة GGML_OP_DUP:
        ggml_vk_cpy(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_مجموعه_ردیفها:
        ggml_vk_مجموعه_ردیفها(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_SILU_پشت:
        ggml_vk_silu_پشت(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_NORM:
        ggml_vk_norm(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_گروه_NORM:
        ggml_vk_گروه_norm(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_RMS_NORM:
        ggml_vk_rms_norm(ctx, compute_ctx, cgraph, گره_idx, (عائم *)گره->op_params);
        اكسر;
    حالة GGML_OP_RMS_NORM_پشت:
        ggml_vk_rms_norm_پشت(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_L2_NORM:
        ggml_vk_l2_norm(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_UNARY:
        إذا (ctx->fused_topk_moe_نما != TOPK_MOE_شمار) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, گره_idx);
            اكسر;
        }
        إذا (ctx->رقم_additional_fused_ops) {
            ggml_vk_unary_mul(ctx, compute_ctx, cgraph, گره_idx);
            اكسر;
        }

        بدل (ggml_get_unary_op(گره)) {
        حالة GGML_UNARY_OP_ELU:
        حالة GGML_UNARY_OP_EXP:
        حالة GGML_UNARY_OP_EXPM1:
        حالة GGML_UNARY_OP_SILU:
        حالة GGML_UNARY_OP_GELU:
        حالة GGML_UNARY_OP_GELU_ERF:
        حالة GGML_UNARY_OP_GELU_سریع:
        حالة GGML_UNARY_OP_RELU:
        حالة GGML_UNARY_OP_NEG:
        حالة GGML_UNARY_OP_TANH:
        حالة GGML_UNARY_OP_SIGMOID:
        حالة GGML_UNARY_OP_HARDSIGMOID:
        حالة GGML_UNARY_OP_HARDSWISH:
        حالة GGML_UNARY_OP_ABS:
        حالة GGML_UNARY_OP_SOFTPLUS:
        حالة GGML_UNARY_OP_گام:
        حالة GGML_UNARY_OP_دور:
        حالة GGML_UNARY_OP_CEIL:
        حالة GGML_UNARY_OP_FLOOR:
        حالة GGML_UNARY_OP_TRUNC:
        حالة GGML_UNARY_OP_SGN:
            ggml_vk_unary(ctx, compute_ctx, src0, گره);
            اكسر;
        حالة GGML_UNARY_OP_XIELU:
            ggml_vk_xielu(ctx, compute_ctx, src0, گره);
            اكسر;
        افتراضي:
            أرجع خطأ;
        }
        اكسر;
    حالة GGML_OP_GLU:
        بدل (ggml_get_glu_op(گره)) {
        حالة GGML_GLU_OP_GEGLU:
        حالة GGML_GLU_OP_REGLU:
        حالة GGML_GLU_OP_SWIGLU:
        حالة GGML_GLU_OP_SWIGLU_OAI:
        حالة GGML_GLU_OP_GEGLU_ERF:
        حالة GGML_GLU_OP_GEGLU_سریع:
        حالة GGML_GLU_OP_SWIGLU_CLAMP:
            ggml_vk_glu(ctx, compute_ctx, src0, src1, گره);
            اكسر;
        افتراضي:
            أرجع خطأ;
        }
        اكسر;
    حالة GGML_OP_DIAG_ماسک_INF:
        ggml_vk_diag_ماسک_inf(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_SOFT_MAX:
        إذا (ctx->fused_topk_moe_نما != TOPK_MOE_شمار) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, گره_idx);
        } وإلا {
            ggml_vk_soft_max(ctx, compute_ctx, src0, src1, src2, گره);
        }

        اكسر;
    حالة GGML_OP_SOFT_MAX_پشت:
        ggml_vk_soft_max_پشت(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_ROPE:
        ggml_vk_rope(ctx, compute_ctx, cgraph, گره_idx, خطأ);

        اكسر;
    حالة GGML_OP_ROPE_پشت:
        ggml_vk_rope(ctx, compute_ctx, cgraph, گره_idx, درست);

        اكسر;
    حالة GGML_OP_ARGSORT:
        إذا (ctx->fused_topk_moe_نما != TOPK_MOE_شمار) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, گره_idx);
        } وإلا {
            ggml_vk_argsort(ctx, compute_ctx, src0, گره);
        }

        اكسر;
    حالة GGML_OP_بالاترین_K:
        ggml_vk_topk(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_جمع:
        ggml_vk_جمع(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_جمع_ردیفها:
        ggml_vk_جمع_ردیفها(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_CUMSUM:
        ggml_vk_cumsum(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_DSV4_HC_COMB:
        ggml_vk_dsv4_hc_comb(ctx, compute_ctx, src0, src1, src2, گره);

        اكسر;
    حالة GGML_OP_DSV4_HC_PRE:
        ggml_vk_dsv4_hc_pre(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_DSV4_HC_POST:
        ggml_vk_dsv4_hc_post(ctx, compute_ctx, src0, src1, src2, src3, گره);

        اكسر;
    حالة GGML_OP_میانگین:
        ggml_vk_میانگین(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_ARGMAX:
        ggml_vk_argmax(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_CROSS_ENTROPY_LOSS:
        ggml_vk_cross_entropy_loss(ctx, compute_ctx, گره);

        اكسر;
    حالة GGML_OP_CROSS_ENTROPY_LOSS_پشت:
        ggml_vk_cross_entropy_loss_پشت(ctx, compute_ctx, گره);

        اكسر;
    حالة GGML_OP_شمار_EQUAL:
        ggml_vk_شمار_equal(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_SOLVE_TRI:
        ggml_vk_solve_tri(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_IM2_ستون:
        ggml_vk_im2col(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_IM2_ستون_3_D:
        ggml_vk_im2col_3d(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_TIMESTEP_EMBEDDING:
        ggml_vk_timestep_embedding(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_COL2IM_1D:
        ggml_vk_col2im_1d(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_CONV_TRANSPOSE_1D:
        ggml_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_POOL_1D:
        ggml_vk_pool_1d(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_POOL_2D:
        ggml_vk_pool_2d(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_CONV_2D:
    حالة GGML_OP_CONV_TRANSPOSE_2D:
        ggml_vk_conv_2d(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_CONV_3D:
        ggml_vk_conv_3d(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_CONV_2D_DW:
        ggml_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, گره);

        اكسر;
    حالة GGML_OP_LEAKY_RELU:
        ggml_vk_leaky_relu(ctx, compute_ctx, src0, گره);

        اكسر;
    حالة GGML_OP_MUL_MAT:
        ggml_vk_mul_mat(ctx, compute_ctx, cgraph, گره_idx);

        اكسر;
    حالة GGML_OP_MUL_MAT_ID:
        ggml_vk_mul_mat_id(ctx, compute_ctx, cgraph, گره_idx);

        اكسر;

    حالة GGML_OP_FLASH_ATTN_EXT:
        ggml_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, گره->src[4], گره);

        اكسر;

    حالة GGML_OP_RWKV_WKV6:
        ggml_vk_rwkv_wkv6(ctx, compute_ctx, گره);

        اكسر;

    حالة GGML_OP_RWKV_WKV7:
        ggml_vk_rwkv_wkv7(ctx, compute_ctx, گره);

        اكسر;

    حالة GGML_OP_GATED_خطی_ATTN:
        ggml_vk_gated_خطی_attn(ctx, compute_ctx, گره);

        اكسر;

    حالة GGML_OP_LIGHTNING_INDEXER:
        ggml_vk_lightning_indexer(ctx, compute_ctx, گره);

        اكسر;

    حالة GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, گره);

        اكسر;

    حالة GGML_OP_SSM_پیمایش:
        ggml_vk_ssm_پیمایش(ctx, compute_ctx, گره);

        اكسر;

    حالة GGML_OP_SSM_CONV:
        ggml_vk_ssm_conv(ctx, compute_ctx, cgraph, گره_idx);

        اكسر;

    حالة GGML_OP_OPT_گام_ADAMW:
        ggml_vk_opt_گام_adamw(ctx, compute_ctx, گره);

        اكسر;

    حالة GGML_OP_OPT_گام_SGD:
        ggml_vk_opt_گام_sgd(ctx, compute_ctx, src0, src1, src2, گره);

        اكسر;
    افتراضي:
        أرجع خطأ;
    }

    // the submit path below can end the command buffer, so close the region first
    dbg.close();

    ctx->tensor_ctxs[گره_idx] = compute_ctx;

#إذا defined(GGML_VULKAN_CHECK_RESULTS)
    // Force context reset on each node so that each tensor ends up in its own context
    // and can be run and compared to its CPU equivalent separately
    آخرین_گره = درست;
#پایان_اگر

    إذا (submit || آخرین_گره) {
        ggml_vk_ctx_انتها(compute_ctx);

        // TODO probably it'd be better to pass a exit_node flag to ggml_vk_compute_forward
        إذا (آخرین_گره) {
            compute_ctx->exit_tensor_idx = گره_idx_begin;
        }
        وإلا {
            compute_ctx->exit_tensor_idx = -1;
        }

        ctx->compute_ctx.reset();

        ggml_vk_compute_forward(ctx, cgraph, گره_begin, گره_idx_begin, almost_ready);
    }
    أرجع درست;
}

فراغ ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ggml_tensor * tensor, صحيح tensor_idx, منطقي almost_ready) {
    GGML_استفادهنشده(cgraph);
    GGML_استفادهنشده(tensor);

    VK_LOG_DEBUG("ggml_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << ggml_op_name(tensor->op) << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

    vk_context subctx = ctx->tensor_ctxs[tensor_idx].قفل();

    // Only run if ctx hasn't been submitted yet
    إذا (!subctx->seqs.خالی()) {
#اگر_تعریف_شده GGML_VULKAN_CHECK_RESULTS
        ggml_vk_check_results_0(ctx, cgraph, tensor_idx);
#پایان_اگر

        // Do staging buffer copies
        لكل (خودکار& cpy : subctx->درون_memcpys) {
            کپی_حافظه(cpy.dst, cpy.src, cpy.n);
        }

        لكل (خودکار& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        إذا (ctx->device->serialize_submissions) {
            ggml_vk_submit(subctx, ctx->fence);
        } وإلا إذا (almost_ready && !ctx->almost_ready_fence_pending) {
            ggml_vk_submit(subctx, ctx->almost_ready_fence);
            ctx->almost_ready_fence_pending = درست;
        } وإلا {
            ggml_vk_submit(subctx, {});
        }
        ctx->submit_pending = درست;

#اگر_تعریف_شده GGML_VULKAN_CHECK_RESULTS
        ggml_vk_synchronize(ctx);
        ggml_vk_check_results_1(ctx, cgraph, tensor_idx);
#پایان_اگر
    }

    إذا (tensor_idx == subctx->exit_tensor_idx) {
        // Do staging buffer copies
        لكل (خودکار& cpy : subctx->بیرون_memcpys) {
            کپی_حافظه(cpy.dst, cpy.src, cpy.n);
        }
        subctx->درون_memcpys.clear();
        subctx->بیرون_memcpys.clear();
        subctx->memsets.clear();
    }
}

فراغ ggml_vk_گراف_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_graph_cleanup()");
    ctx->prealloc_y_آخرین_pipeline_استفاده_شده = {};
    ctx->prealloc_y_آخرین_tensor_استفاده_شده = اشاره_تهی;
    ctx->prealloc_y_آخرین_k_padded = خطأ;

    ctx->unsynced_nodes_written.clear();
    ctx->unsynced_nodes_read.clear();
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_تقسیم_k_need_sync = خطأ;

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    إذا (ctx->device->async_use_transfer_صف) {
        ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);
    }

    لكل (size_t i = 0; i < ctx->gc.semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.semaphores[i].s });
    }
    ctx->gc.semaphores.clear();

    لكل (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.tl_semaphores[i].s });
    }
    ctx->gc.tl_semaphores.clear();
    ctx->semaphore_idx = 0;

    ctx->event_idx = 0;

    لكل (خودکار& event : ctx->gc.events) {
        ctx->device->device.resetEvent(event);
    }

    ctx->tensor_ctxs.clear();
    ctx->gc.contexts.clear();
    ctx->pipeline_descriptor_مجموعه_requirements = 0;
    ctx->descriptor_مجموعه_idx = 0;
}

فراغ ggml_vk_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_cleanup(" << ctx->name << ")");
    // discard any unsubmitted command buffers
    ctx->compute_ctx.reset();
    // wait for any pending command buffers to finish
    ggml_vk_synchronize(ctx);

    ggml_vk_گراف_cleanup(ctx);

    ggml_vk_destroy_میانگیر(ctx->prealloc_x);
    ggml_vk_destroy_میانگیر(ctx->prealloc_y);
    ggml_vk_destroy_میانگیر(ctx->prealloc_تقسیم_k);
    ggml_vk_destroy_میانگیر(ctx->prealloc_جمع_کن_rms_partials);
    ggml_vk_destroy_میانگیر(ctx->sync_staging);

    ctx->prealloc_y_آخرین_pipeline_استفاده_شده = اشاره_تهی;
    ctx->prealloc_y_آخرین_tensor_استفاده_شده = اشاره_تهی;
    ctx->prealloc_y_آخرین_k_padded = خطأ;

    ctx->prealloc_اندازه_x = 0;
    ctx->prealloc_اندازه_y = 0;
    ctx->prealloc_اندازه_تقسیم_k = 0;

    لكل (خودکار& event : ctx->gc.events) {
        ctx->device->device.destroyEvent(event);
    }
    ctx->gc.events.clear();

    ctx->device->device.destroyFence(ctx->fence);
    ctx->device->device.destroyFence(ctx->almost_ready_fence);

    لكل (خودکار& pool : ctx->descriptor_pools) {
        ctx->device->device.destroyDescriptorPool(pool);
    }
    ctx->descriptor_pools.clear();
    ctx->descriptor_sets.clear();
    ctx->descriptor_مجموعه_bindings.clear();

    ctx->compute_cmd_pool.destroy(ctx->device->device);
    إذا (ctx->device->async_use_transfer_صف) {
        ctx->device->device.destroySemaphore(ctx->transfer_semaphore.s);

        ctx->transfer_cmd_pool.destroy(ctx->device->device);
    }
    إذا (vk_perf_logger_enabled) {
        ctx->perf_logger->print_timings(درست);
    }
}

صحيح ggml_vk_get_device_شمار() {
    ggml_vk_instance_init();

    أرجع vk_instance.device_indices.size();
}

فراغ ggml_vk_get_device_description(صحيح device, حرف * description, size_t description_اندازه) {
    ggml_vk_instance_init();

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    vk::PhysicalDeviceProperties props;
    devices[device].getProperties(&props);

    snprintf(description, description_اندازه, "%s", props.deviceName.داده());
}

منطقي ggml_backend_میانگیر_is_vk(ggml_backend_میانگیر_t میانگیر) {
    أرجع میانگیر->buft->iface.get_name == ggml_backend_vk_میانگیر_type_name;
}

فراغ ggml_backend_vk_میانگیر_free_میانگیر(ggml_backend_میانگیر_t میانگیر) {
    VK_LOG_حافظه("ggml_backend_vk_buffer_free_buffer()");
    ggml_backend_vk_میانگیر_context * ctx = (ggml_backend_vk_میانگیر_context *)میانگیر->context;
    ggml_vk_destroy_میانگیر(ctx->dev_میانگیر);
    حذف ctx;
}

فراغ * ggml_backend_vk_میانگیر_get_پایه(ggml_backend_میانگیر_t میانگیر) {
    أرجع vk_ptr_پایه;

    استفادهنشده(میانگیر);
}

تعداد ggml_وضعیت ggml_backend_vk_میانگیر_init_tensor(ggml_backend_میانگیر_t میانگیر, ggml_tensor * tensor) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_init_tensor(" << میانگیر << " (" << میانگیر->context << "), " << tensor << ")");
    إذا (tensor->view_src != اشاره_تهی) {
        GGML_ASSERT(tensor->view_src->میانگیر->buft == میانگیر->buft);
    }
    أرجع GGML_وضعیت_SUCCESS;
}

فراغ ggml_backend_vk_میانگیر_memset_tensor(ggml_backend_میانگیر_t میانگیر, ggml_tensor * tensor, uint8_t value, size_t جابه_جایی, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_memset_tensor(" << میانگیر << ", " << tensor << ", " << value << ", " << جابه_جایی << ", " << size << ")");
    ggml_backend_vk_میانگیر_context * buf_ctx = (ggml_backend_vk_میانگیر_context *)میانگیر->context;
    vk_میانگیر buf = buf_ctx->dev_میانگیر;

    إذا (size == 0) {
        أرجع;
    }

    uint32_t val32 = (uint32_t)value * 0x01010101;
    ggml_vk_میانگیر_memset(buf, vk_tensor_جابه_جایی(tensor) + tensor->view_offs + جابه_جایی, val32, size);
}

فراغ ggml_backend_vk_میانگیر_مجموعه_tensor(ggml_backend_میانگیر_t میانگیر, ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor(" << میانگیر << ", " << tensor << ", " << داده << ", " << جابه_جایی << ", " << size << ")");
    ggml_backend_vk_میانگیر_context * buf_ctx = (ggml_backend_vk_میانگیر_context *)میانگیر->context;
    vk_میانگیر buf = buf_ctx->dev_میانگیر;

    إذا (size == 0) {
        أرجع;
    }

    ggml_vk_میانگیر_write(buf, vk_tensor_جابه_جایی(tensor) + tensor->view_offs + جابه_جایی, داده, size);
}

فراغ ggml_backend_vk_میانگیر_مجموعه_tensor_2d(ggml_backend_میانگیر_t میانگیر, ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_داده) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor_2d(" << میانگیر << ", " << tensor << ", " << داده << ", " << جابه_جایی << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_داده << ")");
    ggml_backend_vk_میانگیر_context * buf_ctx = (ggml_backend_vk_میانگیر_context *)میانگیر->context;
    vk_میانگیر buf = buf_ctx->dev_میانگیر;

    إذا (size == 0) {
        أرجع;
    }

    ggml_vk_میانگیر_write_2d(buf, vk_tensor_جابه_جایی(tensor) + tensor->view_offs + جابه_جایی, داده, stride_داده, stride_tensor, size, n_copies);
}

فراغ ggml_backend_vk_میانگیر_get_tensor(ggml_backend_میانگیر_t میانگیر, ثابت ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor(" << میانگیر << ", " << tensor << ", " << داده << ", " << جابه_جایی << ", " << size << ")");
    ggml_backend_vk_میانگیر_context * buf_ctx = (ggml_backend_vk_میانگیر_context *)میانگیر->context;

    إذا (size == 0) {
        أرجع;
    }

    vk_میانگیر buf = buf_ctx->dev_میانگیر;

    ggml_vk_میانگیر_read(buf, vk_tensor_جابه_جایی(tensor) + tensor->view_offs + جابه_جایی, داده, size);
}

فراغ ggml_backend_vk_میانگیر_get_tensor_2d(ggml_backend_میانگیر_t میانگیر, ثابت ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_داده) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor_2d(" << میانگیر << ", " << tensor << ", " << داده << ", " << جابه_جایی << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_داده << ")");
    ggml_backend_vk_میانگیر_context * buf_ctx = (ggml_backend_vk_میانگیر_context *)میانگیر->context;

    إذا (size == 0) {
        أرجع;
    }

    vk_میانگیر buf = buf_ctx->dev_میانگیر;

    ggml_vk_میانگیر_read_2d(buf, vk_tensor_جابه_جایی(tensor) + tensor->view_offs + جابه_جایی, داده, stride_tensor, stride_داده, size, n_copies);
}

منطقي ggml_backend_vk_میانگیر_cpy_tensor(ggml_backend_میانگیر_t میانگیر, ثابت ggml_tensor * src, ggml_tensor * dst) {
    إذا (ggml_nbytes(src) == 0) {
        أرجع درست;
    }

    إذا (ggml_backend_میانگیر_is_vk(src->میانگیر)) {
        ggml_backend_vk_میانگیر_context * src_buf_ctx = (ggml_backend_vk_میانگیر_context *)src->میانگیر->context;
        ggml_backend_vk_میانگیر_context * dst_buf_ctx = (ggml_backend_vk_میانگیر_context *)dst->میانگیر->context;

        vk_میانگیر src_buf = src_buf_ctx->dev_میانگیر;
        vk_میانگیر dst_buf = dst_buf_ctx->dev_میانگیر;

        ggml_vk_میانگیر_copy(dst_buf, vk_tensor_جابه_جایی(dst) + dst->view_offs, src_buf, vk_tensor_جابه_جایی(src) + src->view_offs, ggml_nbytes(src));

        أرجع درست;
    }
    أرجع خطأ;

    استفادهنشده(میانگیر);
}

فراغ ggml_backend_vk_میانگیر_clear(ggml_backend_میانگیر_t میانگیر, uint8_t value) {
    ggml_backend_vk_میانگیر_context * ctx = (ggml_backend_vk_میانگیر_context *)میانگیر->context;

    ggml_vk_میانگیر_memset(ctx->dev_میانگیر, 0, value, میانگیر->size);
}

ثابت حرف * ggml_backend_vk_میانگیر_type_name(ggml_backend_میانگیر_type_t buft) {
    ggml_backend_vk_میانگیر_type_context * ctx = (ggml_backend_vk_میانگیر_type_context *)buft->context;

    أرجع ctx->name.c_str();
}

ggml_backend_میانگیر_t ggml_backend_vk_میانگیر_type_alloc_میانگیر(ggml_backend_میانگیر_type_t buft, size_t size) {
    VK_LOG_حافظه("ggml_backend_vk_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_vk_میانگیر_type_context * ctx = (ggml_backend_vk_میانگیر_type_context *) buft->context;

    vk_میانگیر dev_میانگیر = اشاره_تهی;
    حاول {
        dev_میانگیر = ggml_vk_create_میانگیر_device(ctx->device, size);
    } امسك (ثابت vk::SystemError& e) {
        أرجع اشاره_تهی;
    }

    ggml_backend_vk_میانگیر_context * bufctx = جديد ggml_backend_vk_میانگیر_context(ctx->device, std::move(dev_میانگیر), ctx->name);

    أرجع ggml_backend_میانگیر_init(buft, ggml_backend_vk_میانگیر_interface, bufctx, size);
}

size_t ggml_backend_vk_میانگیر_type_get_alignment(ggml_backend_میانگیر_type_t buft) {
    ggml_backend_vk_میانگیر_type_context * ctx = (ggml_backend_vk_میانگیر_type_context *) buft->context;
    أرجع ctx->device->properties.limits.min_Storage_میانگیر_جابه_جایی_Alignment;
}

size_t ggml_backend_vk_میانگیر_type_get_max_اندازه(ggml_backend_میانگیر_type_t buft) {
    ggml_backend_vk_میانگیر_type_context * ctx = (ggml_backend_vk_میانگیر_type_context *) buft->context;
    أرجع ctx->device->suballocation_بلوک_اندازه;
}

size_t ggml_backend_vk_میانگیر_type_get_alloc_اندازه(ggml_backend_میانگیر_type_t buft, ثابت ggml_tensor * tensor) {
    أرجع ggml_nbytes(tensor);

    استفادهنشده(buft);
}

ggml_backend_میانگیر_type_t ggml_backend_vk_میانگیر_type(size_t dev_رقم) {
    ggml_vk_instance_init();

    VK_LOG_DEBUG("ggml_backend_vk_buffer_type(" << dev_رقم << ")");

    vk_device dev = ggml_vk_get_device(dev_رقم);

    أرجع &dev->میانگیر_type;
}

ساكن ثابت حرف * ggml_backend_vk_host_میانگیر_type_name(ggml_backend_میانگیر_type_t buft) {
    أرجع GGML_VK_NAME "_Host";

    استفادهنشده(buft);
}

ساكن فراغ ggml_backend_vk_host_میانگیر_free_میانگیر(ggml_backend_میانگیر_t میانگیر) {
    VK_LOG_حافظه("ggml_backend_vk_host_buffer_free_buffer()");
    ggml_vk_host_free(میانگیر->context);
}

ساكن ggml_backend_میانگیر_t ggml_backend_vk_host_میانگیر_type_alloc_میانگیر(ggml_backend_میانگیر_type_t buft, size_t size) {
    VK_LOG_حافظه("ggml_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");

    size += 32;  // Behave like the CPU buffer type
    فراغ * ptr = ggml_vk_host_malloc(size);
    إذا (ptr == اشاره_تهی) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory\n");
        // fallback to cpu buffer
        أرجع ggml_backend_buft_alloc_میانگیر(ggml_backend_cpu_میانگیر_type(), size);
    }

    ggml_backend_میانگیر_t میانگیر = ggml_backend_cpu_میانگیر_from_ptr(ptr, size);
    میانگیر->buft = buft;
    میانگیر->iface.free_میانگیر = ggml_backend_vk_host_میانگیر_free_میانگیر;

    أرجع میانگیر;

    استفادهنشده(buft);
}

ساكن size_t ggml_backend_vk_host_میانگیر_type_get_alignment(ggml_backend_میانگیر_type_t buft) {
    أرجع vk_instance.devices[0]->properties.limits.min_حافظه_Map_Alignment;

    استفادهنشده(buft);
}

ساكن size_t ggml_backend_vk_host_میانگیر_type_get_max_اندازه(ggml_backend_میانگیر_type_t buft) {
    أرجع vk_instance.devices[0]->suballocation_بلوک_اندازه;

    استفادهنشده(buft);
}

ggml_backend_میانگیر_type_t ggml_backend_vk_host_میانگیر_type() {
    ساكن بنية ggml_backend_میانگیر_type ggml_backend_vk_میانگیر_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_vk_host_میانگیر_type_name,
            /* .alloc_buffer        = */ ggml_backend_vk_host_میانگیر_type_alloc_میانگیر,
            /* .alloc_buffer_n      = */ اشاره_تهی,
            /* .get_alignment       = */ ggml_backend_vk_host_میانگیر_type_get_alignment,
            /* .get_max_size        = */ ggml_backend_vk_host_میانگیر_type_get_max_اندازه,
            /* .get_alloc_size      = */ ggml_backend_cpu_میانگیر_type()->iface.get_alloc_اندازه,
            /* .get_alloc_size_n    = */ عدم,
            /* .is_host             = */ ggml_backend_cpu_میانگیر_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0),
        /* .context  = */ اشاره_تهی,
    };

    // Make sure device 0 is initialized
    ggml_vk_instance_init();
    ggml_vk_get_device(0);

    أرجع &ggml_backend_vk_میانگیر_type_host;
}

ساكن ثابت حرف * ggml_backend_vk_name(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    أرجع ctx->name.c_str();
}

فراغ ggml_backend_vk_free(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    VK_LOG_DEBUG("ggml_backend_vk_free(" << ctx->name << ")");

    ggml_vk_cleanup(ctx);

    حذف ctx;
    حذف backend;
}

ساكن ggml_backend_میانگیر_type_t ggml_backend_vk_get_default_میانگیر_type(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    أرجع &ctx->device->میانگیر_type;
}

ساكن فراغ ggml_backend_vk_مجموعه_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_داده) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->میانگیر->buft == ggml_backend_vk_get_default_میانگیر_type(backend) || tensor->میانگیر->buft == ggml_backend_vk_host_میانگیر_type()) && "unsupported buffer type");

    إذا (size == 0) {
        أرجع;
    }

    ggml_backend_vk_میانگیر_context * buf_ctx = (ggml_backend_vk_میانگیر_context *)tensor->میانگیر->context;

    vk_context cpy_ctx;

    إذا (ctx->device->async_use_transfer_صف) {
        cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
    } وإلا {
        cpy_ctx = ggml_vk_get_compute_ctx(ctx);
    }

    vk_میانگیر buf = buf_ctx->dev_میانگیر;

    خودکار dst_جابه_جایی = vk_tensor_جابه_جایی(tensor) + tensor->view_offs + جابه_جایی;

    منطقي ret = ggml_vk_میانگیر_write_2d_async(cpy_ctx, buf, dst_جابه_جایی, داده, stride_داده, stride_tensor, size, n_copies);

    إذا (!ret) {
        ثابت size_t staging_اندازه = size * n_copies;
        ggml_vk_ensure_sync_staging_میانگیر(ctx, staging_اندازه);
        ggml_vk_sync_buffers(اشاره_تهی, cpy_ctx);

        std::vector<vk::میانگیر_Copy> slices(1);
        إذا (size == stride_tensor) {
            slices[0].src_جابه_جایی_2 = 0;
            slices[0].dst_جابه_جایی_2 = dst_جابه_جایی;
            slices[0].size = staging_اندازه;
        } وإلا {
            slices.resize(n_copies);
            لكل (size_t i = 0; i < n_copies; i++) {
                slices[i].src_جابه_جایی_2 = i * size;
                slices[i].dst_جابه_جایی_2 = dst_جابه_جایی + i * stride_tensor;
                slices[i].size = size;
            }
        }

        cpy_ctx->s->میانگیر->buf.copy_میانگیر(ctx->sync_staging->میانگیر, buf->میانگیر, slices);

        إذا (size == stride_داده) {
            deferred_memcpy(ctx->sync_staging->ptr, داده, staging_اندازه, &cpy_ctx->درون_memcpys);
        } وإلا {
            لكل (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)ctx->sync_staging->ptr + i * size, (ثابت uint8_t *)داده + i * stride_داده, size, &cpy_ctx->درون_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

ساكن فراغ ggml_backend_vk_مجموعه_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_async(" << size << ")");
    ggml_backend_vk_مجموعه_tensor_2d_async(backend, tensor, داده, جابه_جایی, size, 1, size, size);
}

ساكن فراغ ggml_backend_vk_get_tensor_2d_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_داده) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->میانگیر->buft == ggml_backend_vk_get_default_میانگیر_type(backend) || tensor->میانگیر->buft == ggml_backend_vk_host_میانگیر_type()) && "unsupported buffer type");

    إذا (size == 0) {
        أرجع;
    }

    ggml_backend_vk_میانگیر_context * buf_ctx = (ggml_backend_vk_میانگیر_context *)tensor->میانگیر->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    vk_میانگیر buf = buf_ctx->dev_میانگیر;

    خودکار src_جابه_جایی = vk_tensor_جابه_جایی(tensor) + tensor->view_offs + جابه_جایی;
    منطقي ret = ggml_vk_میانگیر_read_2d_async(compute_ctx, buf, src_جابه_جایی, داده, stride_tensor, stride_داده, size, n_copies);

    إذا (!ret) {
        ثابت size_t staging_اندازه = size * n_copies;
        ggml_vk_ensure_sync_staging_میانگیر(ctx, staging_اندازه);
        ggml_vk_sync_buffers(اشاره_تهی, compute_ctx);

        std::vector<vk::میانگیر_Copy> slices(1);
        إذا (size == stride_tensor) {
            slices[0].src_جابه_جایی_2 = src_جابه_جایی;
            slices[0].dst_جابه_جایی_2 = 0;
            slices[0].size = staging_اندازه;
        } وإلا {
            slices.resize(n_copies);
            لكل (size_t i = 0; i < n_copies; i++) {
                slices[i].src_جابه_جایی_2 = src_جابه_جایی + i * stride_tensor;
                slices[i].dst_جابه_جایی_2 = i * size;
                slices[i].size = size;
            }
        }

        compute_ctx->s->میانگیر->buf.copy_میانگیر(buf->میانگیر, ctx->sync_staging->میانگیر, slices);

        إذا (size == stride_داده) {
            deferred_memcpy(داده, ctx->sync_staging->ptr, staging_اندازه, &compute_ctx->بیرون_memcpys);
        } وإلا {
            لكل (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)داده + i * stride_داده, (ثابت uint8_t *)ctx->sync_staging->ptr + i * size, size, &compute_ctx->بیرون_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

ساكن فراغ ggml_backend_vk_get_tensor_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_async(" << size << ")");
    ggml_backend_vk_get_tensor_2d_async(backend, tensor, داده, جابه_جایی, size, 1, size, size);
}

ساكن منطقي ggml_backend_vk_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_backend_vk_cpy_tensor_async(" << src << " -> " << dst << ", size=" << ggml_nbytes(src) << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend_dst->context;

    // Skip zero-size tensors
    إذا (ggml_nbytes(src) == 0) {
        أرجع درست;
    }

    إذا (dst->میانگیر->buft != ggml_backend_vk_get_default_میانگیر_type(backend_dst)) {
        أرجع خطأ;
    }

    ggml_backend_vk_میانگیر_context * dst_buf_ctx = (ggml_backend_vk_میانگیر_context *)dst->میانگیر->context;
    vk_میانگیر dst_buf = dst_buf_ctx->dev_میانگیر;

    إذا (ggml_backend_میانگیر_is_vk(src->میانگیر)) {
        ggml_backend_vk_میانگیر_context * src_buf_ctx = (ggml_backend_vk_میانگیر_context *)src->میانگیر->context;

        // Async copy only works within the same device
        إذا (src_buf_ctx->dev_میانگیر->device != dst_buf->device) {
            أرجع خطأ;
        }

        vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_میانگیر_copy_async(compute_ctx, dst_buf, vk_tensor_جابه_جایی(dst) + dst->view_offs,
                                   src_buf_ctx->dev_میانگیر, vk_tensor_جابه_جایی(src) + src->view_offs,
                                   ggml_nbytes(src));
        أرجع درست;
    }

    إذا (ggml_backend_میانگیر_is_host(src->میانگیر)) {
        vk_میانگیر pinned_buf = اشاره_تهی;
        size_t pinned_جابه_جایی = 0;
        ggml_vk_host_get(ctx->device, src->داده, pinned_buf, pinned_جابه_جایی);
        إذا (pinned_buf == اشاره_تهی) {
            أرجع خطأ;
        }

        // If the backend is idle, use a CPU copy to avoid GPU synchronization overhead.
        ساكن constexpr size_t max_cpu_copy_اندازه = 128 * 1024;
        ثابت منطقي src_backend_synchronous = backend_src->iface.synchronize == اشاره_تهی;
        ثابت منطقي transfer_idle = !ctx->device->async_use_transfer_صف ||
                                   ctx->transfer_semaphore_آخرین_submitted == ctx->transfer_semaphore.value;
        ثابت منطقي backend_idle = ctx->compute_ctx.expired() && ctx->transfer_ctx.expired() &&
                                  !ctx->submit_pending && !ctx->almost_ready_fence_pending && transfer_idle;
        ثابت منطقي dst_host_coherent =
            (dst_buf->حافظه_property_flags & (vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent)) ==
            (vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent);

        إذا ((backend_src == backend_dst || src_backend_synchronous) && backend_idle && dst_host_coherent && ggml_nbytes(src) <= max_cpu_copy_اندازه) {
            ggml_vk_میانگیر_write(dst_buf, vk_tensor_جابه_جایی(dst) + dst->view_offs, src->داده, ggml_nbytes(src));
            أرجع درست;
        }

        vk_context cpy_ctx;
        إذا (ctx->device->async_use_transfer_صف) {
            cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
        } وإلا {
            cpy_ctx = ggml_vk_get_compute_ctx(ctx);
        }

        أرجع ggml_vk_میانگیر_write_async(cpy_ctx, dst_buf,
                                          vk_tensor_جابه_جایی(dst) + dst->view_offs,
                                          src->داده, ggml_nbytes(src));
    }

    أرجع خطأ;
}

فراغ ggml_vk_synchronize(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_synchronize()");

    منطقي do_transfer = !ctx->compute_ctx.expired();

    إذا (ggml_vk_submit_transfer_ctx(ctx)) {
        ctx->submit_pending = درست;
    }

    vk_context compute_ctx;
    vk_command_میانگیر* cmd_buf = اشاره_تهی;
    إذا (do_transfer) {
        compute_ctx = ctx->compute_ctx.قفل();
        إذا (compute_ctx->s) {
            cmd_buf = compute_ctx->s->میانگیر;
        }

        ggml_vk_ctx_انتها(compute_ctx);

        لكل (خودکار& cpy : compute_ctx->درون_memcpys) {
            کپی_حافظه(cpy.dst, cpy.src, cpy.n);
        }

        إذا (ctx->device->serialize_submissions) {
            ggml_vk_submit(compute_ctx, ctx->fence);
            VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, درست, UINT64_MAX), "synchronize waitForFences", ctx->device);
            ctx->device->device.resetFences({ ctx->fence });
        } وإلا {
            ggml_vk_submit(compute_ctx, {});
        }
        ctx->submit_pending = درست;
    }

    إذا (ctx->submit_pending) {
        إذا (ctx->device->serialize_submissions) {
            ctx->submit_pending = خطأ;
        } وإلا إذا (ctx->device->async_use_transfer_صف && ctx->transfer_semaphore_آخرین_submitted < ctx->transfer_semaphore.value) {
            vk::TimelineSemaphoreSubmitInfo tl_info{
                1, &ctx->transfer_semaphore.value,
                0, اشاره_تهی,
            };
            vk::PipelineStageFlags stage = ctx->device->transfer_صف->stage_flags;
            vk::SubmitInfo si{
                1, &ctx->transfer_semaphore.s, &stage,
                0, اشاره_تهی,
                0, اشاره_تهی,
            };
            si.مجموعه_PNext(&tl_info);
            ctx->device->compute_صف->handle->submit({ si }, ctx->fence);
            ctx->transfer_semaphore_آخرین_submitted = ctx->transfer_semaphore.value;
        } وإلا {
            ctx->device->compute_صف->handle->submit({}, ctx->fence);
        }
        إذا (!ctx->device->serialize_submissions) {
            ggml_vk_wait_for_fence(ctx);
        }
        ctx->submit_pending = خطأ;
        إذا (cmd_buf) {
            cmd_buf->درون_use = خطأ;
            cmd_buf->buf.reset();
        }
    }

    إذا (do_transfer) {
        لكل (خودکار& cpy : compute_ctx->بیرون_memcpys) {
            کپی_حافظه(cpy.dst, cpy.src, cpy.n);
        }
        ctx->compute_ctx.reset();
    }
}

ساكن فراغ ggml_backend_vk_synchronize(ggml_backend_t backend) {
    VK_LOG_DEBUG("ggml_backend_vk_synchronize()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ggml_vk_synchronize(ctx);

    ggml_vk_گراف_cleanup(ctx);
}

منطقي ggml_vk_is_خالی(ggml_tensor * گره) {
    أرجع ggml_is_خالی(گره) || گره->op == GGML_OP_NONE || گره->op == GGML_OP_RESHAPE || گره->op == GGML_OP_TRANSPOSE || گره->op == GGML_OP_VIEW || گره->op == GGML_OP_جایگشت;
}

ساكن منطقي ggml_vk_can_fuse_unary_mul(ثابت بنية ggml_cgraph * cgraph, صحيح unary_idx, صحيح mul_idx) {
    ثابت ggml_tensor * unary = cgraph->nodes[unary_idx];
    ثابت ggml_tensor * mul = cgraph->nodes[mul_idx];

    إذا (ggml_vk_unary_mul_op_شاخص(ggml_get_unary_op(unary)) < 0) {
        أرجع خطأ;
    }
    إذا (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
        أرجع خطأ;
    }
    إذا (unary->type != mul->type) {
        أرجع خطأ;
    }
    إذا (mul->src[0] != unary && mul->src[1] != unary) {
        أرجع خطأ;
    }
    ثابت ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
    إذا (other == اشاره_تهی || other->type != unary->type) {
        أرجع خطأ;
    }
    إذا (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0])) {
        أرجع خطأ;
    }
    // fastmod needs src to tile into dst
    إذا (mul->src[0] == unary) {
        أرجع ggml_can_repeat(other, unary);
    }
    أرجع ggml_can_repeat(unary, mul->src[0]);
}

ساكن منطقي ggml_vk_can_fuse_unary_mul_pair(ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {
    ثابت تعداد ggml_op ops[]    = { GGML_OP_UNARY, GGML_OP_MUL };
    ثابت صحيح           outputs[] = { گره_idx + 1 };
    أرجع ggml_can_fuse_subgraph(cgraph, گره_idx, 2, ops, outputs, 1) &&
           ggml_vk_can_fuse_unary_mul(cgraph, گره_idx, گره_idx + 1);
}

ساكن منطقي ggml_vk_can_fuse_hc_post_gate(ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {
    ثابت ggml_tensor * scale_درون  = cgraph->nodes[گره_idx];
    ثابت ggml_tensor * sigmoid   = cgraph->nodes[گره_idx + 1];
    ثابت ggml_tensor * scale_بیرون = cgraph->nodes[گره_idx + 2];

    // the shader folds scale -> sigmoid -> scale; a bias on either scale is not handled
    أرجع ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
           ggml_get_op_params_f32(scale_درون, 1) == 0.0f &&
           ggml_get_op_params_f32(scale_بیرون, 1) == 0.0f &&
           scale_درون->src[0]->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(scale_درون->src[0], scale_بیرون);
}

منطقي ggml_vk_can_fuse(ثابت ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx, std::initializer_list<تعداد ggml_op> ops) {
    إذا (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL) {
        أرجع ggml_vk_can_fuse_unary_mul_pair(cgraph, گره_idx);
    }

    إذا (!ggml_can_fuse(cgraph, گره_idx, ops)) {
        أرجع خطأ;
    }

    إذا ((ops.size() == 2 || ops.size() == 3 || ops.size() == 4) &&
        ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        ثابت ggml_tensor *rms_norm = cgraph->nodes[گره_idx];
        ثابت ggml_tensor *mul = cgraph->nodes[گره_idx + 1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        // rms_norm only supports f32
        إذا (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
        // if rms_norm is the B operand, then we don't handle broadcast
        إذا (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            أرجع خطأ;
        }
        // rms_norm shader assumes contiguous rows
        إذا (!ggml_is_contiguous_ردیفها(mul->src[0]) || !ggml_is_contiguous_ردیفها(mul->src[1])) {
            أرجع خطأ;
        }

        إذا (ops.size() >= 3 && ops.begin()[2] == GGML_OP_جمع_کن) {
            ثابت ggml_tensor *جمع_کن_2 = cgraph->nodes[گره_idx + 2];
            ثابت ggml_tensor *residual = جمع_کن_2->src[0] == mul ? جمع_کن_2->src[1] : جمع_کن_2->src[0];
            إذا (جمع_کن_2->src[0] != mul && جمع_کن_2->src[1] != mul) {
                أرجع خطأ;
            }
            إذا (residual->type != GGML_TYPE_F32 || جمع_کن_2->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(جمع_کن_2, residual) || !ggml_is_contiguous(residual) ||
                !ggml_is_contiguous(جمع_کن_2) || get_misalign_بایتها(ctx, residual) != 0) {
                أرجع خطأ;
            }

            ثابت ggml_tensor *dst = جمع_کن_2;
            إذا (ops.size() == 4) {
                إذا (ops.begin()[3] != GGML_OP_MUL) {
                    أرجع خطأ;
                }

                ثابت ggml_tensor *post_mul = cgraph->nodes[گره_idx + 3];
                ثابت ggml_tensor *scale = post_mul->src[0] == جمع_کن_2 ? post_mul->src[1] : post_mul->src[0];
                إذا (post_mul->src[0] != جمع_کن_2 && post_mul->src[1] != جمع_کن_2) {
                    أرجع خطأ;
                }
                // The shader reads data_e[0], so the final multiply must use a scalar.
                إذا (scale->type != GGML_TYPE_F32 || post_mul->type != GGML_TYPE_F32 ||
                    ggml_nelements(scale) != 1 || !ggml_is_contiguous(post_mul) ||
                    get_misalign_بایتها(ctx, scale) != 0) {
                    أرجع خطأ;
                }
                dst = post_mul;
            }

            إذا (get_misalign_بایتها(ctx, dst) != 0) {
                أرجع خطأ;
            }
        }
    }

    خودکار ثابت &mm_جمع_کن_ok = [&](ثابت ggml_tensor *mul, ثابت ggml_tensor *جمع_کن_2) {
        ثابت ggml_tensor *bias = جمع_کن_2->src[0] == mul ? جمع_کن_2->src[1] : جمع_کن_2->src[0];

        // mat-vec only
        إذا (ggml_nrows(mul) != 1) {
            أرجع خطأ;
        }
        // shaders assume the types match
        إذا (mul->type != bias->type) {
            أرجع خطأ;
        }
        // shaders reuse the D shape for bias
        إذا (!ggml_are_same_shape(mul, bias) ||
            !ggml_are_same_stride(mul, bias)) {
            أرجع خطأ;
        }
        // unaligned bias isn't handled
        إذا (get_misalign_بایتها(ctx, bias) != 0) {
            أرجع خطأ;
        }
        أرجع درست;
    };

    إذا ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_جمع_کن) {
        // additional constraints specific to this fusion
        ثابت ggml_tensor *mul = cgraph->nodes[گره_idx];
        ثابت ggml_tensor *جمع_کن_2 = cgraph->nodes[گره_idx + 1];

        إذا (!mm_جمع_کن_ok(mul, جمع_کن_2)) {
            أرجع خطأ;
        }
        إذا (ops.size() == 3) {
            إذا (ops.begin()[2] != GGML_OP_جمع_کن) {
                أرجع خطأ;
            }
            إذا (!mm_جمع_کن_ok(جمع_کن_2, cgraph->nodes[گره_idx + 2])) {
                أرجع خطأ;
            }
        }
    }

    خودکار ثابت &mmid_mul_ok = [&](ثابت ggml_tensor *mmid, ثابت ggml_tensor *mul) {
        ثابت ggml_tensor *scale = mul->src[1];

        إذا (mmid != mul->src[0]) {
            أرجع خطأ;
        }
        // mat-vec only
        إذا (!ggml_vk_use_mul_mat_vec_id(cgraph, گره_idx)) {
            أرجع خطأ;
        }
        // shaders assume the types match
        إذا (mmid->type != scale->type) {
            أرجع خطأ;
        }
        // shaders assume the bias is contiguous
        إذا (!ggml_is_contiguous(scale)) {
            أرجع خطأ;
        }
        // unaligned bias isn't handled
        إذا (get_misalign_بایتها(ctx, scale) != 0) {
            أرجع خطأ;
        }
        // shader only indexes by expert index
        إذا (scale->ne[0] != 1 ||
            scale->ne[1] != mul->ne[1] ||
            scale->ne[2] != 1 ||
            scale->ne[3] != 1) {
            أرجع خطأ;
        }
        أرجع درست;
    };

    إذا ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_جمع_کن_ID) {
        // additional constraints specific to this fusion
        ثابت ggml_tensor *mul = cgraph->nodes[گره_idx];
        ثابت ggml_tensor *جمع_کن_2 = cgraph->nodes[گره_idx + 1];
        ثابت ggml_tensor *bias = جمع_کن_2->src[1];

        إذا (mul != جمع_کن_2->src[0]) {
            أرجع خطأ;
        }
        // mat-vec only
        إذا (!ggml_vk_use_mul_mat_vec_id(cgraph, گره_idx)) {
            أرجع خطأ;
        }
        // shaders assume the types match
        إذا (mul->type != bias->type) {
            أرجع خطأ;
        }
        // shaders assume the bias is contiguous
        إذا (!ggml_is_contiguous(bias)) {
            أرجع خطأ;
        }
        // the ID tensor must be the same for mul_mat_id and add_id
        إذا (mul->src[2] != جمع_کن_2->src[2]) {
            أرجع خطأ;
        }
        // unaligned bias isn't handled
        إذا (get_misalign_بایتها(ctx, bias) != 0) {
            أرجع خطأ;
        }

        إذا (ops.size() == 3) {
            إذا (ops.begin()[2] != GGML_OP_MUL) {
                أرجع خطأ;
            }
            ثابت ggml_tensor *mul = cgraph->nodes[گره_idx + 2];
            أرجع mmid_mul_ok(جمع_کن_2, mul);
        }
    }

    إذا (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        ثابت ggml_tensor *mmid = cgraph->nodes[گره_idx];
        ثابت ggml_tensor *mul = cgraph->nodes[گره_idx + 1];

        إذا (!mmid_mul_ok(mmid, mul)) {
            أرجع خطأ;
        }
    }

    أرجع درست;
}

منطقي ggml_vk_can_fuse_ssm_conv(ثابت ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph,
                                      صحيح گره_idx, صحيح رقم_extra) {
    ثابت ggml_tensor * conv = cgraph->nodes[گره_idx];
    إذا (conv->op != GGML_OP_SSM_CONV) {
        أرجع خطأ;
    }

    ثابت ggml_tensor * silu = اشاره_تهی;
    ثابت ggml_tensor * bias = اشاره_تهی;

    إذا (رقم_extra == 1) {
        إذا (!ggml_can_fuse(cgraph, گره_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY })) {
            أرجع خطأ;
        }
        silu = cgraph->nodes[گره_idx + 1];
    } وإلا إذا (رقم_extra == 2) {
        إذا (!ggml_can_fuse(cgraph, گره_idx, { GGML_OP_SSM_CONV, GGML_OP_جمع_کن, GGML_OP_UNARY })) {
            أرجع خطأ;
        }
        ثابت ggml_tensor * جمع_کن_2 = cgraph->nodes[گره_idx + 1];
        silu = cgraph->nodes[گره_idx + 2];
        bias = (جمع_کن_2->src[0] == conv) ? جمع_کن_2->src[1] : جمع_کن_2->src[0];

        إذا (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            أرجع خطأ;
        }
        // bias must be channel-wise (one element per channel of the conv output)
        إذا (ggml_nelements(bias) != conv->ne[0] || bias->ne[0] != conv->ne[0]) {
            أرجع خطأ;
        }
        إذا (جمع_کن_2->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
        // The shader doesn't apply per-tensor offsets, so reject misaligned bias.
        إذا (get_misalign_بایتها(ctx, bias) != 0) {
            أرجع خطأ;
        }
    } وإلا {
        أرجع خطأ;
    }

    إذا (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        أرجع خطأ;
    }
    إذا (conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }
    // The shader writes to the fused dst using its own strides, but the push constants don't
    // carry a per-tensor offset, so the binding must be naturally aligned.
    إذا (get_misalign_بایتها(ctx, silu) != 0) {
        أرجع خطأ;
    }
    أرجع درست;
}

منطقي ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph,
                                      صحيح گره_idx, topk_moe_نما mode) {

    ثابت ggml_tensor * softmax;
    ثابت ggml_tensor * weights;
    ثابت ggml_tensor * get_ردیفها;
    ثابت ggml_tensor * argsort;

    بدل (mode) {
    حالة TOPK_MOE_EARLY_SOFTMAX_NORM:
        softmax = cgraph->nodes[گره_idx + 0];
        weights = cgraph->nodes[گره_idx + 9];
        get_ردیفها = cgraph->nodes[گره_idx + 4];
        argsort = cgraph->nodes[گره_idx + 2];
        اكسر;
    حالة TOPK_MOE_SIGMOID_NORM_BIAS:
        softmax = cgraph->nodes[گره_idx + 0]; // really sigmoid
        weights = cgraph->nodes[گره_idx + 10];
        get_ردیفها = cgraph->nodes[گره_idx + 5];
        argsort = cgraph->nodes[گره_idx + 3];
        إذا (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SIGMOID) {
            أرجع خطأ;
        }
        // bias is expected to be 1D
        إذا (ggml_nrows(cgraph->nodes[گره_idx + 2]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[گره_idx + 2]->src[1])) {
            أرجع خطأ;
        }
        // sigmoid fusion seems to generate infinities on moltenvk
        إذا (ctx->device->driver_id == vk::DriverId::eMoltenvk) {
            أرجع خطأ;
        }
        اكسر;
    حالة TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS:
        softmax = cgraph->nodes[گره_idx + 0]; // really softplus
        weights = cgraph->nodes[گره_idx + 11];
        get_ردیفها = cgraph->nodes[گره_idx + 6];
        argsort = cgraph->nodes[گره_idx + 4];
        إذا (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SOFTPLUS) {
            أرجع خطأ;
        }
        // bias is expected to be 1D
        إذا (ggml_nrows(cgraph->nodes[گره_idx + 3]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[گره_idx + 3]->src[1])) {
            أرجع خطأ;
        }
        اكسر;
    حالة TOPK_MOE_EARLY_SOFTMAX:
        softmax = cgraph->nodes[گره_idx + 0];
        weights = cgraph->nodes[گره_idx + 4];
        get_ردیفها = cgraph->nodes[گره_idx + 4];
        argsort = cgraph->nodes[گره_idx + 2];
        اكسر;
    حالة TOPK_MOE_LATE_SOFTMAX:
        softmax = cgraph->nodes[گره_idx + 4];
        weights = cgraph->nodes[گره_idx + 5];
        get_ردیفها = cgraph->nodes[گره_idx + 2];
        argsort = cgraph->nodes[گره_idx + 0];
        اكسر;
    افتراضي:
        أرجع خطأ;
    }

    ggml_tensor * probs = get_ردیفها->src[0];
    إذا (probs->op != GGML_OP_RESHAPE) {
        أرجع خطأ;
    }
    probs = probs->src[0];
    ggml_tensor * selection_probs = argsort->src[0];

    إذا (probs != selection_probs &&
        mode != TOPK_MOE_SIGMOID_NORM_BIAS &&
        mode != TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        أرجع خطأ;
    }

    إذا (softmax->op == GGML_OP_SOFT_MAX) {
        ثابت عائم * op_params = (ثابت عائم *)softmax->op_params;

        عائم scale = op_params[0];
        عائم max_bias = op_params[1];

        إذا (scale != 1.0f || max_bias != 0.0f) {
            أرجع خطأ;
        }

        // don't fuse when masks or sinks are present
        إذا (softmax->src[1] || softmax->src[2]) {
            أرجع خطأ;
        }
    }

    ثابت صحيح n_expert = softmax->ne[0];
    إذا (n_expert > (1 << (رقم_topk_moe_pipelines-1))) {
        أرجع خطأ;
    }

    إذا (!ctx->device->subgroup_arithmetic ||
        !ctx->device->subgroup_shuffle ||
        !ctx->device->subgroup_require_پر_support ||
        ctx->device->disable_fusion) {
        أرجع خطأ;
    }

    أرجع درست;
}

ساكن منطقي ggml_vk_تطابق_ops(ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx,
                              ثابت std::initializer_list<ggml_op> & ops) {
    إذا (گره_idx + (صحيح) ops.size() > cgraph->n_nodes) {
        أرجع خطأ;
    }
    لكل (size_t j = 0; j < ops.size(); ++j) {
        ثابت ggml_tensor * گره = cgraph->nodes[گره_idx + j];
        إذا (گره->op != ops.begin()[j] ||
            (گره->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            (گره->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            أرجع خطأ;
        }
    }
    أرجع درست;
}

منطقي ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {
    إذا (ctx->device->disable_fusion || !ctx->device->pipeline_topk_radix_qsa) {
        أرجع خطأ;
    }

    ثابت صحيح n_ops = topk_qsa_الگو.size();
    إذا (!ggml_vk_تطابق_ops(cgraph, گره_idx, topk_qsa_الگو) ||
        !ggml_check_یالها(cgraph, گره_idx, topk_qsa_یالها)) {
        أرجع خطأ;
    }

    // elided nodes must be single-use (cpy counts its own src[1] self-reference)
    لكل (صحيح j = 0; j < n_ops - 1; ++j) {
        ثابت ggml_tensor * گره = cgraph->nodes[گره_idx + j];
        ثابت int32_t want = گره->op == GGML_OP_CPY ? 2 : 1;
        إذا (ggml_گره_get_use_شمار(cgraph, گره_idx + j) != want) {
            أرجع خطأ;
        }
    }

    ثابت ggml_tensor * get_ردیفها = cgraph->nodes[گره_idx + 0];
    ثابت ggml_tensor * جمع_کن_2      = cgraph->nodes[گره_idx + n_ops - 2];
    ثابت ggml_tensor * بالاترین_k    = cgraph->nodes[گره_idx + n_ops - 1];

    ثابت ggml_tensor * scores   = get_ردیفها->src[0]; // [n_tps, n_blocks, n_stream]
    ثابت ggml_tensor * cell_blk = get_ردیفها->src[1]; // [n_kv, n_stream]
    ثابت ggml_tensor * expanded = جمع_کن_2->src[0];      // [n_kv, n_tps, n_stream]

    // raw mask: follow the reshape/cpy chain back to the materialized f16 input
    ثابت ggml_tensor * ماسک = جمع_کن_2->src[1];
    طالما (ماسک && (ماسک->op == GGML_OP_RESHAPE || ماسک->op == GGML_OP_CPY)) {
        ماسک = ماسک->src[0];
    }
    إذا (!ماسک || ماسک->type != GGML_TYPE_F16) {
        أرجع خطأ;
    }

    إذا (scores->type != GGML_TYPE_F32 || cell_blk->type != GGML_TYPE_I32 || بالاترین_k->type != GGML_TYPE_I32) {
        أرجع خطأ;
    }
    إذا (!ggml_is_contiguous(scores) || !ggml_is_contiguous(cell_blk) || !ggml_is_contiguous(ماسک) ||
        !ggml_is_contiguous(expanded) || !ggml_is_contiguous(بالاترین_k)) {
        أرجع خطأ;
    }

    ثابت int64_t n_tps    = scores->ne[0];
    ثابت int64_t n_بلوکها = scores->ne[1];
    ثابت int64_t n_جریان = scores->ne[2];
    ثابت int64_t n_kv     = cell_blk->ne[0];
    ثابت int64_t عرض    = بالاترین_k->ne[0];

    // pin the indexer layout the shader's addressing assumes
    إذا (scores->ne[3] != 1 || cell_blk->ne[1] != n_جریان || ggml_nrows(cell_blk) != n_جریان ||
        ggml_nelements(ماسک) != n_kv * n_tps * n_جریان ||
        expanded->ne[0] != n_kv || expanded->ne[1] != n_tps || expanded->ne[2] != n_جریان ||
        بالاترین_k->ne[1] != n_tps || بالاترین_k->ne[2] != n_جریان || بالاترین_k->ne[3] != 1 ||
        n_بلوکها <= 0 || n_kv <= 0 || عرض <= 0 || عرض > n_kv) {
        أرجع خطأ;
    }

    // only worth it in the radix regime; small k uses the faster tournament unfused
    ثابت uint32_t k_min_pipeline = std::max((uint32_t) log2f(عائم(عرض)) + 1, ctx->device->subgroup_اندازه_log2);
    إذا (k_min_pipeline < رقم_topk_pipelines && ctx->device->pipeline_topk_f32[k_min_pipeline]) {
        أرجع خطأ;
    }
    أرجع درست;
}

منطقي ggml_vk_can_fuse_rope_مجموعه_ردیفها(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph,
                                           صحيح گره_idx) {
    ثابت ggml_tensor *rope = cgraph->nodes[گره_idx + 0];
    ثابت ggml_tensor *view = cgraph->nodes[گره_idx + 1];
    ثابت ggml_tensor *مجموعه_ردیفها = cgraph->nodes[گره_idx + 2];

    // The set_rows epilogue uses one index per ne2 slice and does not encode ne3.
    إذا (rope->src[0]->ne[3] != 1) {
        أرجع خطأ;
    }

    إذا (مجموعه_ردیفها->type != GGML_TYPE_F32 && مجموعه_ردیفها->type != GGML_TYPE_F16) {
        أرجع خطأ;
    }

    // The shader reads each aligned I64 index as a uvec2 and uses its low 32 bits.
    إذا (مجموعه_ردیفها->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(مجموعه_ردیفها->src[1]) ||
        مجموعه_ردیفها->nb[0] != ggml_type_اندازه(مجموعه_ردیفها->type) || get_misalign_بایتها(ctx, مجموعه_ردیفها->src[1]) != 0) {
        أرجع خطأ;
    }

    // SET_ROWS consumes one flattened [ne0*ne1] row for each ne2 slice.
    إذا (!ggml_is_contiguous(view) ||
        view->ne[0] != rope->ne[0] * rope->ne[1] || view->ne[1] != rope->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(مجموعه_ردیفها->src[1]) != rope->ne[2]) {
        أرجع خطأ;
    }

    // Only norm/neox/mrope/imrope shaders have the fusion code
    ثابت صحيح mode = ((ثابت int32_t *) rope->op_params)[2];
    إذا (mode != GGML_ROPE_TYPE_نرمال && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        أرجع خطأ;
    }

    أرجع درست;
}

منطقي ggml_vk_can_fuse_rms_norm_مجموعه_ردیفها(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph,
                                               صحيح گره_idx) {
    ثابت ggml_tensor * rms = cgraph->nodes[گره_idx];
    ثابت ggml_tensor * view = cgraph->nodes[گره_idx + 1];
    ثابت ggml_tensor * مجموعه_ردیفها = cgraph->nodes[گره_idx + 2];

    // The RMS kernel reads F32 and writes directly to the F32 or F16 SET_ROWS destination.
    إذا (rms->src[0]->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 ||
        (مجموعه_ردیفها->type != GGML_TYPE_F32 && مجموعه_ردیفها->type != GGML_TYPE_F16) ||
        مجموعه_ردیفها->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(مجموعه_ردیفها->src[1]) ||
        مجموعه_ردیفها->nb[0] != ggml_type_اندازه(مجموعه_ردیفها->type) || get_misalign_بایتها(ctx, مجموعه_ردیفها->src[1]) != 0) {
        أرجع خطأ;
    }
    // As with the ROPE epilogue, each ne2 slice supplies one flattened row and ne3 is not encoded.
    إذا (rms->ne[3] != 1 || !ggml_is_contiguous(rms->src[0]) || !ggml_is_contiguous(view)) {
        أرجع خطأ;
    }
    إذا (view->ne[0] != rms->ne[0] * rms->ne[1] || view->ne[1] != rms->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(مجموعه_ردیفها->src[1]) != rms->ne[2]) {
        أرجع خطأ;
    }

    أرجع درست;
}

منطقي ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {
    GGML_استفادهنشده(ctx);
    إذا (!ggml_can_fuse(cgraph, گره_idx, snake_الگو)) {
        أرجع خطأ;
    }

    ثابت ggml_tensor * mul0     = cgraph->nodes[گره_idx + 0];
    ثابت ggml_tensor * sin_گره = cgraph->nodes[گره_idx + 1];
    ثابت ggml_tensor * sqr      = cgraph->nodes[گره_idx + 2];
    ثابت ggml_tensor * mul1     = cgraph->nodes[گره_idx + 3];
    ثابت ggml_tensor * جمع_کن_2      = cgraph->nodes[گره_idx + 4];

    ثابت ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ثابت ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    ثابت ggml_tensor * inv_b    = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];
    ثابت ggml_tensor * x_درون_جمع_کن = (جمع_کن_2->src[0] == mul1) ? جمع_کن_2->src[1] : جمع_کن_2->src[0];

    إذا (x_درون_جمع_کن != x) {
        أرجع خطأ;
    }
    إذا (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_BF16) {
        أرجع خطأ;
    }
    // Shader bindings: data_a is A_TYPE so it follows x's precision, while
    // data_b and data_c are hardcoded float, so the broadcast operands must
    // be F32 regardless of x's type.
    إذا (a->type     != GGML_TYPE_F32) أرجع خطأ;
    إذا (inv_b->type != GGML_TYPE_F32) أرجع خطأ;
    // Chain intermediates and output share x's precision (single A_TYPE / D_TYPE pipeline).
    إذا (mul0->type     != x->type) أرجع خطأ;
    إذا (sin_گره->type != x->type) أرجع خطأ;
    إذا (sqr->type      != x->type) أرجع خطأ;
    إذا (mul1->type     != x->type) أرجع خطأ;
    إذا (جمع_کن_2->type      != x->type) أرجع خطأ;
    إذا (!ggml_are_same_shape(a, inv_b)) {
        أرجع خطأ;
    }
    إذا (a->ne[0] != 1 || a->ne[1] != x->ne[1]) {
        أرجع خطأ;
    }
    // Dispatch is 2D over (ne0, ne1), so x and add must be 2D and a / inv_b
    // must collapse to [1, C, 1, 1]. Higher dims are not handled by the shader.
    إذا (x->ne[2]     != 1 || x->ne[3]     != 1) أرجع خطأ;
    إذا (جمع_کن_2->ne[2]   != 1 || جمع_کن_2->ne[3]   != 1) أرجع خطأ;
    إذا (a->ne[2]     != 1 || a->ne[3]     != 1) أرجع خطأ;
    إذا (inv_b->ne[2] != 1 || inv_b->ne[3] != 1) أرجع خطأ;
    // Shader uses idx = i0 + i1 * ne0 and reads data_b[i1] / data_c[i1],
    // so every operand must be contiguous.
    إذا (!ggml_is_contiguous(x) || !ggml_is_contiguous(جمع_کن_2) ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(inv_b)) {
        أرجع خطأ;
    }
    أرجع درست;
}

منطقي ggml_vk_tensors_overlap(ثابت ggml_tensor * a, ثابت ggml_tensor * b, منطقي elementwise) {
    ggml_backend_vk_میانگیر_context * a_buf_ctx = (ggml_backend_vk_میانگیر_context *)a->میانگیر->context;
    vk_میانگیر a_buf = a_buf_ctx->dev_میانگیر;
    ggml_backend_vk_میانگیر_context * b_buf_ctx = (ggml_backend_vk_میانگیر_context *)b->میانگیر->context;
    vk_میانگیر b_buf = b_buf_ctx->dev_میانگیر;
    إذا (a_buf == b_buf) {
        خودکار a_پایه = vk_tensor_جابه_جایی(a) + a->view_offs;
        خودکار a_اندازه = ggml_nbytes(a);
        خودکار b_پایه = vk_tensor_جابه_جایی(b) + b->view_offs;
        خودکار b_اندازه = ggml_nbytes(b);

        إذا (elementwise && a_پایه == b_پایه && a_اندازه == b_اندازه) {
            أرجع خطأ;
        }

        إذا ((b_پایه <= a_پایه && a_پایه < b_پایه + b_اندازه) ||
            (a_پایه <= b_پایه && b_پایه < a_پایه + a_اندازه)) {
            أرجع درست;
        }
    }
    أرجع خطأ;
}

منطقي ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph,
                                               صحيح گره_idx) {
    ثابت ggml_tensor *rms = cgraph->nodes[گره_idx + 0];
    ثابت ggml_tensor *mul = cgraph->nodes[گره_idx + 1];
    ثابت ggml_tensor *rope = cgraph->nodes[گره_idx + 2];

    ثابت صحيح mode = ((ثابت int32_t *) rope->op_params)[2];

    // noncontig tensors aren't tested, and don't seem common in practice
    إذا (!ggml_is_contiguous(rms) ||
        !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous(rope)) {
        أرجع خطأ;
    }

    // only norm/neox are handled in the shader
    إذا (mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_نرمال) {
        أرجع خطأ;
    }

    // shared memory size for passing data from mul->rope
    إذا (mul->ne[0] > 1024) {
        أرجع خطأ;
    }

    // conditions for pipeline creation
    إذا (حجم(vk_op_rms_norm_mul_rope_افزودن_constants) > ctx->device->properties.limits.max_افزودن_Constants_اندازه) {
        أرجع خطأ;
    }

    أرجع درست;
}

uint32_t ggml_vk_fuse_multi_جمع_کن(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {

    ثابت ggml_tensor *اولی_گره = cgraph->nodes[گره_idx];
    إذا (اولی_گره->op != GGML_OP_جمع_کن) {
        أرجع 0;
    }

    إذا (!ctx->device->multi_جمع_کن) {
        أرجع 0;
    }

    int32_t رقم_adds = 1;
    طالما (گره_idx + رقم_adds < cgraph->n_nodes &&
           cgraph->nodes[گره_idx + رقم_adds]->op == GGML_OP_جمع_کن &&
           رقم_adds < MAX_FUSED_ADDS) {
        رقم_adds++;
    }

    // The shader currently requires same shapes (but different strides are allowed),
    // everything f32, and no misalignment
    لكل (int32_t i = 0; i < رقم_adds; ++i) {
        ثابت ggml_tensor *بعدی_گره = cgraph->nodes[گره_idx + i];
        إذا (!ggml_are_same_shape(اولی_گره, بعدی_گره->src[0]) ||
            !ggml_are_same_shape(اولی_گره, بعدی_گره->src[1]) ||
            بعدی_گره->type != GGML_TYPE_F32 ||
            بعدی_گره->src[0]->type != GGML_TYPE_F32 ||
            بعدی_گره->src[1]->type != GGML_TYPE_F32 ||
            get_misalign_بایتها(ctx, بعدی_گره) ||
            get_misalign_بایتها(ctx, بعدی_گره->src[0]) ||
            get_misalign_بایتها(ctx, بعدی_گره->src[1])) {
            رقم_adds = i;
        }
    }

    // Verify we can fuse these
    ggml_op adds[MAX_FUSED_ADDS];
    لكل (int32_t i = 0; i < رقم_adds; ++i) {
        adds[i] = GGML_OP_جمع_کن;
    }

    // decrease num_adds if they can't all be fused
    طالما (رقم_adds > 1 && !ggml_can_fuse(cgraph, گره_idx, adds, رقم_adds)) {
        رقم_adds--;
    }

    // a single add is not "fused", so just return zero
    إذا (رقم_adds == 1) {
        أرجع 0;
    }
    أرجع رقم_adds;
}

ساكن int32_t پیدا_کن_اولی_مجموعه(uint32_t x) {
    int32_t ret = 0;
    إذا (!x) {
        أرجع -1;
    }
    طالما (!(x & 1)) {
        x >>= 1;
        ret++;
    }
    أرجع ret;
}

ساكن ggml_وضعیت ggml_backend_vk_گراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    VK_LOG_DEBUG("ggml_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ctx->device->diag_cgraph = اشاره_تهی;
    ctx->device->diag_prev_شروع = -1;
    ctx->device->diag_prev_انتها = -1;

    إذا (vk_instance.debug_utils_support) {
        ctx->device->debug_cmdbuf_idx = 0;
    }

    // queue scope, so it encloses every submit this evaluation makes.
    // closed when the function returns
    ggml_vk_debug_label صف_dbg(ctx->device->compute_صف->handle.get(), "ggml_backend_vk_graph_compute");

    ctx->prealloc_اندازه_جمع_کن_rms_partials_جابه_جایی = 0;
    ctx->do_جمع_کن_rms_partials = خطأ;
    ctx->do_جمع_کن_rms_partials_جابه_جایی_calculation = خطأ;

    صحيح آخرین_گره = cgraph->n_nodes - 1;

    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    طالما (آخرین_گره > 0 && (ggml_vk_is_خالی(cgraph->nodes[آخرین_گره]) || ((cgraph->nodes[آخرین_گره]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0))) {
        آخرین_گره -= 1;
    }

    // Reserve tensor context space for all nodes
    ctx->tensor_ctxs.resize(cgraph->n_nodes);

    منطقي اولی_گره_درون_batch = درست; // true if next node will be first node in a batch
    صحيح submit_گره_idx = 0; // index to first node in a batch

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx;
    إذا (vk_perf_logger_enabled) {
        // allocate/resize the query pool
        إذا (ctx->رقم_queries < cgraph->n_nodes + 1) {
            إذا (ctx->query_pool) {
                ctx->device->device.destroyQueryPool(ctx->query_pool);
            }
            vk::QueryPoolCreateInfo query_create_info;
            query_create_info.queryType = vk::QueryType::eTimestamp;
            query_create_info.query_شمار = cgraph->n_nodes + 100;
            ctx->query_pool = ctx->device->device.createQueryPool(query_create_info);
            ctx->رقم_queries = query_create_info.query_شمار;
            ctx->query_fusion_نامها.resize(ctx->رقم_queries);
            ctx->query_fusion_گره_شمار.resize(ctx->رقم_queries);
            ctx->query_nodes.resize(ctx->رقم_queries);
            ctx->query_گره_idx.resize(ctx->رقم_queries);
        }

        ctx->device->device.resetQueryPool(ctx->query_pool, 0, cgraph->n_nodes+1);
        std::fill(ctx->query_fusion_نامها.begin(), ctx->query_fusion_نامها.end(), اشاره_تهی);
        std::fill(ctx->query_fusion_گره_شمار.begin(), ctx->query_fusion_گره_شمار.end(), 0);
        std::fill(ctx->query_nodes.begin(), ctx->query_nodes.end(), اشاره_تهی);
        std::fill(ctx->query_گره_idx.begin(), ctx->query_گره_idx.end(), 0);

        GGML_ASSERT(ctx->compute_ctx.expired());
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        ctx->query_idx = 0;
        compute_ctx->s->میانگیر->buf.writeTimestamp(vk::Pipeline_Stage_Flag_بیتها::eAllCommands, ctx->query_pool, ctx->query_idx++);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    ctx->prealloc_y_آخرین_pipeline_استفاده_شده = اشاره_تهی;
    ctx->prealloc_y_آخرین_tensor_استفاده_شده = اشاره_تهی;
    ctx->prealloc_y_آخرین_k_padded = خطأ;

    إذا (ctx->prealloc_اندازه_جمع_کن_rms_partials) {
        ggml_vk_preallocate_buffers(ctx, اشاره_تهی);
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        // initialize partial sums to zero.
        ggml_vk_میانگیر_memset_async(compute_ctx, ctx->prealloc_جمع_کن_rms_partials, 0, 0, ctx->prealloc_اندازه_جمع_کن_rms_partials);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    // Submit after enough work has accumulated, to overlap CPU cmdbuffer generation with GPU execution.
    // Estimate the amount of compute work using flops, and submit every 200 GFLOP
    // (and scaled down based on total graph flops, so smaller models submit earlier).
    // Also submit at least every 100 nodes, in case there are workloads without heavy compute.
    uint32_t submitted_nodes = 0;
    uint32_t submit_شمار = 0;
    uint64_t batch_flops = 0;
    uint64_t مجموع_flops = 0;
    uint64_t flops_cap = 200'000'000'000ULL;

    // On weaker AMD GPUs larger submissions can hit a driver timeout, submit more often to avoid this
    إذا (ctx->device->vendor_id == VK_VENDOR_ID_AMD && ctx->device->shader_core_شمار > 0) {
        إذا (ctx->device->architecture == AMD_GCN && ctx->device->shader_core_شمار < 32) {
            flops_cap = 500'000'000ULL * ctx->device->shader_core_شمار;
        } وإلا إذا (ctx->device->architecture != AMD_GCN && ctx->device->shader_core_شمار < 24) {
            flops_cap = 2'000'000'000ULL * ctx->device->shader_core_شمار;
        }
    }
    uint64_t flops_در_هر_submit = std::min(flops_cap, ctx->آخرین_مجموع_flops / 40u);

    خودکار ثابت submit_after = [&](صحيح start, صحيح end) {
        إذا (ctx->device->serialize_submissions) {
            حاول {
                خودکار res = ctx->device->device.waitForFences({ ctx->fence }, درست, UINT64_MAX);
                إذا (res != vk::نتیجه_2::eSuccess) {
                    GGML_LOG_ERROR("ggml_vulkan: waitForFences error during serialized submission\n");
                    ارم vk::SystemError(vk::make_error_code(res), "ggml_vulkan: waitForFences during serialized submission");
                }
            } امسك (vk::Device_گمشده_Error &) {
                ggml_vk_print_device_fault_info(ctx->device);
                GGML_LOG_ERROR("ggml_vulkan: device lost on %s waiting for submission (nodes %d to %d):\n",
                        ctx->device->name.c_str(), start, end);
                ggml_vk_print_گره_list(cgraph, start, end);
                ارم;
            }
            ctx->device->device.resetFences({ ctx->fence });
            ctx->submit_pending = خطأ;
            ctx->device->diag_cgraph = cgraph;
            ctx->device->diag_prev_شروع = start;
            ctx->device->diag_prev_انتها = end;
        }
        اولی_گره_درون_batch = درست;
        submitted_nodes = 0;
        batch_flops = 0;
        إذا (submit_شمار < 3) {
            flops_در_هر_submit *= 2;
        }
        submit_شمار++;
    };

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        إذا (اولی_گره_درون_batch) {
            submit_گره_idx = i;
        }

        {
            خودکار گره_flops = ggml_vk_get_گره_flops(cgraph->nodes[i]);
            مجموع_flops += گره_flops;

            // Flush the current batch before recording a node that would push it over the flop threshold
            إذا (flops_در_هر_submit != 0 && submitted_nodes > 0 && batch_flops + گره_flops >= flops_در_هر_submit) {
                vk_context flush_ctx = ggml_vk_get_compute_ctx(ctx);
                ggml_vk_ctx_انتها(flush_ctx);
                flush_ctx->exit_tensor_idx = -1;
                ctx->compute_ctx.reset();
                ggml_vk_compute_forward(ctx, cgraph, cgraph->nodes[submit_گره_idx], submit_گره_idx, خطأ);
                submit_after(submit_گره_idx, i - 1);
                submit_گره_idx = i;
            }

            batch_flops += گره_flops;
        }

        // op_srcs_fused_elementwise indicates whether an op's srcs all contribute to
        // the fused result in an elementwise-way. This affects whether the memory for
        // the src is allowed to overlap the memory for the destination.
        // The array is sized to handle the largest fusion (asserted later).
        منطقي op_srcs_fused_elementwise[13];

        ctx->fused_topk_moe_نما = TOPK_MOE_شمار;
        ctx->fused_topk_moe_scale = خطأ;
        ctx->fused_topk_qsa = خطأ;
        ctx->fused_hc_post_gate = خطأ;
        ctx->fused_rms_norm_نما = RMS_NORM_شمار;
        ثابت حرف *fusion_string {};
        إذا (!ctx->device->disable_fusion) {
            uint32_t رقم_adds = ggml_vk_fuse_multi_جمع_کن(ctx, cgraph, i);
            إذا (رقم_adds) {
                ctx->رقم_additional_fused_ops = رقم_adds - 1;
                fusion_string = "MULTI_ADD";
                std::fill_n(op_srcs_fused_elementwise, ctx->رقم_additional_fused_ops + 1, درست);
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_جمع_کن, GGML_OP_جمع_کن })) {
                ctx->رقم_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ADD_ADD";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = درست;
                op_srcs_fused_elementwise[2] = درست;
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_جمع_کن })) {
                ctx->رقم_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ADD";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = درست;
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_جمع_کن_ID, GGML_OP_MUL })) {
                ctx->رقم_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ID_ADD_ID_MUL";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = درست;
                op_srcs_fused_elementwise[2] = درست;
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_جمع_کن_ID })) {
                ctx->رقم_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_ADD_ID";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = درست;
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL })) {
                ctx->رقم_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_MUL";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = درست;
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, rms_norm_mul_rope_view_مجموعه_ردیفها_الگو, { i + 4 }) &&
                       ggml_check_یالها(cgraph, i, rms_norm_mul_rope_view_مجموعه_ردیفها_یالها) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i) &&
                       ggml_vk_can_fuse_rope_مجموعه_ردیفها(ctx, cgraph, i + 2)) {
                ctx->رقم_additional_fused_ops = 4;
                ctx->fused_rms_norm_نما = RMS_NORM_MUL_ROPE_VIEW_مجموعه_ردیفها;
                fusion_string = "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = خطأ;
                op_srcs_fused_elementwise[2] = خطأ;
                op_srcs_fused_elementwise[3] = خطأ;
                op_srcs_fused_elementwise[4] = خطأ;
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i)) {
                ctx->رقم_additional_fused_ops = 2;
                ctx->fused_rms_norm_نما = RMS_NORM_MUL_ROPE;
                fusion_string = "RMS_NORM_MUL_ROPE";
                // rope is approximately elementwise - whole rows are done by a single workgroup and it's row-wise
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = درست;
                op_srcs_fused_elementwise[2] = درست;
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, hc_post_gate_الگو, { i + 3 }) &&
                       ggml_check_یالها(cgraph, i, hc_post_gate_یالها) &&
                       ggml_vk_can_fuse_hc_post_gate(cgraph, i)) {
                ctx->رقم_additional_fused_ops = hc_post_gate_الگو.size() - 1;
                ctx->fused_hc_post_gate = درست;
                fusion_string = "HC_POST_GATE";
                std::fill_n(op_srcs_fused_elementwise, ctx->رقم_additional_fused_ops + 1, خطأ);
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_جمع_کن_mul_الگو)) {
                ctx->رقم_additional_fused_ops = 3;
                ctx->fused_rms_norm_نما = RMS_NORM_MUL_جمع_کن_MUL;
                fusion_string = "RMS_NORM_MUL_ADD_MUL";
                std::fill_n(op_srcs_fused_elementwise, 4, درست);
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_جمع_کن_الگو)) {
                ctx->رقم_additional_fused_ops = 2;
                ctx->fused_rms_norm_نما = RMS_NORM_MUL_جمع_کن;
                fusion_string = "RMS_NORM_MUL_ADD";
                std::fill_n(op_srcs_fused_elementwise, 3, درست);
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, rms_norm_view_مجموعه_ردیفها_الگو, { i + 2 }) &&
                       ggml_check_یالها(cgraph, i, rms_norm_view_مجموعه_ردیفها_یالها) &&
                       ggml_vk_can_fuse_rms_norm_مجموعه_ردیفها(ctx, cgraph, i)) {
                ctx->رقم_additional_fused_ops = 2;
                ctx->fused_rms_norm_نما = RMS_NORM_VIEW_مجموعه_ردیفها;
                fusion_string = "RMS_NORM_VIEW_SET_ROWS";
                std::fill_n(op_srcs_fused_elementwise, 3, خطأ);
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                ctx->رقم_additional_fused_ops = 1;
                ctx->fused_rms_norm_نما = RMS_NORM_MUL;
                fusion_string = "RMS_NORM_MUL";
                // rms_norm is not elementwise, but whole rows must be consumed and the scale factor computed before
                // they are overwritten, and one workgroup per row. So close enough.
                op_srcs_fused_elementwise[0] = درست;
                op_srcs_fused_elementwise[1] = درست;
            } وإلا إذا (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL })) {
                ctx->رقم_additional_fused_ops = 1;
                بدل (ggml_get_unary_op(cgraph->nodes[i])) {
                    حالة GGML_UNARY_OP_GELU:     fusion_string = "GELU_MUL";     اكسر;
                    حالة GGML_UNARY_OP_SIGMOID:  fusion_string = "SIGMOID_MUL";  اكسر;
                    حالة GGML_UNARY_OP_SILU:     fusion_string = "SILU_MUL";     اكسر;
                    افتراضي:                     fusion_string = "SOFTPLUS_MUL"; اكسر;
                }
                op_srcs_fused_elementwise[0] = درست;
                op_srcs_fused_elementwise[1] = درست;
            } وإلا إذا (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 2)) {
                ctx->رقم_additional_fused_ops = 2;
                fusion_string = "SSM_CONV_BIAS_SILU";
                // ssm_conv reads multiple input tokens per output, so it's not elementwise w.r.t. its srcs.
                // The downstream add and silu are elementwise on the conv output.
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = درست;
                op_srcs_fused_elementwise[2] = درست;
            } وإلا إذا (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 1)) {
                ctx->رقم_additional_fused_ops = 1;
                fusion_string = "SSM_CONV_SILU";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = درست;
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, rope_view_مجموعه_ردیفها_الگو, { i + 2 }) &&
                       ggml_check_یالها(cgraph, i, rope_view_مجموعه_ردیفها_یالها) &&
                       ggml_vk_can_fuse_rope_مجموعه_ردیفها(ctx, cgraph, i)) {
                ctx->رقم_additional_fused_ops = 2;
                fusion_string = "ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = خطأ;
                op_srcs_fused_elementwise[1] = خطأ;
                op_srcs_fused_elementwise[2] = خطأ;
            } وإلا إذا (ggml_vk_can_fuse_snake(ctx, cgraph, i)) {
                ctx->رقم_additional_fused_ops = 4;
                fusion_string = "SNAKE";
                // elementwise=true: snake.comp is safe under exact aliasing because each
                // thread reads data_x[idx] into a register before writing data_d[idx]
                // with a data dependency on that register. The overlap check still
                // rejects partial overlaps (different base or size).
                std::fill_n(op_srcs_fused_elementwise, 5, درست);
            } وإلا إذا (ggml_vk_can_fuse_topk_qsa(ctx, cgraph, i)) {
                ctx->رقم_additional_fused_ops = topk_qsa_الگو.size() - 1;
                ctx->fused_topk_qsa = درست;
                fusion_string = "TOPK_QSA";
                std::fill_n(op_srcs_fused_elementwise, ctx->رقم_additional_fused_ops + 1, خطأ);
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax_norm, { i + 3, i + 9 }) &&
                       ggml_check_یالها(cgraph, i, topk_moe_early_softmax_norm_یالها) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX_NORM)) {
                ctx->رقم_additional_fused_ops = topk_moe_early_softmax_norm.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ماسک |= 1 << 3;
                ctx->fused_topk_moe_نما = TOPK_MOE_EARLY_SOFTMAX_NORM;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX_NORM";
                std::fill_n(op_srcs_fused_elementwise, ctx->رقم_additional_fused_ops + 1, خطأ);
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sigmoid_norm_bias, { i + 4, i + 10 }) &&
                       ggml_check_یالها(cgraph, i, topk_moe_sigmoid_norm_bias_یالها) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SIGMOID_NORM_BIAS)) {
                ctx->رقم_additional_fused_ops = topk_moe_sigmoid_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ماسک |= 1 << 4;
                ctx->fused_topk_moe_نما = TOPK_MOE_SIGMOID_NORM_BIAS;
                fusion_string = "TOPK_MOE_SIGMOID_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->رقم_additional_fused_ops + 1, خطأ);
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sqrt_softplus_norm_bias, { i + 5, i + 11 }) &&
                       ggml_check_یالها(cgraph, i, topk_moe_sqrt_softplus_norm_bias_یالها) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS)) {
                ctx->رقم_additional_fused_ops = topk_moe_sqrt_softplus_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ماسک |= 1 << 5;
                ctx->fused_topk_moe_نما = TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
                fusion_string = "TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->رقم_additional_fused_ops + 1, خطأ);
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax, { i + 3, i + 4 }) &&
                       ggml_check_یالها(cgraph, i, topk_moe_early_softmax_یالها) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX)) {
                ctx->رقم_additional_fused_ops = topk_moe_early_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ماسک |= 1 << 3;
                ctx->fused_topk_moe_نما = TOPK_MOE_EARLY_SOFTMAX;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->رقم_additional_fused_ops + 1, خطأ);
            } وإلا إذا (ggml_can_fuse_subgraph(cgraph, i, topk_moe_late_softmax, { i + 1, i + 5 }) &&
                       ggml_check_یالها(cgraph, i, topk_moe_late_softmax_یالها) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_LATE_SOFTMAX)) {
                ctx->رقم_additional_fused_ops = topk_moe_late_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ماسک |= 1 << 1;
                ctx->fused_topk_moe_نما = TOPK_MOE_LATE_SOFTMAX;
                fusion_string = "TOPK_MOE_LATE_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->رقم_additional_fused_ops + 1, خطأ);
            }
            إذا (ctx->fused_topk_moe_نما != TOPK_MOE_شمار) {
                // Look for an additional scale op to fuse - occurs in deepseek2 and nemotron3 nano.
                إذا (ggml_can_fuse_subgraph(cgraph, i + ctx->رقم_additional_fused_ops - 1, { GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE }, { i + ctx->رقم_additional_fused_ops + 1 }) ||
                    ggml_can_fuse_subgraph(cgraph, i + ctx->رقم_additional_fused_ops, { GGML_OP_GET_ردیفها, GGML_OP_SCALE }, { i + ctx->رقم_additional_fused_ops + 1 })) {
                    ctx->fused_topk_moe_scale = درست;
                    ctx->رقم_additional_fused_ops++;
                    op_srcs_fused_elementwise[ctx->رقم_additional_fused_ops] = خطأ;
                }
            }
        }
        GGML_ASSERT(ctx->رقم_additional_fused_ops < (صحيح)(حجم(op_srcs_fused_elementwise) / حجم(op_srcs_fused_elementwise[0])));
        ctx->fused_ops_write_ماسک |= 1 << ctx->رقم_additional_fused_ops;

        // Check whether fusion would overwrite src operands while they're still in use.
        // If so, disable fusion.
        إذا (ctx->رقم_additional_fused_ops) {
            // There are up to two output nodes - topk_moe has two.
            uint32_t بیتها = ctx->fused_ops_write_ماسک & ~(1 << ctx->رقم_additional_fused_ops);
            ggml_tensor *output_nodes[2] {};
            output_nodes[0] = cgraph->nodes[i + ctx->رقم_additional_fused_ops];
            إذا (بیتها) {
                صحيح output_idx = پیدا_کن_اولی_مجموعه(بیتها);
                GGML_ASSERT(بیتها == (1u << output_idx));
                output_nodes[1] = cgraph->nodes[i + output_idx];
            }

            منطقي need_disable = خطأ;

            لكل (صحيح j = 0; j < 2; ++j) {
                ggml_tensor *dst = output_nodes[j];
                إذا (!dst) {
                    استمر;
                }
                // Loop over all srcs of all nodes in the fusion. If the src overlaps
                // the destination and the src is not an intermediate node that's being
                // elided, then disable fusion.
                لكل (صحيح k = 0; k <= ctx->رقم_additional_fused_ops; ++k) {
                    لكل (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                        ggml_tensor *src = cgraph->nodes[i + k]->src[s];
                        إذا (!src || src->op == GGML_OP_NONE) {
                            استمر;
                        }
                        إذا (ggml_vk_tensors_overlap(src, dst, op_srcs_fused_elementwise[k])) {
                            منطقي یافت = خطأ;
                            لكل (صحيح n = 0; n < k; ++n) {
                                إذا (cgraph->nodes[i + n] == src) {
                                    یافت = درست;
                                    اكسر;
                                }
                            }
                            إذا (!یافت) {
                                need_disable = درست;
                            }
                        }
                    }
                }
            }
            إذا (need_disable) {
                ctx->رقم_additional_fused_ops = 0;
                ctx->fused_ops_write_ماسک = 1;
                ctx->fused_topk_moe_نما = TOPK_MOE_شمار;
                ctx->fused_topk_moe_scale = خطأ;
                ctx->fused_topk_qsa = خطأ;
                ctx->fused_hc_post_gate = خطأ;
                ctx->fused_rms_norm_نما = RMS_NORM_شمار;
                fusion_string = اشاره_تهی;
            }
        }

        // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
        منطقي almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
        منطقي submit = (submitted_nodes >= ctx->device->max_nodes_در_هر_submit) ||
                      (flops_در_هر_submit != 0 && batch_flops >= flops_در_هر_submit) ||
                      (i + ctx->رقم_additional_fused_ops >= آخرین_گره) ||
                      (almost_ready && !ctx->almost_ready_fence_pending);

        منطقي enqueued = ggml_vk_build_گراف(ctx, cgraph, i, cgraph->nodes[submit_گره_idx], submit_گره_idx, i + ctx->رقم_additional_fused_ops >= آخرین_گره, almost_ready, submit);

        إذا (vk_perf_logger_enabled && enqueued) {
            compute_ctx = ggml_vk_get_compute_ctx(ctx);
            إذا (!vk_perf_logger_concurrent) {
                // track a single node/fusion for the current query
                ctx->query_nodes[ctx->query_idx] = cgraph->nodes[i];
                ctx->query_fusion_نامها[ctx->query_idx] = fusion_string;
                compute_ctx->s->میانگیر->buf.writeTimestamp(vk::Pipeline_Stage_Flag_بیتها::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            } وإلا {
                // track a fusion string and number of fused ops for the current node_idx
                ctx->query_fusion_نامها[i] = fusion_string;
                ctx->query_fusion_گره_شمار[i] = ctx->رقم_additional_fused_ops;
            }
        }

        إذا (enqueued) {
            ++submitted_nodes;

#اگر_تعریف_نشده GGML_VULKAN_CHECK_RESULTS
            إذا (اولی_گره_درون_batch) {
                اولی_گره_درون_batch = خطأ;
            }
#پایان_اگر
        }

        إذا (submit && enqueued) {
            submit_after(submit_گره_idx, i + (صحيح)ctx->رقم_additional_fused_ops);
        }
        i += ctx->رقم_additional_fused_ops;
        ctx->رقم_additional_fused_ops = 0;
        ctx->fused_ops_write_ماسک = 0;
    }

    ctx->آخرین_مجموع_flops = مجموع_flops;

    إذا (vk_perf_logger_enabled) {
        // End the command buffer and submit/wait
        GGML_ASSERT(!ctx->compute_ctx.expired());
        compute_ctx = ctx->compute_ctx.قفل();
        ggml_vk_ctx_انتها(compute_ctx);

        ggml_vk_submit(compute_ctx, ctx->device->fence);
        VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, درست, UINT64_MAX), "GGML_VULKAN_PERF waitForFences", ctx->device);
        ctx->device->device.resetFences({ ctx->device->fence });
        ctx->compute_ctx.reset();

        // Get the results and pass them to the logger
        std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
        VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->query_pool, 0, ctx->query_idx, (cgraph->n_nodes + 1)*حجم(uint64_t), timestamps.داده(), حجم(uint64_t), vk::Query_نتیجه_Flag_بیتها::e64 | vk::Query_نتیجه_Flag_بیتها::eWait), "get timestamp results", ctx->device);
        إذا (!vk_perf_logger_concurrent) {
            // Log each op separately
            لكل (صحيح i = 1; i < ctx->query_idx; i++) {
                خودکار گره = ctx->query_nodes[i];
                خودکار name = ctx->query_fusion_نامها[i];
                ctx->perf_logger->log_timing(گره, name, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        } وإلا {
            // Log each group of nodes
            صحيح prev_گره_idx = 0;
            لكل (صحيح i = 1; i < ctx->query_idx; i++) {
                خودکار cur_گره_idx = ctx->query_گره_idx[i];
                std::vector<ggml_tensor *> nodes;
                std::vector<ثابت حرف *> نامها;
                لكل (صحيح گره_idx = prev_گره_idx; گره_idx < cur_گره_idx; ++گره_idx) {
                    إذا (ggml_op_is_خالی(cgraph->nodes[گره_idx]->op)) {
                        استمر;
                    }
                    nodes.push_back(cgraph->nodes[گره_idx]);
                    نامها.push_back(ctx->query_fusion_نامها[گره_idx]);
                    گره_idx += ctx->query_fusion_گره_شمار[گره_idx];
                }
                prev_گره_idx = cur_گره_idx;
                ctx->perf_logger->log_timing(nodes, نامها, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        }
        ctx->perf_logger->print_timings();
    }

    إذا (!ctx->device->support_async) {
        ggml_vk_synchronize(ctx);
    }

    أرجع GGML_وضعیت_SUCCESS;

    استفادهنشده(backend);
}

فراغ ggml_vk_گراف_optimize(ggml_backend_t backend, بنية ggml_cgraph * graph, بنية ggml_backend_گراف_optimize_params * params)
{
    VK_LOG_DEBUG("ggml_vk_graph_optimize(" << graph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    إذا (ctx->device->disable_گراف_optimize) {
        أرجع;
    }

    خودکار ثابت &is_خالی = [](ثابت ggml_tensor * گره) -> منطقي {
        أرجع گره->op == GGML_OP_NONE || گره->op == GGML_OP_RESHAPE || گره->op == GGML_OP_TRANSPOSE || گره->op == GGML_OP_VIEW || گره->op == GGML_OP_جایگشت;
    };

    خودکار ثابت &is_src_از = [&is_خالی](ثابت ggml_tensor *dst, ثابت ggml_tensor *src) -> منطقي {
        خودکار ثابت &پایه = [](ثابت ggml_tensor * tensor) {
            أرجع tensor->view_src ? tensor->view_src : tensor;
        };
        لكل (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            إذا (dst->src[s] == src) {
                أرجع درست;
            }
            إذا (is_خالی(dst) || is_خالی(src)) {
                استمر;
            }
            // A source view of dst may read storage written through a different view by src.
            إذا (dst->src[s] && پایه(dst->src[s]) == پایه(src)) {
                أرجع درست;
            }
            // Moving dst forward may overwrite storage still read through a view by src.
            إذا (src->src[s] && پایه(dst) == پایه(src->src[s])) {
                أرجع درست;
            }
        }
        // implicit dependency if they view the same tensor
        إذا (پایه(dst) == پایه(src)) {
            أرجع درست;
        }
        أرجع خطأ;
    };

    std::vector<ggml_tensor *> new_ترتیب;
    std::vector<منطقي> استفاده_شده(graph->n_nodes, خطأ);
    std::set<ggml_tensor *> استفاده_شده_گره_مجموعه;

    صحيح اولی_استفادهنشده = 0;

    // scheduled or zero-compute nodes in [lo, hi)
    خودکار ثابت &خالی_or_scheduled_between = [&](صحيح lo, صحيح hi) -> منطقي {
        لكل (صحيح v = lo; v < hi; ++v) {
            إذا (!استفاده_شده[v] && !is_خالی(graph->nodes[v])) {
                أرجع خطأ;
            }
        }
        أرجع درست;
    };
    طالما (اولی_استفادهنشده < graph->n_nodes) {
        std::vector<صحيح> current_مجموعه;

        // Check for fusion patterns and avoid reordering them
        خودکار ثابت &تطابق_الگو = [&](ثابت std::initializer_list<ggml_op> &pattern, صحيح start) -> منطقي {
            إذا (start + (صحيح)pattern.size() <= graph->n_nodes) {
                منطقي is_الگو = درست;
                لكل (size_t j = 0; j < pattern.size(); ++j) {
                    إذا (graph->nodes[start + j]->op != pattern.begin()[j] || استفاده_شده[start + j]) {
                        is_الگو = خطأ;
                    }
                }
                أرجع is_الگو;
            }
            أرجع خطأ;
        };

        خودکار ثابت &keep_الگو = [&](ثابت std::initializer_list<ggml_op> &pattern) -> منطقي {
            إذا (تطابق_الگو(pattern, اولی_استفادهنشده)) {
                لكل (size_t j = 0; j < pattern.size(); ++j) {
                    new_ترتیب.push_back(graph->nodes[اولی_استفادهنشده + j]);
                    استفاده_شده_گره_مجموعه.insert(graph->nodes[اولی_استفادهنشده + j]);
                    استفاده_شده[اولی_استفادهنشده + j] = درست;
                }
                طالما (اولی_استفادهنشده < graph->n_nodes && استفاده_شده[اولی_استفادهنشده]) {
                    اولی_استفادهنشده++;
                }
                أرجع درست;
            }
            أرجع خطأ;
        };

        خودکار ثابت &جمع_کن_الگو_alloc_deps = [&](ثابت std::initializer_list<ggml_op> &pattern, صحيح آخرین_گره) {
            // Keep external inputs alive through the fused output.
            std::set<ggml_tensor *> دیده;
            لكل (size_t j = 0; j < pattern.size(); ++j) {
                ggml_tensor * گره = graph->nodes[اولی_استفادهنشده + j];
                لكل (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                    ggml_tensor * src = گره->src[s];
                    إذا (src && دیده.insert(src).second) {
                        params->جمع_کن_alloc_dep(params->user_داده, src, graph->nodes[آخرین_گره]);
                    }
                }
                دیده.insert(گره);
            }
        };

        خودکار ثابت &keep_topk_moe_الگو = [&](ثابت std::initializer_list<ggml_op> &pattern) -> منطقي {
            إذا (!تطابق_الگو(pattern, اولی_استفادهنشده)) {
                أرجع خطأ;
            }

            صحيح آخرین_گره = اولی_استفادهنشده + (صحيح) pattern.size() - 1;
            // Some TOPK_MOE variants fuse a trailing scale.
            إذا (آخرین_گره + 1 < graph->n_nodes && graph->nodes[آخرین_گره + 1]->op == GGML_OP_SCALE) {
                آخرین_گره++;
            }

            جمع_کن_الگو_alloc_deps(pattern, آخرین_گره);

            أرجع keep_الگو(pattern);
        };

        إذا (keep_topk_moe_الگو(topk_moe_early_softmax_norm)) {
            استمر;
        }
        إذا (keep_topk_moe_الگو(topk_moe_sigmoid_norm_bias)) {
            استمر;
        }
        إذا (keep_topk_moe_الگو(topk_moe_sqrt_softplus_norm_bias)) {
            استمر;
        }
        إذا (keep_topk_moe_الگو(topk_moe_early_softmax)) {
            استمر;
        }
        إذا (keep_topk_moe_الگو(topk_moe_late_softmax)) {
            استمر;
        }
        إذا (keep_الگو(snake_الگو)) {
            استمر;
        }
        إذا (keep_الگو(topk_qsa_الگو)) {
            استمر;
        }

        إذا (keep_الگو(rms_norm_mul_جمع_کن_mul_الگو)) {
            استمر;
        }
        إذا (keep_الگو(rms_norm_mul_جمع_کن_الگو)) {
            استمر;
        }
        إذا (keep_الگو(rms_norm_mul_rope_view_مجموعه_ردیفها_الگو)) {
            استمر;
        }
        إذا (keep_الگو(rms_norm_view_مجموعه_ردیفها_الگو)) {
            استمر;
        }
        إذا (keep_الگو(rope_view_مجموعه_ردیفها_الگو)) {
            استمر;
        }
        إذا (تطابق_الگو(hc_post_gate_الگو, اولی_استفادهنشده)) {
            جمع_کن_الگو_alloc_deps(hc_post_gate_الگو, اولی_استفادهنشده + (صحيح) hc_post_gate_الگو.size() - 1);
            keep_الگو(hc_post_gate_الگو);
            استمر;
        }

        // First, grab the next unused node.
        current_مجموعه.push_back(اولی_استفادهنشده);

        // Loop through the next N nodes. Grab any that don't depend on other nodes that
        // haven't already been run. Nodes that have already been run have used[i] set
        // to true. Allow nodes that depend on the previous node if it's a fusion pattern
        // that we support (e.g. RMS_NORM + MUL).
        // This first pass only grabs "real" (non-view nodes). Second pass grabs view nodes.
        // The goal is to not interleave real and view nodes in a way that breaks fusion.
        ثابت صحيح رقم_مقصد_CHECK = 20;
        لكل (صحيح j = اولی_استفادهنشده+1; j < std::min(اولی_استفادهنشده + رقم_مقصد_CHECK, graph->n_nodes); ++j) {
            إذا (استفاده_شده[j]) {
                استمر;
            }
            إذا (is_خالی(graph->nodes[j])) {
                استمر;
            }
            // Protect every interior QSA node (not just the start): the mask branch is
            // independent, so it gets pulled out and breaks keep_pattern otherwise.
            خودکار ثابت &درون_qsa_الگو = [&](صحيح n) -> منطقي {
                لكل (صحيح o = 0; o < (صحيح) topk_qsa_الگو.size(); ++o) {
                    إذا (n - o >= 0 && تطابق_الگو(topk_qsa_الگو, n - o)) {
                        أرجع درست;
                    }
                }
                أرجع خطأ;
            };
            إذا (تطابق_الگو(topk_moe_early_softmax_norm, j) ||
                تطابق_الگو(topk_moe_sigmoid_norm_bias, j) ||
                تطابق_الگو(topk_moe_sqrt_softplus_norm_bias, j) ||
                تطابق_الگو(topk_moe_early_softmax, j) ||
                تطابق_الگو(topk_moe_late_softmax, j) ||
                تطابق_الگو(snake_الگو, j) ||
                درون_qsa_الگو(j) ||
                تطابق_الگو(rms_norm_mul_جمع_کن_mul_الگو, j) ||
                تطابق_الگو(rms_norm_mul_جمع_کن_الگو, j) ||
                تطابق_الگو(rms_norm_mul_rope_view_مجموعه_ردیفها_الگو, j) ||
                تطابق_الگو(rms_norm_view_مجموعه_ردیفها_الگو, j) ||
                تطابق_الگو(rope_view_مجموعه_ردیفها_الگو, j) ||
                تطابق_الگو(hc_post_gate_الگو, j)) {
                استمر;
            }
            منطقي ok = درست;
            لكل (صحيح c = اولی_استفادهنشده; c < j; ++c) {
                إذا (!استفاده_شده[c] &&
                    is_src_از(graph->nodes[j], graph->nodes[c]) &&
                    !(c == current_مجموعه.back() && graph->nodes[c]->op == GGML_OP_RMS_NORM && graph->nodes[j]->op == GGML_OP_MUL && خالی_or_scheduled_between(c+1, j)) &&
                    !(c == current_مجموعه.back() && graph->nodes[c]->op == GGML_OP_UNARY && graph->nodes[j]->op == GGML_OP_MUL && خالی_or_scheduled_between(c+1, j)) &&
                    !(j == c+1 && c == current_مجموعه.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT && graph->nodes[j]->op == GGML_OP_جمع_کن) &&
                    !(j == c+1 && c == current_مجموعه.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_جمع_کن_ID) &&
                    !(j == c+1 && c == current_مجموعه.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_مجموعه.back() && graph->nodes[c]->op == GGML_OP_جمع_کن && graph->nodes[j]->op == GGML_OP_جمع_کن) &&
                    !(j == c+1 && c == current_مجموعه.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_جمع_کن) &&
                    !(j == c+1 && c == current_مجموعه.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_UNARY)) {
                    ok = خطأ;
                    اكسر;
                }
            }
            إذا (ok) {
                current_مجموعه.push_back(j);

                صحيح rope_idx = j;

                // When we've found RMS_NORM + MUL, try to find a ROPE that uses it
                إذا (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_MUL &&
                    graph->nodes[j-1]->op == GGML_OP_RMS_NORM) {
                    لكل (صحيح k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        إذا (graph->nodes[k]->op == GGML_OP_ROPE &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // Check that other srcs are already valid
                            graph->nodes[k]->src[1]->op == GGML_OP_NONE &&
                            (graph->nodes[k]->src[2] == اشاره_تهی || graph->nodes[k]->src[2]->op == GGML_OP_NONE)) {
                            rope_idx = k;
                            current_مجموعه.push_back(rope_idx);
                            استفاده_شده[rope_idx] = درست;
                            اكسر;
                        }
                    }
                }
                // Look for ROPE/RMS_NORM + VIEW + SET_ROWS and make them consecutive
                إذا (graph->nodes[rope_idx]->op == GGML_OP_ROPE || graph->nodes[rope_idx]->op == GGML_OP_RMS_NORM) {
                    صحيح view_idx = -1;
                    صحيح مجموعه_ردیفها_idx = -1;
                    لكل (صحيح k = rope_idx + 1; k < std::min(rope_idx + 15, graph->n_nodes); ++k) {
                        إذا (استفاده_شده[k]) {
                            استمر;
                        }
                        إذا (view_idx == -1 && graph->nodes[k]->op == GGML_OP_VIEW && graph->nodes[k]->src[0] == graph->nodes[rope_idx]) {
                            view_idx = k;
                            استمر;
                        }
                        إذا (view_idx != -1 && graph->nodes[k]->op == GGML_OP_مجموعه_ردیفها && graph->nodes[k]->src[0] == graph->nodes[view_idx]) {
                            مجموعه_ردیفها_idx = k;
                            اكسر;
                        }
                    }
                    إذا (مجموعه_ردیفها_idx != -1) {
                        ثابت صحيح گره_idxs[] = { rope_idx, view_idx, مجموعه_ردیفها_idx };
                        ثابت ggml_op ops[] = { graph->nodes[rope_idx]->op, GGML_OP_VIEW, GGML_OP_مجموعه_ردیفها };
                        منطقي can_pull = ggml_can_fuse_subgraph_ext(graph, گره_idxs, 3, ops, &مجموعه_ردیفها_idx, 1);

                        لكل (صحيح c = rope_idx + 1; can_pull && c < مجموعه_ردیفها_idx; ++c) {
                            إذا (!استفاده_شده[c] && c != view_idx && !is_خالی(graph->nodes[c]) &&
                                is_src_از(graph->nodes[مجموعه_ردیفها_idx], graph->nodes[c])) {
                                can_pull = خطأ;
                            }
                        }

                        إذا (can_pull) {
                            current_مجموعه.push_back(view_idx);
                            current_مجموعه.push_back(مجموعه_ردیفها_idx);
                            استفاده_شده[view_idx] = درست;
                            استفاده_شده[مجموعه_ردیفها_idx] = درست;
                        }
                    }
                }
                // Look for MUL_MAT_ID + ADD_ID + MUL
                إذا (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_جمع_کن_ID &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT_ID) {
                    لكل (صحيح k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        إذا (graph->nodes[k]->op == GGML_OP_MUL &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || استفاده_شده_گره_مجموعه.find(graph->nodes[k]->src[1]) != استفاده_شده_گره_مجموعه.end())) {
                            current_مجموعه.push_back(k);
                            استفاده_شده[k] = درست;
                            اكسر;
                        }
                    }
                }
                // Look for MUL_MAT + ADD + ADD
                إذا (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_جمع_کن &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT) {
                    لكل (صحيح k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        إذا (graph->nodes[k]->op == GGML_OP_جمع_کن &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || استفاده_شده_گره_مجموعه.find(graph->nodes[k]->src[1]) != استفاده_شده_گره_مجموعه.end())) {
                            current_مجموعه.push_back(k);
                            استفاده_شده[k] = درست;
                            اكسر;
                        }
                    }
                }
                // SSM_CONV + ADD + UNARY: pull the consuming UNARY forward
                إذا (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_جمع_کن &&
                    graph->nodes[j-1]->op == GGML_OP_SSM_CONV) {
                    لكل (صحيح k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        إذا (graph->nodes[k]->op == GGML_OP_UNARY &&
                            graph->nodes[k]->src[0] == graph->nodes[j]) {
                            current_مجموعه.push_back(k);
                            استفاده_شده[k] = درست;
                            اكسر;
                        }
                    }
                }
                // UNARY + MUL: pull the consuming MUL forward
                إذا (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_UNARY) {
                    لكل (صحيح k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ggml_tensor * mul = graph->nodes[k];
                        إذا (mul->op != GGML_OP_MUL || (mul->src[0] != graph->nodes[j] && mul->src[1] != graph->nodes[j])) {
                            استمر;
                        }
                        ggml_tensor * other = (mul->src[0] == graph->nodes[j]) ? mul->src[1] : mul->src[0];
                        // the other src must either be weights or already processed
                        إذا (!(other->op == GGML_OP_NONE || استفاده_شده_گره_مجموعه.find(other) != استفاده_شده_گره_مجموعه.end())) {
                            استمر;
                        }
                        إذا (!ggml_vk_can_fuse_unary_mul(graph, j, k)) {
                            استمر;
                        }
                        current_مجموعه.push_back(k);
                        استفاده_شده[k] = درست;
                        اكسر;
                    }
                }
            }
        }
        // Second pass grabs view nodes.
        // Skip this if it would break a fusion optimization (don't split up add->rms_norm or add->add).
        إذا (graph->nodes[current_مجموعه.back()]->op != GGML_OP_جمع_کن) {
            لكل (صحيح j = اولی_استفادهنشده+1; j < std::min(اولی_استفادهنشده + رقم_مقصد_CHECK, graph->n_nodes); ++j) {
                إذا (استفاده_شده[j]) {
                    استمر;
                }
                إذا (!is_خالی(graph->nodes[j])) {
                    استمر;
                }
                منطقي ok = درست;
                لكل (صحيح c = اولی_استفادهنشده; c < j; ++c) {
                    منطقي c_درون_current_مجموعه = std::find(current_مجموعه.begin(), current_مجموعه.end(), c) != current_مجموعه.end();
                    // skip views whose srcs haven't been processed.
                    إذا (!استفاده_شده[c] &&
                        is_src_از(graph->nodes[j], graph->nodes[c]) &&
                        !c_درون_current_مجموعه) {
                        ok = خطأ;
                        اكسر;
                    }
                }
                إذا (ok) {
                    current_مجموعه.push_back(j);
                }
            }
        }

        // Push the current set into new_order
        لكل (خودکار c : current_مجموعه) {
            new_ترتیب.push_back(graph->nodes[c]);
            استفاده_شده_گره_مجموعه.insert(graph->nodes[c]);
            استفاده_شده[c] = درست;
        }
        طالما (اولی_استفادهنشده < graph->n_nodes && استفاده_شده[اولی_استفادهنشده]) {
            اولی_استفادهنشده++;
        }
    }
    // Replace the graph with the new order.
    لكل (صحيح i = 0; i < graph->n_nodes; ++i) {
        graph->nodes[i] = new_ترتیب[i];
    }
}

ساكن فراغ ggml_backend_vk_event_رکورد(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_record(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
    خودکار* cmd_buf = compute_ctx->s->میانگیر; // retrieve pointer before it gets reset

    إذا (vkev->has_event) {
        // Move existing event into submitted
        vkev->events_submitted.push_back(vkev->event);
    }

    // Grab the next event and record it, create one if necessary
    إذا (vkev->events_free.خالی()) {
        vkev->event = ctx->device->device.createEvent({});
    } وإلا {
        vkev->event = vkev->events_free.back();
        vkev->events_free.pop_back();
    }

    vkev->has_event = درست;

    ggml_vk_مجموعه_event(compute_ctx, vkev->event);

    vkev->tl_semaphore.value++;
    compute_ctx->s->signal_semaphores.push_back(vkev->tl_semaphore);
    ggml_vk_ctx_انتها(compute_ctx);

    ggml_vk_submit(compute_ctx, {});
    ctx->submit_pending = درست;
    vkev->cmd_میانگیر = cmd_buf;
    vkev->cmd_میانگیر_use_counter = cmd_buf->use_counter;
    ctx->compute_ctx.reset();
}

ساكن فراغ ggml_backend_vk_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_wait(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    إذا (vkev->has_event) {
        // Wait for latest event
        ggml_vk_wait_events(compute_ctx, { vkev->event });

        إذا (ctx->device->async_use_transfer_صف) {
            vk_context transfer_ctx = ggml_vk_get_transfer_ctx(ctx);
            transfer_ctx->s->wait_semaphores.push_back(vkev->tl_semaphore);
        }
    }
}

ساكن ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .set_tensor_async        = */ ggml_backend_vk_مجموعه_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_vk_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_vk_مجموعه_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_vk_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_vk_گراف_compute,
    /* .event_record            = */ ggml_backend_vk_event_رکورد,
    /* .event_wait              = */ ggml_backend_vk_event_wait,
    /* .graph_optimize          = */ ggml_vk_گراف_optimize,
};

ساكن ggml_guid_t ggml_backend_vk_guid() {
    ساكن ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    أرجع &guid;
}

ggml_backend_t ggml_backend_vk_init(size_t dev_رقم) {
    VK_LOG_DEBUG("ggml_backend_vk_init(" << dev_رقم << ")");

    ggml_backend_vk_context * ctx = جديد ggml_backend_vk_context;
    ggml_vk_init(ctx, dev_رقم);

    ggml_backend_t vk_backend = جديد ggml_backend {
        /* .guid    = */ ggml_backend_vk_guid(),
        /* .iface   = */ ggml_backend_vk_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), dev_رقم),
        /* .context = */ ctx,
    };

    إذا (!ctx->device->support_async) {
        vk_backend->iface.get_tensor_async = اشاره_تهی;
    }

    أرجع vk_backend;
}

منطقي ggml_backend_is_vk(ggml_backend_t backend) {
    أرجع backend != عدم && ggml_guid_matches(backend->guid, ggml_backend_vk_guid());
}

صحيح ggml_backend_vk_get_device_شمار() {
    أرجع ggml_vk_get_device_شمار();
}

فراغ ggml_backend_vk_get_device_description(صحيح device, حرف * description, size_t description_اندازه) {
    GGML_ASSERT(device < (صحيح) vk_instance.device_indices.size());
    صحيح dev_idx = vk_instance.device_indices[device];
    ggml_vk_get_device_description(dev_idx, description, description_اندازه);
}

فراغ ggml_backend_vk_get_device_حافظه(صحيح device, size_t * حرر, size_t * مجموع) {
    GGML_ASSERT(device < (صحيح) vk_instance.device_indices.size());
    GGML_ASSERT(device < (صحيح) vk_instance.device_supports_membudget.size());

    vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];
    vk::Physical_Device_حافظه_Budget_Properties_EXT budgetprops;
    vk::Physical_Device_حافظه_Properties2 memprops = {};
    ثابت منطقي membudget_supported = vk_instance.device_supports_membudget[device];
    ثابت منطقي is_integrated_gpu = vkdev.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    إذا (membudget_supported) {
        memprops.p_بعدی = &budgetprops;
    }
    vkdev.get_حافظه_Properties2(&memprops);

    *مجموع = 0;
    *حرر = 0;

    لكل (uint32_t i = 0; i < memprops.حافظه_Properties.حافظه_انباشته_شمار; ++i) {
        ثابت vk::حافظه_انباشته & انباشته = memprops.حافظه_Properties.حافظه_Heaps[i];

        إذا (is_integrated_gpu || (انباشته.flags & vk::حافظه_انباشته_Flag_بیتها::eDeviceLocal)) {
            *مجموع += انباشته.size;

            إذا (membudget_supported && i < budgetprops.انباشته_Usage.size()) {
                // Driver may report heapBudget > heap.size so we add a safeguard here
                // See https://github.com/ggml-org/llama.cpp/issues/29277
                ثابت vk::Device_اندازه budget = std::min(budgetprops.انباشته_Budget[i], انباشته.size);
                *حرر += budget > budgetprops.انباشته_Usage[i] ? budget - budgetprops.انباشته_Usage[i] : 0;
            } وإلا {
                *حرر += انباشته.size;
            }
        }
    }
}

ساكن vk::PhysicalDeviceType ggml_backend_vk_get_device_type(صحيح device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (صحيح) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    vk::PhysicalDeviceProperties2 props = {};
    device.getProperties2(&props);

    أرجع props.properties.deviceType;
}

ساكن std::string ggml_backend_vk_get_device_pci_id(صحيح device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (صحيح) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    ثابت std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

    منطقي ext_support = خطأ;

    لكل (ثابت خودکار& properties : ext_props) {
        إذا (strcmp("VK_EXT_pci_bus_info", properties.extensionName) == 0) {
            ext_support = درست;
            اكسر;
        }
    }

    إذا (!ext_support) {
        أرجع "";
    }

    vk::PhysicalDeviceProperties2 props = {};
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pci_bus_info = {};

    props.p_بعدی = &pci_bus_info;

    device.getProperties2(&props);

    ثابت uint32_t pci_domain = pci_bus_info.pciDomain;
    ثابت uint32_t pci_bus = pci_bus_info.pciBus;
    ثابت uint32_t pci_device = pci_bus_info.pciDevice;
    ثابت uint8_t pci_function = (uint8_t) pci_bus_info.pciFunction; // pci function is between 0 and 7, prevent printf overflow warning

    حرف pci_bus_id[16] = {};
    snprintf(pci_bus_id, حجم(pci_bus_id), "%04x:%02x:%02x.%x", pci_domain, pci_bus, pci_device, pci_function);

    أرجع std::string(pci_bus_id);
}

ساكن ثابت حرف * ggml_backend_vk_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    أرجع ctx->name.c_str();
}

ساكن ثابت حرف * ggml_backend_vk_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    أرجع ctx->description.c_str();
}

ساكن فراغ ggml_backend_vk_device_get_حافظه(ggml_backend_dev_t device, size_t * حرر, size_t * مجموع) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)device->context;
    ggml_backend_vk_get_device_حافظه(ctx->device, حرر, مجموع);
}

ساكن ggml_backend_میانگیر_type_t ggml_backend_vk_device_get_میانگیر_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    أرجع ggml_backend_vk_میانگیر_type(ctx->device);
}

ساكن ggml_backend_میانگیر_type_t ggml_backend_vk_device_get_host_میانگیر_type(ggml_backend_dev_t dev) {
    استفادهنشده(dev);
    أرجع ggml_backend_vk_host_میانگیر_type();
}

ساكن تعداد ggml_backend_dev_type ggml_backend_vk_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    أرجع ctx->is_integrated_gpu ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

ساكن فراغ ggml_backend_vk_device_get_props(ggml_backend_dev_t dev, بنية ggml_backend_dev_props * props) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    props->name        = ggml_backend_vk_device_get_name(dev);
    props->description = ggml_backend_vk_device_get_description(dev);
    props->type        = ggml_backend_vk_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.خالی() ? اشاره_تهی : ctx->pci_bus_id.c_str();
    ggml_backend_vk_device_get_حافظه(dev, &props->حافظه_free, &props->حافظه_مجموع);
    props->caps = {
        /* .async                 = */ درست,
        /* .host_buffer           = */ درست,
        /* .buffer_from_host_ptr  = */ خطأ,
        /* .events                = */ درست,
        /* .mmap_support          = */ !ctx->is_integrated_gpu,
    };
}

ساكن ggml_backend_t ggml_backend_vk_device_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    استفادهنشده(params);
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    أرجع ggml_backend_vk_init(ctx->device);
}

ساكن منطقي ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ثابت vk_device& device = ggml_vk_get_device(ctx->device);

    ثابت منطقي استفاده_میکند_bda = (op->op == GGML_OP_IM2_ستون || op->op == GGML_OP_IM2_ستون_3_D) &&
                          device->shader_int64 && device->میانگیر_device_آدرس;

    خودکار ثابت & tensor_اندازه_supported = [&](size_t tensor_اندازه) {
        إذا (tensor_اندازه > device->max_میانگیر_اندازه) {
            أرجع خطأ;
        }
        // For im2col shaders using BDA, maxStorageBufferRange limit doesn't apply.
        // If shader64BitIndexing is enabled, maxStorageBufferRange limit doesn't apply.
        إذا (!استفاده_میکند_bda && !device->shader_64b_indexing) {
            إذا (tensor_اندازه > device->properties.limits.max_Storage_میانگیر_Range) {
                أرجع خطأ;
            }
        }
        أرجع درست;
    };
    // reject any tensors larger than the max buffer size
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        إذا (op->src[i] && !tensor_اندازه_supported(ggml_nbytes(op->src[i]))) {
            أرجع خطأ;
        }
    }
    إذا (!tensor_اندازه_supported(ggml_nbytes(op))) {
        أرجع خطأ;
    }

    بدل (op->op) {
        حالة GGML_OP_UNARY:
            بدل (ggml_get_unary_op(op)) {
                حالة GGML_UNARY_OP_EXP:
                حالة GGML_UNARY_OP_EXPM1:
                حالة GGML_UNARY_OP_ELU:
                حالة GGML_UNARY_OP_GELU:
                حالة GGML_UNARY_OP_GELU_ERF:
                حالة GGML_UNARY_OP_GELU_سریع:
                حالة GGML_UNARY_OP_SILU:
                حالة GGML_UNARY_OP_RELU:
                حالة GGML_UNARY_OP_XIELU:
                حالة GGML_UNARY_OP_NEG:
                حالة GGML_UNARY_OP_TANH:
                حالة GGML_UNARY_OP_SIGMOID:
                حالة GGML_UNARY_OP_HARDSIGMOID:
                حالة GGML_UNARY_OP_HARDSWISH:
                حالة GGML_UNARY_OP_ABS:
                حالة GGML_UNARY_OP_SOFTPLUS:
                حالة GGML_UNARY_OP_گام:
                حالة GGML_UNARY_OP_دور:
                حالة GGML_UNARY_OP_CEIL:
                حالة GGML_UNARY_OP_FLOOR:
                حالة GGML_UNARY_OP_TRUNC:
                حالة GGML_UNARY_OP_SGN:
                    أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                افتراضي:
                    أرجع خطأ;
            }
        حالة GGML_OP_GLU:
            بدل (ggml_get_glu_op(op)) {
                حالة GGML_GLU_OP_GEGLU:
                حالة GGML_GLU_OP_REGLU:
                حالة GGML_GLU_OP_SWIGLU:
                حالة GGML_GLU_OP_SWIGLU_OAI:
                حالة GGML_GLU_OP_GEGLU_ERF:
                حالة GGML_GLU_OP_GEGLU_سریع:
                حالة GGML_GLU_OP_SWIGLU_CLAMP:
                    أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type) &&
                           (!op->src[1] || op->src[1]->type == op->src[0]->type);
                افتراضي:
                    أرجع خطأ;
            }
        حالة GGML_OP_MUL_MAT:
        حالة GGML_OP_MUL_MAT_ID:
            {
                ggml_type src0_type = op->src[0]->type;
                إذا (op->op == GGML_OP_MUL_MAT_ID) {
                    إذا (!ggml_vk_use_mul_mat_vec_id(op)) {
                        ثابت ggml_tensor * ids = op->src[2];
                        // Shaders store row IDs as uint16_t slot and token indices.
                        إذا (ids->ne[0] > int64_t(UINT16_MAX) + 1 || ids->ne[1] > int64_t(UINT16_MAX) + 1) {
                            أرجع خطأ;
                        }
                    }
                    إذا (!device->mul_mat_id_s[src0_type] && !device->mul_mat_id_m[src0_type] && !device->mul_mat_id_l[src0_type]) {
                        // If there's not enough shared memory for row_ids and the result tile, fallback to CPU
                        أرجع خطأ;
                    }
                    إذا (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                        أرجع خطأ;
                    }
                }
                بدل (src0_type) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                    حالة GGML_TYPE_Q1_0:
                    حالة GGML_TYPE_Q2_0:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_Q2_K:
                    حالة GGML_TYPE_Q3_K:
                    حالة GGML_TYPE_Q4_K:
                    حالة GGML_TYPE_Q5_K:
                    حالة GGML_TYPE_Q6_K:
                    حالة GGML_TYPE_IQ1_S:
                    حالة GGML_TYPE_IQ1_M:
                    حالة GGML_TYPE_IQ2_XXS:
                    حالة GGML_TYPE_IQ2_XS:
                    حالة GGML_TYPE_IQ2_S:
                    حالة GGML_TYPE_IQ3_XXS:
                    حالة GGML_TYPE_IQ3_S:
                    حالة GGML_TYPE_IQ4_XS:
                    حالة GGML_TYPE_IQ4_NL:
                    حالة GGML_TYPE_MXFP4:
                    حالة GGML_TYPE_NVFP4:
                    حالة GGML_TYPE_TQ1_0:
                    حالة GGML_TYPE_TQ2_0:
                        اكسر;
                    افتراضي:
                        أرجع خطأ;
                }
                بنية ggml_tensor * a;
                بنية ggml_tensor * b;
                إذا (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } وإلا {
                    a = op->src[2];
                    b = op->src[1];
                }
                إذا (a->ne[3] != b->ne[3]) {
                    أرجع خطأ;
                }
                إذا (!(ggml_vk_dim01_contiguous(op->src[0]) || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) ||
                    !(ggml_vk_dim01_contiguous(op->src[1]) || op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16)) {
                    أرجع خطأ;
                }
                إذا (op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F16) {
                    // We currently don't have a bf16 x f16 shader, or an fp16->bf16 copy shader.
                    // So don't support this combination for now.
                    أرجع خطأ;
                }
                إذا (op->src[1]->type == GGML_TYPE_BF16 && op->src[0]->type != GGML_TYPE_BF16 && op->op != GGML_OP_MUL_MAT) {
                    // BF16 in src1 (widened to F32/F16) is only wired up for MUL_MAT
                    أرجع خطأ;
                }

                أرجع درست;
            }
        حالة GGML_OP_FLASH_ATTN_EXT:
            {
                منطقي coopmat2 = device->coopmat2;
                uint32_t HSK = op->src[1]->ne[0];
                uint32_t HSV = op->src[2]->ne[0];
                إذا ((HSK % 8) != 0 || (HSV % 8) != 0) {
                    أرجع خطأ;
                }
                إذا (op->src[4] && op->src[4]->type != GGML_TYPE_F32) {
                    أرجع خطأ;
                }
                إذا (op->src[0]->type != GGML_TYPE_F32) {
                    أرجع خطأ;
                }
                إذا (op->type != GGML_TYPE_F32) {
                    أرجع خطأ;
                }
                إذا (op->src[3] && op->src[3]->type != GGML_TYPE_F16) {
                    أرجع خطأ;
                }
                خودکار fa_kv_ok = [](ggml_type t) {
                    بدل (t) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_IQ4_NL:
                        أرجع درست;
                    افتراضي:
                        أرجع خطأ;
                    }
                };
                إذا (!fa_kv_ok(op->src[1]->type) || !fa_kv_ok(op->src[2]->type)) {
                    أرجع خطأ;
                }
                إذا ((op->src[1]->type == GGML_TYPE_BF16) != (op->src[2]->type == GGML_TYPE_BF16)) {
                    أرجع خطأ;
                }
                إذا (!coopmat2 && !(device->subgroup_shuffle && device->subgroup_vote)) {
                    // scalar/coopmat1 FA uses subgroupShuffle/subgroupAll
                    أرجع خطأ;
                }
                أرجع درست;
            }
        حالة GGML_OP_GET_ردیفها:
            {
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                    حالة GGML_TYPE_Q1_0:
                    حالة GGML_TYPE_Q2_0:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_Q2_K:
                    حالة GGML_TYPE_Q3_K:
                    حالة GGML_TYPE_Q4_K:
                    حالة GGML_TYPE_Q5_K:
                    حالة GGML_TYPE_Q6_K:
                    حالة GGML_TYPE_IQ1_S:
                    حالة GGML_TYPE_IQ1_M:
                    حالة GGML_TYPE_IQ2_XXS:
                    حالة GGML_TYPE_IQ2_XS:
                    حالة GGML_TYPE_IQ2_S:
                    حالة GGML_TYPE_IQ3_XXS:
                    حالة GGML_TYPE_IQ3_S:
                    حالة GGML_TYPE_IQ4_XS:
                    حالة GGML_TYPE_IQ4_NL:
                    حالة GGML_TYPE_MXFP4:
                    حالة GGML_TYPE_NVFP4:
                    حالة GGML_TYPE_TQ1_0:
                    حالة GGML_TYPE_TQ2_0:
                    حالة GGML_TYPE_I32:
                        أرجع درست;
                    افتراضي:
                        أرجع خطأ;
                }
            }
        حالة GGML_OP_GET_ردیفها_پشت:
            أرجع op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_مجموعه_ردیفها:
            {
                إذا ((op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_F16) ||
                    (op->src[1]->type != GGML_TYPE_I32 && op->src[1]->type != GGML_TYPE_I64)) {
                    أرجع خطأ;
                }
                بدل (op->type) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                    حالة GGML_TYPE_Q1_0:
                    حالة GGML_TYPE_Q2_0:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_IQ4_NL:
                        أرجع درست;
                    افتراضي:
                        أرجع خطأ;
                }
            }
        حالة GGML_OP_CONT:
        حالة GGML_OP_CPY:
        حالة GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1] != اشاره_تهی ? op->src[1]->type : src0_type;

                إذا (src0_type == GGML_TYPE_F32) {
                    بدل (src1_type) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                    حالة GGML_TYPE_Q1_0:
                    حالة GGML_TYPE_Q2_0:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_IQ4_NL:
                        أرجع درست;
                    افتراضي:
                        اكسر;
                    }
                }
                إذا (src1_type == GGML_TYPE_F32) {
                    بدل (src0_type) {
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                    حالة GGML_TYPE_Q1_0:
                    حالة GGML_TYPE_Q2_0:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_IQ4_NL:
                        أرجع درست;
                    افتراضي:
                        اكسر;
                    }
                }

                إذا (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    أرجع درست;
                }

                إذا (
                    (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) ||
                    (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32)
                ) {
                    أرجع درست;
                }

                // We can handle copying from a type to the same type if it's
                // either not quantized or is quantized and contiguous.
                // We use f16 or f32 shaders to do the copy,
                // so the type/block size must be a multiple of 4.
                إذا (src0_type == src1_type &&
                    (!ggml_is_quantized(src0_type) || (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op))) &&
                    (ggml_type_اندازه(src0_type) % 2) == 0) {
                    أرجع درست;
                }
                أرجع خطأ;
            }
        حالة GGML_OP_REPEAT:
            أرجع ggml_type_اندازه(op->type) == ggml_type_اندازه(op->src[0]->type) &&
                  (ggml_type_اندازه(op->type) == حجم(عائم) || ggml_type_اندازه(op->type) == 2);
        حالة GGML_OP_REPEAT_پشت:
            أرجع op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_ROPE:
            أرجع ggml_is_contiguous_ردیفها(op) && ggml_is_contiguous_ردیفها(op->src[0]);
        حالة GGML_OP_ROPE_پشت:
        حالة GGML_OP_NONE:
        حالة GGML_OP_RESHAPE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_جایگشت:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_RMS_NORM:
            أرجع درست;
        حالة GGML_OP_گروه_NORM:
            أرجع ggml_is_contiguous(op->src[0]);
        حالة GGML_OP_NORM:
        حالة GGML_OP_L2_NORM:
            أرجع op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        حالة GGML_OP_جمع_کن:
        حالة GGML_OP_SUB:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
            أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        حالة GGML_OP_جمع_کن_ID:
            أرجع op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32;
        حالة GGML_OP_SILU_پشت:
        حالة GGML_OP_RMS_NORM_پشت:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_SQR:
        حالة GGML_OP_SQRT:
        حالة GGML_OP_SIN:
        حالة GGML_OP_COS:
        حالة GGML_OP_CLAMP:
        حالة GGML_OP_LEAKY_RELU:
            أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        حالة GGML_OP_OPT_گام_ADAMW:
        حالة GGML_OP_OPT_گام_SGD:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_بیرون_PROD:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && op->type == GGML_TYPE_F32;
        حالة GGML_OP_LOG:
        حالة GGML_OP_TRI:
        حالة GGML_OP_DIAG:
            أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        حالة GGML_OP_ARGSORT:
            {
                إذا (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    أرجع خطأ;
                }
                // pipeline_argsort_large_f32 requires vulkan memory model.
                إذا (device->vulkan_حافظه_model) {
                    أرجع درست;
                } وإلا {
                    أرجع op->ne[0] <= (1 << std::min(device->max_workgroup_اندازه_log2, رقم_argsort_pipelines - 1));
                }
            }
        حالة GGML_OP_بالاترین_K:
            {
                إذا (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    أرجع خطأ;
                }
                // large k falls back to radix-select
                ثابت uint32_t min_pipeline =
                    std::max((uint32_t) log2f(عائم(op->ne[0])) + 1, device->subgroup_اندازه_log2);
                إذا (min_pipeline < رقم_topk_pipelines && device->pipeline_topk_f32[min_pipeline]) {
                    أرجع درست;
                }
                أرجع device->pipeline_topk_radix_f32 != اشاره_تهی;
            }
        حالة GGML_OP_UPSCALE:
            إذا (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                إذا ((op->op_params[0] & 0xFF) != GGML_SCALE_نما_BILINEAR) {
                    أرجع خطأ;
                }
            }
            أرجع op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_ACC:
            أرجع op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        حالة GGML_OP_مجموعه:
            أرجع op->src[0]->type == op->src[1]->type && op->src[0]->type == op->type &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_I32);
        حالة GGML_OP_CONCAT: {
            أرجع ggml_vk_concat_supported(op->src[0], op->src[1], op);
        }
        حالة GGML_OP_ADD1:
            أرجع (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16);
        حالة GGML_OP_ARANGE:
            أرجع op->type == GGML_TYPE_F32;
        حالة GGML_OP_FILL:
            أرجع op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
        حالة GGML_OP_SCALE:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_PAD:
        حالة GGML_OP_PAD_REFLECT_1D:
        حالة GGML_OP_ROLL:
            أرجع op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_DIAG_ماسک_INF:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_SOFT_MAX:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && (!op->src[1] || (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16));
        حالة GGML_OP_SOFT_MAX_پشت:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32;
        حالة GGML_OP_جمع:
            أرجع op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguously_allocated(op->src[0]);
        حالة GGML_OP_جمع_ردیفها:
        حالة GGML_OP_میانگین:
            أرجع op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_ردیفها(op->src[0]);
        حالة GGML_OP_CUMSUM:
            {
                إذا (device->subgroup_arithmetic && device->subgroup_require_پر_support) {
                    أرجع op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_ردیفها(op->src[0]);
                }
                أرجع خطأ;
            }
        حالة GGML_OP_DSV4_HC_COMB:
        حالة GGML_OP_DSV4_HC_PRE:
        حالة GGML_OP_DSV4_HC_POST:
            {
                إذا (op->type != GGML_TYPE_F32) {
                    أرجع خطأ;
                }
                لكل (uint32_t i = 0; i < GGML_MAX_SRC; ++i) {
                    إذا (op->src[i] && op->src[i]->type != GGML_TYPE_F32) {
                        أرجع خطأ;
                    }
                }
                // hc is hardcoded to 4 in the shaders. ggml only constrains it
                // to 4 for COMB, so PRE/POST have to be checked here.
                إذا (op->op == GGML_OP_DSV4_HC_PRE && op->src[0]->ne[1] != 4) {
                    أرجع خطأ;
                }
                إذا (op->op == GGML_OP_DSV4_HC_POST && op->src[1]->ne[1] != 4) {
                    أرجع خطأ;
                }
                إذا (op->op == GGML_OP_DSV4_HC_COMB) {
                    أرجع device->pipeline_dsv4_hc_comb_f32 != اشاره_تهی;
                }
                أرجع درست;
            }
        حالة GGML_OP_SOLVE_TRI:
            {
                إذا (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
                    أرجع خطأ;
                }
                ثابت uint32_t N = op->src[0]->ne[0];
                ثابت uint32_t K = op->src[1]->ne[0];
                // K dimension limited to workgroup size
                إذا (K > 1u << device->max_workgroup_اندازه_log2) {
                    أرجع خطأ;
                }
                ثابت uint32_t batch_N = device->properties.limits.max_Compute_Shared_حافظه_اندازه / ((N + K) * حجم(عائم));

                إذا (batch_N == 0) {
                    أرجع خطأ;
                }
                أرجع درست;
            }
        حالة GGML_OP_ARGMAX:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_CROSS_ENTROPY_LOSS:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[0], op->src[1])
                && ggml_is_contiguous(op) && ggml_is_scalar(op) && op->type == GGML_TYPE_F32;
        حالة GGML_OP_CROSS_ENTROPY_LOSS_پشت:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32 && ggml_is_scalar(op->src[0])
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[2]) && op->src[2]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[1], op->src[2])
                && ggml_are_same_shape(op->src[1], op)
                && ggml_is_contiguous(op) && op->type == GGML_TYPE_F32;
        حالة GGML_OP_شمار_EQUAL:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_I32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_I32;
        حالة GGML_OP_IM2_ستون:
            أرجع ggml_is_contiguous(op->src[1])
                && op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        حالة GGML_OP_IM2_ستون_3_D:
            أرجع op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        حالة GGML_OP_TIMESTEP_EMBEDDING:
            أرجع op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_CONV_2D_DW:
            أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)
                && op->src[1]->type == GGML_TYPE_F32;
        حالة GGML_OP_POOL_1D:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_POOL_2D:
            أرجع ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_RWKV_WKV6:
        حالة GGML_OP_RWKV_WKV7:
            أرجع درست; // all inputs are contiguous, see ggml.c
        حالة GGML_OP_GATED_خطی_ATTN:
            // the shader block size is hardcoded to head_size 64
            أرجع op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->src[0]->ne[0] == 64;
        حالة GGML_OP_LIGHTNING_INDEXER:
            {
                ثابت ggml_tensor * q = op->src[0];
                ثابت ggml_tensor * k = op->src[1];
                ثابت ggml_tensor * w = op->src[2];
                ثابت ggml_tensor * m = op->src[3];

                // the q/w/m types and the shape relationships between q, k, w, m and dst
                // are already asserted in ggml_lightning_indexer()
                إذا (!ggml_vk_lightning_indexer_k_type_supported(k->type) || !device->fp16) {
                    أرجع خطأ;
                }

                // the shader block size is hardcoded to head size 128
                إذا (q->ne[0] != 128) {
                    أرجع خطأ;
                }

                // the shader indexes the buffers by element stride, and is dispatched
                // without allow_misalign
                لكل (ثابت ggml_tensor * t : {q, k, w, m, op}) {
                    إذا (t->nb[0] != ggml_type_اندازه(t->type) ||
                        (vk_tensor_جابه_جایی(t) + t->view_offs) % device->properties.limits.min_Storage_میانگیر_جابه_جایی_Alignment != 0) {
                        أرجع خطأ;
                    }
                    // the strides get scaled down from bytes, so the division must be exact
                    لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
                        إذا (t->nb[i] % ggml_type_اندازه(t->type) != 0) {
                            أرجع خطأ;
                        }
                    }
                }
                أرجع درست;
            }
        حالة GGML_OP_GATED_DELTA_NET:
            {
                ثابت uint32_t S_v = op->src[2]->ne[0];
                إذا (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128) {
                    أرجع خطأ;
                }
                لكل (صحيح i = 0; i < 6; i++) {
                    إذا (op->src[i] == اشاره_تهی || op->src[i]->type != GGML_TYPE_F32) {
                        أرجع خطأ;
                    }
                }
                أرجع op->type == GGML_TYPE_F32;
            }
        حالة GGML_OP_SSM_پیمایش:
            {
                لكل (صحيح i = 0; i < 6; i++) {
                    إذا (op->src[i] && ggml_is_quantized(op->src[i]->type)) {
                        أرجع خطأ;
                    }
                }
                إذا (op->src[6] && op->src[6]->type != GGML_TYPE_I32) {
                    أرجع خطأ;
                }
                إذا (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    أرجع خطأ;
                }

                ثابت uint32_t d_state = op->src[0]->ne[0];
                ثابت uint32_t سر_dim = op->src[0]->ne[1];

                منطقي is_mamba2 = (op->src[3] && op->src[3]->nb[1] == حجم(عائم));
                إذا (!is_mamba2) {
                    أرجع خطأ;
                }

                إذا ((d_state != 128 && d_state != 256) || سر_dim % 16 != 0) {
                    أرجع خطأ;
                }

                size_t shmem_اندازه = d_state * حجم(عائم);

                إذا (shmem_اندازه > device->properties.limits.max_Compute_Shared_حافظه_اندازه) {
                    أرجع خطأ;
                }

                إذا (!device->subgroup_basic) {
                    أرجع خطأ;
                }

                أرجع درست;
            }
        حالة GGML_OP_SSM_CONV:
            أرجع op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_CONV_TRANSPOSE_1D:
            أرجع op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        حالة GGML_OP_COL2IM_1D:
            أرجع (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16 ||
                    op->src[0]->type == GGML_TYPE_BF16) &&
                   op->type == op->src[0]->type &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op);
        حالة GGML_OP_CONV_2D:
        حالة GGML_OP_CONV_TRANSPOSE_2D:
            {
                ثابت منطقي transpose = op->op == GGML_OP_CONV_TRANSPOSE_2D;
                ثابت int64_t خروجی = !transpose ? op->src[0]->ne[3] : op->src[0]->ne[2];
                ثابت int64_t ورودی  = !transpose ? op->src[0]->ne[2] : op->src[0]->ne[3];

                // Channel-contiguous format is not supported yet.
                أرجع ((op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    (op->src[0]->nb[0] == حجم(عائم) || op->src[0]->nb[0] == حجم(ggml_fp16_t) ) &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->type == GGML_TYPE_F32 &&
                    خروجی == op->ne[2] &&
                    ورودی == op->src[1]->ne[2] &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op));
            }
        حالة GGML_OP_CONV_3D:
            أرجع (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op);
        افتراضي:
            أرجع خطأ;
    }

    استفادهنشده(dev);
}

ساكن منطقي ggml_backend_vk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_میانگیر_type_t buft) {
    إذا (buft->iface.get_name != ggml_backend_vk_میانگیر_type_name) {
        أرجع خطأ;
    }

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ggml_backend_vk_میانگیر_type_context * buft_ctx = (ggml_backend_vk_میانگیر_type_context *)buft->context;

    أرجع buft_ctx->device->idx == ctx->device;
}

ساكن منطقي ggml_backend_vk_device_offload_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_backend_vk_device_context * dev_ctx = (ggml_backend_vk_device_context *)dev->context;

    أرجع ggml_vk_get_op_batch_اندازه(op) >= dev_ctx->op_offload_min_batch_اندازه;
}

ساكن ggml_backend_event_t ggml_backend_vk_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    خودکار device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = جديد vk_event;
    إذا (!vkev) {
        أرجع اشاره_تهی;
    }

    // No events initially, they get created on demand
    vkev->has_event = خطأ;

    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.مجموعه_PNext(&tci);
    vkev->tl_semaphore = { device->device.createSemaphore(ci), 0 };

    أرجع جديد ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ vkev,
    };
}

ساكن فراغ ggml_backend_vk_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    خودکار device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = (vk_event *)event->context;

    device->device.destroySemaphore(vkev->tl_semaphore.s);
    لكل (خودکار& event : vkev->events_free) {
        device->device.destroyEvent(event);
    }
    لكل (خودکار& event : vkev->events_submitted) {
        device->device.destroyEvent(event);
    }
    إذا (vkev->has_event) {
        device->device.destroyEvent(vkev->event);
    }
    حذف vkev;
    حذف event;
}

ساكن فراغ ggml_backend_vk_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_device_event_synchronize(backend=" << dev << ", event=" << event << ")");
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    خودکار device = ggml_vk_get_device(ctx->device);
    vk_event *vkev = (vk_event *)event->context;

    // Only do something if the event has actually been used
    إذا (vkev->has_event) {
        vk::Semaphore sem = vkev->tl_semaphore.s;
        uint64_t val = vkev->tl_semaphore.value;
        vk::SemaphoreWaitInfo swi{vk::SemaphoreWaitFlags{}, sem, val};
        VK_CHECK(device->device.waitSemaphores(swi, UINT64_MAX), "event_synchronize", device);

        // Reset and move submitted events
        لكل (خودکار& event : vkev->events_submitted) {
            device->device.resetEvent(event);
        }
        vkev->events_free.insert(vkev->events_free.end(), vkev->events_submitted.begin(), vkev->events_submitted.end());
        vkev->events_submitted.clear();

        // Finished using current command buffer so we flag for reuse
        إذا (vkev->cmd_میانگیر) {
            // Only flag for reuse if it hasn't been reused already
            إذا (vkev->cmd_میانگیر_use_counter == vkev->cmd_میانگیر->use_counter) {
                vkev->cmd_میانگیر->درون_use = خطأ;
                vkev->cmd_میانگیر->buf.reset();
            }
            vkev->cmd_میانگیر = اشاره_تهی;
        }
    }
}

ساكن ggml_backend_میانگیر_t ggml_backend_vk_device_میانگیر_from_host_ptr(ggml_backend_dev_t dev, فراغ * ptr, size_t size, size_t max_tensor_اندازه) {
    VK_LOG_DEBUG("ggml_backend_vk_device_buffer_from_host_ptr(backend=" << dev << ", ptr=" << ptr << ", size=" << size << ")");
    GGML_استفادهنشده(max_tensor_اندازه);

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    خودکار device = ggml_vk_get_device(ctx->device);

    vk_میانگیر buf = ggml_vk_میانگیر_from_host_ptr(device, ptr, size);

    إذا (!buf) {
        أرجع {};
    }

    ggml_backend_vk_میانگیر_context * bufctx = جديد ggml_backend_vk_میانگیر_context(device, std::move(buf), device->name);

    ggml_backend_میانگیر_t ret = ggml_backend_میانگیر_init(ggml_backend_vk_device_get_میانگیر_type(dev), ggml_backend_vk_میانگیر_interface, bufctx, size);

    أرجع ret;
}

ساكن ثابت بنية ggml_backend_device_i ggml_backend_vk_device_i = {
    /* .get_name             = */ ggml_backend_vk_device_get_name,
    /* .get_description      = */ ggml_backend_vk_device_get_description,
    /* .get_memory           = */ ggml_backend_vk_device_get_حافظه,
    /* .get_type             = */ ggml_backend_vk_device_get_type,
    /* .get_props            = */ ggml_backend_vk_device_get_props,
    /* .init_backend         = */ ggml_backend_vk_device_init,
    /* .get_buffer_type      = */ ggml_backend_vk_device_get_میانگیر_type,
    /* .get_host_buffer_type = */ ggml_backend_vk_device_get_host_میانگیر_type,
    /* .buffer_from_host_ptr = */ ggml_backend_vk_device_میانگیر_from_host_ptr,
    /* .supports_op          = */ ggml_backend_vk_device_supports_op,
    /* .supports_buft        = */ ggml_backend_vk_device_supports_buft,
    /* .offload_op           = */ ggml_backend_vk_device_offload_op,
    /* .event_new            = */ ggml_backend_vk_device_event_new,
    /* .event_free           = */ ggml_backend_vk_device_event_free,
    /* .event_synchronize    = */ ggml_backend_vk_device_event_synchronize,
};

ساكن ثابت حرف * ggml_backend_vk_reg_get_name(ggml_backend_reg_t reg) {
    استفادهنشده(reg);
    أرجع GGML_VK_NAME;
}

ساكن size_t ggml_backend_vk_reg_get_device_شمار(ggml_backend_reg_t reg) {
    استفادهنشده(reg);
    أرجع ggml_backend_vk_get_device_شمار();
}

ساكن ggml_backend_dev_t ggml_backend_vk_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    ساكن std::vector<ggml_backend_dev_t> devices;

    ساكن منطقي initialized = خطأ;

    {
        ساكن std::mutex mutex;
        std::lock_guard<std::mutex> قفل(mutex);
        إذا (!initialized) {
            ثابت صحيح min_batch_اندازه = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
            لكل (صحيح i = 0; i < ggml_backend_vk_get_device_شمار(); i++) {
                ggml_backend_vk_device_context * ctx = جديد ggml_backend_vk_device_context;
                حرف desc[256];
                ggml_backend_vk_get_device_description(i, desc, حجم(desc));
                ctx->device = i;
                ctx->name = GGML_VK_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->is_integrated_gpu = ggml_backend_vk_get_device_type(i) == vk::PhysicalDeviceType::eIntegratedGpu;
                ctx->pci_bus_id = ggml_backend_vk_get_device_pci_id(i);
                ctx->op_offload_min_batch_اندازه = min_batch_اندازه;
                devices.push_back(جديد ggml_backend_device {
                    /* .iface   = */ ggml_backend_vk_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                });
            }
            initialized = درست;
        }
    }

    GGML_ASSERT(device < devices.size());
    أرجع devices[device];
}

ساكن ثابت بنية ggml_backend_reg_i ggml_backend_vk_reg_i = {
    /* .get_name         = */ ggml_backend_vk_reg_get_name,
    /* .get_device_count = */ ggml_backend_vk_reg_get_device_شمار,
    /* .get_device       = */ ggml_backend_vk_reg_get_device,
    /* .get_proc_address = */ عدم,
};

ggml_backend_reg_t ggml_backend_vk_reg() {
    ساكن ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_نسخه,
        /* .iface       = */ ggml_backend_vk_reg_i,
        /* .context     = */ اشاره_تهی,
    };
    حاول {
        ggml_vk_instance_init();
        أرجع &reg;
    } امسك (ثابت vk::SystemError& e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: System error: " << e.what());
        أرجع اشاره_تهی;
    } امسك (ثابت std::exception &e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: " << e.what());
        أرجع اشاره_تهی;
    } امسك (...) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: unknown exception during Vulkan init");
        أرجع اشاره_تهی;
    }
}

منطقي ggml_vk_instance_layer_settings_موجود() {
#اگر_تعریف_شده GGML_VULKAN_VALIDATE
    // Check if validation layer provides the extension
    ثابت std::string layer_name = "VK_LAYER_KHRONOS_validation";
    لكل (ثابت خودکار& layer : vk::enumerateInstanceLayerProperties()) {
        إذا (layer_name == layer.layerName.داده()) {
            لكل (ثابت خودکار& ext : vk::enumerateInstanceExtensionProperties(layer_name)) {
                إذا (strcmp("VK_EXT_layer_settings", ext.extensionName.داده()) == 0) {
                    أرجع درست;
                }
            }
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Validation layer or layer extension VK_EXT_layer_settings not found." << std::پایان_سطر;
#پایان_اگر
    أرجع خطأ;
}

منطقي ggml_vk_instance_portability_enumeration_ext_موجود(ثابت std::vector<vk::ExtensionProperties>& instance_extensions) {
#اگر_تعریف_شده __APPLE__
    // Check for portability enumeration extension for MoltenVK support
    لكل (ثابت خودکار& properties : instance_extensions) {
        إذا (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            أرجع درست;
        }
    }
    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::پایان_سطر;
#پایان_اگر
    أرجع خطأ;

    استفادهنشده(instance_extensions);
}

منطقي ggml_vk_instance_debug_utils_ext_موجود(
    ثابت std::vector<vk::ExtensionProperties> & instance_extensions) {
    // Check for portability enumeration extension for MoltenVK support
    لكل (ثابت خودکار & properties : instance_extensions) {
        إذا (strcmp("VK_EXT_debug_utils", properties.extensionName) == 0) {
            أرجع درست;
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found." << std::پایان_سطر;
    أرجع خطأ;

    استفادهنشده(instance_extensions);
}

منطقي ggml_vk_device_is_supported(ثابت vk::PhysicalDevice & vkdev) {
    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_بعدی = اشاره_تهی;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_بعدی = &vk11_features;

    vkGetPhysicalDeviceFeatures2(vkdev, &device_features2);

    أرجع vk11_features.storage_Buffer16_بیت_Access;
}

منطقي ggml_vk_khr_cooperative_ماتریس_support(ثابت vk::PhysicalDeviceProperties& props, ثابت vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch) {
    بدل (props.vendorID) {
    حالة VK_VENDOR_ID_INTEL:
        // Only allowing Xe2/Xe3 GPU and integrated Xe GPUs at the moment since older hardware (ex. Arc A770) has performance regressions.
        أرجع (arch == vk_device_architecture::INTEL_XE2) ||
            (arch == vk_device_architecture::INTEL_XE1 && props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu && driver_props.driverID == vk::DriverId::eIntelProprietaryWindows);
    حالة VK_VENDOR_ID_AMD:
        إذا (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::e_Amd_Open_منبع) {
            // Workaround for AMD proprietary driver reporting support on all GPUs
            أرجع arch == vk_device_architecture::AMD_RDNA3 || arch == vk_device_architecture::AMD_RDNA4;
        }
        أرجع درست;
    حالة VK_VENDOR_ID_QUALCOMM:
        // Only allow Adreno GPUs with hardware matrix cores (Gen 6+).
        أرجع arch == vk_device_architecture::QUALCOMM_ADRENO;
    افتراضي:
        أرجع درست;
    }
}

uint32_t ggml_vk_intel_shader_core_شمار(ثابت vk::PhysicalDevice& vkdev) {
    VkPhysicalDeviceProperties2 props = vkdev.getProperties2();

    إذا (props.properties.vendorID != VK_VENDOR_ID_INTEL) {
        أرجع 0;
    }

    ثابت uint32_t device_id = props.properties.deviceID;

    بدل (device_id) {
    حالة 0x56A6:  // A310
        أرجع 6;
    حالة 0x5693:  // A370M
    حالة 0x56A5:  // A380
    حالة 0x56B1:  // Pro A40/A50
        أرجع 8;
    حالة 0x5697:  // A530M
        أرجع 12;
    حالة 0x5692:  // A550M
    حالة 0x56B3:  // Pro A60
        أرجع 16;
    حالة 0x56A2:  // A580
        أرجع 24;
    حالة 0x5691:  // A730M
    حالة 0x56A1:  // A750
        أرجع 28;
    حالة 0x56A0:  // A770
    حالة 0x5690:  // A770M
        أرجع 32;
    حالة 0xE212:  // Pro B50
        أرجع 16;
    حالة 0xE20C:  // B570
        أرجع 18;
    حالة 0xE20B:  // B580
    حالة 0xE211:  // Pro B60
    حالة 0xE222:  // Pro B65
        أرجع 20;
    حالة 0xE223:  // Pro B70
        أرجع 32;
    حالة 0xB080:  // PTL Xe3 LPG 2x6 (12 subslices)
        أرجع 12;
    افتراضي:
        أرجع 0;
    }
}

منطقي ggml_vk_intel_windows_driver_درون_range(uint32_t driver_نسخه, uint32_t پایینی_major, uint32_t پایینی_minor, uint32_t فوقانی_major, uint32_t فوقانی_minor) {
#إذا defined(_WIN32)
    // Intel Windows encodes xxx.yyyy as [31:14].[13:0].
    ثابت uint32_t major = driver_نسخه >> 14;
    ثابت uint32_t minor = driver_نسخه & 0x3fff;

    ثابت منطقي ge_پایینی = major > پایینی_major || (major == پایینی_major && minor >= پایینی_minor);
    ثابت منطقي lt_فوقانی = major < فوقانی_major || (major == فوقانی_major && minor < فوقانی_minor);

    أرجع ge_پایینی && lt_فوقانی;
#وإلا
    GGML_استفادهنشده(driver_نسخه);
    GGML_استفادهنشده(پایینی_major);
    GGML_استفادهنشده(پایینی_minor);
    GGML_استفادهنشده(فوقانی_major);
    GGML_استفادهنشده(فوقانی_minor);
    أرجع درست;
#پایان_اگر
}

GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)


// out-of-lined header method definitions

فراغ vk_صف_handle_synchronized::submit(vk::ArrayProxy<ثابت vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    إذا (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    std::lock_guard<std::mutex> guard(mutex);
    حاول {
        queue.submit(submits, fence);
    } امسك (vk::Device_گمشده_Error &) {
        إذا (خودکار dev = device.قفل()) {
            ggml_vk_print_device_گمشده_info(dev);
        }
        ارم;
    }
}

فراغ vk_صف_handle_unsynchronized::submit(vk::ArrayProxy<ثابت vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    إذا (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    حاول {
        queue.submit(submits, fence);
    } امسك (vk::Device_گمشده_Error &) {
        إذا (خودکار dev = device.قفل()) {
            ggml_vk_print_device_گمشده_info(dev);
        }
        ارم;
    }
}

vk_device_struct::~vk_device_struct() {
    // the build failed before the device was created, nothing to destroy
    إذا (device == VK_NULL_HANDLE) {
        أرجع;
    }

    VK_LOG_DEBUG("destroy device " << name);

    device.destroyFence(fence);

    ggml_vk_destroy_میانگیر(sync_staging);

    إذا (compute_صف) compute_صف->cmd_pool.destroy(device);
    إذا (transfer_صف) transfer_صف->cmd_pool.destroy(device);

    // Explicitly clear to ensure queues drop their shared_ptrs to handles
    // before the Vulkan logical device instance is destroyed
    compute_صف.reset();
    transfer_صف.reset();

    لكل (خودکار& pipeline : all_pipelines) {
        إذا (pipeline.expired()) {
            استمر;
        }

        vk_pipeline pl = pipeline.قفل();
        ggml_vk_destroy_pipeline(device, pl);
    }
    all_pipelines.clear();

    device.destroy_Descriptor_مجموعه_Layout(dsl);

    device.destroy();
}

فراغ vk_perf_logger::print_timings(منطقي force) {
    إذا (timings.خالی()) {
        أرجع;
    }
    print_شمار++;
    إذا ((print_شمار % vk_perf_logger_frequency) != 0 && !force) {
        أرجع;
    }
    print_شمار = 0;
    uint64_t مجموع_all_op_times = 0;
    std::cerr << "----------------\nVulkan Timings:" << std::پایان_سطر;
    لكل (ثابت خودکار & t : timings) {
        uint64_t مجموع_op_times = 0;
        لكل (ثابت خودکار & زمان : t.second) {
            مجموع_op_times += زمان;
        }
        std::cerr << t.اولی << ": " << t.second.size() << " x " << (مجموع_op_times / t.second.size() / 1000.0)
                  << " us = " << (مجموع_op_times / 1000.0) << " us";

        // If we have as many flops entries as timing entries for the op, then compute and log the flops/S.
        خودکار it = flops.find(t.اولی);
        إذا (it != flops.end() && (it->second).size() == t.second.size()) {
            uint64_t مجموع_op_flops = 0;
            لكل (ثابت خودکار & elem : it->second) {
                مجموع_op_flops += elem;
            }
            std::cerr << " ("
                      << (مزدوج(مجموع_op_flops) / (1000.0 * 1000.0 * 1000.0)) /
                             (مزدوج(مجموع_op_times) / (1000.0 * 1000.0 * 1000.0))
                      << " GFLOPS/s)";
        }

        مجموع_all_op_times += مجموع_op_times;

        std::cerr << std::پایان_سطر;
    }

    إذا (timings.size() > 0) {
        std::cerr << "Total time: " << مجموع_all_op_times / 1000.0 << " us." << std::پایان_سطر;
    }

    timings.clear();
    flops.clear();
}

std::string vk_perf_logger::get_گره_fusion_name(ثابت ggml_tensor * گره, ثابت حرف *fusion_name, uint64_t *n_flops) {
    *n_flops = ggml_vk_get_گره_flops(گره);
    std::string fusion_str;
    إذا (fusion_name) {
        fusion_str = fusion_name + std::string(" ");
    }
    إذا (گره->op == GGML_OP_UNARY) {
        أرجع fusion_str + ggml_unary_op_name(ggml_get_unary_op(گره));
    }
    إذا (گره->op == GGML_OP_MUL_MAT || گره->op == GGML_OP_MUL_MAT_ID) {
        ثابت uint64_t m     = گره->ne[0];
        ثابت uint64_t n     = گره->ne[1];
        ثابت uint64_t k     = گره->src[1]->ne[0];
        ثابت uint64_t batch = گره->ne[2] * گره->ne[3];
        std::string    name  = ggml_op_name(گره->op);
        إذا ((گره->op == GGML_OP_MUL_MAT && n <= mul_mat_vec_max_تعداد_ستون) ||
            (گره->op == GGML_OP_MUL_MAT_ID && گره->src[2]->ne[1] == 1)) {
            name += "_VEC";
        }
        name += " ";
        name += ggml_type_name(گره->src[0]->type);
        name += " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
        إذا (گره->op == GGML_OP_MUL_MAT_ID) {
            name += " n_expert=" + std::to_string(گره->src[0]->ne[2]);
        }
        إذا (batch > 1) {
            name += " batch=" + std::to_string(batch);
        }
        أرجع fusion_str + name;
    }
    إذا (گره->op == GGML_OP_CONV_2D || گره->op == GGML_OP_CONV_TRANSPOSE_2D) {
        std::string   name    = ggml_op_name(گره->op);
        ثابت ggml_tensor * knl = گره->src[0];
        uint64_t      Cout    = گره->ne[2];
        uint64_t      اندازه_K  = گره->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        uint64_t      اندازه_N_2  = گره->ne[3] * گره->ne[0] * گره->ne[1];
        name += " M=Cout=" + std::to_string(Cout) + ", K=Cin*KW*KH=" + std::to_string(اندازه_K) +
                ", N=N*OW*OH=" + std::to_string(اندازه_N_2);
        أرجع fusion_str + name;
    }
    إذا (گره->op == GGML_OP_RMS_NORM) {
        std::string   name    = ggml_op_name(گره->op);
        name += "(" + std::to_string(گره->ne[0]) + "," + std::to_string(گره->ne[1]) + "," + std::to_string(گره->ne[2]) + "," + std::to_string(گره->ne[3]) + ")";
        أرجع fusion_str + name;
    }
    إذا (گره->op == GGML_OP_FLASH_ATTN_EXT) {
        ثابت ggml_tensor * dst = گره;
        ثابت ggml_tensor * q = گره->src[0];
        ثابت ggml_tensor * k = گره->src[1];
        ثابت ggml_tensor * v = گره->src[2];
        ثابت ggml_tensor * m = گره->src[3];
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(گره->op) <<
            " dst(" << dst->ne[0] << "," << dst->ne[1] << "," << dst->ne[2] << "," << dst->ne[3] << "), " <<
            " q(" << q->ne[0] << "," << q->ne[1] << "," << q->ne[2] << "," << q->ne[3] << "), " <<
            " k(" << k->ne[0] << "," << k->ne[1] << "," << k->ne[2] << "," << k->ne[3] << "), " <<
            " v(" << v->ne[0] << "," << v->ne[1] << "," << v->ne[2] << "," << v->ne[3] << "), " <<
            " m(" << (m?m->ne[0]:0) << "," << (m?m->ne[1]:0) << "," << (m?m->ne[2]:0) << "," << (m?m->ne[3]:0) << ")";
        أرجع name.str();
    }
    إذا (گره->op == GGML_OP_بالاترین_K) {
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(گره->op) <<
            " K=" << گره->ne[0] <<
            " (" << گره->src[0]->ne[0] << "," << گره->src[0]->ne[1] << "," << گره->src[0]->ne[2] << "," << گره->src[0]->ne[3] << ")";
        أرجع name.str();
    }
    أرجع fusion_str + ggml_op_name(گره->op);
}

ggml_backend_vk_میانگیر_context::~ggml_backend_vk_میانگیر_context() {
    ggml_vk_destroy_میانگیر(dev_میانگیر);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, ثابت std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2) {
    إذا (!vk_instance.debug_utils_support || ctx->s == اشاره_تهی) {
        أرجع;
    }
    begin(ctx, pipeline_name + " (" + std::to_string(wg0) + "," + std::to_string(wg1) + "," + std::to_string(wg2) + ")");
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, ثابت ggml_cgraph * cgraph, صحيح گره_idx, صحيح n_fused) {
    إذا (!vk_instance.debug_utils_support || ctx->s == اشاره_تهی) {
        أرجع;
    }
    std::string name = ggml_op_name(cgraph->nodes[گره_idx]->op);
    لكل (صحيح i = 1; i <= n_fused; i++) {
        name += "+";
        name += ggml_op_name(cgraph->nodes[گره_idx + i]->op);
    }
    name += " ";
    name += cgraph->nodes[گره_idx]->name;
    begin(ctx, name);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_صف_handle * handle, ثابت حرف * name) {
    إذا (!vk_instance.debug_utils_support || handle == اشاره_تهی) {
        أرجع;
    }
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<عائم, 4>{1.0f, 1.0f, 1.0f, 1.0f};

    qhandle = handle;
    std::lock_guard<vk_صف_handle> guard(*qhandle);
    vk_instance.pfn_vk_صف_Begin_Debug_Utils_Label_EXT(qhandle->queue, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

فراغ ggml_vk_debug_label::close() {
    إذا (subctx != اشاره_تهی) {
        // close on the current command buffer, which may differ from the one begin used
        إذا (subctx->s != اشاره_تهی) {
            vk_instance.pfn_vk_Cmd_انتها_Debug_Utils_Label_EXT(subctx->s->میانگیر->buf);
        }
        subctx->debug_labels.pop_back();
        subctx = اشاره_تهی;
    }
    إذا (qhandle != اشاره_تهی) {
        std::lock_guard<vk_صف_handle> guard(*qhandle);
        vk_instance.pfn_vk_صف_انتها_Debug_Utils_Label_EXT(qhandle->queue);
        qhandle = اشاره_تهی;
    }
}

فراغ ggml_vk_debug_label::begin(vk_context & ctx, ثابت std::string & name) {
    إذا (!vk_instance.debug_utils_support || ctx->s == اشاره_تهی) {
        أرجع;
    }
    subctx = ctx.get();
    subctx->debug_labels.push_back(name);
    ggml_vk_cmd_label_begin(subctx->s->میانگیر->buf, subctx->debug_labels.back().c_str());
}

