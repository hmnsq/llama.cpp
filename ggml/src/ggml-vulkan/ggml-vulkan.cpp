#అన్తర్భావ "ggml-vulkan-common.h"

స్థిర vk_device_architecture get_device_architecture(స్థిరరాశి vk::PhysicalDevice& device) {
    vk::PhysicalDeviceProperties props = device.getProperties();

    ఒకవేళ (props.vendorID == VK_VENDOR_ID_AMD) {
        స్థిరరాశి std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        తార్కిక amd_shader_core_properties = అబద్ధం;
        తార్కిక integer_dot_లబ్ధం = అబద్ధం;
        తార్కిక subgroup_కొలత_control = అబద్ధం;
        తార్కిక shader_float8 = అబద్ధం;

        కొరకు (స్థిరరాశి స్వయంచాలక& properties : ext_props) {
            ఒకవేళ (strcmp("VK_AMD_shader_core_properties", properties.extensionName) == 0) {
                amd_shader_core_properties = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_లబ్ధం = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_కొలత_control = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_EXT_shader_float8", properties.extensionName) == 0) {
                shader_float8 = నిజం;
            }
        }

        ఒకవేళ (!amd_shader_core_properties || !integer_dot_లబ్ధం || !subgroup_కొలత_control) {
            తిరిగి vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceShaderCorePropertiesAMD shader_core_props_amd;
        vk::Physical_Device_Shader_Integer_Dot_లబ్ధం_Properties_KHR integer_dot_props;
        vk::Physical_Device_Subgroup_కొలత_Control_Properties_EXT subgroup_కొలత_control_props;

        props2.p_తదుపరి = &shader_core_props_amd;
        shader_core_props_amd.p_తదుపరి = &integer_dot_props;
        integer_dot_props.p_తదుపరి = &subgroup_కొలత_control_props;

        device.getProperties2(&props2);

        ఒకవేళ (subgroup_కొలత_control_props.max_Subgroup_కొలత == 64 && subgroup_కొలత_control_props.min_Subgroup_కొలత == 64) {
            తిరిగి vk_device_architecture::AMD_GCN;
        }
        ఒకవేళ (subgroup_కొలత_control_props.max_Subgroup_కొలత == 64 && subgroup_కొలత_control_props.min_Subgroup_కొలత == 32) {
            // RDNA
            ఒకవేళ (shader_core_props_amd.wavefronts_చొప్పున_Simd == 20) {
                తిరిగి vk_device_architecture::AMD_RDNA1;
            }
            ఒకవేళ (shader_float8) {
                తిరిగి vk_device_architecture::AMD_RDNA4;
            }
            ఒకవేళ (integer_dot_props.integer_Dot_Product4x8_బిట్టు_Packed_Mixed_Signedness_Accelerated) {
                తిరిగి vk_device_architecture::AMD_RDNA3;
            }
            తిరిగి vk_device_architecture::AMD_RDNA2;
        }
    } లేకపోతే ఒకవేళ (props.vendorID == VK_VENDOR_ID_INTEL) {
        స్థిరరాశి std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        తార్కిక subgroup_కొలత_control = అబద్ధం;
        తార్కిక integer_dot_లబ్ధం = అబద్ధం;

        కొరకు (స్థిరరాశి స్వయంచాలక& properties : ext_props) {
            ఒకవేళ (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_కొలత_control = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_లబ్ధం = నిజం;
            }
        }

        ఒకవేళ (!subgroup_కొలత_control || !integer_dot_లబ్ధం) {
            తిరిగి vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::Physical_Device_Subgroup_కొలత_Control_Properties_EXT subgroup_కొలత_control_props;
        vk::Physical_Device_Shader_Integer_Dot_లబ్ధం_Properties_KHR integer_dot_props;

        props2.p_తదుపరి = &subgroup_కొలత_control_props;
        subgroup_కొలత_control_props.p_తదుపరి = &integer_dot_props;
        device.getProperties2(&props2);

        ఒకవేళ (subgroup_కొలత_control_props.min_Subgroup_కొలత == 16) {
            // Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8.
            // Minimum subgroup size matches the SIMD width so we distinguish architecture by checking this value.
            // https://www.intel.com/content/www/us/en/content-details/824434/2024-intel-tech-tour-xe2-and-lunar-lake-s-gpu.html
            // https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
            తిరిగి vk_device_architecture::INTEL_XE2;
        } లేకపోతే ఒకవేళ (subgroup_కొలత_control_props.min_Subgroup_కొలత == 8 &&
                 integer_dot_లబ్ధం && integer_dot_props.integer_Dot_Product4x8_బిట్టు_Packed_Signed_Accelerated) {
            తిరిగి vk_device_architecture::INTEL_XE1;
        }
    } లేకపోతే ఒకవేళ (props.vendorID == VK_VENDOR_ID_NVIDIA) {
        స్థిరరాశి std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        తార్కిక cooperative_మాతృక = అబద్ధం;
        తార్కిక sm_builtins = అబద్ధం;

        // Detect "pre-turing" based on lack of coopmat support.
        కొరకు (స్థిరరాశి స్వయంచాలక& properties : ext_props) {
            ఒకవేళ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_మాతృక = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = నిజం;
            }
        }

        ఒకవేళ (!cooperative_మాతృక) {
            తిరిగి vk_device_architecture::NVIDIA_PRE_TURING;
        }

        ఒకవేళ (sm_builtins) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;

            props2.p_తదుపరి = &sm_props;

            device.getProperties2(&props2);

            // Turing has 32, following architectures have 48
            ఒకవేళ (sm_props.shader_Warps_చొప్పున_SM == 32) {
                తిరిగి vk_device_architecture::NVIDIA_TURING;
            }
        }
    } లేకపోతే ఒకవేళ(props.vendorID == VK_VENDOR_ID_QUALCOMM){
        స్థిరరాశి std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        తార్కిక cooperative_మాతృక = అబద్ధం;
        తార్కిక cooperative_మాతృక_conversion = అబద్ధం;

        కొరకు (స్థిరరాశి స్వయంచాలక& properties : ext_props) {
            ఒకవేళ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_మాతృక = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_QCOM_cooperative_matrix_conversion", properties.extensionName) == 0) {
                cooperative_మాతృక_conversion = నిజం;
            }
        }

        ఒకవేళ (cooperative_మాతృక && cooperative_మాతృక_conversion) {
            తిరిగి vk_device_architecture::QUALCOMM_ADRENO;
        }
    }
    తిరిగి vk_device_architecture::OTHER;
}

తార్కిక ggml_vk_lightning_indexer_k_type_supported(ggml_type type) {
    తిరిగి std::find(lightning_indexer_k_types.begin(), lightning_indexer_k_types.end(), type) != lightning_indexer_k_types.end();
}
శూన్య ggml_vk_print_device_fault_info(స్థిరరాశి vk_device& device) {
    ఒకవేళ (!device->device_fault || !device->pfn_vkGetDeviceFaultInfoEXT) {
        తిరిగి;
    }

    VkDeviceFaultCountsEXT fault_counts {};
    fault_counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    Vk_ఫలితం res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, శూన్యసూచిక);
    ఒకవేళ (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (counts) failed: %d\n", res);
        తిరిగి;
    }

    std::vector<Vk_Device_Fault_చిరునామా_Info_EXT> చిరునామా_infos(fault_counts.చిరునామా_Info_లెక్క);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor_infos(fault_counts.vendor_Info_లెక్క);

    VkDeviceFaultInfoEXT fault_info {};
    fault_info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    fault_info.p_చిరునామా_Infos = చిరునామా_infos.దత్తాంశం();
    fault_info.pVendorInfos = vendor_infos.దత్తాంశం();

    res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, &fault_info);
    ఒకవేళ (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (info) failed: %d\n", res);
        తిరిగి;
    }

    ఒకవేళ (fault_counts.చిరునామా_Info_లెక్క == 0 && fault_counts.vendor_Info_లెక్క == 0 && fault_info.description[0] == '\0') {
        తిరిగి;
    }

    ఒకవేళ (fault_info.description[0] != '\0') {
        GGML_LOG_ERROR("ggml_vulkan: device fault on %s: %s\n", device->name.c_str(), fault_info.description);
    }

    కొరకు (uint32_t i = 0; i < fault_counts.చిరునామా_Info_లెక్క; i++) {
        స్థిరరాశి స్వయంచాలక& info = చిరునామా_infos[i];
        GGML_LOG_CONT("  address fault %u: type=%d address=0x%llx precision=0x%llx\n",
                i, (పూర్ణాంకం)info.చిరునామా_Type,
                (గుర్తులేని దీర్ఘ దీర్ఘ)info.reported_చిరునామా,
                (గుర్తులేని దీర్ఘ దీర్ఘ)info.చిరునామా_Precision);
    }
    కొరకు (uint32_t i = 0; i < fault_counts.vendor_Info_లెక్క; i++) {
        స్థిరరాశి స్వయంచాలక& info = vendor_infos[i];
        GGML_LOG_CONT("  vendor fault %u: %s (code=0x%llx data=0x%llx)\n",
                i, info.description,
                (గుర్తులేని దీర్ఘ దీర్ఘ)info.vendor_Fault_సంకేతం,
                (గుర్తులేని దీర్ఘ దీర్ఘ)info.vendor_Fault_దత్తాంశం);
    }
}
uint64_t ggml_vk_get_కణుపు_flops(స్థిరరాశి ggml_tensor * కణుపు) {
    ఒకవేళ (కణుపు->op == GGML_OP_MUL_MAT || కణుపు->op == GGML_OP_MUL_MAT_ID) {
        స్థిరరాశి uint64_t m     = కణుపు->ne[0];
        స్థిరరాశి uint64_t n     = కణుపు->ne[1];
        స్థిరరాశి uint64_t k     = కణుపు->src[1]->ne[0];
        స్థిరరాశి uint64_t batch = కణుపు->ne[2] * కణుపు->ne[3];
        తిరిగి m * n * (k + (k - 1)) * batch;
    }
    ఒకవేళ (కణుపు->op == GGML_OP_CONV_2D || కణుపు->op == GGML_OP_CONV_TRANSPOSE_2D) {
        స్థిరరాశి ggml_tensor * knl = కణుపు->src[0];
        స్థిరరాశి uint64_t Cout  = కణుపు->ne[2];
        స్థిరరాశి uint64_t కొలత_K = కణుపు->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        స్థిరరాశి uint64_t కొలత_N_2 = కణుపు->ne[3] * కణుపు->ne[0] * కణుపు->ne[1];
        తిరిగి Cout * కొలత_N_2 * (కొలత_K + (కొలత_K - 1));
    }
    ఒకవేళ (కణుపు->op == GGML_OP_CONV_3D) {
        స్థిరరాశి ggml_tensor * knl = కణుపు->src[0];
        స్థిరరాశి uint64_t OC     = ggml_get_op_params_i32(కణుపు, 11);
        స్థిరరాశి uint64_t IC     = ggml_get_op_params_i32(కణుపు, 9);
        స్థిరరాశి uint64_t కొలత_K = IC * knl->ne[0] * knl->ne[1] * knl->ne[2];
        స్థిరరాశి uint64_t కొలత_N_2 = కణుపు->ne[3] / OC * కణుపు->ne[0] * కణుపు->ne[1] * కణుపు->ne[2];
        తిరిగి OC * కొలత_N_2 * (కొలత_K + (కొలత_K - 1));
    }
    ఒకవేళ (కణుపు->op == GGML_OP_FLASH_ATTN_EXT) {
        స్థిరరాశి ggml_tensor * q = కణుపు->src[0];
        స్థిరరాశి ggml_tensor * k = కణుపు->src[1];
        స్థిరరాశి ggml_tensor * v = కణుపు->src[2];
        తిరిగి 2ull * q->ne[1] * q->ne[2] * (k->ne[0] + v->ne[0]) * k->ne[1] * q->ne[3];
    }
    తిరిగి 0;
}
శూన్య ggml_vk_print_కణుపు_list(స్థిరరాశి ggml_cgraph * cgraph, పూర్ణాంకం start, పూర్ణాంకం end) {
    uint64_t మొత్తం_flops = 0;
    పూర్ణాంకం n_ops = 0;
    కొరకు (పూర్ణాంకం j = start; j <= end && j < cgraph->n_nodes; j++) {
        uint64_t flops = ggml_vk_get_కణుపు_flops(cgraph->nodes[j]);
        మొత్తం_flops += flops;
        n_ops++;
        ఒకవేళ (flops > 0) {
            GGML_LOG_CONT("  node %d: %s (%s) [%.2f GFLOP]\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op),
                    flops / 1e9);
        } లేకపోతే {
            GGML_LOG_CONT("  node %d: %s (%s)\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op));
        }
    }
    GGML_LOG_CONT("  total: %d ops, %.2f GFLOP\n", n_ops, మొత్తం_flops / 1e9);
}
శూన్య ggml_vk_print_device_కోల్పోయిన_info(స్థిరరాశి vk_device& device) {
    ggml_vk_print_device_fault_info(device);
    ఒకవేళ (device->serialize_submissions && device->diag_cgraph != శూన్యసూచిక && device->diag_prev_ప్రారంభం >= 0) {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s, likely caused by previous submission (nodes %d to %d):\n",
                device->name.c_str(), device->diag_prev_ప్రారంభం, device->diag_prev_చివర);
        ggml_vk_print_కణుపు_list(device->diag_cgraph, device->diag_prev_ప్రారంభం, device->diag_prev_చివర);
    } లేకపోతే {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s\n", device->name.c_str());
    }
}
శూన్య * స్థిరరాశి vk_ptr_పునాది = (శూన్య *)(uintptr_t) 0x1000;  // NOLINT

uint64_t vk_tensor_స్థానభ్రంశం(స్థిరరాశి ggml_tensor * tensor) {
    ఒకవేళ (tensor->view_src) {
        తిరిగి (uint8_t *) tensor->view_src->దత్తాంశం - (uint8_t *) vk_ptr_పునాది;
    }
    తిరిగి (uint8_t *) tensor->దత్తాంశం - (uint8_t *) vk_ptr_పునాది;
}

size_t ggml_vk_tensor_బఫరు_స్థానభ్రంశం(స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor * t) {
    // vk_tensor_offset() is relative to vk_ptr_base, but mapped host tensors need an offset relative to their Vulkan buffer.
    ఒకవేళ (ctx->device->uma) {
        vk_బఫరు buf = శూన్యసూచిక;
        size_t off = 0;
        ggml_vk_host_get(ctx->device, t->దత్తాంశం, buf, off);
        ఒకవేళ (buf) {
            తిరిగి off;
        }
    }
    తిరిగి (size_t)(vk_tensor_స్థానభ్రంశం(t) + t->view_offs);
}
size_t ggml_vk_descriptor_స్థానభ్రంశం(size_t tensor_స్థానభ్రంశం, size_t alignment, size_t type_కొలత) {
    // Move the descriptor back until its distance to the tensor is divisible by the tensor type size.
    size_t descriptor_స్థానభ్రంశం = tensor_స్థానభ్రంశం & ~(alignment - 1);
    అయితే ((tensor_స్థానభ్రంశం - descriptor_స్థానభ్రంశం) % type_కొలత != 0) {
        GGML_ASSERT(descriptor_స్థానభ్రంశం >= alignment);
        descriptor_స్థానభ్రంశం -= alignment;
    }

    తిరిగి descriptor_స్థానభ్రంశం;
}
uint32_t get_misalign_బైట్లు(స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor * t) {
    స్థిరరాశి size_t tensor_స్థానభ్రంశం = ggml_vk_tensor_బఫరు_స్థానభ్రంశం(ctx, t);
    స్థిరరాశి size_t descriptor_స్థానభ్రంశం = ggml_vk_descriptor_స్థానభ్రంశం(
        tensor_స్థానభ్రంశం, ctx->device->properties.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment, ggml_type_కొలత(t->type));
    GGML_ASSERT(tensor_స్థానభ్రంశం - descriptor_స్థానభ్రంశం <= UINT32_MAX);
    తిరిగి tensor_స్థానభ్రంశం - descriptor_స్థానభ్రంశం;
}

uint32_t ggml_vk_concat_unit_కొలత(ggml_type type) {
    స్థిరరాశి uint32_t type_కొలత = ggml_type_కొలత(type);

    ఒకవేళ (!ggml_is_quantized(type)) {
        తిరిగి type_కొలత;
    }

    // Use the widest existing concat shader that evenly divides a quant block.
    ఒకవేళ (type_కొలత % 8 == 0) {
        తిరిగి 8;
    }
    ఒకవేళ (type_కొలత % 4 == 0) {
        తిరిగి 4;
    }
    ఒకవేళ (type_కొలత % 2 == 0) {
        తిరిగి 2;
    }
    తిరిగి 1;
}
తార్కిక ggml_vk_concat_supported(స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * dst) {
    ఒకవేళ (src0->type != src1->type || src0->type != dst->type) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_quantized(src0->type)) {
        స్థిరరాశి size_t type_కొలత = ggml_type_కొలత(src0->type);
        తిరిగి type_కొలత == 1 || type_కొలత == 2 || type_కొలత == 4 || type_కొలత == 8;
    }

    // Quantized tensor rows are block-aligned when created.
    తిరిగి ggml_is_contiguous_అడ్డువరుసలు(src0) && ggml_is_contiguous_అడ్డువరుసలు(src1) && ggml_is_contiguous_అడ్డువరుసలు(dst);
}
స్థిర తార్కిక vk_instance_initialized = అబద్ధం;

vk_instance_t vk_instance;

స్థిర Vk_Device_కొలత ggml_vk_get_max_బఫరు_range(స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి vk_బఫరు &buf, స్థిరరాశి Vk_Device_కొలత స్థానభ్రంశం) {
    స్థిరరాశి Vk_Device_కొలత range = std::min(Vk_Device_కొలత{buf->size - స్థానభ్రంశం},
                                        Vk_Device_కొలత{ctx->device->properties.limits.max_Storage_బఫరు_Range});
    తిరిగి range;
}

శూన్య ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx) {
    // Use waitForFences while most of the graph executes. Hopefully the CPU can sleep
    // during this wait.
    ఒకవేళ (ctx->almost_ready_fence_pending) {
        VK_CHECK(ctx->device->device.waitForFences({ ctx->almost_ready_fence }, నిజం, UINT64_MAX), "almost_ready_fence", ctx->device);
        ctx->device->device.resetFences({ ctx->almost_ready_fence });
        ctx->almost_ready_fence_pending = అబద్ధం;
    }

    // Spin (w/pause) waiting for the graph to finish executing.
    vk::ఫలితం_2 ఫలితం;
    కొరకు (;;) {
        ప్రయత్నించు {
            ఫలితం = ctx->device->device.get_Fence_స్థితి(ctx->fence);
        } పట్టుకో (vk::Device_కోల్పోయిన_Error &) {
            ggml_vk_print_device_కోల్పోయిన_info(ctx->device);
            GGML_LOG_ERROR("ggml_vulkan: getFenceStatus at %s:%d\n", __FILE__, __LINE__);
            విసురు;
        }
        ఒకవేళ (ఫలితం == vk::ఫలితం_2::eSuccess) {
            ఆపు;
        }
        ఒకవేళ (ఫలితం != vk::ఫలితం_2::eNotReady) {
            GGML_LOG_ERROR("ggml_vulkan: error %s at %s:%d\n", to_string(ఫలితం).c_str(), __FILE__, __LINE__);
            విసురు vk::SystemError(vk::make_error_code(ఫలితం), "ggml_vulkan: getFenceStatus");
        }
        కొరకు (uint32_t i = 0; i < 100; ++i) {
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

స్థిర తార్కిక ggml_vk_strip_decode_సదిశ(స్థిరరాశి uint32_t * code, size_t మాట_లెక్క, std::vector<uint32_t> & బహిర్గతం) {
    స్థిర స్థిరరాశి అక్షరం k_Decode_సదిశ_Ext[] = "SPV_NV_cooperative_matrix_decode_vector";

    ఒకవేళ (మాట_లెక్క < 5) {
        తిరిగి అబద్ధం;
    }

    తార్కిక ఉపయోగిస్తుంది_decode_సదిశ = అబద్ధం;
    కొరకు (size_t pos = 5; pos < మాట_లెక్క; ) {
        uint32_t మాట = code[pos];
        uint32_t wc   = మాట >> spv::మాట_లెక్క_జరుపు;
        uint32_t op   = మాట & spv::Op_సంకేతం_ముసుగు;
        GGML_ASSERT(wc > 0 && pos + wc <= మాట_లెక్క);
        ఒకవేళ (op == spv::OpExtension && wc >= 2) {
            స్థిరరాశి అక్షరం * s = reinterpret_cast<స్థిరరాశి అక్షరం *>(&code[pos + 1]);
            ఒకవేళ (strcmp(s, k_Decode_సదిశ_Ext) == 0) {
                ఉపయోగిస్తుంది_decode_సదిశ = నిజం;
                ఆపు;
            }
        }
        pos += wc;
    }

    ఒకవేళ (!ఉపయోగిస్తుంది_decode_సదిశ) {
        తిరిగి అబద్ధం;
    }

    VK_LOG_DEBUG("ggml_vk_strip_decode_vector: stripping SPV_NV_cooperative_matrix_decode_vector");

    // Bulk-copy unchanged runs and only break the run when an instruction needs to
    // be dropped or patched. Use reserve + insert/push_back so the destination buffer
    // is touched exactly once (no zero-initialization pass from resize()).
    బహిర్గతం.clear();
    బహిర్గతం.reserve(మాట_లెక్క);

    size_t నిరంతరం_ప్రారంభం = 0;
    స్వయంచాలక flush_నిరంతరం = [&](size_t పైకి_గమ్యం) {
        ఒకవేళ (పైకి_గమ్యం > నిరంతరం_ప్రారంభం) {
            బహిర్గతం.insert(బహిర్గతం.end(), code + నిరంతరం_ప్రారంభం, code + పైకి_గమ్యం);
        }
    };

    కొరకు (size_t pos = 5; pos < మాట_లెక్క; ) {
        uint32_t మాట = code[pos];
        uint32_t wc   = మాట >> spv::మాట_లెక్క_జరుపు;
        uint32_t op   = మాట & spv::Op_సంకేతం_ముసుగు;
        GGML_ASSERT(wc > 0 && pos + wc <= మాట_లెక్క);

        ఒకవేళ (op == spv::OpExtension && wc >= 2) {
            స్థిరరాశి అక్షరం * s = reinterpret_cast<స్థిరరాశి అక్షరం *>(&code[pos + 1]);
            ఒకవేళ (strcmp(s, k_Decode_సదిశ_Ext) == 0) {
                flush_నిరంతరం(pos);
                pos += wc;
                నిరంతరం_ప్రారంభం = pos;
                కొనసాగు;
            }
        }

        ఒకవేళ (op == spv::OpCapability && wc == 2 && code[pos + 1] == k_Spv_Capability_Cooperative_మాతృక_Decode_సదిశ_NV) {
            flush_నిరంతరం(pos);
            pos += wc;
            నిరంతరం_ప్రారంభం = pos;
            కొనసాగు;
        }

        ఒకవేళ (op == k_Spv_Op_Cooperative_మాతృక_ఎక్కించు_Tensor_NV) {
            // [opcode/wc][ResultType][Result][Pointer][Object][TensorLayout][MemOperand mask][mem extras...][TA mask][ta extras...]
            GGML_ASSERT(wc >= 8);

            uint32_t mem_ముసుగు = code[pos + 6];
            size_t   cur      = pos + 7;
            // Each of these MemoryAccess bits (when set) carries one trailing operand.
            cur += (mem_ముసుగు & 0x2)     ? 1 : 0; // Aligned
            cur += (mem_ముసుగు & 0x8)     ? 1 : 0; // MakePointerAvailable
            cur += (mem_ముసుగు & 0x10)    ? 1 : 0; // MakePointerVisible
            cur += (mem_ముసుగు & 0x10000) ? 1 : 0; // AliasScopeINTELMask
            cur += (mem_ముసుగు & 0x20000) ? 1 : 0; // NoAliasINTELMask
            GGML_ASSERT(cur < pos + wc);

            uint32_t ta_ముసుగు = code[cur];
            ఒకవేళ ((ta_ముసుగు & k_Spv_Tensor_Addressing_Decode_సదిశ_Func_బిట్టు) == 0) {
                pos += wc;
                కొనసాగు; // leave instruction inside the current unchanged run
            }

            flush_నిరంతరం(pos);

            // Append unchanged prefix of the instruction (header through the mem-extras).
            size_t inst_ప్రారంభం = బహిర్గతం.size();
            size_t pre_n      = cur - pos;
            బహిర్గతం.insert(బహిర్గతం.end(), code + pos, code + pos + pre_n);

            // Emit TA mask with the DecodeVectorFunc bit cleared.
            బహిర్గతం.push_back(ta_ముసుగు & ~k_Spv_Tensor_Addressing_Decode_సదిశ_Func_బిట్టు);

            // TA extras: TensorView (0x1) and DecodeFunc (0x2) are kept verbatim;
            // DecodeVectorFunc (0x4) is dropped along with its trailing id operand.
            size_t keep_ta_extras = ((ta_ముసుగు & 0x1) ? 1 : 0) + ((ta_ముసుగు & 0x2) ? 1 : 0);
            ఒకవేళ (keep_ta_extras) {
                బహిర్గతం.insert(బహిర్గతం.end(), code + cur + 1, code + cur + 1 + keep_ta_extras);
            }

            GGML_ASSERT(wc == pre_n + 1 + keep_ta_extras + 1);

            // Patch the instruction header with the new (one-shorter) word count.
            uint32_t new_wc = wc - 1;
            బహిర్గతం[inst_ప్రారంభం] = (new_wc << spv::మాట_లెక్క_జరుపు) | op;

            pos += wc;
            నిరంతరం_ప్రారంభం = pos;
            కొనసాగు;
        }

        pos += wc;
    }

    flush_నిరంతరం(మాట_లెక్క);
    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_vk_roll_bk_loop(స్థిరరాశి uint32_t * code, size_t మాట_లెక్క, std::vector<uint32_t> & బహిర్గతం) {
    ఒకవేళ (మాట_లెక్క < 5) {
        తిరిగి అబద్ధం;
    }

    నిర్మాణం vk_spv_loop {
        size_t   శీర్షిక;
        size_t   end;
        uint32_t control;
    };

    std::vector<vk_spv_loop> loops;

    // Collect a list of all loops in the module.
    కొరకు (size_t pos = 5; pos < మాట_లెక్క; ) {
        స్థిరరాశి uint32_t wc = code[pos] >> spv::మాట_లెక్క_జరుపు;
        స్థిరరాశి uint32_t op = code[pos] & spv::Op_సంకేతం_ముసుగు;
        ఒకవేళ (wc == 0 || pos + wc > మాట_లెక్క) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (op == spv::Op_Loop_విలీనం && wc >= 4) { loops.push_back({ pos, 0, code[pos + 3] }); }

        ఒకవేళ (op == spv::OpLabel && wc >= 2) {
            కొరకు (స్వయంచాలక & l : loops) {
                ఒకవేళ (l.end == 0 && code[l.శీర్షిక + 1] == code[pos + 1]) { l.end = pos; }
            }
        }

        pos += wc;
    }

    స్వయంచాలక encloses = [](స్థిరరాశి vk_spv_loop & a, స్థిరరాశి vk_spv_loop & b) {
        తిరిగి a.శీర్షిక < b.శీర్షిక && b.శీర్షిక < a.end;
    };

    // Find the BK loop.
    స్థిరరాశి vk_spv_loop * bk = శూన్యసూచిక;
    కొరకు (స్థిరరాశి స్వయంచాలక & h : loops) {
        ఒకవేళ (h.control != spv::Loop_Control_Unroll_ముసుగు) {
            కొనసాగు;
        }
        స్థిరరాశి vk_spv_loop * మాతృ = శూన్యసూచిక;
        తార్కిక has_బిడ్డ = అబద్ధం;
        కొరకు (స్థిరరాశి స్వయంచాలక & g : loops) {
            ఒకవేళ (encloses(g, h) && (!మాతృ || g.శీర్షిక > మాతృ->శీర్షిక)) {
                మాతృ = &g;
            }
            ఒకవేళ (encloses(h, g)) {
                has_బిడ్డ = నిజం;
            }
        }
        // BK loop should be the last loop nested inside the loop with no hint
        // and have at least one child loop.
        ఒకవేళ (మాతృ &&
            మాతృ->control == spv::Loop_Control_ముసుగు_None &&
            has_బిడ్డ &&
            (!bk || h.శీర్షిక > bk->శీర్షిక)) {
            bk = &h;
        }
    }
    ఒకవేళ (!bk) {
        తిరిగి అబద్ధం;
    }

    // set DontUnroll instead of Unroll
    బహిర్గతం.కేటాయించు(code, code + మాట_లెక్క);
    బహిర్గతం[bk->శీర్షిక + 3] = spv::Loop_Control_Dont_Unroll_ముసుగు;
    తిరిగి నిజం;
}

స్థిర శూన్య ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, size_t spv_కొలత, స్థిరరాశి శూన్య* spv_దత్తాంశం, స్థిరరాశి std::string entrypoint,
                                         uint32_t parameter_లెక్క, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         తార్కిక disable_robustness, తార్కిక require_నిండిన_subgroups, uint32_t required_subgroup_కొలత) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_లెక్క <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_నిండిన_subgroups << ", " << required_subgroup_కొలత << ")");
    GGML_ASSERT(parameter_లెక్క > 0);
    GGML_ASSERT(parameter_లెక్క <= MAX_PARAMETER_లెక్క);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_కొలత, reinterpret_cast<స్థిరరాశి uint32_t *>(spv_దత్తాంశం));

    // Patch SPIR-V to enable supported FP16 float controls, avoiding the need
    // for separate shader variants.
    std::vector<uint32_t> spirv;
    ఒకవేళ (device->float_controls_rte_fp16 || device->float_controls_denorm_preserve_fp16) {
        స్థిరరాశి uint32_t* spv_పదాలు = reinterpret_cast<స్థిరరాశి uint32_t *>(spv_దత్తాంశం);
        size_t మాట_లెక్క = spv_కొలత / పరిమాణం(uint32_t);
        spirv.కేటాయించు(spv_పదాలు, spv_పదాలు + మాట_లెక్క);

        // Find insertion points respecting SPIR-V layout order:
        //   Header(5) -> OpCapability -> OpExtension -> ... -> OpEntryPoint -> OpExecutionMode -> ...
        size_t pos = 5; // skip header
        size_t cap_చొప్పించు_pos = pos;
        size_t ext_చొప్పించు_pos = pos;
        size_t exec_చొప్పించు_pos = pos;
        uint32_t నమోదు_బిందువు_id = 0;

        అయితే (pos < spirv.size()) {
            uint32_t opcode = spirv[pos] & spv::Op_సంకేతం_ముసుగు;
            uint32_t len    = spirv[pos] >> spv::మాట_లెక్క_జరుపు;
            ఒకవేళ (len == 0) ఆపు;

            ఒకవేళ (opcode == spv::OpCapability) {
                cap_చొప్పించు_pos = pos + len;
                ext_చొప్పించు_pos = pos + len;
            } లేకపోతే ఒకవేళ (opcode == spv::OpExtension) {
                ext_చొప్పించు_pos = pos + len;
            } లేకపోతే ఒకవేళ (opcode == spv::Op_నమోదు_బిందువు) {
                నమోదు_బిందువు_id = spirv[pos + 2];
                exec_చొప్పించు_pos = pos + len;
            } లేకపోతే ఒకవేళ (opcode == spv::Op_Execution_బహుళకం || opcode == spv::Op_Execution_బహుళకం_Id) {
                exec_చొప్పించు_pos = pos + len;
            } లేకపోతే ఒకవేళ (నమోదు_బిందువు_id != 0) {
                ఆపు;
            }

            pos += len;
        }

        // Insert from latest position first so earlier indices stay valid.

        ఒకవేళ (device->float_controls_rte_fp16) {
            // OpExecutionMode %entrypoint RoundingModeRTE 16
            uint32_t exec_బహుళకం[] = { (4u << spv::మాట_లెక్క_జరుపు) | spv::Op_Execution_బహుళకం, నమోదు_బిందువు_id, spv::Execution_బహుళకం_Rounding_బహుళకం_RTE, 16 };
            spirv.insert(spirv.begin() + exec_చొప్పించు_pos, std::begin(exec_బహుళకం), std::end(exec_బహుళకం));
        }

        ఒకవేళ (device->float_controls_denorm_preserve_fp16) {
            // OpExecutionMode %entrypoint DenormPreserve 16
            uint32_t exec_బహుళకం[] = { (4u << spv::మాట_లెక్క_జరుపు) | spv::Op_Execution_బహుళకం, నమోదు_బిందువు_id, spv::Execution_బహుళకం_Denorm_Preserve, 16 };
            spirv.insert(spirv.begin() + exec_చొప్పించు_pos, std::begin(exec_బహుళకం), std::end(exec_బహుళకం));
        }

        // OpExtension "SPV_KHR_float_controls"
        స్థిరరాశి అక్షరం ext_str[] = "SPV_KHR_float_controls";
        size_t ext_str_పదాలు = CEIL_DIV(పరిమాణం(ext_str), పరిమాణం(uint32_t));
        std::vector<uint32_t> extension(1 + ext_str_పదాలు, 0);
        extension[0] = (uint32_t)((1 + ext_str_పదాలు) << spv::మాట_లెక్క_జరుపు) | spv::OpExtension;
        జ్ఞాపకం_నకలు(&extension[1], ext_str, పరిమాణం(ext_str));
        spirv.insert(spirv.begin() + ext_చొప్పించు_pos, extension.begin(), extension.end());

        ఒకవేళ (device->float_controls_rte_fp16) {
            // OpCapability RoundingModeRTE
            uint32_t capability[] = { (2u << spv::మాట_లెక్క_జరుపు) | spv::OpCapability, spv::Capability_Rounding_బహుళకం_RTE };
            spirv.insert(spirv.begin() + cap_చొప్పించు_pos, std::begin(capability), std::end(capability));
        }

        ఒకవేళ (device->float_controls_denorm_preserve_fp16) {
            // OpCapability DenormPreserve
            uint32_t capability[] = { (2u << spv::మాట_లెక్క_జరుపు) | spv::OpCapability, spv::CapabilityDenormPreserve };
            spirv.insert(spirv.begin() + cap_చొప్పించు_pos, std::begin(capability), std::end(capability));
        }

        shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * పరిమాణం(uint32_t), spirv.దత్తాంశం());
    }

#ఒకవేళ defined(GGML_VULKAN_COOPMAT2_DECODE_సదిశ_GLSLC_SUPPORT)
    ఒకవేళ (device->coopmat2 && !device->coopmat2_decode_సదిశ) {
        స్థిరరాశి uint32_t * src   = spirv.ఖాళీలు() ? reinterpret_cast<స్థిరరాశి uint32_t *>(spv_దత్తాంశం) : spirv.దత్తాంశం();
        size_t           src_n = spirv.ఖాళీలు() ? spv_కొలత / పరిమాణం(uint32_t) : spirv.size();
        std::vector<uint32_t> stripped;
        ఒకవేళ (ggml_vk_strip_decode_సదిశ(src, src_n, stripped)) {
            spirv = std::move(stripped);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * పరిమాణం(uint32_t), spirv.దత్తాంశం());
        }
    }
#అయితే_ముగింపు

#ఒకవేళ VK_శీర్షిక_సంస్కరణ >= 287
    // Roll the mul_mm BK loop on Asahi Linux. Skip bf16 and the mul_mmq pipelines.
    ఒకవేళ (device->driver_id == vk::DriverId::eMesaHoneykrisp &&
        pipeline->name.rfind("matmul", 0) == 0 &&
        pipeline->name.find("bf16") == std::string::npos &&
        pipeline->name.find("q8_1") == std::string::npos) {
        స్థిరరాశి uint32_t * src   = spirv.ఖాళీలు() ? reinterpret_cast<స్థిరరాశి uint32_t *>(spv_దత్తాంశం) : spirv.దత్తాంశం();
        size_t           src_n = spirv.ఖాళీలు() ? spv_కొలత / పరిమాణం(uint32_t) : spirv.size();
        std::vector<uint32_t> rolled;
        ఒకవేళ (ggml_vk_roll_bk_loop(src, src_n, rolled)) {
            spirv = std::move(rolled);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * పరిమాణం(uint32_t), spirv.దత్తాంశం());
        }
    }
#అయితే_ముగింపు

    ప్రయత్నించు {
        pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);
    } పట్టుకో (స్థిరరాశి vk::SystemError& e) {
        std::cerr << "ggml_vulkan: shader module creation failed for " << pipeline->name << ": " << e.what() << std::పంక్తి_ముగింపు;
        విసురు;
    }

    vk::నెట్టు_Constant_Range pcr(
        vk::Shader_Stage_Flag_బిట్లు::eCompute,
        0,
        pipeline->నెట్టు_constant_కొలత
    );

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
    ప్రయత్నించు {
        pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);
    } పట్టుకో (స్థిరరాశి vk::SystemError& e) {
        std::cerr << "ggml_vulkan: pipeline layout creation failed for " << pipeline->name << ": " << e.what() << std::పంక్తి_ముగింపు;
        విసురు;
    }

    std::vector<vk::Specialization_Map_నమోదు> specialization_entries(specialization_constants.size());

    కొరకు (size_t i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].స్థానభ్రంశం = i * పరిమాణం(uint32_t);
        specialization_entries[i].size = పరిమాణం(uint32_t);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.దత్తాంశం(),
        specialization_constants.size() * పరిమాణం(uint32_t),
        specialization_constants.దత్తాంశం()
    );

    vk::PipelineShaderStageCreateFlags pipeline_shader_stage_create_flags{};

    ఒకవేళ (device->subgroup_require_నిండిన_support && require_నిండిన_subgroups) {
        pipeline_shader_stage_create_flags |= vk::Pipeline_Shader_Stage_Create_Flag_బిట్లు::e_Require_నిండిన_Subgroups_EXT;
    }

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            pipeline_shader_stage_create_flags,
            vk::Shader_Stage_Flag_బిట్లు::eCompute,
            pipeline->shader_module,
            entrypoint.c_str(),
            &specialization_info);

    vk::Pipeline_Shader_Stage_Required_Subgroup_కొలత_Create_Info_EXT pipeline_shader_stage_required_subgroup_కొలత_create_info;
    pipeline_shader_stage_required_subgroup_కొలత_create_info.required_Subgroup_కొలత = required_subgroup_కొలత;
    ఒకవేళ (device->subgroup_కొలత_control && required_subgroup_కొలత > 0) {
        GGML_ASSERT(device->subgroup_min_కొలత <= required_subgroup_కొలత && required_subgroup_కొలత <= device->subgroup_max_కొలత);
        pipeline_shader_create_info.సమితి_PNext(&pipeline_shader_stage_required_subgroup_కొలత_create_info);
    }

    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        device->pipeline_executable_properties_support ?
            vk::Pipeline_Create_Flag_బిట్లు::eCaptureStatisticsKHR :
            vk::PipelineCreateFlags{},
        pipeline_shader_create_info,
        pipeline->layout);

    vk::PipelineRobustnessCreateInfoEXT rci;

    ఒకవేళ (device->pipeline_robustness && disable_robustness) {
        rci.storageBuffers = vk::Pipeline_Robustness_బఫరు_Behavior_EXT::eDisabled;
        rci.ఏకరీతి_Buffers = vk::Pipeline_Robustness_బఫరు_Behavior_EXT::eDisabled;
        compute_pipeline_create_info.సమితి_PNext(&rci);
    }

#ఒకవేళ defined(VK_EXT_shader_64bit_indexing)
    vk::PipelineCreateFlags2CreateInfo pipelineFlags2CreateInfo;
    ఒకవేళ (pipeline->is_64b_indexing)
    {
        pipelineFlags2CreateInfo.flags = vk::PipelineCreateFlagBits2::e64_బిట్టు_Indexing_EXT;
        ఒకవేళ (device->pipeline_executable_properties_support) {
            pipelineFlags2CreateInfo.flags |= vk::PipelineCreateFlagBits2::eCaptureStatisticsKHR;
        }
        pipelineFlags2CreateInfo.సమితి_PNext(compute_pipeline_create_info.p_తదుపరి);
        compute_pipeline_create_info.సమితి_PNext(&pipelineFlags2CreateInfo);
    }
#అయితే_ముగింపు

    ప్రయత్నించు {
        pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;
    } పట్టుకో (స్థిరరాశి vk::SystemError& e) {
        std::cerr << "ggml_vulkan: compute pipeline creation failed for " << pipeline->name << ": " << e.what() << std::పంక్తి_ముగింపు;
        విసురు;
    }

    ఒకవేళ (vk_instance.debug_utils_support) {
        vk::DebugUtilsObjectNameInfoEXT duoni;
        duoni.objectType = vk::ObjectType::ePipeline;
        duoni.pObjectName = pipeline->name.c_str();
        duoni.objectHandle = /*reinterpret_cast*/(uint64_t)(static_cast<VkPipeline>(pipeline->pipeline));
        vk_instance.pfn_vk_సమితి_Debug_Utils_Object_Name_EXT(device->device, &static_cast<VkDebugUtilsObjectNameInfoEXT &>(duoni));
    }

    ఒకవేళ (device->pipeline_executable_properties_support) {
        vk::PipelineExecutableInfoKHR executableInfo;
        executableInfo.pipeline = pipeline->pipeline;

        స్వయంచాలక statistics = device->device.getPipelineExecutableStatisticsKHR(executableInfo);

        తార్కిక print_stats = !vk_pipeline_stats_filter.ఖాళీలు() &&
                           pipeline->name.find(vk_pipeline_stats_filter) != std::string::npos;
        ఒకవేళ (print_stats) {
            std::cerr << "ggml_vulkan: pipeline stats for " << pipeline->name << ":" << std::పంక్తి_ముగింపు;
        }

        కొరకు (స్వయంచాలక & s : statistics) {
            ఒకవేళ (print_stats) {
                std::cerr << "ggml_vulkan:   " << s.name.దత్తాంశం() << ": ";
                ఎంపిక (s.format) {
                    స్థితి vk::PipelineExecutableStatisticFormatKHR::eBool32:
                        std::cerr << (s.value.b32 ? "true" : "false");
                        ఆపు;
                    స్థితి vk::PipelineExecutableStatisticFormatKHR::eInt64:
                        std::cerr << s.value.i64;
                        ఆపు;
                    స్థితి vk::PipelineExecutableStatisticFormatKHR::eUint64:
                        std::cerr << s.value.u64;
                        ఆపు;
                    స్థితి vk::PipelineExecutableStatisticFormatKHR::eFloat64:
                        std::cerr << s.value.f64;
                        ఆపు;
                }
                std::cerr << std::పంక్తి_ముగింపు;
            }
            // "Register Count" is reported by NVIDIA drivers.
            ఒకవేళ (strcmp(s.name, "Register Count") == 0) {
                VK_LOG_DEBUG(pipeline->name << " " << s.name << ": " << s.value.u64 << " registers");
                pipeline->రిజిస్టరు_లెక్క = (uint32_t)s.value.u64;
            }
        }
    }

    {
        std::lock_guard<std::mutex> guard(device->compile_mutex);
        device->all_pipelines.push_back(pipeline);
        pipeline->compiled = నిజం;
        pipeline->compile_pending = అబద్ధం;
    }
    device->compile_cv.notify_all();
}

శూన్య ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_pipeline_destroy_pipeline(" << pipeline->name << ")");
    device.destroyPipelineLayout(pipeline->layout);

    device.destroyShaderModule(pipeline->shader_module);

    device.destroyPipeline(pipeline->pipeline);
}

శూన్య ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n) {
    VK_LOG_DEBUG("ggml_pipeline_request_descriptor_sets(" << pipeline->name << ", " << n << ")");
    ctx->pipeline_descriptor_సమితి_requirements += n;
    ఒకవేళ (!pipeline->compiled) {
        ggml_vk_ఎక్కించు_shaders(ctx->device, pipeline);
    }
    ggml_pipeline_allocate_descriptor_sets(ctx);
}

శూన్య ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {

    ఒకవేళ (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_సమితి_requirements) {
        // Enough descriptors are available
        తిరిగి;
    }

    vk_device& device = ctx->device;

    // Grow by 50% to avoid frequent allocations
    uint32_t needed = std::max(3 * ctx->descriptor_sets.size() / 2, size_t{ctx->pipeline_descriptor_సమితి_requirements});
    uint32_t గమ్యం_alloc = needed - ctx->descriptor_sets.size();
    uint32_t pool_మిగిలిన = VK_DEVICE_DESCRIPTOR_POOL_కొలత - ctx->descriptor_sets.size() % VK_DEVICE_DESCRIPTOR_POOL_కొలత;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_కొలత;

    అయితే (గమ్యం_alloc > 0) {
        స్థిరరాశి uint32_t alloc_లెక్క = std::min(pool_మిగిలిన, గమ్యం_alloc);
        గమ్యం_alloc -= alloc_లెక్క;
        pool_మిగిలిన = VK_DEVICE_DESCRIPTOR_POOL_కొలత;

        ఒకవేళ (pool_idx >= ctx->descriptor_pools.size()) {
            vk::Descriptor_Pool_కొలత descriptor_pool_కొలత(vk::DescriptorType::e_Storage_బఫరు, MAX_PARAMETER_లెక్క * VK_DEVICE_DESCRIPTOR_POOL_కొలత);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_కొలత, descriptor_pool_కొలత);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::Descriptor_సమితి_Layout> layouts(alloc_లెక్క);
        కొరకు (uint32_t i = 0; i < alloc_లెక్క; i++) {
            layouts[i] = device->dsl;
        }
        vk::Descriptor_సమితి_Allocate_Info descriptor_సమితి_alloc_info(ctx->descriptor_pools[pool_idx], alloc_లెక్క, layouts.దత్తాంశం());
        std::vector<vk::Descriptor_సమితి> sets = device->device.allocateDescriptorSets(descriptor_సమితి_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
    ctx->descriptor_సమితి_bindings.resize(ctx->descriptor_sets.size());
}

స్థిర vk_command_బఫరు* ggml_vk_create_cmd_బఫరు(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_create_cmd_buffer()");
    vk::Command_బఫరు_Allocate_Info command_బఫరు_alloc_info(
        p.pool,
        vk::Command_బఫరు_Level::ePrimary,
        1);
    స్థిరరాశి std::vector<vk::Command_బఫరు> cmd_buffers = device->device.allocateCommandBuffers(command_బఫరు_alloc_info);
    p.cmd_buffers.push_back({ cmd_buffers.front(), 0, నిజం });
    తిరిగి &p.cmd_buffers[p.cmd_buffers.size()-1];
}

శూన్య ggml_vk_submit(vk_context& ctx, vk::Fence fence) {
    ఒకవేళ (ctx->seqs.ఖాళీలు()) {
        ఒకవేళ (fence) {
            ctx->p->q->handle->submit({}, fence);
        }
        తిరిగి;
    }
    VK_LOG_DEBUG("ggml_vk_submit(" << ctx << ", " << fence << ")");

    std::vector<std::vector<uint64_t>> tl_wait_vals;
    std::vector<std::vector<uint64_t>> tl_signal_vals;
    std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
    std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
    std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
    std::vector<vk::SubmitInfo> submit_infos;
    పూర్ణాంకం idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    size_t reserve = 0;

    కొరకు (స్థిరరాశి స్వయంచాలక& sequence : ctx->seqs) {
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

    కొరకు (స్థిరరాశి స్వయంచాలక& sequence : ctx->seqs) {
        కొరకు (స్థిరరాశి స్వయంచాలక& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            tl_wait_vals.push_back({});
            tl_wait_semaphores.push_back({});
            tl_signal_vals.push_back({});
            tl_signal_semaphores.push_back({});
            కొరకు (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(ctx->p->q->stage_flags);
                tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
                tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
            }
            కొరకు (size_t i = 0; i < submission.signal_semaphores.size(); i++) {
                tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
                tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
            }
            tl_submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_vals[idx].దత్తాంశం(),
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_vals[idx].దత్తాంశం(),
            });
            tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
            tl_submit_infos[idx].p_తదుపరి = శూన్యసూచిక;
            vk::SubmitInfo si{
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_semaphores[idx].దత్తాంశం(),
                stage_flags[idx].దత్తాంశం(),
                1,
                &submission.బఫరు->buf,
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_semaphores[idx].దత్తాంశం(),
            };
            si.సమితి_PNext(&tl_submit_infos[idx]);
            submit_infos.push_back(si);
        }
    }

    ctx->p->q->handle->submit(submit_infos, fence);

    ctx->seqs.clear();
}

uint32_t ggml_vk_కనుగొను_వరుస_family_సూచీ(std::vector<vk::వరుస_Family_Properties>& వరుస_family_props, స్థిరరాశి vk::వరుస_Flags& required, స్థిరరాశి vk::వరుస_Flags& avoid, int32_t compute_సూచీ, uint32_t min_అంకె_queues) {
    VK_LOG_DEBUG("ggml_vk_find_queue_family_index()");
    స్థిరరాశి uint32_t qfsize = వరుస_family_props.size();

    // Try with avoid preferences first
    కొరకు (uint32_t i = 0; i < qfsize; i++) {
        ఒకవేళ (వరుస_family_props[i].వరుస_లెక్క >= min_అంకె_queues && (compute_సూచీ < 0 || i != (uint32_t) compute_సూచీ) && వరుస_family_props[i].వరుస_Flags_2 & required && !(వరుస_family_props[i].వరుస_Flags_2 & avoid)) {
            తిరిగి i;
        }
    }

    // Fall back to only required
    కొరకు (size_t i = 0; i < qfsize; i++) {
        ఒకవేళ (వరుస_family_props[i].వరుస_లెక్క >= min_అంకె_queues && (compute_సూచీ < 0 || i != (uint32_t) compute_సూచీ) && వరుస_family_props[i].వరుస_Flags_2 & required) {
            తిరిగి i;
        }
    }

    // Fall back to reusing compute queue
    కొరకు (size_t i = 0; i < qfsize; i++) {
        ఒకవేళ (వరుస_family_props[i].వరుస_లెక్క >= min_అంకె_queues && వరుస_family_props[i].వరుస_Flags_2 & required) {
            తిరిగి i;
        }
    }

    // Fall back to ignoring min_num_queries
    కొరకు (size_t i = 0; i < qfsize; i++) {
        ఒకవేళ (వరుస_family_props[i].వరుస_Flags_2 & required) {
            తిరిగి i;
        }
    }

    // All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
    // Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
    ఒకవేళ (compute_సూచీ >= 0) {
        తిరిగి compute_సూచీ;
    }

    std::cerr << "ggml_vulkan: No suitable queue family index found." << std::పంక్తి_ముగింపు;

    కొరకు(స్వయంచాలక &q_family : వరుస_family_props) {
        std::cerr << "Queue number: "  + std::to_string(q_family.వరుస_లెక్క) << " flags: " + to_string(q_family.వరుస_Flags_2) << std::పంక్తి_ముగింపు;
    }
    abort();
}

std::unique_ptr<vk_వరుస> ggml_vk_create_వరుస(vk_device& device, uint32_t వరుస_family_సూచీ, uint32_t వరుస_సూచీ, vk::PipelineStageFlags&& stage_flags, తార్కిక transfer_only) {
    VK_LOG_DEBUG("ggml_vk_create_queue()");
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    స్వయంచాలక q = std::make_unique<vk_వరుస>();
    q->వరుస_family_సూచీ = వరుస_family_సూచీ;
    q->transfer_only = transfer_only;

    std::shared_ptr<vk_వరుస_handle> h;
    vk::Device_వరుస_Info2 వరుస_info2{};
    వరుస_info2.వరుస_Family_సూచీ = వరుస_family_సూచీ;
    వరుస_info2.వరుస_సూచీ_2 = వరుస_సూచీ;

    ఒకవేళ (device->has_internally_synchronized_queues) {
        h = std::make_shared<vk_వరుస_handle_unsynchronized>();
        వరుస_info2.flags = eInternallySynchronizedKHR;
    } లేకపోతే {
        h = std::make_shared<vk_వరుస_handle_synchronized>();
    }

    h->queue = device->device.getQueue2(వరుస_info2);
    h->device = device;
    // Avoid concurrent submissions on NVIDIA due to driver bug.
    ఒకవేళ (device->vendor_id == VK_VENDOR_ID_NVIDIA) {
        h->device_submit_mutex = &device->వరుస_submit_mutex;
    }
    q->handle = h;

    q->cmd_pool.init(device, q.get());

    q->stage_flags = stage_flags;
    తిరిగి q;
}

std::unique_ptr<vk_వరుస> ggml_vk_create_aliased_వరుస(vk_device& device, స్థిరరాశి std::unique_ptr<vk_వరుస>& ఆధారం) {
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    స్వయంచాలక q = std::make_unique<vk_వరుస>();
    q->handle = ఆధారం->handle;
    q->వరుస_family_సూచీ = ఆధారం->వరుస_family_సూచీ;
    q->stage_flags = ఆధారం->stage_flags;
    q->transfer_only = ఆధారం->transfer_only;
    q->cmd_pool.init(device, q.get());
    తిరిగి q;
}

vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p) {
    vk_context ఫలితం = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_context(" << ఫలితం << ")");
    ctx->gc.contexts.emplace_back(ఫలితం);
    ఫలితం->p = &p;
    తిరిగి ఫలితం;
}

vk_context ggml_vk_create_temporary_context(vk_command_pool& p) {
    vk_context ఫలితం = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_temporary_context(" << ఫలితం << ")");
    ఫలితం->p = &p;
    తిరిగి ఫలితం;
}

స్థిర vk_semaphore * ggml_vk_create_ద్విమాన_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::e_ద్విమాన, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.సమితి_PNext(&tci);
    vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
    ctx->gc.semaphores.push_back({ semaphore, 0 });
    తిరిగి &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}

స్థిర vk_semaphore * ggml_vk_create_timeline_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    ఒకవేళ (ctx->semaphore_idx >= ctx->gc.tl_semaphores.size()) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.సమితి_PNext(&tci);
        vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
        ctx->gc.tl_semaphores.push_back({ semaphore, 0 });
    }
    తిరిగి &ctx->gc.tl_semaphores[ctx->semaphore_idx++];
}

స్థిర vk::Event ggml_vk_create_event(ggml_backend_vk_context * ctx) {
    ఒకవేళ (ctx->event_idx >= ctx->gc.events.size()) {
        ctx->gc.events.push_back(ctx->device->device.createEvent({}));
    }
    తిరిగి ctx->gc.events[ctx->event_idx++];
}

శూన్య ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_command_pool_cleanup()");

    // Requires command buffers to be done
    device->device.resetCommandPool(p.pool);
    // Don't clear the command buffers and mark them as not in use.
    // This allows us to reuse them
    కొరకు (స్వయంచాలక& cmd_బఫరు : p.cmd_buffers) {
        cmd_బఫరు.లోపల_use = అబద్ధం;
    }
}

శూన్య ggml_vk_వరుస_command_pools_cleanup(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_queue_command_pools_cleanup()");

    // Arbitrary frequency to cleanup/reuse command buffers
    స్థిర constexpr uint32_t cleanup_frequency = 10;

    ఒకవేళ (device->compute_వరుస && device->compute_వరుస->cmd_pool.buffers_లోపల_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->compute_వరుస->cmd_pool);
    }
    ఒకవేళ (device->transfer_వరుస && device->transfer_వరుస->cmd_pool.buffers_లోపల_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->transfer_వరుస->cmd_pool);
    }
}

vk_subbuffer ggml_vk_subbuffer(స్థిరరాశి ggml_backend_vk_context* ctx, స్థిరరాశి vk_బఫరు& buf, size_t స్థానభ్రంశం) {
    తిరిగి { buf, స్థానభ్రంశం, ggml_vk_get_max_బఫరు_range(ctx, buf, స్థానభ్రంశం) };
}

శూన్య ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_sync_buffers()");

    స్థిరరాశి తార్కిక transfer_వరుస = subctx->p->q->transfer_only;

    ఒకవేళ (ctx) {
        ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_విభజించు_k_need_sync = అబద్ధం;
    }

    subctx->s->బఫరు->buf.pipelineBarrier(
        subctx->p->q->stage_flags,
        subctx->p->q->stage_flags,
        {},
        { {
          { !transfer_వరుస ? (vk::Access_Flag_బిట్లు::eShaderRead | vk::Access_Flag_బిట్లు::eShaderWrite | vk::Access_Flag_బిట్లు::eTransferRead | vk::Access_Flag_బిట్లు::eTransferWrite) : (vk::Access_Flag_బిట్లు::eTransferRead | vk::Access_Flag_బిట్లు::eTransferWrite) },
          { !transfer_వరుస ? (vk::Access_Flag_బిట్లు::eShaderRead | vk::Access_Flag_బిట్లు::eShaderWrite | vk::Access_Flag_బిట్లు::eTransferRead | vk::Access_Flag_బిట్లు::eTransferWrite) : (vk::Access_Flag_బిట్లు::eTransferRead | vk::Access_Flag_బిట్లు::eTransferWrite) }
        } },
        {},
        {}
    );
}

స్థిర శూన్య ggml_vk_reset_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->బఫరు->buf.resetEvent(
        event,
        ctx->p->q->stage_flags
    );
}

శూన్య ggml_vk_సమితి_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->బఫరు->buf.సమితి_Event(
        event,
        ctx->p->q->stage_flags
    );
}

శూన్య ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events) {
    VK_LOG_DEBUG("ggml_vk_wait_events()");
    ఒకవేళ (events.ఖాళీలు()) {
        తిరిగి;
    }

    ctx->s->బఫరు->buf.waitEvents(
        events,
        ctx->p->q->stage_flags,
        ctx->p->q->stage_flags,
        {},
        {},
        {}
    );
}

స్థిర vk_fa_tuning_params get_fa_tuning_params_scalar(స్థిరరాశి vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_అడ్డువరుసలు, uint32_t n_kv, ggml_type k_type, ggml_type v_type, తార్కిక f32acc) {

    vk_fa_tuning_params ఫలితం{};
    ఫలితం.path = FA_SCALAR;

    ఒకవేళ (device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Disable subgroup use due to performance issues when enforcing subgroup sizes
        ఫలితం.subgroup_కొలత = 32;
        ఫలితం.disable_subgroups = నిజం;
    } లేకపోతే ఒకవేళ (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN) {
        ఫలితం.subgroup_కొలత = n_అడ్డువరుసలు < 4 ? 32 : device->subgroup_కొలత;
    } లేకపోతే {
        ఫలితం.subgroup_కొలత = device->subgroup_కొలత;
    }

    // Row split splits the workgroup so that synchronization only has to happen within subgroups, which avoids barriers
    uint32_t అడ్డువరుస_విభజించు_max_hsk = 64;
    ఒకవేళ (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN && !device->uma) {
        అడ్డువరుస_విభజించు_max_hsk = n_అడ్డువరుసలు <= 8 ? 64 : 128;
    }
    ఫలితం.అడ్డువరుస_విభజించు = (n_అడ్డువరుసలు < 4 || hsk <= అడ్డువరుస_విభజించు_max_hsk) ? 1 : 4;

    ఒకవేళ (ఫలితం.subgroup_కొలత > 32 && (n_అడ్డువరుసలు < 4 || hsk < (ఫలితం.అడ్డువరుస_విభజించు == 1 ? 128 : 64))) {
        ఫలితం.workgroup_కొలత = ఫలితం.subgroup_కొలత * 2;
    } లేకపోతే {
        ఫలితం.workgroup_కొలత = ఫలితం.subgroup_కొలత * 4;
    }

    స్థిరరాశి uint32_t D = hsk | hsv;

    స్థిరరాశి తార్కిక reduce_ఖండం_అడ్డువరుసలు = D & 8 || n_kv < 1024 || device->vendor_id == VK_VENDOR_ID_INTEL;

    ఒకవేళ (n_అడ్డువరుసలు == 1) {
        ఫలితం.ఖండం_అడ్డువరుసలు = 1;
        ఫలితం.ఖండం_నిలువుల_సంఖ్య = 64;
    } లేకపోతే {
        // row_split 1 means higher register use per row, so block size has to be adjusted
        ఒకవేళ (ఫలితం.అడ్డువరుస_విభజించు == 1) {
            ఫలితం.ఖండం_అడ్డువరుసలు = n_అడ్డువరుసలు == 2 ? 2 : ((n_అడ్డువరుసలు <= 4 || reduce_ఖండం_అడ్డువరుసలు) ? 4 : 8);
        } లేకపోతే {
            ఫలితం.ఖండం_అడ్డువరుసలు = n_అడ్డువరుసలు <= 4 ? 4 : ((n_అడ్డువరుసలు <= 8 || reduce_ఖండం_అడ్డువరుసలు) ? 8 : 16);
        }

        ఫలితం.ఖండం_నిలువుల_సంఖ్య = (D & 8) ? 64 : 32;
    }

    స్థిరరాశి uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit

    ఫలితం.d_విభజించు = std::min(std::min(ఫలితం.subgroup_కొలత, 8u), D_lsb / 4);

    ఫలితం.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    ఒకవేళ (!reduce_ఖండం_అడ్డువరుసలు && !ggml_vk_flash_attn_scalar_shmem_support(device, ఫలితం, hsk, hsv, f32acc, k_type, v_type)) {
        ఫలితం.ఖండం_అడ్డువరుసలు /= 2;
    }

    // On AMD RDNA, for small head sizes and big batch size the shader uses few registers, so too many subgroups get scheduled
    // at once and end up thrashing the cache. Fix this by setting a large (unused) shmem buffer that reduces occupancy.
    // This targets an occupancy of 4 subgroups per SIMD.
    ఒకవేళ (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత == 65536) {
        ఒకవేళ (device->architecture != AMD_GCN && n_అడ్డువరుసలు >= 64 && hsk <= 128) {
            // 30kb target for hsk > 64, 26kb for <= 64 due to smaller workgroup size
            // Values are guessed, tested on RDNA2
            ఫలితం.హద్దు_occupancy_shmem = (hsk <= 64 ? 26 : 30) * 1024 / 4 / 4;
        } లేకపోతే ఒకవేళ (device->architecture == AMD_GCN && n_అడ్డువరుసలు <= 8 && hsk >= 256) {
            // Same thing for GCN, with an occupancy target of 2 subgroups per SIMD.
            // Here low-batch FA with large head size is affected.
            // n_rows < 4 switch because workgroup size switches from 128 to 256 there.
            ఫలితం.హద్దు_occupancy_shmem = (n_అడ్డువరుసలు < 4 ? 14 : 26) * 1024 / 4 / 4;
        }
    }

    తిరిగి ఫలితం;
}

స్థిర vk_fa_tuning_params get_fa_tuning_params_coopmat1(స్థిరరాశి vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_అడ్డువరుసలు, uint32_t n_kv, ggml_type k_type, ggml_type v_type, తార్కిక f32acc) {
    GGML_వాడని(n_అడ్డువరుసలు);
    GGML_వాడని(n_kv);
    GGML_వాడని(k_type);
    GGML_వాడని(v_type);
    GGML_వాడని(f32acc);

    vk_fa_tuning_params ఫలితం{};
    ఫలితం.path = FA_COOPMAT1;

    స్థిరరాశి uint32_t D = hsk | hsv;

    స్థిరరాశి uint32_t coopmat_ఖండం_అడ్డువరుసలు = 16;
    స్థిరరాశి uint32_t coopmat_ఖండం_నిలువుల_సంఖ్య = 16;

    స్థిరరాశి uint32_t అంకె_subgroups = 4;

    ఫలితం.ఖండం_అడ్డువరుసలు = coopmat_ఖండం_అడ్డువరుసలు;
    ఫలితం.ఖండం_నిలువుల_సంఖ్య = coopmat_ఖండం_నిలువుల_సంఖ్య * అంకె_subgroups;
    ఫలితం.అడ్డువరుస_విభజించు = అంకె_subgroups;
    ఫలితం.subgroup_కొలత = device->subgroup_కొలత;
    ఫలితం.workgroup_కొలత = అంకె_subgroups * ఫలితం.subgroup_కొలత;

    స్థిరరాశి uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit
    ఫలితం.d_విభజించు = std::min(std::min(ఫలితం.subgroup_కొలత, 8u), D_lsb / 4);

    ఫలితం.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    తిరిగి ఫలితం;
}

స్థిర vk_fa_tuning_params get_fa_tuning_params_coopmat2(స్థిరరాశి vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_అడ్డువరుసలు, uint32_t n_kv, ggml_type k_type, ggml_type v_type, తార్కిక f32acc) {
    GGML_వాడని(n_kv);
    GGML_వాడని(f32acc);

    vk_fa_tuning_params ఫలితం{};
    ఫలితం.path = FA_COOPMAT2;

    స్థిరరాశి uint32_t D = hsk | hsv;

    స్థిరరాశి తార్కిక small_అడ్డువరుసలు = n_అడ్డువరుసలు < 32;

    ఒకవేళ (small_అడ్డువరుసలు) {
        ఫలితం.ఖండం_అడ్డువరుసలు = 32;
        ఫలితం.ఖండం_నిలువుల_సంఖ్య = 32;
    } లేకపోతే ఒకవేళ (ggml_is_quantized(k_type) || ggml_is_quantized(v_type) || hsk >= 256 || hsv >= 256) {
        ఫలితం.ఖండం_అడ్డువరుసలు = (hsk >= 512 || hsv >= 512) ? 32 : 64;
        ఫలితం.ఖండం_నిలువుల_సంఖ్య = 32;
    } లేకపోతే {
        ఫలితం.ఖండం_అడ్డువరుసలు = 64;
        ఫలితం.ఖండం_నిలువుల_సంఖ్య = 64;
    }

    ఫలితం.subgroup_కొలత = device->subgroup_కొలత;
    ఫలితం.workgroup_కొలత = (small_అడ్డువరుసలు && (D % 32) == 0) ? 256 : 128;

    తిరిగి ఫలితం;
}

vk_fa_tuning_params get_fa_tuning_params(స్థిరరాశి vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_అడ్డువరుసలు, uint32_t n_kv, ggml_type k_type, ggml_type v_type, తార్కిక f32acc) {
    Fa_సంకేతం_మార్గం path = device->coopmat2 ? FA_COOPMAT2 :
                      device->coopmat1_fa_support ? FA_COOPMAT1 : FA_SCALAR;

    ఒకవేళ (path == FA_COOPMAT2 && k_type == GGML_TYPE_BF16 && !device->coopmat2_bf16_support) {
        path = FA_COOPMAT1;
    }
    ఒకవేళ (path == FA_COOPMAT1 && k_type == GGML_TYPE_BF16 && !device->coopmat_bf16_support) {
        path = FA_SCALAR;
    }

    ఒకవేళ (path == FA_COOPMAT1 && device->architecture == vk_device_architecture::NVIDIA_TURING) {
        // Nvidia compiler bug, see https://github.com/ggml-org/llama.cpp/pull/19075#issuecomment-3820716090
        path = FA_SCALAR;
    }

    ఒకవేళ (path == FA_COOPMAT1) {
        తార్కిక shape_ok = (f32acc && device->coopmat_support_16x16x16_f32acc) ||
                        (!f32acc && device->coopmat_support_16x16x16_f16acc);
        స్థిరరాశి vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_అడ్డువరుసలు, n_kv, k_type, v_type, f32acc);
        తార్కిక shmem_ok = ggml_vk_flash_attn_coopmat_shmem_support(device, params, hsk, hsv, f32acc, k_type, v_type);

        ఒకవేళ (!shape_ok || !shmem_ok) {
            path = FA_SCALAR;
        }
    }

    // scalar is faster than coopmat when N==1
    ఒకవేళ (n_అడ్డువరుసలు == 1 && (path == FA_COOPMAT1 || path == FA_COOPMAT2)) {
        path = FA_SCALAR;
    }

    ఎంపిక (path) {
    స్థితి FA_SCALAR:
        తిరిగి get_fa_tuning_params_scalar(device, hsk, hsv, n_అడ్డువరుసలు, n_kv, k_type, v_type, f32acc);
    స్థితి FA_COOPMAT1:
        తిరిగి get_fa_tuning_params_coopmat1(device, hsk, hsv, n_అడ్డువరుసలు, n_kv, k_type, v_type, f32acc);
    స్థితి FA_COOPMAT2:
        తిరిగి get_fa_tuning_params_coopmat2(device, hsk, hsv, n_అడ్డువరుసలు, n_kv, k_type, v_type, f32acc);
    అప్రమేయ:
        విసురు std::runtime_error("unsupported FaCodePath");
    }
}

vk_fa_pipeline_state get_fa_pipeline_state(స్థిరరాశి vk_device& device, స్థిరరాశి vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, తార్కిక aligned, తార్కిక f32acc,
                                                  తార్కిక use_ముసుగు, తార్కిక use_ముసుగు_opt, తార్కిక use_logit_softcap, తార్కిక use_sparse, ggml_type k_type, ggml_type v_type) {
    స్థిరరాశి తార్కిక old_amd_windows = device->vendor_id == VK_VENDOR_ID_AMD && device->driver_id == vk::DriverId::eAmdProprietary &&
                                 (device->architecture == AMD_GCN || device->architecture == AMD_RDNA1 || device->architecture == AMD_RDNA2);

    uint32_t flags = (use_ముసుగు_opt      ? 1 : 0) |
                     (use_ముసుగు          ? 2 : 0) |
                     (use_logit_softcap ? 4 : 0) |
                     (old_amd_windows   ? 8 : 0) |
                     (use_sparse        ? 16 : 0);

    స్థిరరాశి uint32_t subgroup_కొలత = params.disable_subgroups ? 0 : params.subgroup_కొలత;

    తిరిగి vk_fa_pipeline_state{hsk, hsv, params.ఖండం_అడ్డువరుసలు, params.ఖండం_నిలువుల_సంఖ్య, params.d_విభజించు, params.అడ్డువరుస_విభజించు, params.shmem_staging, params.path, params.workgroup_కొలత, subgroup_కొలత, aligned, f32acc, flags, params.హద్దు_occupancy_shmem, k_type, v_type};
}

స్థిర uint32_t fa_ఖండం_బైట్లు(ggml_type t) {
    ఒకవేళ (t == GGML_TYPE_F32) {
        తిరిగి 16u;
    }
    తిరిగి (uint32_t) ggml_type_కొలత(t);
}

స్థిర std::vector<uint32_t> get_fa_spec_constants(స్థిరరాశి vk_fa_pipeline_state& state) {
    తిరిగి {
        /* 0 WorkGroupSize   */ state.workgroup_కొలత,
        /* 1 Br              */ state.Br,
        /* 2 Bc              */ state.Bc,
        /* 3 HSK             */ state.HSK,
        /* 4 HSV             */ state.HSV,
        /* 5 Clamp           */ static_cast<uint32_t>(!state.aligned),
        /* 6 D_split         */ state.D_విభజించు,
        /* 7 row_split       */ state.అడ్డువరుస_విభజించు,
        /* 8 SubGroupSize    */ state.subgroup_కొలత,
        /* 9 SHMEM_STAGING   */ state.shmem_staging ? 1u : 0u,
        /*10 Flags           */ state.flags,
        /*11 LIMIT_OCCUPANCY_SHMEM */ state.హద్దు_occupancy_shmem,
        /*12 FaTypeK         */ static_cast<uint32_t>(state.k_type),
        /*13 FaTypeV         */ static_cast<uint32_t>(state.v_type),
        /*14 FaBlockBytesK   */ fa_ఖండం_బైట్లు(state.k_type),
        /*15 FaBlockBytesV   */ fa_ఖండం_బైట్లు(state.v_type),
    };
}

స్థిర తార్కిక ggml_vk_matmul_shmem_support(స్థిరరాశి vk_device& device, స్థిరరాశి std::vector<uint32_t>& warptile, తార్కిక mul_mat_id, ggml_type src0_type) {

    uint32_t lut_కొలత = 0;
    ఎంపిక (src0_type) {
    స్థితి GGML_TYPE_IQ1_S:
    స్థితి GGML_TYPE_IQ1_M:
        // Regular matmul uses the compact uint16_t IQ1 grid; the expanded
        // uint32_t grid is only enabled for the q8_1/int-dot vector path.
        lut_కొలత = 2*2048;
        ఆపు;
    స్థితి GGML_TYPE_IQ2_XXS:
        lut_కొలత = 8*256;
        ఆపు;
    స్థితి GGML_TYPE_IQ2_XS:
        lut_కొలత = 8*512;
        ఆపు;
    స్థితి GGML_TYPE_IQ2_S:
        lut_కొలత = 8*1024;
        ఆపు;
    స్థితి GGML_TYPE_IQ3_XXS:
        lut_కొలత = 4*256;
        ఆపు;
    స్థితి GGML_TYPE_IQ3_S:
        lut_కొలత = 4*512;
        ఆపు;
    స్థితి GGML_TYPE_IQ4_NL:
    స్థితి GGML_TYPE_IQ4_XS:
    స్థితి GGML_TYPE_MXFP4:
        lut_కొలత = 4*16;
        ఆపు;
    స్థితి GGML_TYPE_NVFP4:
        // Same kvalues budget as MXFP4 plus ue4m3_fp32_lut[128] (types.glsl, DATA_A_NVFP4).
        lut_కొలత = 4*16 + 128u * (uint32_t)పరిమాణం(దశాంశం);
        ఆపు;
    అప్రమేయ:
        ఆపు;
    }

    // Needs to be kept up to date on shader changes
    // Needs to stay aligned with ggml_vk_mul_mm_spec.
    స్థిరరాశి తార్కిక intel_shmem_stride_pad_zero = device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
                                              device->driver_id == vk::DriverId::eIntelProprietaryWindows;
    స్థిరరాశి uint32_t bank_conflict_స్థానభ్రంశం = intel_shmem_stride_pad_zero ? 0 : (device->coopmat_support ? 8 : 1);
    స్థిరరాశి uint32_t type_కొలత = device->fp16 ? పరిమాణం(ggml_fp16_t) : పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t warps = warptile[0] / warptile[10];

    స్థిరరాశి uint32_t ఎక్కించు_bufs = (warptile[1] + warptile[2]) * (warptile[3] + bank_conflict_స్థానభ్రంశం) * type_కొలత;
    స్థిరరాశి uint32_t mmid_అడ్డువరుస_ids = mul_mat_id ? (warptile[2] * 2 * పరిమాణం(uint16_t)) : 0;
    స్థిరరాశి uint32_t coopmat_stage = device->coopmat_support ? warptile[7] * warptile[8] / warps * పరిమాణం(దశాంశం) : 0;

    స్థిరరాశి uint32_t మొత్తం_కొలత = ఎక్కించు_bufs + mmid_అడ్డువరుస_ids + coopmat_stage + lut_కొలత;
    స్థిరరాశి తార్కిక supported = మొత్తం_కొలత <= device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత;

    VK_LOG_DEBUG("ggml_vk_matmul_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", supported=" << supported);

    తిరిగి supported;
}

స్థిర తార్కిక ggml_vk_matmul_int_shmem_support(స్థిరరాశి vk_device& device, స్థిరరాశి std::vector<uint32_t>& warptile, తార్కిక mul_mat_id, ggml_type src0_type) {

    // FLOAT_TYPE in the shader is float16_t with fp16 support, otherwise float.
    స్థిరరాశి uint32_t fp_కొలత   = device->fp16 ? 2u : 4u;
    స్థిరరాశి uint32_t fp_align  = fp_కొలత;
    స్థిరరాశి uint32_t fp2_కొలత  = 2u * fp_కొలత;
    స్థిరరాశి uint32_t fp2_align = device->fp16 ? 4u : 8u;

    నిర్మాణం member { uint32_t size, align; };
    స్వయంచాలక std430_కొలత = [](std::initializer_list<member> members) {
        uint32_t off = 0, struct_align = 1;
        కొరకు (స్థిరరాశి స్వయంచాలక &m : members) {
            off = (off + m.align - 1) & ~(m.align - 1);
            off += m.size;
            struct_align = std::max(struct_align, m.align);
        }
        తిరిగి (off + struct_align - 1) & ~(struct_align - 1);
    };

    uint32_t ఖండం_a_కొలత = 0;
    ఎంపిక (src0_type) {
        స్థితి GGML_TYPE_Q2_0:    ఖండం_a_కొలత = std430_కొలత({{32, 4}, {fp_కొలత,  fp_align}});                  ఆపు; // qs[8] + dm
        స్థితి GGML_TYPE_Q4_0:    ఖండం_a_కొలత = std430_కొలత({{16, 4}, {fp_కొలత,  fp_align}});                  ఆపు; // qs[16/4] + dm
        స్థితి GGML_TYPE_Q4_1:    ఖండం_a_కొలత = std430_కొలత({{16, 4}, {fp2_కొలత, fp2_align}});                 ఆపు; // qs[16/4] + dm(vec2)
        స్థితి GGML_TYPE_Q5_0:    ఖండం_a_కొలత = std430_కొలత({{16, 4}, {4, 4}, {fp_కొలత,  fp_align}});          ఆపు; // qs[16/4] + qh + dm
        స్థితి GGML_TYPE_Q5_1:    ఖండం_a_కొలత = std430_కొలత({{16, 4}, {4, 4}, {fp2_కొలత, fp2_align}});         ఆపు; // qs[16/4] + qh + dm(vec2)
        స్థితి GGML_TYPE_Q8_0:    ఖండం_a_కొలత = std430_కొలత({{32, 4}, {fp_కొలత,  fp_align}});                  ఆపు; // qs[8] + dm
        స్థితి GGML_TYPE_IQ4_XS:  ఖండం_a_కొలత = std430_కొలత({{32, 4}, {fp_కొలత,  fp_align}});                  ఆపు; // qs[8] + d
        స్థితి GGML_TYPE_MXFP4:   ఖండం_a_కొలత = std430_కొలత({{32, 4}, {fp_కొలత,  fp_align}});                  ఆపు; // qs[8] + d
        స్థితి GGML_TYPE_IQ4_NL:  ఖండం_a_కొలత = std430_కొలత({{32, 4}, {fp_కొలత,  fp_align}});                  ఆపు; // qs[8] + d
        స్థితి GGML_TYPE_NVFP4:   ఖండం_a_కొలత = std430_కొలత({{32, 4}, {fp2_కొలత, fp2_align}});                 ఆపు; // qs[8] + d_scales(vec2)
        స్థితి GGML_TYPE_Q2_K:    ఖండం_a_కొలత = std430_కొలత({{ 8, 4}, {2, 2}, {fp2_కొలత, fp2_align}});         ఆపు; // qs[2] + scales(u8vec2) + dm(vec2)
        స్థితి GGML_TYPE_Q3_K:    ఖండం_a_కొలత = std430_కొలత({{16, 4}, {fp2_కొలత, fp2_align}});                 ఆపు; // qs[4] + d_scales(vec2)
        స్థితి GGML_TYPE_Q4_K:    ఖండం_a_కొలత = std430_కొలత({{16, 4}, {fp2_కొలత, fp2_align}});                 ఆపు; // qs[4] + dm(vec2)
        స్థితి GGML_TYPE_Q5_K:    ఖండం_a_కొలత = std430_కొలత({{32, 4}, {fp2_కొలత, fp2_align}});                 ఆపు; // qs[8] + dm(vec2)
        స్థితి GGML_TYPE_Q6_K:    ఖండం_a_కొలత = std430_కొలత({{32, 4}, {fp2_కొలత, fp2_align}});                 ఆపు; // qs[8] + d_scales(vec2)
        స్థితి GGML_TYPE_IQ3_S:   ఖండం_a_కొలత = std430_కొలత({{32, 4}, {fp_కొలత,  fp_align}});                  ఆపు; // qs[8] + d
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }

    // IQ3_S also copies its 512-entry grid into shared memory (types.glsl, init_iq_shmem)
    స్థిరరాశి uint32_t lut_కొలత = (src0_type == GGML_TYPE_IQ3_S) ? 4*512 : 0;

    // block_b_cache: { int32_t qs[8]; FLOAT_TYPEV2 ds; }
    స్థిరరాశి uint32_t ఖండం_b_కొలత = std430_కొలత({{32, 4}, {fp2_కొలత, fp2_align}});

    స్థిరరాశి uint32_t BM = warptile[1];
    స్థిరరాశి uint32_t BN = warptile[2];
    // mul_mmq.comp: BK_STEP=1 for MUL_MAT_ID, 4 otherwise.
    స్థిరరాశి uint32_t BK_అడుగు = mul_mat_id ? 1u : 4u;

    స్థిరరాశి uint32_t buf_a_కొలత = BM * BK_అడుగు * ఖండం_a_కొలత;
    స్థిరరాశి uint32_t buf_b_కొలత = BN * BK_అడుగు * ఖండం_b_కొలత;
    స్థిరరాశి uint32_t mmid_అడ్డువరుస_ids = mul_mat_id ? (BN * 2u * (uint32_t)పరిమాణం(uint16_t)) : 0u;

    స్థిరరాశి uint32_t మొత్తం_కొలత = buf_a_కొలత + buf_b_కొలత + mmid_అడ్డువరుస_ids + lut_కొలత;
    స్థిరరాశి తార్కిక supported = మొత్తం_కొలత <= device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత;

    VK_LOG_DEBUG("ggml_vk_matmul_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << మొత్తం_కొలత << ", supported=" << supported);

    తిరిగి supported;
}

స్థిర తార్కిక ggml_vk_matmul_cm1_int_shmem_support(స్థిరరాశి vk_device& device, స్థిరరాశి std::vector<uint32_t>& warptile, తార్కిక mul_mat_id, ggml_type src0_type) {

    తార్కిక kscales2 = అబద్ధం;    // two scale sets per block
    తార్కిక has_dm   = అబద్ధం;    // d+m as vec2 + b-side sum
    తార్కిక has_kvalues = అబద్ధం;
    ఎంపిక (src0_type) {
        స్థితి GGML_TYPE_Q4_0: స్థితి GGML_TYPE_Q5_0: స్థితి GGML_TYPE_Q8_0:
            ఆపు;
        స్థితి GGML_TYPE_Q4_1: స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q4_K: స్థితి GGML_TYPE_Q5_K:
            has_dm = నిజం;                          ఆపు;
        స్థితి GGML_TYPE_IQ4_NL: స్థితి GGML_TYPE_IQ4_XS: స్థితి GGML_TYPE_MXFP4:
            has_kvalues = నిజం;                     ఆపు;
        స్థితి GGML_TYPE_Q3_K: స్థితి GGML_TYPE_Q6_K:
            kscales2 = నిజం;                        ఆపు;
        స్థితి GGML_TYPE_NVFP4:
            kscales2 = నిజం; has_kvalues = నిజం;    ఆపు;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t BM         = warptile[1];
    స్థిరరాశి uint32_t BN         = warptile[2];

    స్థిరరాశి uint32_t BK      = 32;
    స్థిరరాశి uint32_t BK_అడుగు = mul_mat_id ? 2u : 4u;
    స్థిరరాశి uint32_t QPITCH  = BK_అడుగు * (BK / 4u) + 4u;
    స్థిరరాశి uint32_t KSCALES = kscales2 ? 2u : 1u;

    uint32_t మొత్తం = 0;
    మొత్తం += BM * QPITCH * (uint32_t)పరిమాణం(uint32_t);   // buf_a_qs
    మొత్తం += BN * QPITCH * (uint32_t)పరిమాణం(uint32_t);   // buf_b_qs
    మొత్తం += has_dm ? (BM * BK_అడుగు * 2u * (uint32_t)పరిమాణం(దశాంశం))   // buf_a_dm (vec2)
                    : (BM * BK_అడుగు * KSCALES * (uint32_t)పరిమాణం(దశాంశం)); // buf_a_d
    మొత్తం += BN * BK_అడుగు * (uint32_t)పరిమాణం(దశాంశం);     // buf_b_d
    ఒకవేళ (has_dm) {
        మొత్తం += BN * BK_అడుగు * (uint32_t)పరిమాణం(దశాంశం); // buf_b_s
    }
    ఒకవేళ (has_kvalues) {
        మొత్తం += 16u * (uint32_t)పరిమాణం(int8_t);         // cm1_kvalues[16]
    }
    ఒకవేళ (src0_type == GGML_TYPE_NVFP4 && !device->ocp_fp4) {
        మొత్తం += 128u * (uint32_t)పరిమాణం(దశాంశం);         // ue4m3_fp32_lut[128]
    }
    ఒకవేళ (mul_mat_id) {
        మొత్తం += BN * 2u * (uint32_t)పరిమాణం(uint16_t);   // row_ids[BN] (u16vec2)
    }

    స్థిరరాశి తార్కిక supported = మొత్తం <= device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత;

    VK_LOG_DEBUG("ggml_vk_matmul_cm1_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << మొత్తం << ", supported=" << supported);

    తిరిగి supported;
}

స్థిర స్థిరరాశి std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
    {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

స్థిర స్థిరరాశి std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
};

స్థిర std::vector<GpuPipelineConfig> gpu_pipeline_configs = {
    {
        vk_device_architecture::AMD_RDNA1,
        {
            rdna1_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_కొలత
    },
    {
        vk_device_architecture::AMD_RDNA2,
        {
            rdna2_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_కొలత
    },
};

uint32_t get_subgroup_కొలత(స్థిరరాశి std::string &pipeline_name, స్థిరరాశి vk_device_architecture &arch) {
    కొరకు (స్థిరరాశి స్వయంచాలక &config : gpu_pipeline_configs) {
        ఒకవేళ (config.arch == arch) {
            స్వయంచాలక pipIt = config.pipelines.find(pipeline_name);
            ఒకవేళ (pipIt != config.pipelines.end()) {
                తిరిగి pipIt->second;
            }
            std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
            std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                      [](స్థిరరాశి స్వయంచాలక &a, స్థిరరాశి స్వయంచాలక &b) { తిరిగి a.మొదటి.size() > b.మొదటి.size(); });
            కొరకు (స్థిరరాశి స్వయంచాలక &నమోదు_2 : sorted_pipelines) {
                ఒకవేళ (pipeline_name.find(నమోదు_2.మొదటి) != std::string::npos) {
                    తిరిగి నమోదు_2.second;
                }
            }
            తిరిగి config.default_subgroup_కొలత;
        }
    }
    తిరిగి 0; // If no matching configuration is found
}

స్థిర తార్కిక ggml_vk_fa_type_needs_shmem(ggml_type type) {
    ఎంపిక (type) {
    స్థితి GGML_TYPE_IQ4_NL:
        తిరిగి నిజం;
    అప్రమేయ:
        తిరిగి అబద్ధం;
    }
}

స్థిర తార్కిక ggml_vk_fa_scalar_ఉపయోగిస్తుంది_mmq(స్థిరరాశి vk_device& device, ggml_type k_type, ggml_type v_type) {
#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    తిరిగి device->integer_dot_లబ్ధం && device->subgroup_clustered &&
           !ggml_vk_fa_type_needs_shmem(v_type) &&
           (k_type == GGML_TYPE_Q4_0 || k_type == GGML_TYPE_Q4_1 ||
            k_type == GGML_TYPE_Q5_0 || k_type == GGML_TYPE_Q5_1 ||
            k_type == GGML_TYPE_Q8_0);
#లేకపోతే
    GGML_వాడని(device);
    GGML_వాడని(k_type);
    GGML_వాడని(v_type);
    తిరిగి అబద్ధం;
#అయితే_ముగింపు
}

శూన్య ggml_vk_ఎక్కించు_shaders(vk_device& device, vk_pipeline requested) {
    VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");

    // some shaders have a minimum subgroup size
    స్థిరరాశి uint32_t subgroup_కొలత_8 = std::max(device->subgroup_కొలత, 8u);
    స్థిరరాశి uint32_t subgroup_కొలత_16 = std::max(device->subgroup_కొలత, 16u);
    స్థిరరాశి uint32_t subgroup_కొలత_32 = std::max(device->subgroup_కొలత, 32u);

    // clamp WARP for l_/m_ warptiles so WM <= BM (breaks on subgroupSize > 64)
    స్థిరరాశి uint32_t mm_warp_8  = std::min(subgroup_కొలత_8,  64u);
    స్థిరరాశి uint32_t mm_warp_16 = std::min(subgroup_కొలత_16, 64u);

    స్థిరరాశి uint32_t mul_mat_subgroup_కొలత = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_కొలత_control) ? device->subgroup_min_కొలత : device->subgroup_కొలత;
    స్థిరరాశి uint32_t mul_mat_subgroup_కొలత_8 = std::max(mul_mat_subgroup_కొలత, 8u);
    స్థిరరాశి uint32_t mul_mat_subgroup_కొలత_16 = std::max(mul_mat_subgroup_కొలత, 16u);
    స్థిరరాశి uint32_t mul_mat_subgroup_కొలత_32 = std::max(mul_mat_subgroup_కొలత, 32u);
    స్థిరరాశి uint32_t mul_mat_mm_warp_8  = std::min(mul_mat_subgroup_కొలత_8,  64u);
    స్థిరరాశి uint32_t mul_mat_mm_warp_16 = std::min(mul_mat_subgroup_కొలత_16, 64u);

    స్థిరరాశి తార్కిక subgroup_min_కొలత_16 = (!device->subgroup_కొలత_control && device->subgroup_కొలత >= 16) ||
                                      (device->subgroup_కొలత_control && device->subgroup_max_కొలత >= 16);

    // mulmat
    // Warptile layout (indices match mul_mm.comp constantIDs):
    //   [0..9]  : BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK
    //   [10]    : WARP / required_subgroup_size (read via WARP_SIZE_IDX)
    స్థిర constexpr size_t WARP_కొలత_IDX = 10;
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
    స్థిరరాశి తార్కిక cm1_use_wave32 = device->vendor_id == VK_VENDOR_ID_AMD &&
                                device->subgroup_కొలత_control &&
                                device->subgroup_min_కొలత <= 32 && device->subgroup_max_కొలత >= 32;
    స్థిరరాశి uint32_t cm1_sg = cm1_use_wave32 ? 32 : device->subgroup_కొలత;

    vk_pipeline wait_pipeline;
    CompileTask claimed_task {};
    తార్కిక has_claimed_task = అబద్ధం;

    // The rest of the walk reads and writes shared device state, so hold the
    // lock until we're done deciding what to compile.
    std::unique_lock<std::mutex> compile_తాళం(device->compile_mutex);

    ఒకవేళ (device->coopmat2) {
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
        స్థిరరాశి uint32_t mmqid_bk = device->coopmat2_decode_సదిశ ? 64u : 32u;
        l_warptile_mmqid = { 256, 128, 128, mmqid_bk, 1 };
        m_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        s_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        l_mmqid_wg_denoms = { 128, 128, 1 };
        m_mmqid_wg_denoms = { 128, 64, 1 };
        s_mmqid_wg_denoms = { 128, 64, 1 };

        l_align = 128;
        m_align =  64;
        s_align =  32;
    } లేకపోతే {
        // Matrix cores require different warp group sizes
        స్థిరరాశి uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
        స్థిరరాశి uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
        స్థిరరాశి uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
        స్థిరరాశి uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
        స్థిరరాశి uint32_t tn_m = device->coopmat_support ? device->coopmat_n : 2;
        స్థిరరాశి uint32_t tn_s = device->coopmat_support ? device->coopmat_n : 2;
        స్థిరరాశి uint32_t tk_l = device->coopmat_support ? device->coopmat_k : 1;
        స్థిరరాశి uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
        స్థిరరాశి uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;

        స్థిరరాశి uint32_t itm = device->coopmat_int_m;
        స్థిరరాశి uint32_t itn = device->coopmat_int_n;
        స్థిరరాశి uint32_t itk = device->coopmat_int_k;

        స్థిరరాశి uint32_t s_warptile_wm = device->subgroup_కొలత == 8 ? 8 : 32;

        l_warptile = { 128,             128, 128, 16, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile = { 128,              64,  64, 16, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile = { subgroup_కొలత_32, 32,  32, 16, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_కొలత_8 };

        l_warptile_mmq = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile_mmq = { 128,              64,  64, 32, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile_mmq = { subgroup_కొలత_32, 32,  32, 32, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_కొలత_8 };

        // Integer MMQ has a smaller shared memory profile, but heavier register use
        l_warptile_mmq_int = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int = { 128,              64,  64, 32, mm_warp_8,     32, 2, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int = { subgroup_కొలత_32, 32,  32, 32, s_warptile_wm, 32, 2, 2, 1, 1, subgroup_కొలత_8 };

        స్థిరరాశి స్వయంచాలక cm1_bs = [cm1_sg](uint32_t bm, uint32_t bn) {
            తిరిగి cm1_sg * (bm / std::min(cm1_sg, bm)) * (bn / 32);
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
        s_warptile_mmq_int_k = { subgroup_కొలత_32,   32,  32, 32, s_warptile_wm, 32, 1, 2, 1, 1, subgroup_కొలత_8 };

        l_warptile_id = { 128,                      128, 128, 16, mul_mat_mm_warp_16 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_16 };
        m_warptile_id = { 128,                       64,  64, 16, mul_mat_mm_warp_16,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_16 };
        s_warptile_id = { mul_mat_subgroup_కొలత_16,  32,  32, 16, s_warptile_wm,          32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_కొలత_16 };

        l_warptile_mmqid = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_8 };
        m_warptile_mmqid = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_8 };
        s_warptile_mmqid = { mul_mat_subgroup_కొలత_32,   32,  32, 32, s_warptile_wm,         32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_కొలత_8 };

        l_warptile_mmqid_int = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, 4, 4, 1, mul_mat_mm_warp_8 };
        m_warptile_mmqid_int = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, 2, 2, 1, mul_mat_mm_warp_8 };
        s_warptile_mmqid_int = { mul_mat_subgroup_కొలత_32,   32,  32, 32, s_warptile_wm,         32, 2, 2, 1, 1, mul_mat_subgroup_కొలత_8 };

        l_warptile_mmqid_int_k = { 128,                     128, 128, 32, mul_mat_mm_warp_16 * 2, 64, 1, 4, 4, 1, mul_mat_mm_warp_16 };
        m_warptile_mmqid_int_k = { 128,                      64,  64, 32, mul_mat_mm_warp_16,     32, 1, 2, 2, 1, mul_mat_mm_warp_16 };
        s_warptile_mmqid_int_k = { mul_mat_subgroup_కొలత_32, 32,  32, 32, s_warptile_wm,          32, 1, 2, 1, 1, mul_mat_subgroup_కొలత_16 };

        // chip specific tuning
        ఒకవేళ ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
            m_warptile_mmq = m_warptile_mmq_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
            m_warptile_mmqid = m_warptile_mmqid_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
        } లేకపోతే ఒకవేళ (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary) {
            // This is intentionally using tx_m values, slight performance increase
            l_warptile = { 256, 128, 128, 16, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq_int_k = { 256, 128, 128, 32, mm_warp_16, 64, 1, 4, 2, 1, mm_warp_16 };
        } లేకపోతే ఒకవేళ (device->vendor_id == VK_VENDOR_ID_QUALCOMM && device->coopmat_support) {
            m_warptile     = { 64, 64, 64, 16, 64, 64, 1, tm_l, tn_l, tk_l, 64 };
            m_warptile_mmq = { 64, 64, 64, 32, 64, 64, 1, tm_m, tn_m, tk_m, 64 };
        }

        l_mmq_wg_denoms = l_wg_denoms = {128, 128, 1 };
        m_mmq_wg_denoms = m_wg_denoms = { 64,  64, 1 };
        s_mmq_wg_denoms = s_wg_denoms = { 32,  32, 1 };
        l_align = 128;
        m_align =  64;
        s_align =  32;

        ఒకవేళ (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support) {
            // Xe1/Xe2/Xe3 with coopmat enabled - warptile performance tuning
            l_warptile = { 512, 128, 128, 16, mm_warp_8, 32, 2, tm_l, tn_l, tk_l, mm_warp_8 };
            ఒకవేళ (device->architecture == INTEL_XE1) {
                l_warptile_mmq  = { 512, 256, 128, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 256, 128, 1 };
                l_align         = 32;  //set as BK
            } లేకపోతే {
                l_warptile_mmq  = { 512, 128, 256, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 128, 256, 1 };
                l_align         = 32;  //set as BK
            }
        }

        స్థిరరాశి తార్కిక use_cm1_int = device->coopmat_int_support &&
                                 (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);

        కొరకు (uint32_t i = 0; i < GGML_TYPE_లెక్క; ++i) {
            ggml_type t = (ggml_type)i;
            // Disable medium and large matrix multiplication if not enough shared memory is available
            // Check mmq warptiles as the largest configuration
            // Throw an error if not enough for any matrix multiplication is available
            ఒకవేళ (!ggml_vk_matmul_shmem_support(device, s_warptile_mmq, అబద్ధం, t)) {
                std::cerr << "ggml_vulkan: Error: Shared memory size too small for matrix multiplication." << std::పంక్తి_ముగింపు;
                విసురు std::runtime_error("Shared memory size too small for matrix multiplication.");
            } లేకపోతే ఒకవేళ (!ggml_vk_matmul_shmem_support(device, m_warptile_mmq, అబద్ధం, t)) {
                device->mul_mat_m[i] = అబద్ధం;
                device->mul_mat_l[i] = అబద్ధం;
            } లేకపోతే ఒకవేళ (!ggml_vk_matmul_shmem_support(device, l_warptile_mmq, అబద్ధం, t)) {
                device->mul_mat_l[i] = అబద్ధం;
            }

            // Disable mul_mat_id if not enough shared memory is available
            ఒకవేళ (!ggml_vk_matmul_shmem_support(device, s_warptile_mmqid, నిజం, t)) {
                device->mul_mat_id_s[i] = అబద్ధం;
                device->mul_mat_id_m[i] = అబద్ధం;
                device->mul_mat_id_l[i] = అబద్ధం;
            } లేకపోతే ఒకవేళ (!ggml_vk_matmul_shmem_support(device, m_warptile_mmqid, నిజం, t)) {
                device->mul_mat_id_m[i] = అబద్ధం;
                device->mul_mat_id_l[i] = అబద్ధం;
            } లేకపోతే ఒకవేళ (!ggml_vk_matmul_shmem_support(device, l_warptile_mmqid, నిజం, t)) {
                device->mul_mat_id_l[i] = అబద్ధం;
            }

            // The q8_1 mmq path has its own (larger) shmem layout, check it separately.
            // K-quants and IQ3_S use the _int_k warptiles, others use _int.
            // cm1 splits k-tiles on the KSCALES==2 types and shares tiles between dense/id.
            స్థిరరాశి తార్కిక is_k_quant = (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                                     t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                                     t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ3_S);
            స్థిరరాశి తార్కిక cm1_k_tile = (t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q6_K ||
                                     t == GGML_TYPE_NVFP4);

            స్థిరరాశి స్వయంచాలక & s_int   = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmq_int_k     : s_warptile_mmq_int);
            స్థిరరాశి స్వయంచాలక & m_int   = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmq_int_k     : m_warptile_mmq_int);
            స్థిరరాశి స్వయంచాలక & l_int   = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmq_int_k     : l_warptile_mmq_int);
            స్థిరరాశి స్వయంచాలక & s_intid = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmqid_int_k   : s_warptile_mmqid_int);
            స్థిరరాశి స్వయంచాలక & m_intid = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmqid_int_k   : m_warptile_mmqid_int);
            స్థిరరాశి స్వయంచాలక & l_intid = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmqid_int_k   : l_warptile_mmqid_int);

            స్థిరరాశి స్వయంచాలక int_shmem_support = [&](స్థిరరాశి std::vector<uint32_t>& wt, తార్కిక id) {
                తిరిగి use_cm1_int ? ggml_vk_matmul_cm1_int_shmem_support(device, wt, id, t)
                                   : ggml_vk_matmul_int_shmem_support(device, wt, id, t);
            };

            ఒకవేళ (!int_shmem_support(s_int, అబద్ధం)) {
                device->mul_mat_s_int[i] = అబద్ధం;
                device->mul_mat_m_int[i] = అబద్ధం;
                device->mul_mat_l_int[i] = అబద్ధం;
            } లేకపోతే ఒకవేళ (!int_shmem_support(m_int, అబద్ధం)) {
                device->mul_mat_m_int[i] = అబద్ధం;
                device->mul_mat_l_int[i] = అబద్ధం;
            } లేకపోతే ఒకవేళ (!int_shmem_support(l_int, అబద్ధం)) {
                device->mul_mat_l_int[i] = అబద్ధం;
            }

            ఒకవేళ (!int_shmem_support(s_intid, నిజం)) {
                device->mul_mat_id_s_int[i] = అబద్ధం;
                device->mul_mat_id_m_int[i] = అబద్ధం;
                device->mul_mat_id_l_int[i] = అబద్ధం;
            } లేకపోతే ఒకవేళ (!int_shmem_support(m_intid, నిజం)) {
                device->mul_mat_id_m_int[i] = అబద్ధం;
                device->mul_mat_id_l_int[i] = అబద్ధం;
            } లేకపోతే ఒకవేళ (!int_shmem_support(l_intid, నిజం)) {
                device->mul_mat_id_l_int[i] = అబద్ధం;
            }
        }
    }

    స్వయంచాలక స్థిరరాశి &ggml_vk_create_pipeline = [&](vk_device& device, vk_pipeline& పునాది_pipeline, స్థిరరాశి అక్షరం *name, size_t spv_కొలత, స్థిరరాశి శూన్య* spv_దత్తాంశం, స్థిరరాశి అక్షరం *entrypoint,
                                              uint32_t parameter_లెక్క, uint32_t నెట్టు_constant_కొలత, std::array<uint32_t, 3> wg_denoms, స్థిరరాశి std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, తార్కిక disable_robustness = అబద్ధం, తార్కిక require_నిండిన_subgroups = అబద్ధం, uint32_t required_subgroup_కొలత = 0) {

        ఒకవేళ (!require_నిండిన_subgroups && required_subgroup_కొలత == 0) {
            required_subgroup_కొలత = get_subgroup_కొలత(name, device->architecture);
        }

        vk_pipeline *ptr = &పునాది_pipeline;

        పూర్ణాంకం అంకె_pipelines = 1;
#ఒకవేళ defined(VK_EXT_shader_64bit_indexing)
        ఒకవేళ (device->shader_64b_indexing) {
            అంకె_pipelines = 2;
        }
#అయితే_ముగింపు
        కొరకు (పూర్ణాంకం i = 0; i < అంకె_pipelines; ++i, ptr = &(*ptr)->తదుపరి) {
            vk_pipeline &pipeline = *ptr;
            ఒకవేళ (!pipeline) {
                pipeline = std::make_shared<vk_pipeline_struct>();
            }
            ఒకవేళ (!pipeline->initialized) {
                pipeline->name = name;
                pipeline->parameter_లెక్క = parameter_లెక్క;
                pipeline->నెట్టు_constant_కొలత = నెట్టు_constant_కొలత;
                pipeline->wg_denoms = wg_denoms;
                pipeline->align = align;
                pipeline->initialized = నిజం;
#ఒకవేళ defined(VK_EXT_shader_64bit_indexing)
                pipeline->is_64b_indexing = (i == 1);
#అయితే_ముగింపు
            }

            // We only care about the pipeline this call asked for; the rest
            // (including the 64-bit indexing variant) are handled by their
            // own request_descriptor_sets / load_shaders calls.
            ఒకవేళ (pipeline.get() != requested.get()) {
                కొనసాగు;
            }

            ఒకవేళ (pipeline->compiled) {
                కొనసాగు;
            }

            wait_pipeline = pipeline;

            ఒకవేళ (!pipeline->compile_pending) {
                pipeline->compile_pending = నిజం;
                claimed_task.pipeline = pipeline;
                claimed_task.spv_కొలత = spv_కొలత;
                claimed_task.spv_దత్తాంశం = spv_దత్తాంశం;
                claimed_task.entrypoint = entrypoint;
                claimed_task.parameter_లెక్క = parameter_లెక్క;
                claimed_task.wg_denoms = wg_denoms;
                claimed_task.specialization_constants = specialization_constants;
                claimed_task.disable_robustness = disable_robustness;
                claimed_task.require_నిండిన_subgroups = require_నిండిన_subgroups;
                claimed_task.required_subgroup_కొలత = required_subgroup_కొలత;
                has_claimed_task = నిజం;
            }
        }
    };

    స్వయంచాలక స్థిరరాశి &ggml_vk_create_pipeline2 = [&](vk_device& device, vk_pipeline& pipeline, స్థిరరాశి std::string &name, size_t spv_కొలత, స్థిరరాశి శూన్య* spv_దత్తాంశం, స్థిరరాశి అక్షరం *entrypoint,
                                              uint32_t parameter_లెక్క, uint32_t నెట్టు_constant_కొలత, std::array<uint32_t, 3> wg_denoms, స్థిరరాశి std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, తార్కిక disable_robustness = అబద్ధం, తార్కిక require_నిండిన_subgroups = అబద్ధం, uint32_t required_subgroup_కొలత = 0) {
        తిరిగి ggml_vk_create_pipeline(device, pipeline, name.c_str(), spv_కొలత, spv_దత్తాంశం, entrypoint,
                                       parameter_లెక్క, నెట్టు_constant_కొలత, wg_denoms, specialization_constants,
                                       align, disable_robustness, require_నిండిన_subgroups, required_subgroup_కొలత);
    };

    // FA scalar has two SPIR-V modules (MMQ vs non-MMQ); FA cm1 has one. K/V
    // quant type is selected at runtime via the FaTypeK / FaTypeV spec constants.

    కొరకు (స్వయంచాలక &fa : device->pipeline_flash_attn_f32_f16) {
        ఒకవేళ (fa.మొదటి.path != FA_SCALAR) కొనసాగు;
        స్థిరరాశి uint32_t Br = fa.మొదటి.Br;
        స్థిరరాశి uint32_t Bc = fa.మొదటి.Bc;
        స్థిరరాశి తార్కిక aligned = fa.మొదటి.aligned;
        స్థిరరాశి తార్కిక f32acc = fa.మొదటి.f32acc;
        స్థిరరాశి uint32_t fa_sgs = fa.మొదటి.subgroup_కొలత;
        స్థిరరాశి తార్కిక fa_ds = fa.మొదటి.subgroup_కొలత == 0;

        స్థిరరాశి తార్కిక bf16_kv = fa.మొదటి.k_type == GGML_TYPE_BF16;
        స్థిరరాశి తార్కిక use_mmq = ggml_vk_fa_scalar_ఉపయోగిస్తుంది_mmq(device, fa.మొదటి.k_type, fa.మొదటి.v_type);
        స్థిరరాశి శూన్య * spv_దత్తాంశం = శూన్యసూచిక;
        size_t spv_కొలత = 0;
        స్థిరరాశి అక్షరం *name = శూన్యసూచిక;
        ఒకవేళ (bf16_kv) {
            spv_దత్తాంశం = flash_attn_f32_f16_fp32_దత్తాంశం;
            spv_కొలత = flash_attn_f32_f16_fp32_len;
            name = aligned ? "flash_attn_f32_bf16_aligned" : "flash_attn_f32_bf16";
        } లేకపోతే ఒకవేళ (use_mmq) {
#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ఒకవేళ (device->fp16) {
                ఒకవేళ (f32acc) { spv_దత్తాంశం = flash_attn_f32_f16_int8_దత్తాంశం;        spv_కొలత = flash_attn_f32_f16_int8_len; }
                లేకపోతే        { spv_దత్తాంశం = flash_attn_f32_f16_f16acc_int8_దత్తాంశం; spv_కొలత = flash_attn_f32_f16_f16acc_int8_len; }
            } లేకపోతే {
                spv_దత్తాంశం = flash_attn_f32_f16_fp32_int8_దత్తాంశం;
                spv_కొలత = flash_attn_f32_f16_fp32_int8_len;
            }
#అయితే_ముగింపు
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        } లేకపోతే {
            ఒకవేళ (device->fp16) {
                ఒకవేళ (device->dot2_f16) {
                    ఒకవేళ (f32acc) { spv_దత్తాంశం = flash_attn_f32_f16_dot2_దత్తాంశం;        spv_కొలత = flash_attn_f32_f16_dot2_len; }
                    లేకపోతే        { spv_దత్తాంశం = flash_attn_f32_f16_dot2_f16acc_దత్తాంశం; spv_కొలత = flash_attn_f32_f16_dot2_f16acc_len; }
                } లేకపోతే {
                    ఒకవేళ (f32acc) { spv_దత్తాంశం = flash_attn_f32_f16_దత్తాంశం;        spv_కొలత = flash_attn_f32_f16_len; }
                    లేకపోతే        { spv_దత్తాంశం = flash_attn_f32_f16_f16acc_దత్తాంశం; spv_కొలత = flash_attn_f32_f16_f16acc_len; }
                }
            } లేకపోతే {
                spv_దత్తాంశం = flash_attn_f32_f16_fp32_దత్తాంశం;
                spv_కొలత = flash_attn_f32_f16_fp32_len;
            }
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        }
        ggml_vk_create_pipeline(device, fa.second, name, spv_కొలత, spv_దత్తాంశం, "main", 8,
                                పరిమాణం(vk_flash_attn_నెట్టు_constants), {Br, 1, 1},
                                get_fa_spec_constants(fa.మొదటి), aligned ? Bc : 1, నిజం,
                                !fa_ds, !fa_ds ? fa_sgs : 0);
    }

#ఒకవేళ defined(VK_KHR_cooperative_మాతృక) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ఒకవేళ (device->coopmat1_fa_support) {
        కొరకు (స్వయంచాలక &fa : device->pipeline_flash_attn_f32_f16) {
            ఒకవేళ (fa.మొదటి.path != FA_COOPMAT1) కొనసాగు;
            స్థిరరాశి uint32_t Br = fa.మొదటి.Br;
            స్థిరరాశి uint32_t Bc = fa.మొదటి.Bc;
            స్థిరరాశి తార్కిక aligned = fa.మొదటి.aligned;
            స్థిరరాశి తార్కిక f32acc = fa.మొదటి.f32acc;
            స్థిరరాశి uint32_t fa_sgs = fa.మొదటి.subgroup_కొలత;
            స్థిరరాశి తార్కిక fa_ds = fa.మొదటి.subgroup_కొలత == 0;

            స్థిరరాశి తార్కిక bf16_kv = fa.మొదటి.k_type == GGML_TYPE_BF16;

            స్థిరరాశి శూన్య * spv_దత్తాంశం;
            size_t spv_కొలత;
            స్థిరరాశి అక్షరం *name;
            ఒకవేళ (bf16_kv) {
#ఒకవేళ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                ఒకవేళ (!device->coopmat_bf16_support) కొనసాగు;
                spv_దత్తాంశం = flash_attn_f32_f16_bf16_cm1_దత్తాంశం;
                spv_కొలత = flash_attn_f32_f16_bf16_cm1_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm1" : "flash_attn_f32_bf16_cm1";
#లేకపోతే
                కొనసాగు;
#అయితే_ముగింపు
            } లేకపోతే {
                ఒకవేళ (f32acc) { spv_దత్తాంశం = flash_attn_f32_f16_cm1_దత్తాంశం;        spv_కొలత = flash_attn_f32_f16_cm1_len; }
                లేకపోతే        { spv_దత్తాంశం = flash_attn_f32_f16_f16acc_cm1_దత్తాంశం; spv_కొలత = flash_attn_f32_f16_f16acc_cm1_len; }
                name = aligned ? "flash_attn_f32_f16_aligned_cm1" : "flash_attn_f32_f16_cm1";
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_కొలత, spv_దత్తాంశం, "main", 8,
                                    పరిమాణం(vk_flash_attn_నెట్టు_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.మొదటి), aligned ? Bc : 1, నిజం,
                                    !fa_ds, !fa_ds ? fa_sgs : 0);
        }
    }
#అయితే_ముగింపు

#ఒకవేళ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    ఒకవేళ (device->coopmat2) {
        కొరకు (స్వయంచాలక &fa : device->pipeline_flash_attn_f32_f16) {
            ఒకవేళ (fa.మొదటి.path != FA_COOPMAT2) కొనసాగు;
            స్థిరరాశి uint32_t Br = fa.మొదటి.Br;
            స్థిరరాశి uint32_t Bc = fa.మొదటి.Bc;
            స్థిరరాశి తార్కిక aligned = fa.మొదటి.aligned;
            స్థిరరాశి తార్కిక f32acc = fa.మొదటి.f32acc;

            స్థిరరాశి తార్కిక bf16_kv = fa.మొదటి.k_type == GGML_TYPE_BF16;
            స్థిరరాశి శూన్య * spv_దత్తాంశం;
            size_t spv_కొలత;
            స్థిరరాశి అక్షరం * name;
            ఒకవేళ (bf16_kv) {
#ఒకవేళ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                ఒకవేళ (!device->coopmat2_bf16_support) కొనసాగు;
                spv_దత్తాంశం = flash_attn_f32_f16_bf16_cm2_దత్తాంశం;
                spv_కొలత = flash_attn_f32_f16_bf16_cm2_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm2" : "flash_attn_f32_bf16_cm2";
#లేకపోతే
                కొనసాగు;
#అయితే_ముగింపు
            } లేకపోతే ఒకవేళ (aligned) {
                ఒకవేళ (f32acc) { spv_దత్తాంశం = flash_attn_f32_f16_cm2_దత్తాంశం;        spv_కొలత = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_aligned_f32acc_cm2"; }
                లేకపోతే        { spv_దత్తాంశం = flash_attn_f32_f16_f16acc_cm2_దత్తాంశం; spv_కొలత = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_aligned_f16acc_cm2"; }
            } లేకపోతే {
                ఒకవేళ (f32acc) { spv_దత్తాంశం = flash_attn_f32_f16_cm2_దత్తాంశం;        spv_కొలత = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_f32acc_cm2"; }
                లేకపోతే        { spv_దత్తాంశం = flash_attn_f32_f16_f16acc_cm2_దత్తాంశం; spv_కొలత = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_f16acc_cm2"; }
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_కొలత, spv_దత్తాంశం, "main", 8,
                                    పరిమాణం(vk_flash_attn_నెట్టు_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.మొదటి), aligned ? Bc : 1, నిజం, అబద్ధం, 0);
        }
    }
#అయితే_ముగింపు

    స్వయంచాలక స్థిరరాశి &ggml_vk_mul_mm_spec = [&device](std::vector<uint32_t> spec, తార్కిక aligned) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        ఒకవేళ (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=12: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=13: APPLY_SLM_A_RESHAPE = true
        }
        తిరిగి spec;
    };

    స్వయంచాలక స్థిరరాశి &ggml_vk_mul_mm_spec_quant = [&device](std::vector<uint32_t> spec, తార్కిక aligned, uint32_t type) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        spec.push_back(type);               // constantID=12: MmTypeA
        ఒకవేళ (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=13: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=14: APPLY_SLM_A_RESHAPE = true
        }
        తిరిగి spec;
    };

    స్థిర స్థిరరాశి ggml_type non_lut_quant_types[] = {
        GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
    };

#నిర్వచించు FOR_EACH_LUT_TYPE_NONFP4(X) \
    X(GGML_TYPE_IQ1_S,   iq1_s)   \
    X(GGML_TYPE_IQ1_M,   iq1_m)   \
    X(GGML_TYPE_IQ2_XXS, iq2_xxs) \
    X(GGML_TYPE_IQ2_XS,  iq2_xs)  \
    X(GGML_TYPE_IQ2_S,   iq2_s)   \
    X(GGML_TYPE_IQ3_XXS, iq3_xxs) \
    X(GGML_TYPE_IQ3_S,   iq3_s)   \
    X(GGML_TYPE_IQ4_XS,  iq4_xs)  \
    X(GGML_TYPE_IQ4_NL,  iq4_nl)
#నిర్వచించు FOR_EACH_LUT_FP4_TYPE(X) \
    X(GGML_TYPE_MXFP4,   mxfp4)   \
    X(GGML_TYPE_NVFP4,   nvfp4)
#నిర్వచించు FOR_EACH_LUT_TYPE(X) \
    FOR_EACH_LUT_TYPE_NONFP4(X)  \
    FOR_EACH_LUT_FP4_TYPE(X)

    స్థిరరాశి పూర్ణాంకం mul_mat_id_param_లెక్క = 5;

    ఉపయోగించి spec_fn_t = std::function<std::vector<uint32_t>(స్థిరరాశి std::vector<uint32_t>&, తార్కిక)>;
    స్వయంచాలక స్థిరరాశి &create_mm_pipelines = [&](
        స్థిరరాశి vk_matmul_pipeline_కీలకం& కీలకం,
        స్థిరరాశి std::vector<vk_tile_config>& tile_configs,
        స్థిరరాశి std::string& shader_name, size_t spv_len, స్థిరరాశి శూన్య* spv_దత్తాంశం,
        uint32_t నెట్టు_constant_కొలత, uint32_t param_లెక్క,
        స్థిరరాశి spec_fn_t& spec_fn,
        తార్కిక disable_robustness = అబద్ధం, తార్కిక require_నిండిన_subgroups = అబద్ధం, uint32_t required_subgroup_కొలత = 0,
        తార్కిక create_aligned = నిజం, తార్కిక pin_subgroup_గమ్యం_warp = అబద్ధం
    ) {
        స్వయంచాలక& vec = device->pipeline_matmul[కీలకం];
        స్థిరరాశి తార్కిక మొదటి_call = vec.ఖాళీలు();
        కొరకు (size_t i = 0; i < tile_configs.size(); i++) {
            స్థిరరాశి స్వయంచాలక& tc = tile_configs[i];

            // Intel coopmat1 pins the required subgroup size to each warptile's WARP element.
            స్థిరరాశి uint32_t rsgs = pin_subgroup_గమ్యం_warp ? tc.warptile[WARP_కొలత_IDX] : required_subgroup_కొలత;
            స్థిరరాశి తార్కిక     rfs  = require_నిండిన_subgroups || pin_subgroup_గమ్యం_warp;

            ఒకవేళ (మొదటి_call) {
                vk_matmul_pipeline_pair pair{};
                pair.align = tc.align;
                std::string suffix = "_" + std::to_string(i);
                pair.unaligned = std::make_shared<vk_pipeline_struct>();
                ఒకవేళ (create_aligned) {
                    pair.aligned = std::make_shared<vk_pipeline_struct>();
                }
                vec.push_back(pair);
            }

            ggml_vk_create_pipeline(device, vec[i].unaligned,
                vec[i].unaligned->name.ఖాళీలు() ? (shader_name + "_" + std::to_string(i)).c_str() : vec[i].unaligned->name.c_str(),
                spv_len, spv_దత్తాంశం, "main", param_లెక్క, నెట్టు_constant_కొలత,
                tc.wg_denoms, spec_fn(tc.warptile, అబద్ధం), 1,
                disable_robustness, rfs, rsgs);

            ఒకవేళ (vec[i].aligned) {
                ggml_vk_create_pipeline(device, vec[i].aligned,
                    vec[i].aligned->name.ఖాళీలు() ? (shader_name + "_aligned_" + std::to_string(i)).c_str() : vec[i].aligned->name.c_str(),
                    spv_len, spv_దత్తాంశం, "main", param_లెక్క, నెట్టు_constant_కొలత,
                    tc.wg_denoms, spec_fn(tc.warptile, నిజం), tc.align,
                    disable_robustness, rfs, rsgs);
            }
        }
    };

    స్వయంచాలక filter_tc = [&](స్థిరరాశి std::vector<vk_tile_config>& configs, ggml_type type, తార్కిక is_id, తార్కిక is_int = అబద్ధం) -> std::vector<vk_tile_config> {
        std::vector<vk_tile_config> ఫలితం;
        తార్కిక enabled[3];
        ఒకవేళ (is_int) {
            enabled[0] = is_id ? device->mul_mat_id_s_int[type] : device->mul_mat_s_int[type];
            enabled[1] = is_id ? device->mul_mat_id_m_int[type] : device->mul_mat_m_int[type];
            enabled[2] = is_id ? device->mul_mat_id_l_int[type] : device->mul_mat_l_int[type];
        } లేకపోతే {
            enabled[0] = is_id ? device->mul_mat_id_s[type] : device->mul_mat_s[type];
            enabled[1] = is_id ? device->mul_mat_id_m[type] : device->mul_mat_m[type];
            enabled[2] = is_id ? device->mul_mat_id_l[type] : device->mul_mat_l[type];
        }
        కొరకు (size_t i = 0; i < configs.size() && i < 3; i++) {
            ఒకవేళ (enabled[i]) ఫలితం.push_back(configs[i]);
        }
        తిరిగి ఫలితం;
    };

    std::vector<vk_tile_config> tc_mm = {{s_warptile, s_wg_denoms, s_align}, {m_warptile, m_wg_denoms, m_align}, {l_warptile, l_wg_denoms, l_align}};
    std::vector<vk_tile_config> tc_mmq = {{s_warptile_mmq, s_mmq_wg_denoms, s_align}, {m_warptile_mmq, m_mmq_wg_denoms, m_align}, {l_warptile_mmq, l_mmq_wg_denoms, l_align}};

#ఒకవేళ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    ఒకవేళ (device->coopmat2) {
        స్వయంచాలక స్థిరరాశి &ggml_vk_mul_mm_cm2_spec = [&](std::vector<uint32_t> spec, తార్కిక aligned, uint32_t type = UINT32_MAX) {
            spec.push_back(aligned ? 1u : 0u);        // ALIGNED
            spec.push_back(device->subgroup_కొలత);     // subgroup_size
            ఒకవేళ (type != UINT32_MAX) {
                spec.push_back(type);                  // MmTypeA
                spec.push_back((uint32_t)ggml_type_కొలత((ggml_type)type)); // MmABlockBytes
            }
            తిరిగి spec;
        };

        std::vector<vk_tile_config> tc_mmq_k = {{s_warptile_mmq_k, s_mmq_wg_denoms_k, s_align}, {m_warptile_mmq_k, m_mmq_wg_denoms_k, m_align}, {l_warptile_mmq_k, l_mmq_wg_denoms_k, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmqid_wg_denoms, s_align}, {m_warptile_mmqid, m_mmqid_wg_denoms, m_align}, {l_warptile_mmqid, l_mmqid_wg_denoms, l_align}};

        spec_fn_t cm2_spec = [&](స్థిరరాశి std::vector<uint32_t>& wt, తార్కిక a) { తిరిగి ggml_vk_mul_mm_cm2_spec(wt, a); };

        // F16 x F16
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, అబద్ధం, నిజం},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm2_len, matmul_f16_f16acc_cm2_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, cm2_spec, నిజం);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc_mm, "matmul_f16",        matmul_f16_cm2_len,        matmul_f16_cm2_దత్తాంశం,        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, cm2_spec, నిజం);
#ఒకవేళ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ఒకవేళ (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, అబద్ధం, అబద్ధం}, tc_mm, "matmul_bf16", matmul_bf16_cm2_len, matmul_bf16_cm2_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, cm2_spec, నిజం);
        }
#అయితే_ముగింపు
        కొరకు (స్థిరరాశి స్వయంచాలక type : non_lut_quant_types) {
            // regression in unified shader on Ampere
            ఒకవేళ (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                కొనసాగు;
            }
            స్వయంచాలక& tc = ((type >= GGML_TYPE_Q2_K && type <= GGML_TYPE_Q6_K) || type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) ? tc_mmq_k : tc_mmq;
            spec_fn_t qs = [&, type](స్థిరరాశి std::vector<uint32_t>& wt, తార్కిక a) { తిరిగి ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, అబద్ధం, నిజం},  tc, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm2_len, matmul_quant_f16_f16acc_cm2_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, qs, నిజం);
            create_mm_pipelines({type, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc, "matmul_quant_f16",        matmul_quant_f16_cm2_len,        matmul_quant_f16_cm2_దత్తాంశం,        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, qs, నిజం);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, అబద్ధం, నిజం},  tc_mmq_k, "matmul_q4_k_f16_f16acc", matmul_q4_k_f16_f16acc_cm2_len, matmul_q4_k_f16_f16acc_cm2_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, cm2_spec, నిజం);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc_mmq_k, "matmul_q4_k_f16",        matmul_q4_k_f16_cm2_len,        matmul_q4_k_f16_cm2_దత్తాంశం,        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, cm2_spec, నిజం);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, అబద్ధం, నిజం},  tc_mmq_k, "matmul_q5_k_f16_f16acc", matmul_q5_k_f16_f16acc_cm2_len, matmul_q5_k_f16_f16acc_cm2_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, cm2_spec, నిజం);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc_mmq_k, "matmul_q5_k_f16",        matmul_q5_k_f16_cm2_len,        matmul_q5_k_f16_cm2_దత్తాంశం,        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, cm2_spec, నిజం);
#నిర్వచించు X_CM2(TYPE, tstr) \
        { స్వయంచాలక tc = filter_tc(tc_mmq, TYPE, అబద్ధం); \
          ఒకవేళ (!tc.ఖాళీలు()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, అబద్ధం, నిజం},  tc, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm2_len, matmul_##tstr##_f16_f16acc_cm2_data, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, cm2_spec, నిజం); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm2_len,        matmul_##tstr##_f16_cm2_data,        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, cm2_spec, నిజం); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2)
#ఒకవేళ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ఒకవేళ (device->ocp_fp4) {
#నిర్వచించు X_CM2_OCP(TYPE, tstr) \
            { స్వయంచాలక tc = filter_tc(tc_mmq, TYPE, అబద్ధం); \
              ఒకవేళ (!tc.ఖాళీలు()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, అబద్ధం, నిజం},  tc, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm2_len, matmul_##tstr##_f16_ocp_f16acc_cm2_data, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, cm2_spec, నిజం); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm2_len,        matmul_##tstr##_f16_ocp_cm2_data,        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, cm2_spec, నిజం); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_OCP)
#నిర్వచనం_రద్దు X_CM2_OCP
        } లేకపోతే
#అయితే_ముగింపు
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2)
        }
#నిర్వచనం_రద్దు X_CM2

        GGML_ASSERT(device->subgroup_ballot);

        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, నిజం, నిజం},  tc_mm, "matmul_id_subgroup_f16_f16acc", matmul_id_subgroup_f16_f16acc_cm2_len, matmul_id_subgroup_f16_f16acc_cm2_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), 5, cm2_spec, నిజం);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, నిజం, అబద్ధం}, tc_mm, "matmul_id_subgroup_f16",        matmul_id_subgroup_f16_cm2_len,        matmul_id_subgroup_f16_cm2_దత్తాంశం,        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), 5, cm2_spec, నిజం);
#ఒకవేళ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ఒకవేళ (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, నిజం, అబద్ధం}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm2_len, matmul_id_subgroup_bf16_cm2_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), 5, cm2_spec, నిజం);
        }
#అయితే_ముగింపు
        కొరకు (స్థిరరాశి స్వయంచాలక type : non_lut_quant_types) {
            ఒకవేళ (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                కొనసాగు;
            }
            spec_fn_t qs_id = [&, type](స్థిరరాశి std::vector<uint32_t>& wt, తార్కిక a) { తిరిగి ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, నిజం, నిజం},  tc_mmqid, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm2_len, matmul_id_subgroup_quant_f16_f16acc_cm2_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), 5, qs_id, నిజం);
            create_mm_pipelines({type, GGML_TYPE_F16, నిజం, అబద్ధం}, tc_mmqid, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm2_len,        matmul_id_subgroup_quant_f16_cm2_దత్తాంశం,        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), 5, qs_id, నిజం);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, నిజం, నిజం},  tc_mmqid, "matmul_id_subgroup_q4_k_f16_f16acc", matmul_id_subgroup_q4_k_f16_f16acc_cm2_len, matmul_id_subgroup_q4_k_f16_f16acc_cm2_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), 5, cm2_spec, నిజం);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, నిజం, అబద్ధం}, tc_mmqid, "matmul_id_subgroup_q4_k_f16",        matmul_id_subgroup_q4_k_f16_cm2_len,        matmul_id_subgroup_q4_k_f16_cm2_దత్తాంశం,        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), 5, cm2_spec, నిజం);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, నిజం, నిజం},  tc_mmqid, "matmul_id_subgroup_q5_k_f16_f16acc", matmul_id_subgroup_q5_k_f16_f16acc_cm2_len, matmul_id_subgroup_q5_k_f16_f16acc_cm2_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), 5, cm2_spec, నిజం);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, నిజం, అబద్ధం}, tc_mmqid, "matmul_id_subgroup_q5_k_f16",        matmul_id_subgroup_q5_k_f16_cm2_len,        matmul_id_subgroup_q5_k_f16_cm2_దత్తాంశం,        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), 5, cm2_spec, నిజం);
#నిర్వచించు X_CM2_ID(TYPE, tstr) \
        { స్వయంచాలక tc = filter_tc(tc_mmqid, TYPE, నిజం); \
          ఒకవేళ (!tc.ఖాళీలు()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, నిజం, నిజం},  tc, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm2_data, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), 5, cm2_spec, నిజం); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, నిజం, అబద్ధం}, tc, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm2_len,        matmul_id_subgroup_##tstr##_f16_cm2_data,        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), 5, cm2_spec, నిజం); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2_ID)
#ఒకవేళ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ఒకవేళ (device->ocp_fp4) {
#నిర్వచించు X_CM2_ID_OCP(TYPE, tstr) \
            { స్వయంచాలక tc = filter_tc(tc_mmqid, TYPE, నిజం); \
              ఒకవేళ (!tc.ఖాళీలు()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, నిజం, నిజం},  tc, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_data, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), 5, cm2_spec, నిజం); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, నిజం, అబద్ధం}, tc, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm2_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm2_data,        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), 5, cm2_spec, నిజం); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID_OCP)
#నిర్వచనం_రద్దు X_CM2_ID_OCP
        } లేకపోతే
#అయితే_ముగింపు
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID)
        }
#నిర్వచనం_రద్దు X_CM2_ID
    } లేకపోతే
#అయితే_ముగింపు  // defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
#ఒకవేళ defined(VK_KHR_cooperative_మాతృక) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ఒకవేళ (device->coopmat_support) {
        spec_fn_t cm1_spec = [&](స్థిరరాశి std::vector<uint32_t>& wt, తార్కిక a) { తిరిగి ggml_vk_mul_mm_spec(wt, a); };

        // Intel coopmat1 pins each pipeline's required subgroup size to its warptile WARP element.
        స్థిరరాశి తార్కిక cm1_pin = device->vendor_id == VK_VENDOR_ID_INTEL;

        // Intel coopmat1 uses a dedicated large-tile config for quant matmul_id.
        std::vector<vk_tile_config> tc_mmq_id = tc_mmq;
        ఒకవేళ (cm1_pin) {
            tc_mmq_id[2] = { { 512, 128, 128, 32, 32, 32, 2, device->coopmat_m, device->coopmat_n, device->coopmat_k, 32 }, { 128, 128, 1 }, 32 };
        }

        స్వయంచాలక cm1_create = [&](vk_matmul_pipeline_కీలకం కీలకం, స్థిరరాశి std::vector<vk_tile_config>& tc_పునాది,
                              స్థిరరాశి std::string& name, size_t len, స్థిరరాశి శూన్య* దత్తాంశం, uint32_t pc_కొలత, uint32_t pc) {
            స్వయంచాలక tc = filter_tc(tc_పునాది, కీలకం.type_a, కీలకం.mul_mat_id);
            ఒకవేళ (!tc.ఖాళీలు()) create_mm_pipelines(కీలకం, tc, name, len, దత్తాంశం, pc_కొలత, pc, cm1_spec, అబద్ధం, నిజం, 0, నిజం, cm1_pin);
        };
        స్వయంచాలక cm1_create_quant = [&](vk_matmul_pipeline_కీలకం కీలకం, స్థిరరాశి std::vector<vk_tile_config>& tc_పునాది,
                                    స్థిరరాశి std::string& name, size_t len, స్థిరరాశి శూన్య* దత్తాంశం, uint32_t pc_కొలత, uint32_t pc) {
            spec_fn_t qs = [&, type_a=కీలకం.type_a](స్థిరరాశి std::vector<uint32_t>& wt, తార్కిక a) { తిరిగి ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
            స్వయంచాలక tc = filter_tc(tc_పునాది, కీలకం.type_a, కీలకం.mul_mat_id);
            ఒకవేళ (!tc.ఖాళీలు()) create_mm_pipelines(కీలకం, tc, name, len, దత్తాంశం, pc_కొలత, pc, qs, అబద్ధం, నిజం, 0, నిజం, cm1_pin);
        };
        // int8 MMQ helper: per-type cm1 shader, warptile passed as-is (carries DEVICE_ARCH in
        // spec constant WARP_SIZE_IDX+1), subgroup size pinned to the warptile WARP element, no aligned variant.
        స్వయంచాలక cm1_create_mmq = [&](vk_matmul_pipeline_కీలకం కీలకం, స్థిరరాశి std::vector<vk_tile_config>& tc_పునాది,
                                  స్థిరరాశి std::string& name, size_t len, స్థిరరాశి శూన్య* దత్తాంశం, uint32_t pc_కొలత, uint32_t pc) {
            spec_fn_t తటస్థం = [](స్థిరరాశి std::vector<uint32_t>& wt, తార్కిక) { తిరిగి wt; };
            స్వయంచాలక tc = filter_tc(tc_పునాది, కీలకం.type_a, కీలకం.mul_mat_id, నిజం);
            ఒకవేళ (!tc.ఖాళీలు()) create_mm_pipelines(కీలకం, tc, name, len, దత్తాంశం, pc_కొలత, pc, తటస్థం, అబద్ధం, అబద్ధం, 0, అబద్ధం, నిజం);
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
        స్థిరరాశి తార్కిక rdna3 = device->architecture == AMD_RDNA3;
        స్థిరరాశి తార్కిక rdna4 = device->architecture == AMD_RDNA4;

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, అబద్ధం, అబద్ధం}, tc_mm, "matmul_f32_f32",     matmul_f32_f32_cm1_len,     matmul_f32_f32_cm1_దత్తాంశం,     పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
        cm1_create({GGML_TYPE_F32, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc_mm, "matmul_f32_f16",     matmul_f32_f16_cm1_len,     matmul_f32_f16_cm1_దత్తాంశం,     పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
        ఒకవేళ (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, అబద్ధం, నిజం},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm1_len, matmul_f16_f16acc_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, అబద్ధం, నిజం},  tc_mm, "matmul_f16_f32_f16acc", matmul_f16_f32_f16acc_cm1_len, matmul_f16_f32_f16acc_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
        }
        ఒకవేళ (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc_mm, "matmul_f16",      matmul_f16_cm1_len,      matmul_f16_cm1_దత్తాంశం,      పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, అబద్ధం, అబద్ధం}, tc_mm, "matmul_f16_f32",  matmul_f16_f32_cm1_len,  matmul_f16_f32_cm1_దత్తాంశం,  పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
        }
#ఒకవేళ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ఒకవేళ (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, అబద్ధం, అబద్ధం}, tc_mm, "matmul_bf16", matmul_bf16_cm1_len, matmul_bf16_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
        }
#అయితే_ముగింపు
        కొరకు (స్థిరరాశి స్వయంచాలక type : non_lut_quant_types) {
            ఒకవేళ (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, అబద్ధం, నిజం},  tc_mmq, "matmul_quant_f32_f16acc", matmul_quant_f32_f16acc_cm1_len, matmul_quant_f32_f16acc_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, అబద్ధం, నిజం},  tc_mmq, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm1_len, matmul_quant_f16_f16acc_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            }
            ఒకవేళ (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, అబద్ధం, అబద్ధం}, tc_mmq, "matmul_quant_f32",        matmul_quant_f32_cm1_len,        matmul_quant_f32_cm1_దత్తాంశం,        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc_mmq, "matmul_quant_f16",        matmul_quant_f16_cm1_len,        matmul_quant_f16_cm1_దత్తాంశం,        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#నిర్వచించు X_CM1(TYPE, tstr) \
        ఒకవేళ (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, అబద్ధం, నిజం},  tc_mmq, "matmul_" #tstr "_f32_f16acc", matmul_##tstr##_f32_f16acc_cm1_len, matmul_##tstr##_f32_f16acc_cm1_data, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, అబద్ధం, నిజం},  tc_mmq, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm1_len, matmul_##tstr##_f16_f16acc_cm1_data, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); \
        } \
        ఒకవేళ (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, అబద్ధం, అబద్ధం}, tc_mmq, "matmul_" #tstr "_f32",        matmul_##tstr##_f32_cm1_len,        matmul_##tstr##_f32_cm1_data,        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc_mmq, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm1_len,        matmul_##tstr##_f16_cm1_data,        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1)
#ఒకవేళ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ఒకవేళ (device->ocp_fp4) {
#నిర్వచించు X_CM1_OCP(TYPE, tstr) \
            ఒకవేళ (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, అబద్ధం, నిజం},  tc_mmq, "matmul_" #tstr "_f32_ocp_f16acc", matmul_##tstr##_f32_ocp_f16acc_cm1_len, matmul_##tstr##_f32_ocp_f16acc_cm1_data, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, అబద్ధం, నిజం},  tc_mmq, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm1_len, matmul_##tstr##_f16_ocp_f16acc_cm1_data, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); \
            } \
            ఒకవేళ (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, అబద్ధం, అబద్ధం}, tc_mmq, "matmul_" #tstr "_f32_ocp",        matmul_##tstr##_f32_ocp_cm1_len,        matmul_##tstr##_f32_ocp_cm1_data,        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc_mmq, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm1_len,        matmul_##tstr##_f16_ocp_cm1_data,        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_OCP)
#నిర్వచనం_రద్దు X_CM1_OCP
        } లేకపోతే
#అయితే_ముగింపు
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1)
        }
#నిర్వచనం_రద్దు X_CM1

        ఒకవేళ (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_q4_0_q8_1",   matmul_q4_0_q8_1_cm1_len,   matmul_q4_0_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            ఒకవేళ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_q4_1_q8_1",   matmul_q4_1_q8_1_cm1_len,   matmul_q4_1_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_q5_0_q8_1",   matmul_q5_0_q8_1_cm1_len,   matmul_q5_0_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            ఒకవేళ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_q5_1_q8_1",   matmul_q5_1_q8_1_cm1_len,   matmul_q5_1_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_q8_0_q8_1",   matmul_q8_0_q8_1_cm1_len,   matmul_q8_0_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_iq4_nl_q8_1", matmul_iq4_nl_q8_1_cm1_len, matmul_iq4_nl_q8_1_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_cm1_len, matmul_iq4_xs_q8_1_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_mxfp4_q8_1",  matmul_mxfp4_q8_1_cm1_len,  matmul_mxfp4_q8_1_cm1_దత్తాంశం,  పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_cm1_int_k, "matmul_q3_k_q8_1",   matmul_q3_k_q8_1_cm1_len,   matmul_q3_k_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            ఒకవేళ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_q4_k_q8_1",   matmul_q4_k_q8_1_cm1_len,   matmul_q4_k_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); }
            ఒకవేళ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_q5_k_q8_1",   matmul_q5_k_q8_1_cm1_len,   matmul_q5_k_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_cm1_int_k, "matmul_q6_k_q8_1",   matmul_q6_k_q8_1_cm1_len,   matmul_q6_k_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            ఒకవేళ (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_cm1_int_k, "matmul_nvfp4_q8_1",  matmul_nvfp4_q8_1_cm1_len,  matmul_nvfp4_q8_1_cm1_దత్తాంశం,  పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); }
        }

        GGML_ASSERT(device->subgroup_ballot);

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mm, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_cm1_len, matmul_id_subgroup_f32_f32_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
        ఒకవేళ (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, నిజం, నిజం},  tc_mm, "matmul_id_subgroup_f16_f16acc",     matmul_id_subgroup_f16_f16acc_cm1_len,     matmul_id_subgroup_f16_f16acc_cm1_దత్తాంశం,     పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, నిజం, నిజం},  tc_mm, "matmul_id_subgroup_f16_f32_f16acc", matmul_id_subgroup_f16_f32_f16acc_cm1_len, matmul_id_subgroup_f16_f32_f16acc_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
        }
        ఒకవేళ (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, నిజం, అబద్ధం}, tc_mm, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_cm1_len,     matmul_id_subgroup_f16_cm1_దత్తాంశం,     పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mm, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_cm1_len, matmul_id_subgroup_f16_f32_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
        }
#ఒకవేళ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ఒకవేళ (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, నిజం, అబద్ధం}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm1_len, matmul_id_subgroup_bf16_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
        }
#అయితే_ముగింపు
        కొరకు (స్థిరరాశి స్వయంచాలక type : non_lut_quant_types) {
            ఒకవేళ (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, నిజం, నిజం},  tc_mmq_id, "matmul_id_subgroup_quant_f32_f16acc", matmul_id_subgroup_quant_f32_f16acc_cm1_len, matmul_id_subgroup_quant_f32_f16acc_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                cm1_create_quant({type, GGML_TYPE_F16, నిజం, నిజం},  tc_mmq_id, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm1_len, matmul_id_subgroup_quant_f16_f16acc_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            }
            ఒకవేళ (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mmq_id, "matmul_id_subgroup_quant_f32",        matmul_id_subgroup_quant_f32_cm1_len,        matmul_id_subgroup_quant_f32_cm1_దత్తాంశం,        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                cm1_create_quant({type, GGML_TYPE_F16, నిజం, అబద్ధం}, tc_mmq_id, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm1_len,        matmul_id_subgroup_quant_f16_cm1_దత్తాంశం,        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#నిర్వచించు X_CM1_ID(TYPE, tstr) \
        ఒకవేళ (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, నిజం, నిజం},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_f16acc", matmul_id_subgroup_##tstr##_f32_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_f16acc_cm1_data, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క); \
            cm1_create({TYPE, GGML_TYPE_F16, నిజం, నిజం},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm1_data, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క); \
        } \
        ఒకవేళ (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32",        matmul_id_subgroup_##tstr##_f32_cm1_len,        matmul_id_subgroup_##tstr##_f32_cm1_data,        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క); \
            cm1_create({TYPE, GGML_TYPE_F16, నిజం, అబద్ధం}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm1_len,        matmul_id_subgroup_##tstr##_f16_cm1_data,        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1_ID)
#ఒకవేళ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ఒకవేళ (device->ocp_fp4) {
#నిర్వచించు X_CM1_ID_OCP(TYPE, tstr) \
            ఒకవేళ (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, నిజం, నిజం},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp_f16acc", matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_data, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క); \
                cm1_create({TYPE, GGML_TYPE_F16, నిజం, నిజం},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_data, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క); \
            } \
            ఒకవేళ (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp",        matmul_id_subgroup_##tstr##_f32_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f32_ocp_cm1_data,        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క); \
                cm1_create({TYPE, GGML_TYPE_F16, నిజం, అబద్ధం}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm1_data,        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID_OCP)
#నిర్వచనం_రద్దు X_CM1_ID_OCP
        } లేకపోతే
#అయితే_ముగింపు
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID)
        }
#నిర్వచనం_రద్దు X_CM1_ID

        ఒకవేళ (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_0_q8_1",   matmul_id_subgroup_q4_0_q8_1_cm1_len,   matmul_id_subgroup_q4_0_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            cm1_create_mmq({GGML_TYPE_Q4_1,   GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_1_q8_1",   matmul_id_subgroup_q4_1_q8_1_cm1_len,   matmul_id_subgroup_q4_1_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_0_q8_1",   matmul_id_subgroup_q5_0_q8_1_cm1_len,   matmul_id_subgroup_q5_0_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            cm1_create_mmq({GGML_TYPE_Q5_1,   GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_1_q8_1",   matmul_id_subgroup_q5_1_q8_1_cm1_len,   matmul_id_subgroup_q5_1_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_id_subgroup_q8_0_q8_1",   matmul_id_subgroup_q8_0_q8_1_cm1_len,   matmul_id_subgroup_q8_0_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_nl_q8_1", matmul_id_subgroup_iq4_nl_q8_1_cm1_len, matmul_id_subgroup_iq4_nl_q8_1_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_cm1_len, matmul_id_subgroup_iq4_xs_q8_1_cm1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_id_subgroup_mxfp4_q8_1",  matmul_id_subgroup_mxfp4_q8_1_cm1_len,  matmul_id_subgroup_mxfp4_q8_1_cm1_దత్తాంశం,  పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q3_k_q8_1",   matmul_id_subgroup_q3_k_q8_1_cm1_len,   matmul_id_subgroup_q3_k_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            cm1_create_mmq({GGML_TYPE_Q4_K,   GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_k_q8_1",   matmul_id_subgroup_q4_k_q8_1_cm1_len,   matmul_id_subgroup_q4_k_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            cm1_create_mmq({GGML_TYPE_Q5_K,   GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_k_q8_1",   matmul_id_subgroup_q5_k_q8_1_cm1_len,   matmul_id_subgroup_q5_k_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q6_k_q8_1",   matmul_id_subgroup_q6_k_q8_1_cm1_len,   matmul_id_subgroup_q6_k_q8_1_cm1_దత్తాంశం,   పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
            ఒకవేళ (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmq_cm1_int_k, "matmul_id_subgroup_nvfp4_q8_1",  matmul_id_subgroup_nvfp4_q8_1_cm1_len,  matmul_id_subgroup_nvfp4_q8_1_cm1_దత్తాంశం,  పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క); }
        }
    } లేకపోతే
#అయితే_ముగింపు  // defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    {
        // Helper for subgroup path with dot2 selection and filtering
        స్వయంచాలక sg_create = [&](vk_matmul_pipeline_కీలకం కీలకం, స్థిరరాశి std::vector<vk_tile_config>& tc_పునాది,
                             స్థిరరాశి std::string& name, size_t len, స్థిరరాశి శూన్య* దత్తాంశం, uint32_t pc_కొలత, uint32_t pc,
                             uint32_t rsgs = 0) {
            స్వయంచాలక tc = filter_tc(tc_పునాది, కీలకం.type_a, కీలకం.mul_mat_id);
            ఒకవేళ (!tc.ఖాళీలు()) create_mm_pipelines(కీలకం, tc, name, len, దత్తాంశం, pc_కొలత, pc,
                [&](స్థిరరాశి std::vector<uint32_t>& wt, తార్కిక a) { తిరిగి ggml_vk_mul_mm_spec(wt, a); },
                అబద్ధం, rsgs > 0, rsgs);
        };
        స్వయంచాలక sg_create_quant = [&](vk_matmul_pipeline_కీలకం కీలకం, స్థిరరాశి std::vector<vk_tile_config>& tc_పునాది,
                                   స్థిరరాశి std::string& name, size_t len, స్థిరరాశి శూన్య* దత్తాంశం, uint32_t pc_కొలత, uint32_t pc,
                                   uint32_t rsgs = 0) {
            స్వయంచాలక tc = filter_tc(tc_పునాది, కీలకం.type_a, కీలకం.mul_mat_id);
            ఒకవేళ (!tc.ఖాళీలు()) {
                spec_fn_t qs = [&, type_a=కీలకం.type_a](స్థిరరాశి std::vector<uint32_t>& wt, తార్కిక a) { తిరిగి ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
                create_mm_pipelines(కీలకం, tc, name, len, దత్తాంశం, pc_కొలత, pc, qs, అబద్ధం, rsgs > 0, rsgs);
            }
        };
#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        స్వయంచాలక sg_create_mmq = [&](vk_matmul_pipeline_కీలకం కీలకం, స్థిరరాశి std::vector<vk_tile_config>& tc_పునాది,
                                 స్థిరరాశి std::string& name, size_t len, స్థిరరాశి శూన్య* దత్తాంశం, uint32_t pc_కొలత, uint32_t pc,
                                 uint32_t rsgs = 0) {
            స్వయంచాలక tc = filter_tc(tc_పునాది, కీలకం.type_a, కీలకం.mul_mat_id, నిజం);
            ఒకవేళ (!tc.ఖాళీలు()) {
                spec_fn_t తటస్థం = [](స్థిరరాశి std::vector<uint32_t>& wt, తార్కిక) { తిరిగి wt; };
                create_mm_pipelines(కీలకం, tc, name, len, దత్తాంశం, pc_కొలత, pc, తటస్థం, అబద్ధం, rsgs > 0, rsgs, అబద్ధం);
            }
        };
#అయితే_ముగింపు

        std::vector<vk_tile_config> tc_id = {{s_warptile_id, s_wg_denoms, s_align}, {m_warptile_id, m_wg_denoms, m_align}, {l_warptile_id, l_wg_denoms, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid, l_mmq_wg_denoms, l_align}};

        ఒకవేళ (device->fp16) {
            // FP16 subgroup path - with dot2 runtime selection
            #నిర్వచించు SPV_DOT2(NAME) (device->dot2_f16 ? NAME ## _dot2_len : NAME ## _len), (device->dot2_f16 ? NAME ## dot2_దత్తాంశం : NAME ## దత్తాంశం_2)
            #నిర్వచించు SPV_DOT2_F16ACC(NAME) (device->dot2_f16 ? NAME ## _dot2_f16acc_len : NAME ## _f16acc_len), (device->dot2_f16 ? NAME ## dot2_f16acc_దత్తాంశం : NAME ## f16acc_దత్తాంశం)

            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, అబద్ధం, అబద్ధం}, tc_mm, "matmul_f32_f32", SPV_DOT2(matmul_f32_f32), పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc_mm, "matmul_f32_f16", SPV_DOT2(matmul_f32_f16), పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, అబద్ధం, నిజం},  tc_mm, "matmul_f16_f16acc",     SPV_DOT2_F16ACC(matmul_f16),     పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc_mm, "matmul_f16",            SPV_DOT2(matmul_f16),            పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, అబద్ధం, నిజం},  tc_mm, "matmul_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_f16_f32), పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, అబద్ధం, అబద్ధం}, tc_mm, "matmul_f16_f32",        SPV_DOT2(matmul_f16_f32),        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            // BF16 - no dot2
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, అబద్ధం, అబద్ధం}, tc_mm, "matmul_bf16", matmul_bf16_len, matmul_bf16_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);

            కొరకు (స్థిరరాశి స్వయంచాలక type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, అబద్ధం, నిజం},  tc_mmq, "matmul_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_quant_f32), పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_quant({type, GGML_TYPE_F32, అబద్ధం, అబద్ధం}, tc_mmq, "matmul_quant_f32",        SPV_DOT2(matmul_quant_f32),        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            }
    #నిర్వచించు X_SG(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, అబద్ధం, నిజం},  tc_mmq, "matmul_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_##tstr##_f32), పరిమాణం(vk_mat_mat_నెట్టు_constants), 3); \
            sg_create({TYPE, GGML_TYPE_F32, అబద్ధం, అబద్ధం}, tc_mmq, "matmul_" #tstr "_f32",        SPV_DOT2(matmul_##tstr##_f32),        పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG)
#నిర్వచనం_రద్దు X_SG

#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ఒకవేళ (device->integer_dot_లబ్ధం) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_len, matmul_q2_0_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_len, matmul_q4_0_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_len, matmul_q4_1_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_len, matmul_q5_0_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_len, matmul_q5_1_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_len, matmul_q8_0_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_mxfp4_q8_1", matmul_mxfp4_q8_1_len, matmul_mxfp4_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_len, matmul_iq4_xs_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_len, matmul_q2_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_len, matmul_q3_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_len, matmul_q4_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_len, matmul_q5_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_len, matmul_q6_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_len, matmul_iq3_s_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            }
#అయితే_ముగింపు

            ఒకవేళ (device->subgroup_ballot && device->subgroup_require_నిండిన_support && subgroup_min_కొలత_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_id, "matmul_id_subgroup_f32_f32", SPV_DOT2(matmul_id_subgroup_f32_f32), పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, నిజం, నిజం},  tc_id, "matmul_id_subgroup_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_subgroup_f16),     పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, నిజం, అబద్ధం}, tc_id, "matmul_id_subgroup_f16",            SPV_DOT2(matmul_id_subgroup_f16),            పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, నిజం, నిజం},  tc_id, "matmul_id_subgroup_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_f16_f32), పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_id, "matmul_id_subgroup_f16_f32",        SPV_DOT2(matmul_id_subgroup_f16_f32),        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, నిజం, అబద్ధం}, tc_id, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_len, matmul_id_subgroup_bf16_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                కొరకు (స్థిరరాశి స్వయంచాలక type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, నిజం, నిజం},  tc_mmqid, "matmul_id_subgroup_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_quant_f32), పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత);
                    sg_create_quant({type, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mmqid, "matmul_id_subgroup_quant_f32",        SPV_DOT2(matmul_id_subgroup_quant_f32),        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత);
                }
        #నిర్వచించు X_SG_ID_SUB(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, నిజం, నిజం},  tc_mmqid, "matmul_id_subgroup_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_##tstr##_f32), పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత); \
                sg_create({TYPE, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32",        SPV_DOT2(matmul_id_subgroup_##tstr##_f32),        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB)
#నిర్వచనం_రద్దు X_SG_ID_SUB
#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                ఒకవేళ (device->integer_dot_లబ్ధం) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_subgroup_q2_0_q8_1", matmul_id_subgroup_q2_0_q8_1_len, matmul_id_subgroup_q2_0_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_subgroup_q4_0_q8_1", matmul_id_subgroup_q4_0_q8_1_len, matmul_id_subgroup_q4_0_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_subgroup_q4_1_q8_1", matmul_id_subgroup_q4_1_q8_1_len, matmul_id_subgroup_q4_1_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_subgroup_q5_0_q8_1", matmul_id_subgroup_q5_0_q8_1_len, matmul_id_subgroup_q5_0_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_subgroup_q5_1_q8_1", matmul_id_subgroup_q5_1_q8_1_len, matmul_id_subgroup_q5_1_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_subgroup_q8_0_q8_1", matmul_id_subgroup_q8_0_q8_1_len, matmul_id_subgroup_q8_0_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_subgroup_mxfp4_q8_1", matmul_id_subgroup_mxfp4_q8_1_len, matmul_id_subgroup_mxfp4_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_len, matmul_id_subgroup_iq4_xs_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int_k, "matmul_id_subgroup_q2_k_q8_1", matmul_id_subgroup_q2_k_q8_1_len, matmul_id_subgroup_q2_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int_k, "matmul_id_subgroup_q3_k_q8_1", matmul_id_subgroup_q3_k_q8_1_len, matmul_id_subgroup_q3_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int_k, "matmul_id_subgroup_q4_k_q8_1", matmul_id_subgroup_q4_k_q8_1_len, matmul_id_subgroup_q4_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int_k, "matmul_id_subgroup_q5_k_q8_1", matmul_id_subgroup_q5_k_q8_1_len, matmul_id_subgroup_q5_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int_k, "matmul_id_subgroup_q6_k_q8_1", matmul_id_subgroup_q6_k_q8_1_len, matmul_id_subgroup_q6_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int_k, "matmul_id_subgroup_iq3_s_q8_1", matmul_id_subgroup_iq3_s_q8_1_len, matmul_id_subgroup_iq3_s_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                }
#అయితే_ముగింపు
            } లేకపోతే {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mm, "matmul_id_f32_f32", SPV_DOT2(matmul_id_f32_f32), పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, నిజం, నిజం},  tc_mm, "matmul_id_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_f16),     పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, నిజం, అబద్ధం}, tc_mm, "matmul_id_f16",            SPV_DOT2(matmul_id_f16),            పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, నిజం, నిజం},  tc_mm, "matmul_id_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_f16_f32), పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mm, "matmul_id_f16_f32",        SPV_DOT2(matmul_id_f16_f32),        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, నిజం, అబద్ధం}, tc_mm, "matmul_id_bf16", matmul_id_bf16_len, matmul_id_bf16_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                కొరకు (స్థిరరాశి స్వయంచాలక type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, నిజం, నిజం},  tc_mmqid, "matmul_id_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_quant_f32), పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_quant({type, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mmqid, "matmul_id_quant_f32",        SPV_DOT2(matmul_id_quant_f32),        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                }
        #నిర్వచించు X_SG_ID(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, నిజం, నిజం},  tc_mmqid, "matmul_id_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_##tstr##_f32), పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క); \
                sg_create({TYPE, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mmqid, "matmul_id_" #tstr "_f32",        SPV_DOT2(matmul_id_##tstr##_f32),        పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                FOR_EACH_LUT_TYPE(X_SG_ID)
#నిర్వచనం_రద్దు X_SG_ID
#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                ఒకవేళ (device->integer_dot_లబ్ధం) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_q2_0_q8_1", matmul_id_q2_0_q8_1_len, matmul_id_q2_0_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_q4_0_q8_1", matmul_id_q4_0_q8_1_len, matmul_id_q4_0_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_q4_1_q8_1", matmul_id_q4_1_q8_1_len, matmul_id_q4_1_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_q5_0_q8_1", matmul_id_q5_0_q8_1_len, matmul_id_q5_0_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_q5_1_q8_1", matmul_id_q5_1_q8_1_len, matmul_id_q5_1_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_q8_0_q8_1", matmul_id_q8_0_q8_1_len, matmul_id_q8_0_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_mxfp4_q8_1", matmul_id_mxfp4_q8_1_len, matmul_id_mxfp4_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int, "matmul_id_iq4_xs_q8_1", matmul_id_iq4_xs_q8_1_len, matmul_id_iq4_xs_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int_k, "matmul_id_q2_k_q8_1", matmul_id_q2_k_q8_1_len, matmul_id_q2_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int_k, "matmul_id_q3_k_q8_1", matmul_id_q3_k_q8_1_len, matmul_id_q3_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int_k, "matmul_id_q4_k_q8_1", matmul_id_q4_k_q8_1_len, matmul_id_q4_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int_k, "matmul_id_q5_k_q8_1", matmul_id_q5_k_q8_1_len, matmul_id_q5_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int_k, "matmul_id_q6_k_q8_1", matmul_id_q6_k_q8_1_len, matmul_id_q6_k_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, నిజం, అబద్ధం}, tc_mmqid_int_k, "matmul_id_iq3_s_q8_1", matmul_id_iq3_s_q8_1_len, matmul_id_iq3_s_q8_1_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                }
#అయితే_ముగింపు
            }
            #నిర్వచనం_రద్దు SPV_DOT2
            #నిర్వచనం_రద్దు SPV_DOT2_F16ACC
        } లేకపోతే {
            // FP32-only fallback path
            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, అబద్ధం, అబద్ధం}, tc_mm, "matmul_f32_f32", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc_mm, "matmul_f32_f16", matmul_f32_f16_fp32_len, matmul_f32_f16_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, అబద్ధం, అబద్ధం}, tc_mm, "matmul_f16",     matmul_f16_fp32_len,     matmul_f16_fp32_దత్తాంశం,     పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, అబద్ధం, అబద్ధం}, tc_mm, "matmul_f16_f32", matmul_f16_f32_fp32_len, matmul_f16_f32_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, అబద్ధం, అబద్ధం}, tc_mm, "matmul_bf16",  matmul_bf16_fp32_len,    matmul_bf16_fp32_దత్తాంశం,    పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);

            కొరకు (స్థిరరాశి స్వయంచాలక type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, అబద్ధం, అబద్ధం}, tc_mmq, "matmul_quant_f32", matmul_quant_f32_fp32_len, matmul_quant_f32_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            }
    #నిర్వచించు X_SG_FP32(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, అబద్ధం, అబద్ధం}, tc_mmq, "matmul_" #tstr "_f32", matmul_##tstr##_f32_fp32_len, matmul_##tstr##_f32_fp32_data, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG_FP32)
#నిర్వచనం_రద్దు X_SG_FP32

#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ఒకవేళ (device->integer_dot_లబ్ధం) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_fp32_len, matmul_q2_0_q8_1_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_fp32_len, matmul_q4_0_q8_1_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_fp32_len, matmul_q4_1_q8_1_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_fp32_len, matmul_q5_0_q8_1_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_fp32_len, matmul_q5_1_q8_1_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_fp32_len, matmul_q8_0_q8_1_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_fp32_len, matmul_iq4_xs_q8_1_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_fp32_len, matmul_q2_k_q8_1_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_fp32_len, matmul_q3_k_q8_1_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_fp32_len, matmul_q4_k_q8_1_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_fp32_len, matmul_q5_k_q8_1_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_fp32_len, matmul_q6_k_q8_1_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, అబద్ధం, అబద్ధం}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_fp32_len, matmul_iq3_s_q8_1_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3);
            }
#అయితే_ముగింపు

            ఒకవేళ (device->subgroup_ballot && device->subgroup_require_నిండిన_support && subgroup_min_కొలత_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_id, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_fp32_len, matmul_id_subgroup_f32_f32_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, నిజం, అబద్ధం}, tc_id, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_fp32_len,     matmul_id_subgroup_f16_fp32_దత్తాంశం,     పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_id, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_fp32_len, matmul_id_subgroup_f16_f32_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, నిజం, అబద్ధం}, tc_id, "matmul_id_subgroup_bf16",  matmul_id_subgroup_bf16_fp32_len,    matmul_id_subgroup_bf16_fp32_దత్తాంశం,    పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత_16);
                కొరకు (స్థిరరాశి స్వయంచాలక type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mmqid, "matmul_id_subgroup_quant_f32", matmul_id_subgroup_quant_f32_fp32_len, matmul_id_subgroup_quant_f32_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత);
                }
        #నిర్వచించు X_SG_ID_SUB_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32", matmul_id_subgroup_##tstr##_f32_fp32_len, matmul_id_subgroup_##tstr##_f32_fp32_data, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, mul_mat_subgroup_కొలత);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB_FP32)
#నిర్వచనం_రద్దు X_SG_ID_SUB_FP32
            } లేకపోతే {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mm, "matmul_id_f32_f32", matmul_id_f32_f32_fp32_len, matmul_id_f32_f32_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, నిజం, అబద్ధం}, tc_mm, "matmul_id_f16",     matmul_id_f16_fp32_len,     matmul_id_f16_fp32_దత్తాంశం,     పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mm, "matmul_id_f16_f32", matmul_id_f16_f32_fp32_len, matmul_id_f16_f32_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, నిజం, అబద్ధం}, tc_mm, "matmul_id_bf16",  matmul_id_bf16_fp32_len,    matmul_id_bf16_fp32_దత్తాంశం,    పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                కొరకు (స్థిరరాశి స్వయంచాలక type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mmqid, "matmul_id_quant_f32", matmul_id_quant_f32_fp32_len, matmul_id_quant_f32_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                }
        #నిర్వచించు X_SG_ID_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, నిజం, అబద్ధం}, tc_mmqid, "matmul_id_" #tstr "_f32", matmul_id_##tstr##_f32_fp32_len, matmul_id_##tstr##_f32_fp32_data, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క);
                FOR_EACH_LUT_TYPE(X_SG_ID_FP32)
#నిర్వచనం_రద్దు X_SG_ID_FP32
            }
        }
    }
#నిర్వచనం_రద్దు FOR_EACH_LUT_TYPE
#నిర్వచనం_రద్దు FOR_EACH_LUT_TYPE_NONFP4
#నిర్వచనం_రద్దు FOR_EACH_LUT_FP4_TYPE
    // BF16 fallback for coopmat devices without bf16 coopmat support
    ఒకవేళ ((device->coopmat2 || device->coopmat_support)
#ఒకవేళ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        && !device->coopmat_bf16_support
#అయితే_ముగింపు
        ) {
        స్థిరరాశి uint32_t s_warptile_wm_bf16 = device->subgroup_కొలత == 8 ? 8 : 32;
        std::vector<vk_tile_config> tc_bf16_fb = {
            {{ subgroup_కొలత_32, 32, 32, 16, s_warptile_wm_bf16, 32, 2, 2, 2, 1, subgroup_కొలత_8 }, {32, 32, 1}, s_align},
            {{ 128, 64, 64, 16, mm_warp_8, 32, 2, 4, 2, 1, mm_warp_8 }, {64, 64, 1}, m_align},
            {{ 128, 128, 128, 16, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 }, {128, 128, 1}, l_align},
        };
        స్వయంచాలక tc_bf16_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, అబద్ధం);
        స్వయంచాలక tc_bf16_id_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, నిజం);
        spec_fn_t bf16_spec = [&](స్థిరరాశి std::vector<uint32_t>& wt, తార్కిక a) { తిరిగి ggml_vk_mul_mm_spec(wt, a); };
        ఒకవేళ (!tc_bf16_filtered.ఖాళీలు()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, అబద్ధం, అబద్ధం}, tc_bf16_filtered, "matmul_bf16", matmul_bf16_fp32_len, matmul_bf16_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_నెట్టు_constants), 3, bf16_spec);
        }
        ఒకవేళ (!tc_bf16_id_filtered.ఖాళీలు()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, నిజం, అబద్ధం}, tc_bf16_id_filtered, "matmul_id_bf16", matmul_id_bf16_fp32_len, matmul_id_bf16_fp32_దత్తాంశం, పరిమాణం(vk_mat_mat_id_నెట్టు_constants), mul_mat_id_param_లెక్క, bf16_spec);
        }
    }

    // Set up tile selector functions
    ఒకవేళ (device->coopmat2) {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t shader_core_లెక్క,
                                          స్థిరరాశి std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            ఒకవేళ (configs.size() <= 1) తిరిగి 0;
            uint32_t చివరి = (uint32_t)configs.size() - 1;
            ఒకవేళ (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                తిరిగి (n > crossover) ? 1 : 0;
            }
            // 3+ configs: s=0, m=1, l=2
            స్థిరరాశి uint32_t tiles_l = CEIL_DIV(m, configs[చివరి].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[చివరి].unaligned->wg_denoms[1]);
            స్థిరరాశి uint32_t tiles_m = CEIL_DIV(m, configs[1].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[1].unaligned->wg_denoms[1]);
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            తార్కిక prefer_large = tiles_m > shader_core_లెక్క || tiles_l > shader_core_లెక్క ||
                                (tiles_l <= shader_core_లెక్క / 3 && tiles_m > shader_core_లెక్క / 2);
            ఒకవేళ (n > crossover_large && prefer_large) తిరిగి చివరి;
            uint32_t crossover_medium_m = configs[0].unaligned->wg_denoms[0];
            uint32_t crossover_medium_n = configs[0].unaligned->wg_denoms[1];
            ఒకవేళ (m > crossover_medium_m && n > crossover_medium_n) తిరిగి 1;
            తిరిగి 0;
        };
        device->matmul_id_tile_selector = [](uint32_t /*m*/, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                             స్థిరరాశి std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            ఒకవేళ (configs.size() <= 1) తిరిగి 0;
            uint32_t చివరి = (uint32_t)configs.size() - 1;
            ఒకవేళ (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                తిరిగి (n > crossover) ? 1 : 0;
            }
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            ఒకవేళ (n > crossover_large) తిరిగి చివరి;
            uint32_t crossover_medium = configs[0].unaligned->wg_denoms[1];
            ఒకవేళ (n > crossover_medium) తిరిగి 1;
            తిరిగి 0;
        };
    } లేకపోతే {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                          స్థిరరాశి std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            ఒకవేళ (configs.size() <= 1) తిరిగి 0;
            ఒకవేళ (m <= 32 || n <= 32) తిరిగి 0;
            ఒకవేళ (configs.size() == 2) తిరిగి 1;
            ఒకవేళ (m <= 64 || n <= 64) తిరిగి 1;
            తిరిగి (uint32_t)configs.size() - 1;
        };
        device->matmul_id_tile_selector = device->matmul_tile_selector;
    }

    // mul mat vec

    // the number of rows computed per shader depends on GPU model and quant
    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;
    uint32_t rm_stdq_int = 1;
    uint32_t rm_kq_int = 1;
    స్వయంచాలక స్థిరరాశి &rm_iq_int = [](uint32_t i) { తిరిగి i == 0 ? 8u : 4u; };
    ఒకవేళ (device->vendor_id == VK_VENDOR_ID_AMD) {
        ఒకవేళ (device->architecture == AMD_GCN) {
            rm_stdq = 2;
            rm_kq = 4;
            rm_stdq_int = 4;
        }
    } లేకపోతే ఒకవేళ (device->vendor_id == VK_VENDOR_ID_INTEL) {
        rm_stdq = 2;
        rm_stdq_int = 2;
    }
    // RDNA3/4: above four columns, static 4 rows for all types bench faster than the default
    స్థిరరాశి తార్కిక is_rdna3_or_4 = device->vendor_id == VK_VENDOR_ID_AMD && (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);
    స్వయంచాలక స్థిరరాశి &rm_int_n = [&](uint32_t అడ్డువరుసలు, uint32_t i) { తిరిగి (is_rdna3_or_4 && i >= 4) ? 4u : అడ్డువరుసలు; };
    // RDNA3/4 and NVIDIA except pre-Turing: use 4 rows for MUL_MAT_ID MMVQ.
    స్వయంచాలక స్థిరరాశి &rm_id = [&](uint32_t అడ్డువరుసలు) {
        ఒకవేళ (device->vendor_id == VK_VENDOR_ID_NVIDIA &&
            device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) {
            తిరిగి 4u;
        }
        తిరిగి is_rdna3_or_4 ? 4u : అడ్డువరుసలు;
    };
    uint32_t rm_iq = 2 * rm_kq;

    స్థిరరాశి తార్కిక use_subgroups = device->subgroup_arithmetic;
    // The Imagination proprietary compiler rejects the subgroup-only dequant mul_mat_vec
    // shaders that require a subgroup size >= 16; fall back to shared-memory reduction.
    స్థిరరాశి తార్కిక is_imagination_proprietary =
        device->driver_id == vk::DriverId::eImaginationProprietary;
    // Ensure a subgroup size >= 16 is available
    స్థిరరాశి తార్కిక use_subgroups16 = use_subgroups && subgroup_min_కొలత_16 && !is_imagination_proprietary;

    స్థిరరాశి uint32_t subgroup_కొలత = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_కొలత_control && device->subgroup_min_కొలత <= 16 && device->subgroup_max_కొలత >= 16) ? 16 : device->subgroup_కొలత;
    స్థిరరాశి uint32_t subgroup_size16 = std::max(subgroup_కొలత, 16u);

    స్థిరరాశి uint32_t force_subgroup_కొలత = use_subgroups ? subgroup_కొలత : 0;
    స్థిరరాశి uint32_t force_subgroup_size16 = use_subgroups16 ? subgroup_size16 : 0;
    స్థిర constexpr uint32_t mul_mat_vec_అంకె_bindings = 5;
    స్థిర constexpr uint32_t mul_mat_vec_id_అంకె_bindings = 6;

#ఒకవేళ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
#నిర్వచించు OCP_DMMV_LEN(NAME, REDUC)  (device->ocp_fp4 ? NAME ## _ocp_len[REDUC]  : NAME ## _len[REDUC])
#నిర్వచించు OCP_DMMV_దత్తాంశం(NAME, REDUC) (device->ocp_fp4 ? NAME ## ocp_దత్తాంశం[REDUC] : NAME ## దత్తాంశం_2[REDUC])
#లేకపోతే
#నిర్వచించు OCP_DMMV_LEN(NAME, REDUC)  NAME ## _len[REDUC]
#నిర్వచించు OCP_DMMV_దత్తాంశం(NAME, REDUC) NAME ## దత్తాంశం_2[REDUC]
#అయితే_ముగింపు

    కొరకు (uint32_t w = 0; w < DMMV_WG_కొలత_లెక్క; ++w) {
        స్థిరరాశి uint32_t wg_కొలత_subgroup   = (w == DMMV_WG_కొలత_SUBGROUP) ? subgroup_కొలత : (subgroup_కొలత * 4);
        స్థిరరాశి uint32_t wg_కొలత_subgroup16 = (w == DMMV_WG_కొలత_SUBGROUP) ? subgroup_size16 : (subgroup_size16 * 4);

        స్థిరరాశి shader_reduction_బహుళకం reduc = (use_subgroups && w == DMMV_WG_కొలత_SUBGROUP) ? SHADER_REDUCTION_బహుళకం_SUBGROUP :
                                            (use_subgroups && w == DMMV_WG_కొలత_LARGE) ? SHADER_REDUCTION_బహుళకం_HYBRID :
                                            SHADER_REDUCTION_బహుళకం_SHMEM;

        స్థిరరాశి shader_reduction_బహుళకం reduc16 = (use_subgroups16 && w == DMMV_WG_కొలత_SUBGROUP) ? SHADER_REDUCTION_బహుళకం_SUBGROUP :
                                              (use_subgroups16 && w == DMMV_WG_కొలత_LARGE) ? SHADER_REDUCTION_బహుళకం_HYBRID :
                                              SHADER_REDUCTION_బహుళకం_SHMEM;

        కొరకు (uint32_t i = 0; i < mul_mat_vec_max_నిలువుల_సంఖ్య; ++i) {
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f32_f32",  arr_dmmv_f32_f32_f32_len[reduc],  arr_dmmv_f32_f32_f32_దత్తాంశం[reduc],  "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {1, 1, 1}, {wg_కొలత_subgroup, 1, i+1}, 1, అబద్ధం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32",  arr_dmmv_f16_f32_f32_len[reduc],  arr_dmmv_f16_f32_f32_దత్తాంశం[reduc],  "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2, 1, 1}, {wg_కొలత_subgroup, 2, i+1}, 1, అబద్ధం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f32_f32", arr_dmmv_bf16_f32_f32_len[reduc], arr_dmmv_bf16_f32_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2, 1, 1}, {wg_కొలత_subgroup, 2, i+1}, 1, అబద్ధం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f32_f32", arr_dmmv_q1_0_f32_f32_len[reduc], arr_dmmv_q1_0_f32_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f32_f32", arr_dmmv_q2_0_f32_f32_len[reduc], arr_dmmv_q2_0_f32_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f32_f32", arr_dmmv_q4_0_f32_f32_len[reduc], arr_dmmv_q4_0_f32_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f32_f32", arr_dmmv_q4_1_f32_f32_len[reduc], arr_dmmv_q4_1_f32_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f32_f32", arr_dmmv_q5_0_f32_f32_len[reduc], arr_dmmv_q5_0_f32_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f32_f32", arr_dmmv_q5_1_f32_f32_len[reduc], arr_dmmv_q5_1_f32_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f32_f32", arr_dmmv_q8_0_f32_f32_len[reduc], arr_dmmv_q8_0_f32_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {1*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 1*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f32_f32", arr_dmmv_q2_k_f32_f32_len[reduc16], arr_dmmv_q2_k_f32_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f32_f32", arr_dmmv_q3_k_f32_f32_len[reduc16], arr_dmmv_q3_k_f32_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f32_f32", arr_dmmv_q5_k_f32_f32_len[reduc16], arr_dmmv_q5_k_f32_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f32_f32", arr_dmmv_tq1_0_f32_f32_len[reduc16], arr_dmmv_tq1_0_f32_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f32_f32", arr_dmmv_tq2_0_f32_f32_len[reduc16], arr_dmmv_tq2_0_f32_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f32_f32",   arr_dmmv_iq1_s_f32_f32_len[reduc16],   arr_dmmv_iq1_s_f32_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f32_f32",   arr_dmmv_iq1_m_f32_f32_len[reduc16],   arr_dmmv_iq1_m_f32_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f32_f32", arr_dmmv_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_iq2_xxs_f32_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f32_f32",  arr_dmmv_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_iq2_xs_f32_f32_దత్తాంశం[reduc16],  "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f32_f32",   arr_dmmv_iq2_s_f32_f32_len[reduc16],   arr_dmmv_iq2_s_f32_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f32_f32", arr_dmmv_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_iq3_xxs_f32_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f32_f32",   arr_dmmv_iq3_s_f32_f32_len[reduc16],   arr_dmmv_iq3_s_f32_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f32_f32",  arr_dmmv_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_iq4_xs_f32_f32_దత్తాంశం[reduc16],  "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f32_f32",  arr_dmmv_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_iq4_nl_f32_f32_దత్తాంశం[reduc16],  "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f32_f32, reduc16), OCP_DMMV_దత్తాంశం(arr_dmmv_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f32_f32, reduc16), OCP_DMMV_దత్తాంశం(arr_dmmv_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f16_f32",  arr_dmmv_f32_f16_f32_len[reduc],  arr_dmmv_f32_f16_f32_దత్తాంశం[reduc],  "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {1, 1, 1}, {wg_కొలత_subgroup, 1, i+1}, 1, అబద్ధం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f16_f32",  arr_dmmv_f16_f16_f32_len[reduc],  arr_dmmv_f16_f16_f32_దత్తాంశం[reduc],  "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2, 1, 1}, {wg_కొలత_subgroup, 2, i+1}, 1, అబద్ధం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f16_f32", arr_dmmv_bf16_f16_f32_len[reduc], arr_dmmv_bf16_f16_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2, 1, 1}, {wg_కొలత_subgroup, 2, i+1}, 1, అబద్ధం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f16_f32", arr_dmmv_q1_0_f16_f32_len[reduc], arr_dmmv_q1_0_f16_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f16_f32", arr_dmmv_q2_0_f16_f32_len[reduc], arr_dmmv_q2_0_f16_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f16_f32", arr_dmmv_q4_0_f16_f32_len[reduc], arr_dmmv_q4_0_f16_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f16_f32", arr_dmmv_q4_1_f16_f32_len[reduc], arr_dmmv_q4_1_f16_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f16_f32", arr_dmmv_q5_0_f16_f32_len[reduc], arr_dmmv_q5_0_f16_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f16_f32", arr_dmmv_q5_1_f16_f32_len[reduc], arr_dmmv_q5_1_f16_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f16_f32", arr_dmmv_q8_0_f16_f32_len[reduc], arr_dmmv_q8_0_f16_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {1*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 1*rm_stdq, i+1}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f16_f32", arr_dmmv_q2_k_f16_f32_len[reduc16], arr_dmmv_q2_k_f16_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f16_f32", arr_dmmv_q3_k_f16_f32_len[reduc16], arr_dmmv_q3_k_f16_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f16_f32", arr_dmmv_q5_k_f16_f32_len[reduc16], arr_dmmv_q5_k_f16_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f16_f32", arr_dmmv_tq1_0_f16_f32_len[reduc16], arr_dmmv_tq1_0_f16_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f16_f32", arr_dmmv_tq2_0_f16_f32_len[reduc16], arr_dmmv_tq2_0_f16_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f16_f32",   arr_dmmv_iq1_s_f16_f32_len[reduc16],   arr_dmmv_iq1_s_f16_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f16_f32",   arr_dmmv_iq1_m_f16_f32_len[reduc16],   arr_dmmv_iq1_m_f16_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f16_f32", arr_dmmv_iq2_xxs_f16_f32_len[reduc16], arr_dmmv_iq2_xxs_f16_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f16_f32",  arr_dmmv_iq2_xs_f16_f32_len[reduc16],  arr_dmmv_iq2_xs_f16_f32_దత్తాంశం[reduc16],  "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f16_f32",   arr_dmmv_iq2_s_f16_f32_len[reduc16],   arr_dmmv_iq2_s_f16_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f16_f32", arr_dmmv_iq3_xxs_f16_f32_len[reduc16], arr_dmmv_iq3_xxs_f16_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f16_f32",   arr_dmmv_iq3_s_f16_f32_len[reduc16],   arr_dmmv_iq3_s_f16_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f16_f32",  arr_dmmv_iq4_xs_f16_f32_len[reduc16],  arr_dmmv_iq4_xs_f16_f32_దత్తాంశం[reduc16],  "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f16_f32",  arr_dmmv_iq4_nl_f16_f32_len[reduc16],  arr_dmmv_iq4_nl_f16_f32_దత్తాంశం[reduc16],  "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f16_f32, reduc16), OCP_DMMV_దత్తాంశం(arr_dmmv_mxfp4_f16_f32, reduc16), "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f16_f32, reduc16), OCP_DMMV_దత్తాంశం(arr_dmmv_nvfp4_f16_f32, reduc16), "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq, i+1}, 1, నిజం, use_subgroups16, force_subgroup_size16);

#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ఒకవేళ (device->integer_dot_లబ్ధం) {
                స్థిరరాశి uint32_t subgroup_కొలత_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_కొలత_control) ? device->subgroup_min_కొలత : device->subgroup_కొలత;
                స్థిరరాశి uint32_t wg_కొలత_subgroup_int = (w == DMMV_WG_కొలత_SUBGROUP) ? subgroup_కొలత_int : (subgroup_కొలత_int * 4);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_q8_1_f32", arr_dmmv_q2_0_q8_1_f32_len[reduc], arr_dmmv_q2_0_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_కొలత_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_q8_1_f32", arr_dmmv_q4_0_q8_1_f32_len[reduc], arr_dmmv_q4_0_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_కొలత_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_q8_1_f32", arr_dmmv_q4_1_q8_1_f32_len[reduc], arr_dmmv_q4_1_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_కొలత_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_q8_1_f32", arr_dmmv_q5_0_q8_1_f32_len[reduc], arr_dmmv_q5_0_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_కొలత_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_q8_1_f32", arr_dmmv_q5_1_q8_1_f32_len[reduc], arr_dmmv_q5_1_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_కొలత_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_q8_1_f32", arr_dmmv_q8_0_q8_1_f32_len[reduc], arr_dmmv_q8_0_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_కొలత_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_MXFP4][i], "mul_mat_vec_mxfp4_q8_1_f32", arr_dmmv_mxfp4_q8_1_f32_len[reduc], arr_dmmv_mxfp4_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_int_n(2*rm_stdq_int, i), 1, 1}, {wg_కొలత_subgroup_int, rm_int_n(2*rm_stdq_int, i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_q8_1_f32", arr_dmmv_q2_k_q8_1_f32_len[reduc], arr_dmmv_q2_k_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_కొలత_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_q8_1_f32", arr_dmmv_q3_k_q8_1_f32_len[reduc], arr_dmmv_q3_k_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_కొలత_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_q8_1_f32", arr_dmmv_q4_k_q8_1_f32_len[reduc], arr_dmmv_q4_k_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_కొలత_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_q8_1_f32", arr_dmmv_q5_k_q8_1_f32_len[reduc], arr_dmmv_q5_k_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_కొలత_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_q8_1_f32", arr_dmmv_q6_k_q8_1_f32_len[reduc], arr_dmmv_q6_k_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_కొలత_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_S][i], "mul_mat_vec_iq1_s_q8_1_f32", arr_dmmv_iq1_s_q8_1_f32_len[reduc], arr_dmmv_iq1_s_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {1*rm_iq_int(i), 1, 1}, {wg_కొలత_subgroup_int, 1*rm_iq_int(i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_M][i], "mul_mat_vec_iq1_m_q8_1_f32", arr_dmmv_iq1_m_q8_1_f32_len[reduc], arr_dmmv_iq1_m_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {1*rm_iq_int(i), 1, 1}, {wg_కొలత_subgroup_int, 1*rm_iq_int(i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ4_XS][i], "mul_mat_vec_iq4_xs_q8_1_f32", arr_dmmv_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_iq4_xs_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_నెట్టు_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_కొలత_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, నిజం, use_subgroups, subgroup_కొలత_int);

            }
#అయితే_ముగింపు // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
        }

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F32 ], "mul_mat_vec_id_f32_f32",        arr_dmmv_id_f32_f32_f32_len[reduc],     arr_dmmv_id_f32_f32_f32_దత్తాంశం[reduc],     "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {1, 1, 1}, {wg_కొలత_subgroup, 1}, 1, అబద్ధం, use_subgroups, force_subgroup_కొలత);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F16 ], "mul_mat_vec_id_f16_f32",        arr_dmmv_id_f16_f32_f32_len[reduc],     arr_dmmv_id_f16_f32_f32_దత్తాంశం[reduc],     "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {2, 1, 1}, {wg_కొలత_subgroup, 2}, 1, అబద్ధం, use_subgroups, force_subgroup_కొలత);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_BF16], "mul_mat_vec_id_bf16_f32",       arr_dmmv_id_bf16_f32_f32_len[reduc],    arr_dmmv_id_bf16_f32_f32_దత్తాంశం[reduc],    "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {2, 1, 1}, {wg_కొలత_subgroup, 2}, 1, అబద్ధం, use_subgroups, force_subgroup_కొలత);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q1_0], "mul_mat_vec_id_q1_0_f32",       arr_dmmv_id_q1_0_f32_f32_len[reduc],    arr_dmmv_id_q1_0_f32_f32_దత్తాంశం[reduc],    "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_f32",       arr_dmmv_id_q2_0_f32_f32_len[reduc],    arr_dmmv_id_q2_0_f32_f32_దత్తాంశం[reduc],    "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_f32",       arr_dmmv_id_q4_0_f32_f32_len[reduc],    arr_dmmv_id_q4_0_f32_f32_దత్తాంశం[reduc],    "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_f32",       arr_dmmv_id_q4_1_f32_f32_len[reduc],    arr_dmmv_id_q4_1_f32_f32_దత్తాంశం[reduc],    "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_f32",       arr_dmmv_id_q5_0_f32_f32_len[reduc],    arr_dmmv_id_q5_0_f32_f32_దత్తాంశం[reduc],    "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_f32",       arr_dmmv_id_q5_1_f32_f32_len[reduc],    arr_dmmv_id_q5_1_f32_f32_దత్తాంశం[reduc],    "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {2*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 2*rm_stdq}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_f32",       arr_dmmv_id_q8_0_f32_f32_len[reduc],    arr_dmmv_id_q8_0_f32_f32_దత్తాంశం[reduc],    "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {1*rm_stdq, 1, 1}, {wg_కొలత_subgroup, 1*rm_stdq}, 1, నిజం, use_subgroups, force_subgroup_కొలత);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_f32",       arr_dmmv_id_q2_k_f32_f32_len[reduc16],    arr_dmmv_id_q2_k_f32_f32_దత్తాంశం[reduc16],    "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_f32",       arr_dmmv_id_q3_k_f32_f32_len[reduc16],    arr_dmmv_id_q3_k_f32_f32_దత్తాంశం[reduc16],    "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_f32",       arr_dmmv_id_q4_k_f32_f32_len[reduc16],    arr_dmmv_id_q4_k_f32_f32_దత్తాంశం[reduc16],    "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_f32",       arr_dmmv_id_q5_k_f32_f32_len[reduc16],    arr_dmmv_id_q5_k_f32_f32_దత్తాంశం[reduc16],    "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_f32",       arr_dmmv_id_q6_k_f32_f32_len[reduc16],    arr_dmmv_id_q6_k_f32_f32_దత్తాంశం[reduc16],    "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ1_0], "mul_mat_vec_id_tq1_0_f32",     arr_dmmv_id_tq1_0_f32_f32_len[reduc16],   arr_dmmv_id_tq1_0_f32_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ2_0], "mul_mat_vec_id_tq2_0_f32",     arr_dmmv_id_tq2_0_f32_f32_len[reduc16],   arr_dmmv_id_tq2_0_f32_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_kq, 1, 1}, {wg_కొలత_subgroup16, rm_kq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_S],   "mul_mat_vec_id_iq1_s_f32",   arr_dmmv_id_iq1_s_f32_f32_len[reduc16],   arr_dmmv_id_iq1_s_f32_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_M],   "mul_mat_vec_id_iq1_m_f32",   arr_dmmv_id_iq1_m_f32_f32_len[reduc16],   arr_dmmv_id_iq1_m_f32_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XXS], "mul_mat_vec_id_iq2_xxs_f32", arr_dmmv_id_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq2_xxs_f32_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XS],  "mul_mat_vec_id_iq2_xs_f32",  arr_dmmv_id_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq2_xs_f32_f32_దత్తాంశం[reduc16],  "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_S],   "mul_mat_vec_id_iq2_s_f32",   arr_dmmv_id_iq2_s_f32_f32_len[reduc16],   arr_dmmv_id_iq2_s_f32_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_XXS], "mul_mat_vec_id_iq3_xxs_f32", arr_dmmv_id_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq3_xxs_f32_f32_దత్తాంశం[reduc16], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_S],   "mul_mat_vec_id_iq3_s_f32",   arr_dmmv_id_iq3_s_f32_f32_len[reduc16],   arr_dmmv_id_iq3_s_f32_f32_దత్తాంశం[reduc16],   "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_XS],  "mul_mat_vec_id_iq4_xs_f32",  arr_dmmv_id_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq4_xs_f32_f32_దత్తాంశం[reduc16],  "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_NL],  "mul_mat_vec_id_iq4_nl_f32",  arr_dmmv_id_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_id_iq4_nl_f32_f32_దత్తాంశం[reduc16],  "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_MXFP4],   "mul_mat_vec_id_mxfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_mxfp4_f32_f32, reduc16), OCP_DMMV_దత్తాంశం(arr_dmmv_id_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq}, 1, నిజం, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_NVFP4],   "mul_mat_vec_id_nvfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_nvfp4_f32_f32, reduc16), OCP_DMMV_దత్తాంశం(arr_dmmv_id_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_iq, 1, 1}, {wg_కొలత_subgroup16, rm_iq}, 1, నిజం, use_subgroups16, force_subgroup_size16);

#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        ఒకవేళ (device->integer_dot_లబ్ధం) {
            స్థిరరాశి uint32_t subgroup_కొలత_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_కొలత_control) ? device->subgroup_min_కొలత : device->subgroup_కొలత;
            స్థిరరాశి uint32_t wg_కొలత_subgroup_int = (w == DMMV_WG_కొలత_SUBGROUP) ? subgroup_కొలత_int : (subgroup_కొలత_int * 4);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_q8_1_f32", arr_dmmv_id_q2_0_q8_1_f32_len[reduc], arr_dmmv_id_q2_0_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_కొలత_subgroup_int, rm_id(2*rm_kq_int)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_q8_1_f32", arr_dmmv_id_q4_0_q8_1_f32_len[reduc], arr_dmmv_id_q4_0_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_కొలత_subgroup_int, rm_id(1*rm_stdq_int)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_q8_1_f32", arr_dmmv_id_q4_1_q8_1_f32_len[reduc], arr_dmmv_id_q4_1_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_కొలత_subgroup_int, rm_id(1*rm_stdq_int)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_q8_1_f32", arr_dmmv_id_q5_0_q8_1_f32_len[reduc], arr_dmmv_id_q5_0_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_కొలత_subgroup_int, rm_id(1*rm_stdq_int)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_q8_1_f32", arr_dmmv_id_q5_1_q8_1_f32_len[reduc], arr_dmmv_id_q5_1_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_కొలత_subgroup_int, rm_id(1*rm_stdq_int)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_q8_1_f32", arr_dmmv_id_q8_0_q8_1_f32_len[reduc], arr_dmmv_id_q8_0_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_కొలత_subgroup_int, rm_id(1*rm_stdq_int)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_MXFP4], "mul_mat_vec_id_mxfp4_q8_1_f32", arr_dmmv_id_mxfp4_q8_1_f32_len[reduc], arr_dmmv_id_mxfp4_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_id(2*rm_stdq_int), 1, 1}, {wg_కొలత_subgroup_int, rm_id(2*rm_stdq_int)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_q8_1_f32", arr_dmmv_id_q2_k_q8_1_f32_len[reduc], arr_dmmv_id_q2_k_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_కొలత_subgroup_int, rm_id(2*rm_kq_int)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_q8_1_f32", arr_dmmv_id_q3_k_q8_1_f32_len[reduc], arr_dmmv_id_q3_k_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_కొలత_subgroup_int, rm_id(1*rm_kq_int)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_q8_1_f32", arr_dmmv_id_q4_k_q8_1_f32_len[reduc], arr_dmmv_id_q4_k_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_కొలత_subgroup_int, rm_id(1*rm_kq_int)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_q8_1_f32", arr_dmmv_id_q5_k_q8_1_f32_len[reduc], arr_dmmv_id_q5_k_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_కొలత_subgroup_int, rm_id(1*rm_kq_int)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_q8_1_f32", arr_dmmv_id_q6_k_q8_1_f32_len[reduc], arr_dmmv_id_q6_k_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_కొలత_subgroup_int, rm_id(1*rm_kq_int)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_S], "mul_mat_vec_id_iq1_s_q8_1_f32", arr_dmmv_id_iq1_s_q8_1_f32_len[reduc], arr_dmmv_id_iq1_s_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {1*rm_iq_int(0), 1, 1}, {wg_కొలత_subgroup_int, 1*rm_iq_int(0)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_M], "mul_mat_vec_id_iq1_m_q8_1_f32", arr_dmmv_id_iq1_m_q8_1_f32_len[reduc], arr_dmmv_id_iq1_m_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {1*rm_iq_int(0), 1, 1}, {wg_కొలత_subgroup_int, 1*rm_iq_int(0)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ4_XS], "mul_mat_vec_id_iq4_xs_q8_1_f32", arr_dmmv_id_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_id_iq4_xs_q8_1_f32_దత్తాంశం[reduc], "main", mul_mat_vec_id_అంకె_bindings, పరిమాణం(vk_mat_vec_id_నెట్టు_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_కొలత_subgroup_int, rm_id(1*rm_stdq_int)}, 1, నిజం, use_subgroups, subgroup_కొలత_int);
        }
#అయితే_ముగింపు // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
    }

#నిర్వచనం_రద్దు OCP_DMMV_దత్తాంశం
#నిర్వచనం_రద్దు OCP_DMMV_LEN

#ఒకవేళ !defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    GGML_వాడని(rm_stdq_int);
    GGML_వాడని(rm_kq_int);
    GGML_వాడని(is_rdna3_or_4);
    GGML_వాడని(rm_int_n);
    GGML_వాడని(rm_id);
    GGML_వాడని(rm_iq_int);
#అయితే_ముగింపు

    // dequant shaders
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F32 ], "f32_to_f16",   dequant_f32_len,  dequant_f32_దత్తాంశం,  "main", 2, 5 * పరిమాణం(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q1_0], "dequant_q1_0", dequant_q1_0_len, dequant_q1_0_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 8, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_0], "dequant_q2_0", dequant_q2_0_len, dequant_q2_0_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_0], "dequant_q4_0", dequant_q4_0_len, dequant_q4_0_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_1], "dequant_q4_1", dequant_q4_1_len, dequant_q4_1_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_transpose[GGML_TYPE_Q8_0], "dequant_q8_0_transpose", dequant_q8_0_transpose_len, dequant_q8_0_transpose_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ1_0], "dequant_tq1_0", dequant_tq1_0_len, dequant_tq1_0_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ2_0], "dequant_tq2_0", dequant_tq2_0_len, dequant_tq2_0_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_S],   "dequant_iq1_s",   dequant_iq1_s_len,   dequant_iq1_s_దత్తాంశం,   "main", 2, 5 * పరిమాణం(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_M],   "dequant_iq1_m",   dequant_iq1_m_len,   dequant_iq1_m_దత్తాంశం,   "main", 2, 5 * పరిమాణం(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XXS], "dequant_iq2_xxs", dequant_iq2_xxs_len, dequant_iq2_xxs_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XS],  "dequant_iq2_xs",  dequant_iq2_xs_len,  dequant_iq2_xs_దత్తాంశం,  "main", 2, 5 * పరిమాణం(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_S],   "dequant_iq2_s",   dequant_iq2_s_len,   dequant_iq2_s_దత్తాంశం,   "main", 2, 5 * పరిమాణం(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_XXS], "dequant_iq3_xxs", dequant_iq3_xxs_len, dequant_iq3_xxs_దత్తాంశం, "main", 2, 5 * పరిమాణం(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_S],   "dequant_iq3_s",   dequant_iq3_s_len,   dequant_iq3_s_దత్తాంశం,   "main", 2, 5 * పరిమాణం(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_XS],  "dequant_iq4_xs",  dequant_iq4_xs_len,  dequant_iq4_xs_దత్తాంశం,  "main", 2, 5 * పరిమాణం(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_NL],  "dequant_iq4_nl",  dequant_iq4_nl_len,  dequant_iq4_nl_దత్తాంశం,  "main", 2, 5 * పరిమాణం(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_MXFP4],   "dequant_mxfp4",   dequant_mxfp4_len,   dequant_mxfp4_దత్తాంశం,   "main", 2, 5 * పరిమాణం(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_NVFP4],   "dequant_nvfp4",   dequant_nvfp4_len,   dequant_nvfp4_దత్తాంశం,   "main", 2, 5 * పరిమాణం(uint32_t), {256 * 16, 1, 1}, {}, 1);

    // get_rows
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_F32 ], "get_rows_f32",  get_అడ్డువరుసలు_f32_len,  get_అడ్డువరుసలు_f32_దత్తాంశం,  "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_F16 ], "get_rows_f16",  get_అడ్డువరుసలు_f16_len,  get_అడ్డువరుసలు_f16_దత్తాంశం,  "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_BF16], "get_rows_bf16", get_అడ్డువరుసలు_bf16_len, get_అడ్డువరుసలు_bf16_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_Q1_0], "get_rows_q1_0", get_అడ్డువరుసలు_q1_0_len, get_అడ్డువరుసలు_q1_0_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_Q2_0], "get_rows_q2_0", get_అడ్డువరుసలు_q2_0_len, get_అడ్డువరుసలు_q2_0_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_Q4_0], "get_rows_q4_0", get_అడ్డువరుసలు_q4_0_len, get_అడ్డువరుసలు_q4_0_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_Q4_1], "get_rows_q4_1", get_అడ్డువరుసలు_q4_1_len, get_అడ్డువరుసలు_q4_1_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_Q5_0], "get_rows_q5_0", get_అడ్డువరుసలు_q5_0_len, get_అడ్డువరుసలు_q5_0_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_Q5_1], "get_rows_q5_1", get_అడ్డువరుసలు_q5_1_len, get_అడ్డువరుసలు_q5_1_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_Q8_0], "get_rows_q8_0", get_అడ్డువరుసలు_q8_0_len, get_అడ్డువరుసలు_q8_0_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_Q2_K], "get_rows_q2_k", get_అడ్డువరుసలు_q2_k_len, get_అడ్డువరుసలు_q2_k_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_Q3_K], "get_rows_q3_k", get_అడ్డువరుసలు_q3_k_len, get_అడ్డువరుసలు_q3_k_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_Q4_K], "get_rows_q4_k", get_అడ్డువరుసలు_q4_k_len, get_అడ్డువరుసలు_q4_k_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_Q5_K], "get_rows_q5_k", get_అడ్డువరుసలు_q5_k_len, get_అడ్డువరుసలు_q5_k_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_Q6_K], "get_rows_q6_k", get_అడ్డువరుసలు_q6_k_len, get_అడ్డువరుసలు_q6_k_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_TQ1_0], "get_rows_tq1_0", get_అడ్డువరుసలు_tq1_0_len, get_అడ్డువరుసలు_tq1_0_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_TQ2_0], "get_rows_tq2_0", get_అడ్డువరుసలు_tq2_0_len, get_అడ్డువరుసలు_tq2_0_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_IQ1_S],   "get_rows_iq1_s",   get_అడ్డువరుసలు_iq1_s_len,   get_అడ్డువరుసలు_iq1_s_దత్తాంశం,   "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_IQ1_M],   "get_rows_iq1_m",   get_అడ్డువరుసలు_iq1_m_len,   get_అడ్డువరుసలు_iq1_m_దత్తాంశం,   "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs", get_అడ్డువరుసలు_iq2_xxs_len, get_అడ్డువరుసలు_iq2_xxs_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs",  get_అడ్డువరుసలు_iq2_xs_len,  get_అడ్డువరుసలు_iq2_xs_దత్తాంశం,  "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_IQ2_S],   "get_rows_iq2_s",   get_అడ్డువరుసలు_iq2_s_len,   get_అడ్డువరుసలు_iq2_s_దత్తాంశం,   "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs", get_అడ్డువరుసలు_iq3_xxs_len, get_అడ్డువరుసలు_iq3_xxs_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_IQ3_S],   "get_rows_iq3_s",   get_అడ్డువరుసలు_iq3_s_len,   get_అడ్డువరుసలు_iq3_s_దత్తాంశం,   "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs",  get_అడ్డువరుసలు_iq4_xs_len,  get_అడ్డువరుసలు_iq4_xs_దత్తాంశం,  "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl",  get_అడ్డువరుసలు_iq4_nl_len,  get_అడ్డువరుసలు_iq4_nl_దత్తాంశం,  "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_MXFP4],   "get_rows_mxfp4",   get_అడ్డువరుసలు_mxfp4_len,   get_అడ్డువరుసలు_mxfp4_దత్తాంశం,   "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_NVFP4],   "get_rows_nvfp4",   get_అడ్డువరుసలు_nvfp4_len,   get_అడ్డువరుసలు_nvfp4_దత్తాంశం,   "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు[GGML_TYPE_I32],     "get_rows_i32",     get_అడ్డువరుసలు_i32_len,     get_అడ్డువరుసలు_i32_దత్తాంశం,     "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_F32 ], "get_rows_f32_f32",  get_అడ్డువరుసలు_f32_f32_len,  get_అడ్డువరుసలు_f32_f32_దత్తాంశం,  "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_F16 ], "get_rows_f16_f32",  get_అడ్డువరుసలు_f16_f32_len,  get_అడ్డువరుసలు_f16_f32_దత్తాంశం,  "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_BF16], "get_rows_bf16_f32", get_అడ్డువరుసలు_bf16_f32_len, get_అడ్డువరుసలు_bf16_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_Q1_0], "get_rows_q1_0_f32", get_అడ్డువరుసలు_q1_0_f32_len, get_అడ్డువరుసలు_q1_0_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_Q2_0], "get_rows_q2_0_f32", get_అడ్డువరుసలు_q2_0_f32_len, get_అడ్డువరుసలు_q2_0_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_Q4_0], "get_rows_q4_0_f32", get_అడ్డువరుసలు_q4_0_f32_len, get_అడ్డువరుసలు_q4_0_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_Q4_1], "get_rows_q4_1_f32", get_అడ్డువరుసలు_q4_1_f32_len, get_అడ్డువరుసలు_q4_1_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_Q5_0], "get_rows_q5_0_f32", get_అడ్డువరుసలు_q5_0_f32_len, get_అడ్డువరుసలు_q5_0_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_అడ్డువరుసలు_q5_1_f32_len, get_అడ్డువరుసలు_q5_1_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_అడ్డువరుసలు_q8_0_f32_len, get_అడ్డువరుసలు_q8_0_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_Q2_K], "get_rows_q2_k_f32", get_అడ్డువరుసలు_q2_k_f32_len, get_అడ్డువరుసలు_q2_k_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_Q3_K], "get_rows_q3_k_f32", get_అడ్డువరుసలు_q3_k_f32_len, get_అడ్డువరుసలు_q3_k_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_Q4_K], "get_rows_q4_k_f32", get_అడ్డువరుసలు_q4_k_f32_len, get_అడ్డువరుసలు_q4_k_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_Q5_K], "get_rows_q5_k_f32", get_అడ్డువరుసలు_q5_k_f32_len, get_అడ్డువరుసలు_q5_k_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_Q6_K], "get_rows_q6_k_f32", get_అడ్డువరుసలు_q6_k_f32_len, get_అడ్డువరుసలు_q6_k_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_TQ1_0], "get_rows_tq1_0_f32", get_అడ్డువరుసలు_tq1_0_f32_len, get_అడ్డువరుసలు_tq1_0_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_TQ2_0], "get_rows_tq2_0_f32", get_అడ్డువరుసలు_tq2_0_f32_len, get_అడ్డువరుసలు_tq2_0_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_IQ1_S],   "get_rows_iq1_s_f32",   get_అడ్డువరుసలు_iq1_s_f32_len,   get_అడ్డువరుసలు_iq1_s_f32_దత్తాంశం,   "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_IQ1_M],   "get_rows_iq1_m_f32",   get_అడ్డువరుసలు_iq1_m_f32_len,   get_అడ్డువరుసలు_iq1_m_f32_దత్తాంశం,   "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs_f32", get_అడ్డువరుసలు_iq2_xxs_f32_len, get_అడ్డువరుసలు_iq2_xxs_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs_f32",  get_అడ్డువరుసలు_iq2_xs_f32_len,  get_అడ్డువరుసలు_iq2_xs_f32_దత్తాంశం,  "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_IQ2_S],   "get_rows_iq2_s_f32",   get_అడ్డువరుసలు_iq2_s_f32_len,   get_అడ్డువరుసలు_iq2_s_f32_దత్తాంశం,   "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs_f32", get_అడ్డువరుసలు_iq3_xxs_f32_len, get_అడ్డువరుసలు_iq3_xxs_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_IQ3_S],   "get_rows_iq3_s_f32",   get_అడ్డువరుసలు_iq3_s_f32_len,   get_అడ్డువరుసలు_iq3_s_f32_దత్తాంశం,   "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs_f32",  get_అడ్డువరుసలు_iq4_xs_f32_len,  get_అడ్డువరుసలు_iq4_xs_f32_దత్తాంశం,  "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl_f32",  get_అడ్డువరుసలు_iq4_nl_f32_len,  get_అడ్డువరుసలు_iq4_nl_f32_దత్తాంశం,  "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_MXFP4],   "get_rows_mxfp4_f32",   get_అడ్డువరుసలు_mxfp4_f32_len,   get_అడ్డువరుసలు_mxfp4_f32_దత్తాంశం,   "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_f32[GGML_TYPE_NVFP4],   "get_rows_nvfp4_f32",   get_అడ్డువరుసలు_nvfp4_f32_len,   get_అడ్డువరుసలు_nvfp4_f32_దత్తాంశం,   "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_అడ్డువరుసలు_వెనుక_f32, "get_rows_back_f32", get_అడ్డువరుసలు_వెనుక_f32_len, get_అడ్డువరుసలు_వెనుక_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {256, 1, 1}, {}, 1, నిజం);

    ggml_vk_create_pipeline(device, device->pipeline_matmul_విభజించు_k_reduce, "split_k_reduce", విభజించు_k_reduce_len, విభజించు_k_reduce_దత్తాంశం, "main", 2, 2 * పరిమాణం(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_విభజించు_k_reduce, "fa_split_k_reduce", fa_విభజించు_k_reduce_len, fa_విభజించు_k_reduce_దత్తాంశం, "main", 3, పరిమాణం(vk_op_flash_attn_విభజించు_k_reduce_నెట్టు_constants), {1, device->subgroup_కొలత, 1}, {device->subgroup_కొలత}, 1, నిజం);

#ఒకవేళ defined(VK_KHR_cooperative_మాతృక) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ఒకవేళ (device->vendor_id == VK_VENDOR_ID_INTEL && (device->architecture == INTEL_XE2 || (device->architecture == INTEL_XE1 && device->coopmat_support && device->uma))) {
        స్వయంచాలక పై_power_యొక్క_2 = [&](uint32_t లోపల) {
            GGML_ASSERT(లోపల != 0);
            ఒకవేళ (లోపల <= 1) తిరిగి 1u;
            uint32_t ret = లోపల - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            తిరిగి ret + 1;
            };

        uint32_t xe_native_sub_సమూహం_కొలత = 16;
        ఒకవేళ (device->architecture == INTEL_XE1) {
            xe_native_sub_సమూహం_కొలత = 8;
        }

        కొరకు (స్వయంచాలక& it : device->pipeline_xe_fa_decode_dual_phases) {
            స్థిరరాశి uint32_t విభజించు_p_chunk = 32;
            స్వయంచాలక HdQk = it.మొదటి;
            స్వయంచాలక& pipelines = it.second;
            uint32_t తల_dim_qk = std::get<0>(HdQk);
            uint32_t తల_dim_pv = std::get<1>(HdQk);
            uint32_t gqa_నిష్పత్తి = std::get<2>(HdQk);
            uint32_t q_len = std::get<3>(HdQk);
            స్థిరరాశి uint32_t బహిర్గతం_dim_చొప్పున_wg = gqa_నిష్పత్తి > 16 ? 8 : 16;
            uint32_t aligned_q_len = పై_power_యొక్క_2(q_len);
            uint32_t సమూహం_sz_ph1 = std::min(std::max(aligned_q_len * xe_native_sub_సమూహం_కొలత, 64u), 256u);
            uint32_t బహిర్గతం_చొప్పున_wg_ph1 = std::min(q_len, 256u / xe_native_sub_సమూహం_కొలత);
            uint32_t aligned_gqa_నిష్పత్తి = పై_power_యొక్క_2(gqa_నిష్పత్తి);
            uint32_t విభజించు_p_చొప్పున_iter_ph2 = 256;
            uint32_t విభజించు_p_చొప్పున_warp = 16;
            uint32_t సమూహం_sz_ph2 = (విభజించు_p_చొప్పున_iter_ph2 / విభజించు_p_చొప్పున_warp) * xe_native_sub_సమూహం_కొలత;
            uint32_t బహిర్గతం_చొప్పున_wg_ph2 = std::min(std::max(16u / aligned_gqa_నిష్పత్తి, 1u), q_len);
            ggml_vk_create_pipeline(device, pipelines.మొదటి, "xe_fa_decode_ph1", fa_decode_ph1_cm1_len, fa_decode_ph1_cm1_దత్తాంశం, "main", 5, పరిమాణం(vk_fa_xe_opt_నెట్టు_constants), { 1, 32, 1 }, { సమూహం_sz_ph1, gqa_నిష్పత్తి, తల_dim_qk, xe_native_sub_సమూహం_కొలత, విభజించు_p_chunk, బహిర్గతం_చొప్పున_wg_ph1 }, 1, అబద్ధం, నిజం, xe_native_sub_సమూహం_కొలత);
            ggml_vk_create_pipeline(device, pipelines.second, "xe_fa_decode_ph2", fa_decode_ph2_cm1_len, fa_decode_ph2_cm1_దత్తాంశం, "main", 5, పరిమాణం(vk_fa_xe_opt_నెట్టు_constants), { 1, 1, 1 }, { సమూహం_sz_ph2, gqa_నిష్పత్తి, తల_dim_pv, బహిర్గతం_చొప్పున_wg_ph2, xe_native_sub_సమూహం_కొలత, విభజించు_p_చొప్పున_iter_ph2, విభజించు_p_chunk, బహిర్గతం_dim_చొప్పున_wg }, 1, అబద్ధం, నిజం, xe_native_sub_సమూహం_కొలత);
        }
    }
#అయితే_ముగింపు

    కొరకు (స్వయంచాలక &it : device->pipeline_fa_ముసుగు_opt) {
        స్వయంచాలక BrBc = it.మొదటి;
        ggml_vk_create_pipeline(device, it.second, "fa_mask_opt", fa_ముసుగు_opt_len, fa_ముసుగు_opt_దత్తాంశం, "main", 2, పరిమాణం(vk_op_flash_attn_ముసుగు_opt_నెట్టు_constants), {1, 1, 1}, {128, 128 / device->subgroup_కొలత, BrBc.మొదటి, BrBc.second}, 1, నిజం, నిజం, device->subgroup_కొలత);
    }

    {
        // Large workgroup so the per-row KV scan parallelizes; capped to device limits.
        స్థిరరాశి uint32_t compact_max = std::min({1024u, device->properties.limits.max_Compute_Work_సమూహం_Invocations, device->properties.limits.max_Compute_Work_సమూహం_కొలత[0]});

        // Fast ballot prefix-sum path when the device supports full subgroups; otherwise
        // a shared-memory prefix-sum fallback. Both emit a deterministic ascending list.
        device->fa_sparse_compact_use_subgroups = device->subgroup_ballot && device->subgroup_require_నిండిన_support;
        ఒకవేళ (device->fa_sparse_compact_use_subgroups) {
            స్థిరరాశి uint32_t compact_wg = std::max(device->subgroup_కొలత, (compact_max / device->subgroup_కొలత) * device->subgroup_కొలత);
            స్థిరరాశి uint32_t compact_అంకె_sg = compact_wg / device->subgroup_కొలత;
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact_subgroup, "fa_sparse_compact_subgroup", fa_sparse_compact_subgroup_len, fa_sparse_compact_subgroup_దత్తాంశం, "main", 2, పరిమాణం(vk_op_flash_attn_sparse_compact_నెట్టు_constants), {1, 1, 1}, {compact_wg, compact_అంకె_sg}, 1, నిజం, నిజం, device->subgroup_కొలత);
        } లేకపోతే {
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact, "fa_sparse_compact", fa_sparse_compact_len, fa_sparse_compact_దత్తాంశం, "main", 2, పరిమాణం(vk_op_flash_attn_sparse_compact_నెట్టు_constants), {1, 1, 1}, {compact_max}, 1, నిజం);
        }
    }

    ఒకవేళ (device->subgroup_clustered && device->subgroup_require_నిండిన_support) {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_subgroup_len, quantize_q8_1_x4_subgroup_దత్తాంశం, "main", 2, పరిమాణం(vk_quantize_q8_1_నెట్టు_constants), {32 * device->subgroup_కొలత / 8, 1, 1}, { device->subgroup_కొలత }, 1, నిజం, నిజం);
    } లేకపోతే {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_len, quantize_q8_1_x4_దత్తాంశం, "main", 2, పరిమాణం(vk_quantize_q8_1_నెట్టు_constants), {32 * device->subgroup_కొలత / 8, 1, 1}, { device->subgroup_కొలత }, 1);
    }

    కొరకు (uint32_t i = 0; i < p021_max_gqa_నిష్పత్తి; ++i) {
        ఒకవేళ (device->subgroup_arithmetic && device->subgroup_require_నిండిన_support) {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_subgroup_కూడు_len, mul_mat_vec_p021_f16_f32_subgroup_కూడు_దత్తాంశం, "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_p021_నెట్టు_constants), {1, 1, 1}, {device->subgroup_కొలత, i + 1}, 1, నిజం, నిజం);
        } లేకపోతే {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_len,              mul_mat_vec_p021_f16_f32_దత్తాంశం,              "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_p021_నెట్టు_constants), {1, 1, 1}, {device->subgroup_కొలత, i + 1}, 1, నిజం);
        }
    }
    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", mul_mat_vec_nc_f16_f32_len, mul_mat_vec_nc_f16_f32_దత్తాంశం, "main", mul_mat_vec_అంకె_bindings, పరిమాణం(vk_mat_vec_nc_నెట్టు_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_norm_f32, "norm_f32", norm_f32_len, norm_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_సమూహం_norm_f32, "group_norm_f32", సమూహం_norm_f32_len, సమూహం_norm_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_నెట్టు_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32, "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {0, 0}, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_f32, "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {0, 1}, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_కూడు_f32, "rms_norm_mul_add_f32", rms_norm_mul_కూడు_f32_len, rms_norm_mul_కూడు_f32_దత్తాంశం, "main", 5, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {0, 1, 0}, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_కూడు_mul_f32, "rms_norm_mul_add_mul_f32", rms_norm_mul_కూడు_f32_len, rms_norm_mul_కూడు_f32_దత్తాంశం, "main", 5, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {0, 1, 1}, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_కూడు_partials_f32, "rms_norm_mul_add_partials_f32", rms_norm_mul_కూడు_partials_f32_len, rms_norm_mul_కూడు_partials_f32_దత్తాంశం, "main", 6, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {0, 1, 0}, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_కూడు_mul_partials_f32, "rms_norm_mul_add_mul_partials_f32", rms_norm_mul_కూడు_partials_f32_len, rms_norm_mul_కూడు_partials_f32_దత్తాంశం, "main", 6, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {0, 1, 1}, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_సమితి_అడ్డువరుసలు_f32_f32, "rms_norm_set_rows_f32_f32", rms_norm_సమితి_అడ్డువరుసలు_f32_f32_len, rms_norm_సమితి_అడ్డువరుసలు_f32_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {0, 0}, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_సమితి_అడ్డువరుసలు_f32_f16, "rms_norm_set_rows_f32_f16", rms_norm_సమితి_అడ్డువరుసలు_f32_f16_len, rms_norm_సమితి_అడ్డువరుసలు_f32_f16_దత్తాంశం, "main", 4, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {0, 0}, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_partials_f32, "rms_norm_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {0, 0}, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_partials_f32, "rms_norm_mul_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {0, 1}, 1, నిజం);

    ఒకవేళ (పరిమాణం(vk_op_rms_norm_mul_rope_నెట్టు_constants) <= device->properties.limits.max_నెట్టు_Constants_కొలత) {
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f32, "rms_norm_mul_rope_f32_f32", rms_norm_mul_rope_f32_f32_len, rms_norm_mul_rope_f32_f32_దత్తాంశం, "main", 7, పరిమాణం(vk_op_rms_norm_mul_rope_నెట్టు_constants), {1, 1, 1}, {0, 1}, 1, నిజం);
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f16, "rms_norm_mul_rope_f32_f16", rms_norm_mul_rope_f32_f16_len, rms_norm_mul_rope_f32_f16_దత్తాంశం, "main", 7, పరిమాణం(vk_op_rms_norm_mul_rope_నెట్టు_constants), {1, 1, 1}, {0, 1}, 1, నిజం);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_వెనుక_f32, "rms_norm_back_f32", rms_norm_వెనుక_f32_len, rms_norm_వెనుక_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_నెట్టు_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_l2_norm_f32, "l2_norm_f32", l2_norm_f32_len, l2_norm_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_32, "cpy_32", cpy_32_len, cpy_32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f16, "cpy_f32_f16", cpy_f32_f16_len, cpy_f32_f16_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_16, "cpy_16", cpy_16_len, cpy_16_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f32, "cpy_f16_f32", cpy_f16_f32_len, cpy_f16_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_bf16,"cpy_f32_bf16",cpy_f32_bf16_len,cpy_f32_bf16_దత్తాంశం,"main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f32,"cpy_bf16_f32",cpy_bf16_f32_len,cpy_bf16_f32_దత్తాంశం,"main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f16,"cpy_bf16_f16",cpy_bf16_f16_len,cpy_bf16_f16_దత్తాంశం,"main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_i32_f32, "cpy_i32_f32", cpy_i32_f32_len, cpy_i32_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_i32, "cpy_f32_i32", cpy_f32_i32_len, cpy_f32_i32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_32, "contig_cpy_32", contig_cpy_32_len, contig_cpy_32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f16, "contig_cpy_f32_f16", contig_cpy_f32_f16_len, contig_cpy_f32_f16_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_16, "contig_cpy_16", contig_cpy_16_len, contig_cpy_16_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f32, "contig_cpy_f16_f32", contig_cpy_f16_f32_len, contig_cpy_f16_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_bf16,"contig_cpy_f32_bf16",contig_cpy_f32_bf16_len,contig_cpy_f32_bf16_దత్తాంశం,"main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f32,"contig_cpy_bf16_f32",contig_cpy_bf16_f32_len,contig_cpy_bf16_f32_దత్తాంశం,"main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f16,"contig_cpy_bf16_f16",contig_cpy_bf16_f16_len,contig_cpy_bf16_f16_దత్తాంశం,"main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_i32_f32, "contig_cpy_i32_f32", contig_cpy_i32_f32_len, contig_cpy_i32_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_i32, "contig_cpy_f32_i32", contig_cpy_f32_i32_len, contig_cpy_f32_i32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_32, "cpy_transpose_32", cpy_transpose_32_len, cpy_transpose_32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_16, "cpy_transpose_16", cpy_transpose_16_len, cpy_transpose_16_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_32, "cpy_transpose_02_32", cpy_transpose_02_32_len, cpy_transpose_02_32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_16, "cpy_transpose_02_16", cpy_transpose_02_16_len, cpy_transpose_02_16_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q1_0], "cpy_f32_q1_0", cpy_f32_q1_0_len, cpy_f32_q1_0_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q2_0], "cpy_f32_q2_0", cpy_f32_q2_0_len, cpy_f32_q2_0_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_0], "cpy_f32_q4_0", cpy_f32_q4_0_len, cpy_f32_q4_0_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_1], "cpy_f32_q4_1", cpy_f32_q4_1_len, cpy_f32_q4_1_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_0], "cpy_f32_q5_0", cpy_f32_q5_0_len, cpy_f32_q5_0_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_1], "cpy_f32_q5_1", cpy_f32_q5_1_len, cpy_f32_q5_1_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q8_0], "cpy_f32_q8_0", cpy_f32_q8_0_len, cpy_f32_q8_0_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_IQ4_NL], "cpy_f32_iq4_nl", cpy_f32_iq4_nl_len, cpy_f32_iq4_nl_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {32, 1, 1}, {}, 1);

#నిర్వచించు సమితి_అడ్డువరుసలు_2(src_idx, src, itype) \
        ggml_vk_create_pipeline(device, device->pipeline_సమితి_అడ్డువరుసలు ## itype [src_idx][GGML_TYPE_F32],  "set_rows_" #src "_f32" #itype,  సమితి_అడ్డువరుసలు_3 ## src ## _f32 ## itype ## _len,  సమితి_అడ్డువరుసలు_3 ## src ## _f32 ## itype ## దత్తాంశం_2,  "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {1}, 1, నిజం); \
        ggml_vk_create_pipeline(device, device->pipeline_సమితి_అడ్డువరుసలు ## itype [src_idx][GGML_TYPE_F16],  "set_rows_" #src "_f16" #itype,  సమితి_అడ్డువరుసలు_3 ## src ## _f16 ## itype ## _len,  సమితి_అడ్డువరుసలు_3 ## src ## _f16 ## itype ## దత్తాంశం_2,  "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {1}, 1, నిజం); \
        ggml_vk_create_pipeline(device, device->pipeline_సమితి_అడ్డువరుసలు ## itype [src_idx][GGML_TYPE_BF16], "set_rows_" #src "_bf16" #itype, సమితి_అడ్డువరుసలు_3 ## src ## _bf16 ## itype ## _len, సమితి_అడ్డువరుసలు_3 ## src ## _bf16 ## itype ## దత్తాంశం_2, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {1}, 1, నిజం); \
        ggml_vk_create_pipeline(device, device->pipeline_సమితి_అడ్డువరుసలు ## itype [src_idx][GGML_TYPE_Q1_0], "set_rows_" #src "_q1_0" #itype, సమితి_అడ్డువరుసలు_3 ## src ## _q1_0 ## itype ## _len, సమితి_అడ్డువరుసలు_3 ## src ## _q1_0 ## itype ## దత్తాంశం_2, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {1}, 1, నిజం); \
        ggml_vk_create_pipeline(device, device->pipeline_సమితి_అడ్డువరుసలు ## itype [src_idx][GGML_TYPE_Q2_0], "set_rows_" #src "_q2_0" #itype, సమితి_అడ్డువరుసలు_3 ## src ## _q2_0 ## itype ## _len, సమితి_అడ్డువరుసలు_3 ## src ## _q2_0 ## itype ## దత్తాంశం_2, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {1}, 1, నిజం); \
        ggml_vk_create_pipeline(device, device->pipeline_సమితి_అడ్డువరుసలు ## itype [src_idx][GGML_TYPE_Q4_0], "set_rows_" #src "_q4_0" #itype, సమితి_అడ్డువరుసలు_3 ## src ## _q4_0 ## itype ## _len, సమితి_అడ్డువరుసలు_3 ## src ## _q4_0 ## itype ## దత్తాంశం_2, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {1}, 1, నిజం); \
        ggml_vk_create_pipeline(device, device->pipeline_సమితి_అడ్డువరుసలు ## itype [src_idx][GGML_TYPE_Q4_1], "set_rows_" #src "_q4_1" #itype, సమితి_అడ్డువరుసలు_3 ## src ## _q4_1 ## itype ## _len, సమితి_అడ్డువరుసలు_3 ## src ## _q4_1 ## itype ## దత్తాంశం_2, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {1}, 1, నిజం); \
        ggml_vk_create_pipeline(device, device->pipeline_సమితి_అడ్డువరుసలు ## itype [src_idx][GGML_TYPE_Q5_0], "set_rows_" #src "_q5_0" #itype, సమితి_అడ్డువరుసలు_3 ## src ## _q5_0 ## itype ## _len, సమితి_అడ్డువరుసలు_3 ## src ## _q5_0 ## itype ## దత్తాంశం_2, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {1}, 1, నిజం); \
        ggml_vk_create_pipeline(device, device->pipeline_సమితి_అడ్డువరుసలు ## itype [src_idx][GGML_TYPE_Q5_1], "set_rows_" #src "_q5_1" #itype, సమితి_అడ్డువరుసలు_3 ## src ## _q5_1 ## itype ## _len, సమితి_అడ్డువరుసలు_3 ## src ## _q5_1 ## itype ## దత్తాంశం_2, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {1}, 1, నిజం); \
        ggml_vk_create_pipeline(device, device->pipeline_సమితి_అడ్డువరుసలు ## itype [src_idx][GGML_TYPE_Q8_0], "set_rows_" #src "_q8_0" #itype, సమితి_అడ్డువరుసలు_3 ## src ## _q8_0 ## itype ## _len, సమితి_అడ్డువరుసలు_3 ## src ## _q8_0 ## itype ## దత్తాంశం_2, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {1}, 1, నిజం); \
        ggml_vk_create_pipeline(device, device->pipeline_సమితి_అడ్డువరుసలు ## itype [src_idx][GGML_TYPE_IQ4_NL], "set_rows_" #src "_iq4_nl" #itype, సమితి_అడ్డువరుసలు_3 ## src ## _iq4_nl ## itype ## _len, సమితి_అడ్డువరుసలు_3 ## src ## _iq4_nl ## itype ## దత్తాంశం_2, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, {1}, 1, నిజం);

    సమితి_అడ్డువరుసలు_2(0, f32, _i32)
    సమితి_అడ్డువరుసలు_2(0, f32, _i64)
    సమితి_అడ్డువరుసలు_2(1, f16, _i32)
    సమితి_అడ్డువరుసలు_2(1, f16, _i64)
#నిర్వచనం_రద్దు సమితి_అడ్డువరుసలు_2


    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q1_0], "cpy_q1_0_f32", cpy_q1_0_f32_len, cpy_q1_0_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {(uint32_t)ggml_blck_కొలత(GGML_TYPE_Q1_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q2_0], "cpy_q2_0_f32", cpy_q2_0_f32_len, cpy_q2_0_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {(uint32_t)ggml_blck_కొలత(GGML_TYPE_Q2_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_0], "cpy_q4_0_f32", cpy_q4_0_f32_len, cpy_q4_0_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {(uint32_t)ggml_blck_కొలత(GGML_TYPE_Q4_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_1], "cpy_q4_1_f32", cpy_q4_1_f32_len, cpy_q4_1_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {(uint32_t)ggml_blck_కొలత(GGML_TYPE_Q4_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_0], "cpy_q5_0_f32", cpy_q5_0_f32_len, cpy_q5_0_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {(uint32_t)ggml_blck_కొలత(GGML_TYPE_Q5_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_1], "cpy_q5_1_f32", cpy_q5_1_f32_len, cpy_q5_1_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {(uint32_t)ggml_blck_కొలత(GGML_TYPE_Q5_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q8_0], "cpy_q8_0_f32", cpy_q8_0_f32_len, cpy_q8_0_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {(uint32_t)ggml_blck_కొలత(GGML_TYPE_Q8_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_IQ4_NL], "cpy_iq4_nl_f32", cpy_iq4_nl_f32_len, cpy_iq4_nl_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {(uint32_t)ggml_blck_కొలత(GGML_TYPE_IQ4_NL), 1, 1}, {}, 1);

    స్వయంచాలక get_suffix = [](తార్కిక src0_f16, తార్కిక src1_f16, తార్కిక dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        తిరిగి s;
    };

#నిర్వచించు CREATE_ద్విమాన(name, namemod, spec, bindings) \
    కొరకు (పూర్ణాంకం s0 : {0,1}) కొరకు (పూర్ణాంకం s1 : {0,1}) కొరకు (పూర్ణాంకం d : {0,1}) \
        ggml_vk_create_pipeline2(device, device->pipeline_ ## name ## namemod[s0][s1][d], \
                                #name + get_suffix(s0, s1, d) + #namemod, name ## _len[s0][s1][d], name ## దత్తాంశం_2[s0][s1][d], \
                                "main", (bindings), పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {512, 1, 1}, spec, 1);

    CREATE_ద్విమాన(కూడు_2, , {0}, 4)
    CREATE_ద్విమాన(కూడు_2, _norepeat, {1}, 4)
    CREATE_ద్విమాన(sub, , {0}, 3)
    CREATE_ద్విమాన(sub, _norepeat, {1}, 3)
    CREATE_ద్విమాన(mul, , {0}, 3)
    CREATE_ద్విమాన(mul, _norepeat, {1}, 3)
    CREATE_ద్విమాన(div, , {0}, 3)
    CREATE_ద్విమాన(div, _norepeat, {1}, 3)
    CREATE_ద్విమాన(కూడు_rms, , {0}, 4)
    CREATE_ద్విమాన(కూడు_rms, _norepeat, {1}, 4)
#నిర్వచనం_రద్దు CREATE_ద్విమాన

    ఒకవేళ (device->multi_కూడు) {
        కొరకు (uint32_t i = 0; i < MAX_FUSED_ADDS; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_multi_కూడు[i],     "multi_add_f32_"     + std::to_string(i+1), multi_కూడు_f32_len,     multi_కూడు_f32_దత్తాంశం,     "main", MAX_PARAMETER_లెక్క, పరిమాణం(vk_op_multi_కూడు_నెట్టు_constants), {512, 1, 1}, {i+2}, 1);
            ggml_vk_create_pipeline2(device, device->pipeline_multi_కూడు_rms[i], "multi_add_rms_f32_" + std::to_string(i+1), multi_కూడు_rms_f32_len, multi_కూడు_rms_f32_దత్తాంశం, "main", MAX_PARAMETER_లెక్క, పరిమాణం(vk_op_multi_కూడు_నెట్టు_constants), {512, 1, 1}, {i+2}, 1);
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_కూడు_id_f32, "add_id_f32", కూడు_id_f32_len, కూడు_id_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_కూడు_id_నెట్టు_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_acc_f32, "acc_f32", acc_f32_len, acc_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {512, 1, 1}, {0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_సమితి_f32, "set_f32", acc_f32_len, acc_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {512, 1, 1}, {0, 0}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_concat_i8, "concat_i8", concat_i8_len, concat_i8_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i16, "concat_i16", concat_i16_len, concat_i16_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i32, "concat_i32", concat_i32_len, concat_i32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i64, "concat_i64", concat_i64_len, concat_i64_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_upscale_సమీప_f32, "upscale_f32", upscale_f32_len, upscale_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_upscale_నెట్టు_constants), {512, 1, 1}, {GGML_SCALE_బహుళకం_సమీప}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_f32, "upscale_f32", upscale_f32_len, upscale_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_upscale_నెట్టు_constants), {512, 1, 1}, {GGML_SCALE_బహుళకం_BILINEAR}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bicubic_f32, "upscale_f32", upscale_f32_len, upscale_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_upscale_నెట్టు_constants), {512, 1, 1}, {GGML_SCALE_బహుళకం_BICUBIC}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_antialias_f32, "upscale_f32", upscale_f32_len, upscale_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_upscale_నెట్టు_constants), {512, 1, 1}, {GGML_SCALE_బహుళకం_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_scale_f32, "scale_f32", scale_f32_len, scale_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_log[0], "log_f32", log_f32_len, log_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_log[1], "log_f16", log_f16_len, log_f16_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_tri[0], "tri_f32", tri_f32_len, tri_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_tri[1], "tri_f16", tri_f16_len, tri_f16_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag[0], "diag_f32", diag_f32_len, diag_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_diag[1], "diag_f16", diag_f16_len, diag_f16_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pad_f32, "pad_f32", pad_f32_len, pad_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_pad_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pad_reflect_1d_f32, "pad_reflect_1d_f32", pad_reflect_1d_f32_len, pad_reflect_1d_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_roll_f32, "roll_f32", roll_f32_len, roll_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i32, "repeat_i32", repeat_i32_len, repeat_i32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_repeat_వెనుక_f32, "repeat_back_f32", repeat_వెనుక_f32_len, repeat_వెనుక_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i16, "repeat_i16", repeat_i16_len, repeat_i16_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);

#నిర్వచించు CREATE_UNARY(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_దత్తాంశం, "main", 2, పరిమాణం(vk_op_unary_నెట్టు_constants), {512, 1, 1}, {}, 1);

    CREATE_UNARY(elu)
    CREATE_UNARY(gelu)
    CREATE_UNARY(gelu_erf)
    CREATE_UNARY(gelu_శీఘ్ర)
    CREATE_UNARY(silu)
    CREATE_UNARY(relu)
    CREATE_UNARY(sqr)
    CREATE_UNARY(వర్గమూలం)
    CREATE_UNARY(జ్యా)
    CREATE_UNARY(కోజ్యా)
    CREATE_UNARY(clamp)
    CREATE_UNARY(leaky_relu)
    CREATE_UNARY(xielu)
    CREATE_UNARY(neg)
    CREATE_UNARY(tanh)
    CREATE_UNARY(sigmoid)
    CREATE_UNARY(hardsigmoid)
    CREATE_UNARY(hardswish)
    CREATE_UNARY(నిరపేక్ష)
    CREATE_UNARY(softplus)
    CREATE_UNARY(అడుగు)
    CREATE_UNARY(round)
    CREATE_UNARY(ceil)
    CREATE_UNARY(floor)
    CREATE_UNARY(trunc)
    CREATE_UNARY(sgn)
    CREATE_UNARY(ఘాతాంకం)
    CREATE_UNARY(expm1)
#నిర్వచనం_రద్దు CREATE_UNARY

// spec constants: {norepeat, op_on_b}
#నిర్వచించు CREATE_UNARY_MUL(name, idx) \
    కొరకు (పూర్ణాంకం dt = 0; dt < 2; ++dt) { \
        స్థిరరాశి size_t len_ = dt ? name ## _mul_f16_len : name ## _mul_f32_len; \
        స్థిరరాశి గుర్తులేని అక్షరం * దత్తాంశం_3 = dt ? name ## mul_f16_దత్తాంశం : name ## mul_f32_దత్తాంశం; \
        స్థిరరాశి std::string dts_ = dt ? "f16" : "f32"; \
        కొరకు (పూర్ణాంకం ob = 0; ob < 2; ++ob) \
            కొరకు (పూర్ణాంకం nr = 0; nr < 2; ++nr) \
                ggml_vk_create_pipeline(device, device->pipeline_unary_mul[(idx)][dt][nr][ob], \
                    (#name "_mul" + std::string(ob ? "_b" : "") + "_" + dts_ + (nr ? "_norepeat" : "")).c_str(), \
                    len_, దత్తాంశం_3, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {512, 1, 1}, \
                    { (uint32_t) nr, (uint32_t) ob }, 1); \
    }

    CREATE_UNARY_MUL(gelu, 0)
    CREATE_UNARY_MUL(sigmoid, 1)
    CREATE_UNARY_MUL(silu, 2)
    CREATE_UNARY_MUL(softplus, 3)
#నిర్వచనం_రద్దు CREATE_UNARY_MUL

    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f16, "add1_f16_f16", add1_f16_f16_len, add1_f16_f16_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f32, "add1_f16_f32", add1_f16_f32_len, add1_f16_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f32_f32, "add1_f32_f32", add1_f32_f32_len, add1_f32_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_arange_f32, "arange_f32", arange_f32_len, arange_f32_దత్తాంశం, "main", 1, పరిమాణం(vk_op_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_fill_f32, "fill_f32", fill_f32_len, fill_f32_దత్తాంశం, "main", 1, పరిమాణం(vk_op_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_fill_f16, "fill_f16", fill_f16_len, fill_f16_దత్తాంశం, "main", 1, పరిమాణం(vk_op_నెట్టు_constants), {512, 1, 1}, {}, 1);

#నిర్వచించు CREATE_GLU(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_glu_నెట్టు_constants), {512, 1, 1}, {}, 1, నిజం);   \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_దత్తాంశం, "main", 3, పరిమాణం(vk_op_glu_నెట్టు_constants), {512, 1, 1}, {}, 1, నిజం);

    CREATE_GLU(geglu)
    CREATE_GLU(reglu)
    CREATE_GLU(swiglu)
    CREATE_GLU(swiglu_oai)
    CREATE_GLU(swiglu_clamp)
    CREATE_GLU(geglu_erf)
    CREATE_GLU(geglu_శీఘ్ర)
#నిర్వచనం_రద్దు CREATE_GLU

    ggml_vk_create_pipeline(device, device->pipeline_silu_వెనుక_f32, "silu_back_f32", silu_వెనుక_f32_len, silu_వెనుక_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag_ముసుగు_inf_f32, "diag_mask_inf_f32", diag_ముసుగు_inf_f32_len, diag_ముసుగు_inf_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_diag_ముసుగు_నెట్టు_constants), {1, 512, 1}, {}, 1, నిజం);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32, "soft_max_f32", soft_max_f32_len, soft_max_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_soft_max_నెట్టు_constants), {1, 1, 1}, { device->subgroup_కొలత }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_wg512, "soft_max_f32_wg512", soft_max_f32_len, soft_max_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_soft_max_నెట్టు_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16, "soft_max_f32_f16", soft_max_f32_f16_len, soft_max_f32_f16_దత్తాంశం, "main", 4, పరిమాణం(vk_op_soft_max_నెట్టు_constants), {1, 1, 1}, { device->subgroup_కొలత }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16_wg512, "soft_max_f32_f16_wg512", soft_max_f32_f16_len, soft_max_f32_f16_దత్తాంశం, "main", 4, పరిమాణం(vk_op_soft_max_నెట్టు_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_వెనుక_f32, "soft_max_back_f32", soft_max_వెనుక_f32_len, soft_max_వెనుక_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_నెట్టు_constants), {1, 1, 1}, { device->subgroup_కొలత }, 1, నిజం);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32,     "soft_max_large1_f32",     soft_max_large1_f32_len,     soft_max_large1_f32_దత్తాంశం,     "main", 6, పరిమాణం(vk_op_soft_max_నెట్టు_constants), {1, 1, 1}, { 128, 4 }, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32,     "soft_max_large2_f32",     soft_max_large2_f32_len,     soft_max_large2_f32_దత్తాంశం,     "main", 6, పరిమాణం(vk_op_soft_max_నెట్టు_constants), {1, 1, 1}, { 128, 4 }, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32,     "soft_max_large3_f32",     soft_max_large3_f32_len,     soft_max_large3_f32_దత్తాంశం,     "main", 6, పరిమాణం(vk_op_soft_max_నెట్టు_constants), {1, 1, 1}, { 128, 4 }, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32_f16, "soft_max_large1_f32_f16", soft_max_large1_f32_f16_len, soft_max_large1_f32_f16_దత్తాంశం, "main", 6, పరిమాణం(vk_op_soft_max_నెట్టు_constants), {1, 1, 1}, { 128, 4 }, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32_f16, "soft_max_large2_f32_f16", soft_max_large2_f32_f16_len, soft_max_large2_f32_f16_దత్తాంశం, "main", 6, పరిమాణం(vk_op_soft_max_నెట్టు_constants), {1, 1, 1}, { 128, 4 }, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32_f16, "soft_max_large3_f32_f16", soft_max_large3_f32_f16_len, soft_max_large3_f32_f16_దత్తాంశం, "main", 6, పరిమాణం(vk_op_soft_max_నెట్టు_constants), {1, 1, 1}, { 128, 4 }, 1, నిజం);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32, "rope_norm_f32", rope_norm_f32_len, rope_norm_f32_దత్తాంశం, "main", 5, పరిమాణం(vk_op_rope_నెట్టు_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32, "rope_neox_f32", rope_neox_f32_len, rope_neox_f32_దత్తాంశం, "main", 5, పరిమాణం(vk_op_rope_నెట్టు_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32, "rope_multi_f32", rope_multi_f32_len, rope_multi_f32_దత్తాంశం, "main", 5, పరిమాణం(vk_op_rope_నెట్టు_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f32, "rope_vision_f32", rope_vision_f32_len, rope_vision_f32_దత్తాంశం, "main", 5, పరిమాణం(vk_op_rope_నెట్టు_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_len, rope_norm_f16_దత్తాంశం, "main", 5, పరిమాణం(vk_op_rope_నెట్టు_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_len, rope_neox_f16_దత్తాంశం, "main", 5, పరిమాణం(vk_op_rope_నెట్టు_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f16, "rope_multi_f16", rope_multi_f16_len, rope_multi_f16_దత్తాంశం, "main", 5, పరిమాణం(vk_op_rope_నెట్టు_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f16, "rope_vision_f16", rope_vision_f16_len, rope_vision_f16_దత్తాంశం, "main", 5, పరిమాణం(vk_op_rope_నెట్టు_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32_f16, "rope_norm_f32_f16", rope_norm_f32_f16_len, rope_norm_f32_f16_దత్తాంశం, "main", 5, పరిమాణం(vk_op_rope_నెట్టు_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32_f16, "rope_neox_f32_f16", rope_neox_f32_f16_len, rope_neox_f32_f16_దత్తాంశం, "main", 5, పరిమాణం(vk_op_rope_నెట్టు_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32_f16, "rope_multi_f32_f16", rope_multi_f32_f16_len, rope_multi_f32_f16_దత్తాంశం, "main", 5, పరిమాణం(vk_op_rope_నెట్టు_constants), {1, 512, 1}, {}, 1);

    కొరకు (uint32_t i = 0; i < అంకె_argsort_pipelines; ++i) {
        uint32_t ఖండం_కొలత_2 = 1u << std::min(i, device->max_workgroup_కొలత_log2);
        ఒకవేళ (i <= device->max_workgroup_కొలత_log2 &&
            2 * పరిమాణం(పూర్ణాంకం) * ఖండం_కొలత_2 <= device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత) {
            స్థిరరాశి uint32_t NCOLS_PADDED_LOG2 = i;
            ggml_vk_create_pipeline2(device, device->pipeline_argsort_f32[i], "argsort_f32_"+std::to_string(i), argsort_f32_len, argsort_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_argsort_నెట్టు_constants), {ఖండం_కొలత_2, 1, 1}, {ఖండం_కొలత_2, NCOLS_PADDED_LOG2}, 1, నిజం);
        }
        స్థిరరాశి uint32_t WG_UNROLL_గుణకం = ఖండం_కొలత_2 > 1 ? 2 : 1;
        ఖండం_కొలత_2 /= WG_UNROLL_గుణకం;
        ggml_vk_create_pipeline2(device, device->pipeline_argsort_large_f32[i], "argsort_large_f32_"+std::to_string(i), argsort_large_f32_len, argsort_large_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_argsort_నెట్టు_constants), {ఖండం_కొలత_2 * WG_UNROLL_గుణకం, 1, 1}, {ఖండం_కొలత_2, WG_UNROLL_గుణకం}, 1, నిజం);
    }

    కొరకు (uint32_t i = 0; i < అంకె_topk_pipelines; ++i) {
        స్థిరరాశి uint32_t ఖండం_కొలత_2 = 1u << i;
        స్థిరరాశి uint32_t NCOLS_PADDED_LOG2 = i;
        ఒకవేళ (i <= device->max_workgroup_కొలత_log2) {
            uint32_t nary_shmem = 2 * పరిమాణం(పూర్ణాంకం) * ఖండం_కొలత_2 +
                                  పరిమాణం(పూర్ణాంకం) * device->subgroup_కొలత +
                                  2 * పరిమాణం(పూర్ణాంకం) +
                                  2 * (ఖండం_కొలత_2 / device->subgroup_కొలత) * పరిమాణం(పూర్ణాంకం);
            ఒకవేళ (device->subgroup_arithmetic && device->subgroup_require_నిండిన_support && device->subgroup_shuffle && device->subgroup_ballot &&
                nary_shmem <= device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_nary_శోధన_f32_len, topk_nary_శోధన_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_topk_నెట్టు_constants), {ఖండం_కొలత_2, 1, 1}, {ఖండం_కొలత_2, device->subgroup_కొలత, device->subgroup_కొలత_log2}, 1, నిజం, నిజం, device->subgroup_కొలత);
            } లేకపోతే ఒకవేళ (2 * పరిమాణం(పూర్ణాంకం) * ఖండం_కొలత_2 <= device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_argsort_f32_len, topk_argsort_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_topk_నెట్టు_constants), {ఖండం_కొలత_2, 1, 1}, {ఖండం_కొలత_2, NCOLS_PADDED_LOG2}, 1, నిజం);
            }
        }
    }

    // large-k fallback: one workgroup per row, radix-select instead of a full sort. The QSA
    // variant (spec constant 1) additionally gathers the qwen4 indexer input on the fly.
    {
        స్థిరరాశి uint32_t ఖండం_కొలత_2 = 1u << std::min(10u, device->max_workgroup_కొలత_log2);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_f32, "topk_radix_f32", topk_radix_select_f32_len, topk_radix_select_f32_దత్తాంశం, "main", 5, పరిమాణం(vk_op_topk_radix_నెట్టు_constants), {ఖండం_కొలత_2, 1, 1}, {ఖండం_కొలత_2, 0}, 1, నిజం);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_qsa, "topk_radix_qsa", topk_radix_select_f32_len, topk_radix_select_f32_దత్తాంశం, "main", 5, పరిమాణం(vk_op_topk_radix_నెట్టు_constants), {ఖండం_కొలత_2, 1, 1}, {ఖండం_కొలత_2, 1}, 1, నిజం);
    }

    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, "argmax_f32", argmax_f32_len, argmax_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_నెట్టు_constants), {1, 1, 1}, { device->subgroup_కొలత }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_కూడిక_అడ్డువరుసలు_f32, "sum_rows_f32", కూడిక_అడ్డువరుసలు_f32_len, కూడిక_అడ్డువరుసలు_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants), {1, 1, 1}, { device->subgroup_కొలత }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32, "cross_entropy_loss_f32", cross_entropy_loss_f32_len, cross_entropy_loss_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_నెట్టు_constants), {1, 1, 1}, { device->subgroup_కొలత }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32_wg512, "cross_entropy_loss_f32_wg512", cross_entropy_loss_f32_len, cross_entropy_loss_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_నెట్టు_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_వెనుక_f32, "cross_entropy_loss_back_f32", cross_entropy_loss_వెనుక_f32_len, cross_entropy_loss_వెనుక_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_నెట్టు_constants), {1, 1, 1}, { device->subgroup_కొలత }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_వెనుక_f32_wg512, "cross_entropy_loss_back_f32_wg512", cross_entropy_loss_వెనుక_f32_len, cross_entropy_loss_వెనుక_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_నెట్టు_constants), {1, 1, 1}, { 512 }, 1);
    // Intel Windows driver in range [32.0.101.8509, 32.0.101.8860) will crash when using fwht kernels so we gate that here
    స్థిరరాశి తార్కిక can_use_fwht = device->driver_id != vk::DriverId::eIntelProprietaryWindows ||
        !ggml_vk_intel_windows_driver_లోపల_range(device->properties.driver_సంస్కరణ_2, 101, 8509, 101, 8860);
    ఒకవేళ (can_use_fwht && device->subgroup_basic && device->subgroup_shuffle) {
        పూర్ణాంకం idx = 0;
        కొరకు (uint32_t n : {64, 128, 256, 512}) {
            ఒకవేళ (device->subgroup_కొలత <= n) {
                ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_fwht_నెట్టు_constants), {1, 1, 1}, { device->subgroup_కొలత, n }, 1, నిజం, నిజం, device->subgroup_కొలత);
            }
            ++idx;
        }
    } లేకపోతే ఒకవేళ (can_use_fwht) {
        పూర్ణాంకం idx = 0;
        కొరకు (uint32_t n : {64, 128, 256, 512}) {
            స్థిరరాశి uint32_t ఖండం_కొలత_3 = std::min(device->subgroup_కొలత, n);
            ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_fwht_నెట్టు_constants), {1, 1, 1}, { ఖండం_కొలత_3, n }, 1);
            ++idx;
        }
    }

    స్థిరరాశి uint32_t cumsum_elem_చొప్పున_thread = (device->vendor_id == VK_VENDOR_ID_AMD || device->vendor_id == VK_VENDOR_ID_INTEL) ? 2 : 4;
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_f32,       "cumsum_f32", cumsum_f32_len, cumsum_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants), {1, 1, 1}, { 256, device->subgroup_కొలత, cumsum_elem_చొప్పున_thread }, 1, నిజం, నిజం, device->subgroup_కొలత);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_small_f32, "cumsum_f32", cumsum_f32_len, cumsum_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants), {1, 1, 1}, { 128, device->subgroup_కొలత, 1 }, 1, నిజం, నిజం, device->subgroup_కొలత);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass1_f32, "cumsum_multipass1_f32", cumsum_multipass1_f32_len, cumsum_multipass1_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants), {256, 1, 1}, { 256, device->subgroup_కొలత }, 1, నిజం, నిజం, device->subgroup_కొలత);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass2_f32, "cumsum_multipass2_f32", cumsum_multipass2_f32_len, cumsum_multipass2_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants), {256, 1, 1}, { 256, device->subgroup_కొలత }, 1, నిజం, నిజం, device->subgroup_కొలత);

    ggml_vk_create_pipeline(device, device->pipeline_లెక్క_equal_i32, "count_equal_i32", లెక్క_equal_i32_len, లెక్క_equal_i32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_నెట్టు_constants), {512, 1, 1}, { device->subgroup_కొలత }, 1);

    ఒకవేళ (device->subgroup_arithmetic && device->subgroup_vote && device->subgroup_require_నిండిన_support) {
        ggml_vk_create_pipeline(device, device->pipeline_లెక్క_experts, "count_experts", లెక్క_experts_subgroup_len, లెక్క_experts_subgroup_దత్తాంశం, "main", 2, పరిమాణం(vk_op_లెక్క_experts_నెట్టు_constants), {1, 1, 1}, {}, 1, నిజం, నిజం);
    } లేకపోతే {
        ggml_vk_create_pipeline(device, device->pipeline_లెక్క_experts, "count_experts", లెక్క_experts_len, లెక్క_experts_దత్తాంశం, "main", 2, పరిమాణం(vk_op_లెక్క_experts_నెట్టు_constants), {1, 1, 1}, {}, 1, నిజం);
    }

    // comb holds a token's 4x4 matrix in one 16-lane slice of a subgroup, so it
    // needs at least 16 lanes, pinned to a known size.
    ఒకవేళ (device->subgroup_basic && device->subgroup_shuffle && device->subgroup_require_నిండిన_support && device->subgroup_కొలత >= 16) {
        స్థిరరాశి uint32_t టోకెన్లు_చొప్పున_workgroup = 4 * (device->subgroup_కొలత / 16);
        ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_comb_f32, "dsv4_hc_comb_f32", dsv4_hc_comb_f32_len, dsv4_hc_comb_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_dsv4_hc_comb_నెట్టు_constants), {టోకెన్లు_చొప్పున_workgroup, 1, 1}, { device->subgroup_కొలత }, 1, నిజం, నిజం, device->subgroup_కొలత);
    }

    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_f32,        "dsv4_hc_pre_f32",        dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_దత్తాంశం,  "main", 3, పరిమాణం(vk_op_dsv4_hc_pre_నెట్టు_constants),  {256, 1, 1}, { 256, 0 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_gated_f32,  "dsv4_hc_pre_gated_f32",  dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_దత్తాంశం,  "main", 3, పరిమాణం(vk_op_dsv4_hc_pre_నెట్టు_constants),  {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_f32,       "dsv4_hc_post_f32",       dsv4_hc_post_f32_len, dsv4_hc_post_f32_దత్తాంశం, "main", 5, పరిమాణం(vk_op_dsv4_hc_post_నెట్టు_constants), {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_nocomb_f32,"dsv4_hc_post_nocomb_f32",dsv4_hc_post_f32_len, dsv4_hc_post_f32_దత్తాంశం, "main", 5, పరిమాణం(vk_op_dsv4_hc_post_నెట్టు_constants), {256, 1, 1}, { 256, 0 }, 1);

    కొరకు (స్వయంచాలక &s : device->pipeline_solve_tri_f32) {
        స్థిరరాశి vk_solve_tri_pipeline_state &state = s.మొదటి;

        // Max number of rows to load at a time, limited by shared memory
        స్థిరరాశి uint32_t batch_N = device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత / ((state.N + state.K) * పరిమాణం(దశాంశం));
        // Need at least K invocations, and prefer a minimum of 128 to spread out loading shared memory
        స్థిరరాశి uint32_t ఖండం_కొలత_3 = std::max(128u, 1u << (uint32_t)ceilf(log2f(దశాంశం(state.K))));

        ggml_vk_create_pipeline(
            device, s.second, "solve_tri_f32",
            solve_tri_f32_len, solve_tri_f32_దత్తాంశం, "main", 3,
            పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {1, 1, 1}, { 0, state.N, state.K, batch_N, ఖండం_కొలత_3 }, 1, నిజం);
    }

#నిర్వచించు IM2_నిలువు(bda) \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32 ## bda ## _len, im2col_f32 ## bda ## దత్తాంశం_2, "main", 2, పరిమాణం(vk_op_im2col_నెట్టు_constants), {512, 1, 1}, { device->subgroup_కొలత }, 1, నిజం);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32, "im2col_3d_f32", im2col_3d_f32 ## bda ## _len, im2col_3d_f32 ## bda ## దత్తాంశం_2, "main", 2, పరిమాణం(vk_op_im2col_3d_నెట్టు_constants), {512, 1, 1}, { 512 }, 1, నిజం);      \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16 ## bda ## _len, im2col_f32_f16 ## bda ## దత్తాంశం_2, "main", 2, పరిమాణం(vk_op_im2col_నెట్టు_constants), {512, 1, 1}, { device->subgroup_కొలత }, 1, నిజం);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", im2col_3d_f32_f16 ## bda ## _len, im2col_3d_f32_f16 ## bda ## దత్తాంశం_2, "main", 2, పరిమాణం(vk_op_im2col_3d_నెట్టు_constants), {512, 1, 1}, { 512 }, 1, నిజం);
    ఒకవేళ (device->shader_int64 && device->బఫరు_device_చిరునామా) {
        IM2_నిలువు(_bda)
    } లేకపోతే {
        IM2_నిలువు()
    }

    ggml_vk_create_pipeline(device, device->pipeline_timestep_embedding_f32, "timestep_embedding_f32", timestep_embedding_f32_len, timestep_embedding_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_timestep_embedding_నెట్టు_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_conv_transpose_1d_f32, "conv_transpose_1d_f32", conv_transpose_1d_f32_len, conv_transpose_1d_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_conv_transpose_1d_నెట్టు_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f32,  "col2im_1d_f32",  col2im_1d_f32_len,  col2im_1d_f32_దత్తాంశం,  "main", 2, పరిమాణం(vk_op_col2im_1d_నెట్టు_constants), {256, 1, 1}, {}, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f16,  "col2im_1d_f16",  col2im_1d_f16_len,  col2im_1d_f16_దత్తాంశం,  "main", 2, పరిమాణం(vk_op_col2im_1d_నెట్టు_constants), {256, 1, 1}, {}, 1, నిజం);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_bf16, "col2im_1d_bf16", col2im_1d_bf16_len, col2im_1d_bf16_దత్తాంశం, "main", 2, పరిమాణం(vk_op_col2im_1d_నెట్టు_constants), {256, 1, 1}, {}, 1, నిజం);

    ggml_vk_create_pipeline(device, device->pipeline_బహిర్గతం_prod_f32, "out_prod_f32", బహిర్గతం_prod_f32_len, బహిర్గతం_prod_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_ద్విమాన_నెట్టు_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_snake_f32,  "snake_f32",  snake_f32_len,  snake_f32_దత్తాంశం,  "main", 4, పరిమాణం(vk_op_snake_నెట్టు_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_f16,  "snake_f16",  snake_f16_len,  snake_f16_దత్తాంశం,  "main", 4, పరిమాణం(vk_op_snake_నెట్టు_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_bf16, "snake_bf16", snake_bf16_len, snake_bf16_దత్తాంశం, "main", 4, పరిమాణం(vk_op_snake_నెట్టు_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pool1d_f32, "pool1d_f32", pool1d_f32_len, pool1d_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_pool1d_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pool2d_f32, "pool2d_f32", pool2d_f32_len, pool2d_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_pool2d_నెట్టు_constants), {1, 1, 1}, {512}, 1);
    ఒకవేళ (device->subgroup_arithmetic && device->subgroup_require_నిండిన_support) {
        ggml_vk_create_pipeline(device, device->pipeline_pool2d_subgroup_f32, "pool2d_subgroup_f32", pool2d_subgroup_f32_len, pool2d_subgroup_f32_దత్తాంశం, "main", 2, పరిమాణం(vk_op_pool2d_నెట్టు_constants), {1, 1, 1}, {128}, 1, అబద్ధం, నిజం, device->subgroup_కొలత);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv6_f32, "rwkv_wkv6_f32", rwkv_wkv6_f32_len, rwkv_wkv6_f32_దత్తాంశం, "main", 7, పరిమాణం(vk_op_rwkv_wkv6_నెట్టు_constants), {1, 1, 1}, {device->subgroup_కొలత}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv7_f32, "rwkv_wkv7_f32", rwkv_wkv7_f32_len, rwkv_wkv7_f32_దత్తాంశం, "main", 8, పరిమాణం(vk_op_rwkv_wkv7_నెట్టు_constants), {1, 1, 1}, {device->subgroup_కొలత}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_gated_రేఖీయ_attn_f32, "gated_linear_attn_f32", gated_రేఖీయ_attn_f32_len, gated_రేఖీయ_attn_f32_దత్తాంశం, "main", 6, పరిమాణం(vk_op_gated_రేఖీయ_attn_నెట్టు_constants), {1, 1, 1}, {}, 1);

    కొరకు (ggml_type k_type : lightning_indexer_k_types) {
        స్థిరరాశి std::string name = "lightning_indexer_" + std::string(ggml_type_name(k_type)) + "_k_f32";
        ggml_vk_create_pipeline(device, device->pipeline_lightning_indexer_f32[k_type], name.c_str(), lightning_indexer_f32_len, lightning_indexer_f32_దత్తాంశం, "main", 5, పరిమాణం(vk_op_lightning_indexer_నెట్టు_constants), {1, 1, 1}, {(uint32_t)k_type, fa_ఖండం_బైట్లు(k_type)}, 1, నిజం);
    }

    {
        స్థిరరాశి uint32_t gdn_sizes[] = {16, 32, 64, 128};
        స్థిరరాశి అక్షరం * gdn_పేర్లు[][2] = {
            {"gated_delta_net_f32_d16",     "gated_delta_net_f32_d16_kda"},
            {"gated_delta_net_f32_d32",     "gated_delta_net_f32_d32_kda"},
            {"gated_delta_net_f32_d64",     "gated_delta_net_f32_d64_kda"},
            {"gated_delta_net_f32_d128",    "gated_delta_net_f32_d128_kda"},
        };
        కొరకు (uint32_t si = 0; si < 4; si++) {
            స్థిరరాశి uint32_t S_V = gdn_sizes[si];
            GGML_ASSERT(is_pow2(S_V));

            // Intel Xe regresses at SIMD32 for this scan; prefer a narrower subgroup.
            uint32_t gdn_subgroup_కొలత = device->subgroup_కొలత;
            ఒకవేళ (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_కొలత_control &&
                device->subgroup_min_కొలత <= 16u && device->subgroup_max_కొలత >= 16u) {
                gdn_subgroup_కొలత = 16u;
            }
            uint32_t lanes_చొప్పున_స్తంభం;
            ఒకవేళ (device->vendor_id == VK_VENDOR_ID_INTEL) {
                // Intel Xe: full-width reduction (min rows/lane) ~10x over the COLS_PER_WG=8 rule.
                lanes_చొప్పున_స్తంభం = std::min(gdn_subgroup_కొలత, S_V);
            } లేకపోతే ఒకవేళ (S_V >= 128u && device->subgroup_clustered) {
                // COLS_PER_WG=8: measured optimum (Ampere sg32->4, Vega20 sg64->8).
                lanes_చొప్పున_స్తంభం = std::max(1u, gdn_subgroup_కొలత / 8u);
            } లేకపోతే {
                // Use largest power-of-two that divides both S_V and subgroup_size so that
                // (1) S_V % lanes_per_column == 0 and (2) S_V % (subgroup_size / lanes_per_column) == 0.
                // This means we don't need extra bounds checking logic in the shader.
                lanes_చొప్పున_స్తంభం = std::min(S_V, gdn_subgroup_కొలత);
            }

            // gated_delta_net.comp relies on S_V % COLS_PER_WG == 0 and
            // S_V % LANES_PER_COLUMN == 0 to avoid bounds checks.
            అయితే (lanes_చొప్పున_స్తంభం > 1u) {
                స్థిరరాశి తార్కిక valid_lanes = (gdn_subgroup_కొలత % lanes_చొప్పున_స్తంభం) == 0 &&
                                         (S_V % lanes_చొప్పున_స్తంభం) == 0;
                స్థిరరాశి uint32_t నిలువుల_సంఖ్య_చొప్పున_wg = valid_lanes ? gdn_subgroup_కొలత / lanes_చొప్పున_స్తంభం : 0;
                ఒకవేళ (valid_lanes && నిలువుల_సంఖ్య_చొప్పున_wg > 0 && (S_V % నిలువుల_సంఖ్య_చొప్పున_wg) == 0) {
                    ఆపు;
                }
                lanes_చొప్పున_స్తంభం >>= 1u;
            }

            GGML_ASSERT((gdn_subgroup_కొలత % lanes_చొప్పున_స్తంభం) == 0);
            GGML_ASSERT((S_V % lanes_చొప్పున_స్తంభం) == 0);
            GGML_ASSERT((S_V % (gdn_subgroup_కొలత / lanes_చొప్పున_స్తంభం)) == 0);

            స్థిరరాశి తార్కిక need_partial_subgroup_reduce = lanes_చొప్పున_స్తంభం != 1u && lanes_చొప్పున_స్తంభం < gdn_subgroup_కొలత;
            స్థిరరాశి తార్కిక use_clustered_reduce = device->subgroup_arithmetic && device->subgroup_clustered && need_partial_subgroup_reduce;
            స్థిరరాశి తార్కిక use_subgroup_reduce = device->subgroup_arithmetic && !need_partial_subgroup_reduce;
            స్థిరరాశి తార్కిక use_subgroup_ops = use_clustered_reduce || use_subgroup_reduce;
            size_t gdn_len;
            స్థిరరాశి శూన్య * gdn_దత్తాంశం;
            ఒకవేళ (use_clustered_reduce) {
                gdn_len = gated_delta_net_f32_len;
                gdn_దత్తాంశం = (స్థిరరాశి శూన్య *)gated_delta_net_f32_దత్తాంశం;
            } లేకపోతే ఒకవేళ (use_subgroup_reduce) {
                gdn_len = gated_delta_net_f32_nocluster_len;
                gdn_దత్తాంశం = (స్థిరరాశి శూన్య *)gated_delta_net_f32_nocluster_దత్తాంశం;
            } లేకపోతే {
                gdn_len = gated_delta_net_f32_shmem_len;
                gdn_దత్తాంశం = (స్థిరరాశి శూన్య *)gated_delta_net_f32_shmem_దత్తాంశం;
            }

            స్థిరరాశి uint32_t నిలువుల_సంఖ్య_చొప్పున_wg = gdn_subgroup_కొలత / lanes_చొప్పున_స్తంభం;
            స్థిరరాశి std::array<uint32_t, 3> wg_denoms = {1u, 1u, నిలువుల_సంఖ్య_చొప్పున_wg};

            కొరకు (uint32_t kda = 0; kda < 2; kda++) {
                ggml_vk_create_pipeline(device, device->pipeline_gated_delta_net[si][kda],
                    gdn_పేర్లు[si][kda], gdn_len, gdn_దత్తాంశం, "main", 7, పరిమాణం(vk_op_gated_delta_net_నెట్టు_constants),
                    wg_denoms, {S_V, kda, gdn_subgroup_కొలత, lanes_చొప్పున_స్తంభం}, 1, నిజం, use_subgroup_ops, gdn_subgroup_కొలత);
            }
        }
    }

    ఒకవేళ (device->subgroup_arithmetic && device->subgroup_require_నిండిన_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_పరిశీలన_f32_d128, "ssm_scan_128_f32", ssm_పరిశీలన_subgroup_f32_len, ssm_పరిశీలన_subgroup_f32_దత్తాంశం, "main", 8, పరిమాణం(vk_op_ssm_పరిశీలన_నెట్టు_constants), {1, 1, 1}, {128, device->subgroup_కొలత}, 1, నిజం, నిజం);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_పరిశీలన_f32_d256, "ssm_scan_256_f32", ssm_పరిశీలన_subgroup_f32_len, ssm_పరిశీలన_subgroup_f32_దత్తాంశం, "main", 8, పరిమాణం(vk_op_ssm_పరిశీలన_నెట్టు_constants), {1, 1, 1}, {256, device->subgroup_కొలత}, 1, నిజం, నిజం);
    } లేకపోతే {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_పరిశీలన_f32_d128, "ssm_scan_128_f32", ssm_పరిశీలన_f32_len, ssm_పరిశీలన_f32_దత్తాంశం, "main", 8, పరిమాణం(vk_op_ssm_పరిశీలన_నెట్టు_constants), {1, 1, 1}, {128, device->subgroup_కొలత, 16}, 1, నిజం, నిజం);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_పరిశీలన_f32_d256, "ssm_scan_256_f32", ssm_పరిశీలన_f32_len, ssm_పరిశీలన_f32_దత్తాంశం, "main", 8, పరిమాణం(vk_op_ssm_పరిశీలన_నెట్టు_constants), {1, 1, 1}, {256, device->subgroup_కొలత, 16}, 1, నిజం, నిజం);
    }

    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_f32,           "ssm_conv_f32",           ssm_conv_f32_len, ssm_conv_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_ssm_conv_నెట్టు_constants), {32, 16, 1}, {32, 16, 0, 0}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_silu_f32,      "ssm_conv_silu_f32",      ssm_conv_f32_len, ssm_conv_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_ssm_conv_నెట్టు_constants), {32, 16, 1}, {32, 16, 0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_bias_silu_f32, "ssm_conv_bias_silu_f32", ssm_conv_f32_len, ssm_conv_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_ssm_conv_నెట్టు_constants), {32, 16, 1}, {32, 16, 1, 1}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_అడుగు_adamw_f32, "opt_step_adamw_f32", opt_అడుగు_adamw_f32_len, opt_అడుగు_adamw_f32_దత్తాంశం, "main", 5, పరిమాణం(vk_op_నెట్టు_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_అడుగు_sgd_f32, "opt_step_sgd_f32", opt_అడుగు_sgd_f32_len, opt_అడుగు_sgd_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_నెట్టు_constants), {512, 1, 1}, {}, 1);

    // conv2d, conv_transpose_2d, conv3d
    కొరకు (uint32_t s = 0; s < CONV_SHAPE_లెక్క; ++s) {
        // smaller WG for the small-tile fallback gives more concurrent WGs per SM
        uint32_t conv2d_WG_కొలత  = (s == CONV_SHAPE_64x32) ? 128 : 256;
        uint32_t use_collectives = 0;  // Enables subgroup ops for preventing the re-calculation of indices.
        uint32_t conv2d_TS_K     = (s == CONV_SHAPE_64x32) ? 4 : 8;
        uint32_t conv2d_SHMEM_PAD = 4;
        vk_conv_ఖండం_కొలత conv2d_BS = vk_conv_ఖండం_sizes[s];
        తార్కిక conv2d_UNROLL = నిజం;

#ఒకవేళ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        ఒకవేళ (device->coopmat2) {
            conv2d_SHMEM_PAD = 8; // 8 float16_t
        }
#అయితే_ముగింపు

        ఒకవేళ (device->vendor_id == VK_VENDOR_ID_INTEL) {
            conv2d_SHMEM_PAD = 0;
            conv2d_UNROLL = అబద్ధం;
        } లేకపోతే ఒకవేళ (device->vendor_id == VK_VENDOR_ID_AMD) {
            conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
            ఒకవేళ (s == CONV_SHAPE_128x128 && device->architecture != vk_device_architecture::AMD_GCN) {
                conv2d_UNROLL = అబద్ధం;
            }
        }

        // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
        తార్కిక allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
                                    device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        తార్కిక allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
                                     device->architecture == vk_device_architecture::AMD_GCN;

        ఒకవేళ (device->subgroup_shuffle &&
            device->vendor_id != VK_VENDOR_ID_INTEL &&   // Do not enable collectives on Intel, see PR 14316.
            allow_collectives_nv &&
            allow_collectives_amd) {
            use_collectives = 1;
            conv2d_BS.CRS   = std::min(
                device->subgroup_కొలత,
                conv2d_BS.CRS);  // CRS block size should be capped at subgroup size for correctness when shuffle is used.
        }

        // cm1 is used only when cm2 is unavailable; capped at 64x128 (due to shared memory size).
        // Requires 16x16x16 f16-acc since that's the fragment shape hard-coded in the shader.
        // Subgroup size must be 32 or 64 (to keep WG_SIZE sane) and we need
        // subgroup_size_control to force the driver to actually use it.
        తార్కిక conv2d_use_cm1 = అబద్ధం;
#ఒకవేళ defined(VK_KHR_cooperative_మాతృక) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        conv2d_use_cm1 = !device->coopmat2 &&
                         device->coopmat_support && device->coopmat_support_16x16x16_f16acc &&
                         device->subgroup_కొలత_control &&
                         (device->subgroup_కొలత == 32 || device->subgroup_కొలత == 64) &&
                         s != CONV_SHAPE_128x128;
#అయితే_ముగింపు

        స్థిరరాశి uint32_t conv2d_cm1_shmem_pad = 8;

        స్వయంచాలక shmem_req = [&](uint32_t pad, తార్కిక csh_store, తార్కిక fp16_shmem) {
            స్థిరరాశి uint32_t elem_కొలత = fp16_shmem ? (uint32_t)పరిమాణం(uint16_t) : (uint32_t)పరిమాణం(దశాంశం);
            స్థిరరాశి uint32_t csh_elems = csh_store ? conv2d_BS.K * conv2d_BS.NPQ : 0u;
            తిరిగి (conv2d_BS.K * (conv2d_BS.CRS + pad) + conv2d_BS.CRS * (conv2d_BS.NPQ + pad) + csh_elems) * elem_కొలత;
        };

        // 2D, transpose-2D, and 3D conv use the same KxCRS @ CRSxNPQ shmem
        // layout. cm1 needs Csh for output, so check before applying cm1 params.
        ఒకవేళ (conv2d_use_cm1 && device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత < shmem_req(conv2d_cm1_shmem_pad, నిజం, నిజం)) {
            conv2d_use_cm1 = అబద్ధం;
        }

        uint32_t conv2d_WM = 16, conv2d_WN = 16;  // cm1 subgroup tile, ignored otherwise
        ఒకవేళ (conv2d_use_cm1) {
            conv2d_SHMEM_PAD = conv2d_cm1_shmem_pad;
            // 16x16x16 fragments; pick WM/WN to keep WG_SIZE at 256
            // (i.e. 8 subgroups for sg=32, 4 subgroups for sg=64).
            స్థిరరాశి తార్కిక sg64 = (device->subgroup_కొలత == 64);
            ఎంపిక (s) {
                స్థితి CONV_SHAPE_64x32:   conv2d_WM = sg64 ? 32 : 16; conv2d_WN = 16; ఆపు;
                స్థితి CONV_SHAPE_64x128:  conv2d_WM = 32; conv2d_WN = sg64 ? 64 : 32; ఆపు;
                స్థితి CONV_SHAPE_32x256:  conv2d_WM = sg64 ? 16 : 32; conv2d_WN = sg64 ? 128 : 32; ఆపు;
                అప్రమేయ: ఆపు;
            }
            స్థిరరాశి uint32_t warps_M = conv2d_BS.K / conv2d_WM;
            స్థిరరాశి uint32_t warps_N = conv2d_BS.NPQ / conv2d_WN;
            conv2d_WG_కొలత         = warps_M * warps_N * device->subgroup_కొలత;
        }

        // stage cm2 accumulator through shmem for coalesced global stores;
        // skipped on 128x128 where the extra Csh footprint hurts occupancy.
        // cm1 always uses the staged path.
        uint32_t conv2d_csh_store = (device->coopmat2 && s != CONV_SHAPE_128x128) ? 1u : 0u;
        ఒకవేళ (conv2d_use_cm1) {
            conv2d_csh_store = 1;
        }

        // shmem is fp16 on cm2/cm1 (matches Csh), fp32 on scalar
        స్థిరరాశి తార్కిక conv2d_use_fp16_shmem = device->coopmat2 || conv2d_use_cm1;

        // shrink CRS if the non-cm1 config still doesn't fit
        ఒకవేళ (device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత < shmem_req(conv2d_SHMEM_PAD, conv2d_csh_store, conv2d_use_fp16_shmem)) {
            GGML_ASSERT(!conv2d_use_cm1);
            conv2d_BS.CRS = 8;
            ఒకవేళ (use_collectives) {
                conv2d_BS.CRS = std::min(device->subgroup_కొలత, conv2d_BS.CRS);
            }
            conv2d_csh_store = 0;
        }

        std::array<uint32_t, 3> wg_denoms = { conv2d_BS.K, 1, 1 };
        std::vector<uint32_t> spec_constants = { conv2d_WG_కొలత, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, use_collectives, conv2d_SHMEM_PAD };

        // cm1 needs a fixed subgroup width to match the WG_SIZE we computed
        స్థిరరాశి uint32_t conv2d_required_subgroup_కొలత = conv2d_use_cm1 ? device->subgroup_కొలత : 0;

#నిర్వచించు CREATE_CONV(name, type_suffix, spv_suffix) \
        కొరకు (స్వయంచాలక &c : device->pipeline_##name##type_suffix[s]) { \
            స్థిరరాశి vk_conv2d_pipeline_state &state = c.మొదటి;  \
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
                పరిమాణం(vk_op_conv2d_నెట్టు_constants), wg_denoms, spec_constants_cpy, 1, నిజం, use_collectives || conv2d_required_subgroup_కొలత, conv2d_required_subgroup_కొలత);    \
        }
#నిర్వచించు CREATE_CONVS(spv_suffix) \
        CREATE_CONV(conv2d, _f32, spv_suffix) \
        CREATE_CONV(conv2d, _f16_f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f16_f32, spv_suffix)
#ఒకవేళ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        ఒకవేళ (device->coopmat2) {
            CREATE_CONVS(_cm2)
        } లేకపోతే
#అయితే_ముగింపు
#ఒకవేళ defined(VK_KHR_cooperative_మాతృక) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        ఒకవేళ (conv2d_use_cm1) {
            CREATE_CONVS(_cm1)
        } లేకపోతే
#అయితే_ముగింపు
        ఒకవేళ (conv2d_UNROLL) {
            CREATE_CONVS(_unroll)
        } లేకపోతే {
            CREATE_CONVS( )
        }
#నిర్వచనం_రద్దు CREATE_CONV
#నిర్వచనం_రద్దు CREATE_CONVS

        std::vector<uint32_t> conv3d_spec_constants = { conv2d_WG_కొలత, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, conv2d_SHMEM_PAD };
#నిర్వచించు CREATE_CONV3D(type_suffix, spv_suffix) \
        కొరకు (స్వయంచాలక &c : device->pipeline_conv3d##type_suffix[s]) { \
            స్థిరరాశి vk_conv3d_pipeline_state &state = c.మొదటి; \
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
                పరిమాణం(vk_op_conv3d_నెట్టు_constants), wg_denoms, spec_constants_cpy, 1, నిజం, conv2d_required_subgroup_కొలత != 0, conv2d_required_subgroup_కొలత); \
        }
#ఒకవేళ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        ఒకవేళ (device->coopmat2) {
            CREATE_CONV3D(_f32, _cm2)
            CREATE_CONV3D(_f16_f32, _cm2)
        } లేకపోతే
#అయితే_ముగింపు
#ఒకవేళ defined(VK_KHR_cooperative_మాతృక) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        ఒకవేళ (conv2d_use_cm1) {
            CREATE_CONV3D(_f32, _cm1)
            CREATE_CONV3D(_f16_f32, _cm1)
        } లేకపోతే
#అయితే_ముగింపు
        ఒకవేళ (conv2d_UNROLL) {
            CREATE_CONV3D(_f32, _unroll)
            CREATE_CONV3D(_f16_f32, _unroll)
        } లేకపోతే {
            CREATE_CONV3D(_f32, )
            CREATE_CONV3D(_f16_f32, )
        }
#నిర్వచనం_రద్దు CREATE_CONV3D
    }

    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f32, "conv2d_dw_whcn_f32", conv2d_dw_whcn_f32_len, conv2d_dw_whcn_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_conv2d_dw_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f32, "conv2d_dw_cwhn_f32", conv2d_dw_cwhn_f32_len, conv2d_dw_cwhn_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_conv2d_dw_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f16_f32, "conv2d_dw_whcn_f16_f32", conv2d_dw_whcn_f16_f32_len, conv2d_dw_whcn_f16_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_conv2d_dw_నెట్టు_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f16_f32, "conv2d_dw_cwhn_f16_f32", conv2d_dw_cwhn_f16_f32_len, conv2d_dw_cwhn_f16_f32_దత్తాంశం, "main", 3, పరిమాణం(vk_op_conv2d_dw_నెట్టు_constants), {512, 1, 1}, {}, 1);

    కొరకు (uint32_t use_నెట్టు = 0; use_నెట్టు < 2; ++use_నెట్టు) {
        కొరకు (uint32_t i = 0; i < అంకె_topk_moe_pipelines; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_topk_moe[i][use_నెట్టు], "topk_moe_f32_"+std::to_string(i), topk_moe_f32_len, topk_moe_f32_దత్తాంశం, "main", 4, పరిమాణం(vk_op_topk_moe_నెట్టు_constants), {1, 1, 1}, {device->subgroup_కొలత, 1u<<i, use_నెట్టు}, 1, నిజం, నిజం, device->subgroup_కొలత);
        }
    }

    // Drop compile_mutex so other threads can walk while we compile.
    compile_తాళం.unlock();

    // Compile what we claimed; create_pipeline_func reacquires compile_mutex
    // at the end to flip compile_pending/compiled and notify waiters.
    ఒకవేళ (has_claimed_task) {
        స్వయంచాలక & task = claimed_task;
        ggml_vk_create_pipeline_func(device, task.pipeline, task.spv_కొలత, task.spv_దత్తాంశం,
                                     task.entrypoint, task.parameter_లెక్క, task.wg_denoms,
                                     task.specialization_constants, task.disable_robustness,
                                     task.require_నిండిన_subgroups, task.required_subgroup_కొలత);
    }

    // Another thread may be compiling the pipeline we need; block on it here.
    ఒకవేళ (wait_pipeline) {
        std::unique_lock<std::mutex> wait_తాళం(device->compile_mutex);
        device->compile_cv.wait(wait_తాళం, [&] {
            తిరిగి wait_pipeline->compiled.load();
        });
    }
}

స్థిర తార్కిక ggml_vk_external_జ్ఞాపకం_host_usable(vk::DriverId driver_id) {
    // Disable external_memory_host until https://github.com/KhronosGroup/MoltenVK/pull/2622
    // is available in the Vulkan SDK.
    తిరిగి driver_id != vk::DriverId::eMoltenvk;
}

స్థిర తార్కిక ggml_vk_host_బఫరు_import_usable(uint32_t vendor_id) {
    // amdgpu re-faults all pages of every imported host allocation on each submission
    తిరిగి vendor_id != VK_VENDOR_ID_AMD;
}

vk_device ggml_vk_get_device(size_t idx) {
    VK_LOG_DEBUG("ggml_vk_get_device(" << idx << ")");

    స్థిర std::mutex mutex;
    std::lock_guard<std::mutex> తాళం(mutex);

    ఒకవేళ (vk_instance.devices[idx] == శూన్యసూచిక) {
        VK_LOG_DEBUG("Initializing new vk_device");
        vk_device device = std::make_shared<vk_device_struct>();

        device->జ్ఞాపకం_logger = std::unique_ptr<vk_జ్ఞాపకం_logger>(కొత్త vk_జ్ఞాపకం_logger());

        size_t dev_అంకె = vk_instance.device_indices[idx];

        std::vector<vk::PhysicalDevice> physical_devices = vk_instance.instance.enumeratePhysicalDevices();

        ఒకవేళ (dev_అంకె >= physical_devices.size()) {
            std::cerr << "ggml_vulkan: Device with index " << dev_అంకె << " does not exist." << std::పంక్తి_ముగింపు;
            విసురు std::runtime_error("Device not found");
        }

        device->physical_device = physical_devices[dev_అంకె];
        స్థిరరాశి std::vector<vk::ExtensionProperties> ext_props = device->physical_device.enumerateDeviceExtensionProperties();

        device->architecture = get_device_architecture(device->physical_device);

        స్థిరరాశి అక్షరం* GGML_VK_PREFER_HOST_జ్ఞాపకం = getenv("GGML_VK_PREFER_HOST_MEMORY");
        device->prefer_host_జ్ఞాపకం = GGML_VK_PREFER_HOST_జ్ఞాపకం != శూన్యసూచిక;

        స్థిరరాశి అక్షరం* GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM = getenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM");
        device->disable_host_visible_vidmem = GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM != శూన్యసూచిక;

        స్థిరరాశి అక్షరం* GGML_VK_ALLOW_SYSMEM_FALLBACK = getenv("GGML_VK_ALLOW_SYSMEM_FALLBACK");
        device->allow_sysmem_fallback = GGML_VK_ALLOW_SYSMEM_FALLBACK != శూన్యసూచిక;

        స్థిరరాశి అక్షరం* GGML_VK_DISABLE_లేఖాచిత్రం_OPTIMIZE = getenv("GGML_VK_DISABLE_GRAPH_OPTIMIZE");
        device->disable_లేఖాచిత్రం_optimize = GGML_VK_DISABLE_లేఖాచిత్రం_OPTIMIZE != శూన్యసూచిక;

        తార్కిక fp16_storage = అబద్ధం;
        తార్కిక fp16_compute = అబద్ధం;
        తార్కిక maintenance4_support = అబద్ధం;
        తార్కిక sm_builtins = అబద్ధం;
        తార్కిక amd_shader_core_properties2 = అబద్ధం;
        తార్కిక pipeline_robustness = అబద్ధం;
        తార్కిక coopmat2_support = అబద్ధం;
        తార్కిక coopmat2_decode_సదిశ_support = అబద్ధం;
        తార్కిక pipeline_executable_properties_support = అబద్ధం;
        తార్కిక internally_sync_support = అబద్ధం;
        device->coopmat_support = అబద్ధం;
        device->integer_dot_లబ్ధం = అబద్ధం;
        device->shader_64b_indexing = అబద్ధం;
        తార్కిక bfloat16_support = అబద్ధం;
        తార్కిక dot2_f16_support = అబద్ధం;
        తార్కిక ocp_microscaling_extension = అబద్ధం;
        తార్కిక shader_float8_extension = అబద్ధం;

        కొరకు (స్థిరరాశి స్వయంచాలక& properties : ext_props) {
            ఒకవేళ (strcmp("VK_KHR_maintenance4", properties.extensionName) == 0) {
                maintenance4_support = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
                fp16_storage = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
                fp16_compute = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_AMD_shader_core_properties2", properties.extensionName) == 0) {
                amd_shader_core_properties2 = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_EXT_pipeline_robustness", properties.extensionName) == 0) {
                pipeline_robustness = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                device->subgroup_కొలత_control = నిజం;
#ఒకవేళ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            } లేకపోతే ఒకవేళ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT")) {
                device->coopmat_support = నిజం;
                device->coopmat_m = 0;
                device->coopmat_n = 0;
                device->coopmat_k = 0;
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            } లేకపోతే ఒకవేళ (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2")) {
                coopmat2_support = నిజం;
#అయితే_ముగింపు
            } లేకపోతే ఒకవేళ (strcmp(VK_NV_COOPERATIVE_మాతృక_DECODE_సదిశ_EXTENSION_NAME, properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
                coopmat2_decode_సదిశ_support = నిజం;
#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            } లేకపోతే ఒకవేళ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
                device->integer_dot_లబ్ధం = నిజం;
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            } లేకపోతే ఒకవేళ (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_BFLOAT16")) {
                bfloat16_support = నిజం;
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
            } లేకపోతే ఒకవేళ (strcmp(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
                ocp_microscaling_extension = నిజం;
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            } లేకపోతే ఒకవేళ (strcmp(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
                shader_float8_extension = నిజం;
#అయితే_ముగింపు
            } లేకపోతే ఒకవేళ (strcmp("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_DOT2")) {
                dot2_f16_support = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_KHR_pipeline_executable_properties", properties.extensionName) == 0) {
                pipeline_executable_properties_support = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_EXT_memory_priority", properties.extensionName) == 0 &&
                       getenv("GGML_VK_ENABLE_MEMORY_PRIORITY")) {
                device->జ్ఞాపకం_ప్రాధాన్యత = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_EXT_external_memory_host", properties.extensionName) == 0) {
                device->external_జ్ఞాపకం_host = నిజం;
#ఒకవేళ defined(VK_EXT_shader_64bit_indexing)
            } లేకపోతే ఒకవేళ (strcmp("VK_EXT_shader_64bit_indexing", properties.extensionName) == 0) {
                device->shader_64b_indexing = నిజం;
#అయితే_ముగింపు
            } లేకపోతే ఒకవేళ (strcmp(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME, properties.extensionName) == 0) {
                internally_sync_support = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_EXT_device_fault", properties.extensionName) == 0) {
                device->device_fault = నిజం;
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
        vk::Physical_Device_Subgroup_కొలత_Control_Properties_EXT subgroup_కొలత_control_props;
        vk::Physical_Device_Shader_Integer_Dot_లబ్ధం_Properties_KHR shader_integer_dot_లబ్ధం_props;
        vk::Physical_Device_External_జ్ఞాపకం_Host_Properties_EXT external_జ్ఞాపకం_host_props;

        props2.p_తదుపరి = &props3;
        props3.p_తదుపరి = &subgroup_props;
        subgroup_props.p_తదుపరి = &driver_props;
        driver_props.p_తదుపరి = &vk11_props;
        vk11_props.p_తదుపరి = &vk12_props;

        Vk_పునాది_బహిర్గతం_Structure * చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&vk12_props;

        ఒకవేళ (maintenance4_support) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&props4;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&props4;
        }
        ఒకవేళ (sm_builtins) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&sm_props;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&sm_props;
        }
        ఒకవేళ (amd_shader_core_properties2) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&amd_shader_core_properties2_props;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&amd_shader_core_properties2_props;
        }
        ఒకవేళ (device->subgroup_కొలత_control) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&subgroup_కొలత_control_props;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&subgroup_కొలత_control_props;
        }

#ఒకవేళ defined(VK_NV_cooperative_matrix2)
        vk::PhysicalDeviceCooperativeMatrix2PropertiesNV coopmat2_props;
        ఒకవేళ (coopmat2_support) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat2_props;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat2_props;
        }
#అయితే_ముగింపు

        ఒకవేళ (device->integer_dot_లబ్ధం) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&shader_integer_dot_లబ్ధం_props;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&shader_integer_dot_లబ్ధం_props;
        }

        ఒకవేళ (device->external_జ్ఞాపకం_host) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&external_జ్ఞాపకం_host_props;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&external_జ్ఞాపకం_host_props;
        }

        device->physical_device.getProperties2(&props2);
        device->properties = props2.properties;
        device->vendor_id = device->properties.vendorID;
        device->driver_id = driver_props.driverID;

        device->external_జ్ఞాపకం_host = device->external_జ్ఞాపకం_host && ggml_vk_external_జ్ఞాపకం_host_usable(device->driver_id);

        // Implementing the async backend interfaces seems broken on older Intel HW,
        // see https://github.com/ggml-org/llama.cpp/issues/17302.
        device->support_async = (device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 std::string(device->properties.deviceName.దత్తాంశం()).find("(DG1)") == std::string::npos) &&
                                getenv("GGML_VK_DISABLE_ASYNC") == శూన్యసూచిక;

        ఒకవేళ (!device->support_async) {
            GGML_LOG_DEBUG("ggml_vulkan: WARNING: Async execution disabled on certain Intel devices.\n");
        }

        స్థిరరాశి అక్షరం* GGML_VK_FORCE_MAX_కేటాయింపు_కొలత = getenv("GGML_VK_FORCE_MAX_ALLOCATION_SIZE");

        ఒకవేళ (GGML_VK_FORCE_MAX_కేటాయింపు_కొలత != శూన్యసూచిక) {
            device->max_జ్ఞాపకం_కేటాయింపు_కొలత = std::stoull(GGML_VK_FORCE_MAX_కేటాయింపు_కొలత);
        } లేకపోతే ఒకవేళ (maintenance4_support) {
            device->max_జ్ఞాపకం_కేటాయింపు_కొలత = std::min(props3.max_జ్ఞాపకం_కేటాయింపు_కొలత_2, props4.max_బఫరు_కొలత_2);
        } లేకపోతే {
            device->max_జ్ఞాపకం_కేటాయింపు_కొలత = props3.max_జ్ఞాపకం_కేటాయింపు_కొలత_2;
        }

        స్థిరరాశి అక్షరం* GGML_VK_FORCE_MAX_బఫరు_కొలత = getenv("GGML_VK_FORCE_MAX_BUFFER_SIZE");

        ఒకవేళ (GGML_VK_FORCE_MAX_బఫరు_కొలత != శూన్యసూచిక) {
            device->max_బఫరు_కొలత = std::stoull(GGML_VK_FORCE_MAX_బఫరు_కొలత);
        } లేకపోతే ఒకవేళ (maintenance4_support) {
            device->max_బఫరు_కొలత = props4.max_బఫరు_కొలత_2;
        } లేకపోతే {
            device->max_బఫరు_కొలత = device->max_జ్ఞాపకం_కేటాయింపు_కొలత;
        }

        స్థిరరాశి అక్షరం* GGML_VK_SUBALLOCATION_ఖండం_కొలత = getenv("GGML_VK_SUBALLOCATION_BLOCK_SIZE");

        ఒకవేళ (GGML_VK_SUBALLOCATION_ఖండం_కొలత != శూన్యసూచిక) {
            device->suballocation_ఖండం_కొలత = std::stoull(GGML_VK_SUBALLOCATION_ఖండం_కొలత);
        } లేకపోతే {
            // Limit batching of allocations to 1GB by default to avoid fragmentation issues
            device->suballocation_ఖండం_కొలత = 1024*1024*1024;
        }
        device->suballocation_ఖండం_కొలత = std::min(device->suballocation_ఖండం_కొలత, device->max_జ్ఞాపకం_కేటాయింపు_కొలత);

        device->subgroup_కొలత = subgroup_props.subgroup_కొలత_2;
        device->subgroup_కొలత_log2 = uint32_t(log2f(దశాంశం(device->subgroup_కొలత)));
        device->uma = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
        ఒకవేళ (sm_builtins) {
            device->shader_core_లెక్క = sm_props.shaderSMCount;
        } లేకపోతే ఒకవేళ (amd_shader_core_properties2) {
            device->shader_core_లెక్క = amd_shader_core_properties2_props.చురుకైన_Compute_Unit_లెక్క;
        } లేకపోతే ఒకవేళ (device->vendor_id == VK_VENDOR_ID_INTEL) {
            device->shader_core_లెక్క = ggml_vk_intel_shader_core_లెక్క(device->physical_device);
        } లేకపోతే {
            device->shader_core_లెక్క = 0;
        }
        device->float_controls_rte_fp16 = vk12_props.shader_Rounding_బహుళకం_RTEFloat16;
        device->float_controls_denorm_preserve_fp16 = vk12_props.shaderDenormPreserveFloat16;

        device->subgroup_basic = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_బిట్లు::eCompute) &&
                                 (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_బిట్లు::eBasic);
        device->subgroup_arithmetic = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_బిట్లు::eCompute) &&
                                      (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_బిట్లు::eArithmetic);
#నిర్వచితమైతే __APPLE__
        // Workaround for subgroup arithmetic failing on MoltenVK with AMD GPUs (issue 15846)
        ఒకవేళ (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_arithmetic = అబద్ధం;
        }
#అయితే_ముగింపు
        device->subgroup_shuffle = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_బిట్లు::eCompute) &&
                                   (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_బిట్లు::eShuffle);
#నిర్వచితమైతే __APPLE__
        ఒకవేళ (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_shuffle = అబద్ధం;
        }
#అయితే_ముగింపు
        device->subgroup_clustered = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_బిట్లు::eCompute) &&
                                     (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_బిట్లు::eClustered);

        device->subgroup_ballot = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_బిట్లు::eCompute) &&
                                  (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_బిట్లు::eBallot);

        device->subgroup_vote = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_బిట్లు::eCompute) &&
                                (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_బిట్లు::eVote);

        // Submit at least every 100 nodes, in case there are workloads without as much matmul.
        device->max_nodes_చొప్పున_submit = 100;
        స్థిరరాశి అక్షరం* GGML_VK_MAX_NODES_చొప్పున_SUBMIT = getenv("GGML_VK_MAX_NODES_PER_SUBMIT");
        ఒకవేళ (GGML_VK_MAX_NODES_చొప్పున_SUBMIT != శూన్యసూచిక) {
            uint32_t max_nodes_చొప్పున_submit = std::stoul(GGML_VK_MAX_NODES_చొప్పున_SUBMIT);
            device->max_nodes_చొప్పున_submit = std::max(max_nodes_చొప్పున_submit, 1u);
        }

        స్థిరరాశి తార్కిక force_disable_f16 = getenv("GGML_VK_DISABLE_F16") != శూన్యసూచిక;

        device->fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

        ఒకవేళ (!ggml_vk_khr_cooperative_మాతృక_support(device->properties, driver_props, device->architecture)) {
            device->coopmat_support = అబద్ధం;
        }

        device->integer_dot_లబ్ధం = device->integer_dot_లబ్ధం && shader_integer_dot_లబ్ధం_props.integer_Dot_Product4x8_బిట్టు_Packed_Signed_Accelerated;

        device->min_imported_host_సూచిక_alignment = external_జ్ఞాపకం_host_props.min_Imported_Host_సూచిక_Alignment;

        device->max_workgroup_కొలత_log2 = uint32_t(log2f(దశాంశం(device->properties.limits.max_Compute_Work_సమూహం_Invocations)));

        std::vector<vk::వరుస_Family_Properties> వరుస_family_props = device->physical_device.get_వరుస_Family_Properties();

        // Try to find a non-graphics compute queue and transfer-focused queues
        // Allow overriding avoiding the graphics queue because it can increase performance on RADV
        స్థిరరాశి తార్కిక allow_graphics_వరుస = (getenv("GGML_VK_ALLOW_GRAPHICS_QUEUE") != శూన్యసూచిక);
        స్థిరరాశి vk::వరుస_Flag_బిట్లు graphics_flag = allow_graphics_వరుస ? (vk::వరుస_Flag_బిట్లు)0 : vk::వరుస_Flag_బిట్లు::eGraphics;
        స్థిరరాశి uint32_t compute_వరుస_family_సూచీ = ggml_vk_కనుగొను_వరుస_family_సూచీ(వరుస_family_props, vk::వరుస_Flag_బిట్లు::eCompute, graphics_flag, -1, 1);
        స్థిరరాశి uint32_t transfer_వరుస_family_సూచీ = ggml_vk_కనుగొను_వరుస_family_సూచీ(వరుస_family_props, vk::వరుస_Flag_బిట్లు::eTransfer, vk::వరుస_Flag_బిట్లు::eCompute | graphics_flag, compute_వరుస_family_సూచీ, 1);

        స్థిరరాశి దశాంశం priorities[] = { 1.0f, 1.0f };
        device->single_వరుస = compute_వరుస_family_సూచీ == transfer_వరుస_family_సూచీ && వరుస_family_props[compute_వరుస_family_సూచీ].వరుస_లెక్క == 1;

        std::vector<vk::Device_వరుస_Create_Info> device_వరుస_create_infos;
        vk::DeviceCreateInfo device_create_info{};
        std::vector<స్థిరరాశి అక్షరం *> device_extensions;
        vk::PhysicalDeviceFeatures device_features = device->physical_device.getFeatures();

        VkPhysicalDeviceFeatures2 device_features2;
        device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        device_features2.p_తదుపరి = శూన్యసూచిక;
        device_features2.features = (VkPhysicalDeviceFeatures)device_features;

        VkPhysicalDeviceVulkan11Features vk11_features;
        vk11_features.p_తదుపరి = శూన్యసూచిక;
        vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        device_features2.p_తదుపరి = &vk11_features;

        VkPhysicalDeviceVulkan12Features vk12_features;
        vk12_features.p_తదుపరి = శూన్యసూచిక;
        vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk11_features.p_తదుపరి = &vk12_features;

        చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&vk12_features;

        VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR internally_synchronized_queues_features{};
        internally_synchronized_queues_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR;
        internally_synchronized_queues_features.p_తదుపరి = శూన్యసూచిక;
        internally_synchronized_queues_features.internallySynchronizedQueues = VK_FALSE;

        ఒకవేళ (internally_sync_support) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&internally_synchronized_queues_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&internally_synchronized_queues_features;
            device_extensions.push_back(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME);
        }

        VkPhysicalDevicePipelineRobustnessFeaturesEXT pl_robustness_features;
        pl_robustness_features.p_తదుపరి = శూన్యసూచిక;
        pl_robustness_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
        pl_robustness_features.pipelineRobustness = VK_FALSE;

        ఒకవేళ (pipeline_robustness) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&pl_robustness_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&pl_robustness_features;
            device_extensions.push_back("VK_EXT_pipeline_robustness");
        }

        Vk_Physical_Device_జ్ఞాపకం_ప్రాధాన్యత_Features_EXT జ్ఞాపకం_ప్రాధాన్యత_features;
        జ్ఞాపకం_ప్రాధాన్యత_features.p_తదుపరి = శూన్యసూచిక;
        జ్ఞాపకం_ప్రాధాన్యత_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_జ్ఞాపకం_ప్రాధాన్యత_FEATURES_EXT;
        జ్ఞాపకం_ప్రాధాన్యత_features.జ్ఞాపకం_ప్రాధాన్యత_2 = VK_FALSE;
        ఒకవేళ (device->జ్ఞాపకం_ప్రాధాన్యత) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&జ్ఞాపకం_ప్రాధాన్యత_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&జ్ఞాపకం_ప్రాధాన్యత_features;
            device_extensions.push_back("VK_EXT_memory_priority");
        }

        Vk_Physical_Device_Subgroup_కొలత_Control_Features_EXT subgroup_కొలత_control_features;
        subgroup_కొలత_control_features.p_తదుపరి = శూన్యసూచిక;
        subgroup_కొలత_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_కొలత_CONTROL_FEATURES_EXT;
        subgroup_కొలత_control_features.compute_నిండిన_Subgroups = అబద్ధం;
        subgroup_కొలత_control_features.subgroup_కొలత_Control = అబద్ధం;

        ఒకవేళ (device->subgroup_కొలత_control) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&subgroup_కొలత_control_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&subgroup_కొలత_control_features;
        }

#ఒకవేళ defined(VK_KHR_cooperative_మాతృక)
        Vk_Physical_Device_Cooperative_మాతృక_Features_KHR coopmat_features;
        coopmat_features.p_తదుపరి = శూన్యసూచిక;
        coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_మాతృక_FEATURES_KHR;
        coopmat_features.cooperative_మాతృక_2 = VK_FALSE;

        ఒకవేళ (device->coopmat_support) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat_features;
        }
#అయితే_ముగింపు

#ఒకవేళ defined(VK_NV_cooperative_matrix2)
        VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
        coopmat2_features.p_తదుపరి = శూన్యసూచిక;
        coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_మాతృక_2_FEATURES_NV;
        ఒకవేళ (coopmat2_support) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat2_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat2_features;
            device_extensions.push_back("VK_NV_cooperative_matrix2");
        }
#అయితే_ముగింపు

        Vk_Physical_Device_Cooperative_మాతృక_Decode_సదిశ_Features_NV coopmat2_decode_సదిశ_features {};
        coopmat2_decode_సదిశ_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_మాతృక_DECODE_సదిశ_FEATURES_NV;
        ఒకవేళ (coopmat2_decode_సదిశ_support) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat2_decode_సదిశ_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat2_decode_సదిశ_features;
            device_extensions.push_back(VK_NV_COOPERATIVE_మాతృక_DECODE_సదిశ_EXTENSION_NAME);
        }

#ఒకవేళ defined(VK_KHR_shader_bfloat16)
        VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
        bfloat16_features.p_తదుపరి = శూన్యసూచిక;
        bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
        ఒకవేళ (bfloat16_support) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&bfloat16_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&bfloat16_features;
            device_extensions.push_back("VK_KHR_shader_bfloat16");
        }
#అయితే_ముగింపు

        VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
        ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
        ఒకవేళ (ocp_microscaling_extension) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&ocp_microscaling_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&ocp_microscaling_features;
            device_extensions.push_back(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME);
        }

        VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
        shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
        ఒకవేళ (shader_float8_extension) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&shader_float8_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&shader_float8_features;
            device_extensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        }

        VkPhysicalDeviceMaintenance4Features maint4_features {};
        maint4_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
        ఒకవేళ (maintenance4_support) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&maint4_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&maint4_features;
            device_extensions.push_back("VK_KHR_maintenance4");
        }

        Vk_Physical_Device_Shader_Integer_Dot_లబ్ధం_Features_KHR shader_integer_dot_లబ్ధం_features {};
        shader_integer_dot_లబ్ధం_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_లబ్ధం_FEATURES_KHR;
        ఒకవేళ (device->integer_dot_లబ్ధం) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&shader_integer_dot_లబ్ధం_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&shader_integer_dot_లబ్ధం_features;
            device_extensions.push_back("VK_KHR_shader_integer_dot_product");
        }

        Vk_Physical_Device_Shader_Mixed_Float_Dot_లబ్ధం_Features_VALVE dot2_features {};
        dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_లబ్ధం_FEATURES_VALVE;
        ఒకవేళ (dot2_f16_support) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&dot2_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&dot2_features;
            device_extensions.push_back("VK_VALVE_shader_mixed_float_dot_product");
        }

        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pep_features {};
        pep_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        ఒకవేళ (pipeline_executable_properties_support) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&pep_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&pep_features;
            device_extensions.push_back("VK_KHR_pipeline_executable_properties");
        }

        ఒకవేళ (device->external_జ్ఞాపకం_host) {
            device_extensions.push_back("VK_EXT_external_memory_host");
        }

#ఒకవేళ defined(VK_EXT_shader_64bit_indexing)
        Vk_Physical_Device_Shader64_బిట్టు_Indexing_Features_EXT shader_64bit_indexing_features {};
        shader_64bit_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_64_బిట్టు_INDEXING_FEATURES_EXT;
        ఒకవేళ (device->shader_64b_indexing) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&shader_64bit_indexing_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&shader_64bit_indexing_features;
            device_extensions.push_back("VK_EXT_shader_64bit_indexing");
        }
#అయితే_ముగింపు

        VkPhysicalDeviceFaultFeaturesEXT fault_features {};
        fault_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
        ఒకవేళ (device->device_fault) {
            చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&fault_features;
            చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&fault_features;
            device_extensions.push_back("VK_EXT_device_fault");
        }

        vkGetPhysicalDeviceFeatures2(device->physical_device, &device_features2);

        device->device_fault = device->device_fault && fault_features.deviceFault;

        device->has_internally_synchronized_queues = internally_synchronized_queues_features.internallySynchronizedQueues;

        // Build queue create infos only after querying whether internally synchronized queues are enabled.
        // getQueue2() later uses the same flag, so creation/retrieval must stay consistent.
        vk::Device_వరుస_Create_Flags వరుస_flags = device->has_internally_synchronized_queues ?
                                                eInternallySynchronizedKHR :
                                                vk::Device_వరుస_Create_Flags();

        ఒకవేళ (compute_వరుస_family_సూచీ != transfer_వరుస_family_సూచీ) {
            device_వరుస_create_infos.push_back({వరుస_flags, compute_వరుస_family_సూచీ, 1, priorities});
            device_వరుస_create_infos.push_back({వరుస_flags, transfer_వరుస_family_సూచీ, 1, priorities + 1});
        } లేకపోతే ఒకవేళ(!device->single_వరుస) {
            device_వరుస_create_infos.push_back({వరుస_flags, compute_వరుస_family_సూచీ, 2, priorities});
        } లేకపోతే {
            device_వరుస_create_infos.push_back({వరుస_flags, compute_వరుస_family_సూచీ, 1, priorities});
        }

        device->pipeline_executable_properties_support = pipeline_executable_properties_support;

        device->fp16 = device->fp16 && vk12_features.shaderFloat16;

#ఒకవేళ defined(VK_KHR_shader_bfloat16)
        device->bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#లేకపోతే
        device->bf16 = అబద్ధం;
#అయితే_ముగింపు

        device->dot2_f16 = dot2_f16_support && dot2_features.shader_Mixed_Float_Dot_లబ్ధం_Float16_Acc_Float32;
        device->ocp_fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                          shader_float8_extension && shader_float8_features.shaderFloat8 &&
                          !getenv("GGML_VK_DISABLE_OCP_FP4");

        device->pipeline_robustness = pl_robustness_features.pipelineRobustness;

        device->multi_కూడు = vk12_props.shader_Rounding_బహుళకం_RTEFloat16 &&
                            device->properties.limits.max_నెట్టు_Constants_కొలత >= పరిమాణం(vk_op_multi_కూడు_నెట్టు_constants) &&
                            getenv("GGML_VK_DISABLE_MULTI_ADD") == శూన్యసూచిక;

        device->shader_int64 = device_features2.features.shaderInt64;
        device->బఫరు_device_చిరునామా = vk12_features.బఫరు_Device_చిరునామా;
        device->vulkan_జ్ఞాపకం_model = vk12_features.vulkan_జ్ఞాపకం_Model;

        ఒకవేళ (device->subgroup_కొలత_control) {
            device->subgroup_min_కొలత = subgroup_కొలత_control_props.min_Subgroup_కొలత;
            device->subgroup_max_కొలత = subgroup_కొలత_control_props.max_Subgroup_కొలత;
            device_extensions.push_back("VK_EXT_subgroup_size_control");
        }

        device->subgroup_కొలత_control = device->subgroup_కొలత_control &&
                (subgroup_కొలత_control_props.required_Subgroup_కొలత_Stages & vk::Shader_Stage_Flag_బిట్లు::eCompute) &&
                subgroup_కొలత_control_features.subgroup_కొలత_Control;

        device->subgroup_require_నిండిన_support = subgroup_కొలత_control_features.compute_నిండిన_Subgroups;

#ఒకవేళ defined(VK_KHR_cooperative_మాతృక)
        device->coopmat_support = device->coopmat_support && coopmat_features.cooperative_మాతృక_2;
        device->coopmat1_fa_support = device->coopmat_support && device->subgroup_require_నిండిన_support;
#అయితే_ముగింపు

        ఒకవేళ (coopmat2_support) {
#ఒకవేళ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            ఒకవేళ (coopmat2_features.cooperative_మాతృక_Workgroup_Scope &&
                coopmat2_features.cooperative_మాతృక_Flexible_Dimensions &&
                coopmat2_features.cooperative_మాతృక_Reductions &&
                coopmat2_features.cooperative_మాతృక_Conversions &&
                coopmat2_features.cooperative_మాతృక_చొప్పున_Element_Operations &&
                coopmat2_features.cooperative_మాతృక_Tensor_Addressing &&
                coopmat2_features.cooperative_మాతృక_ఖండం_Loads &&
                vk12_features.బఫరు_Device_చిరునామా) {

                std::vector<Vk_Cooperative_మాతృక_Flexible_Dimensions_Properties_NV> flexible_dimensions;
                uint32_t count = 0;

                PFN_vk_Get_Physical_Device_Cooperative_మాతృక_Flexible_Dimensions_Properties_NV
                    vk_Get_Physical_Device_Cooperative_మాతృక_Flexible_Dimensions_Properties_NV =
                        (PFN_vk_Get_Physical_Device_Cooperative_మాతృక_Flexible_Dimensions_Properties_NV)
                        vk_instance.instance.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");

                vk_Get_Physical_Device_Cooperative_మాతృక_Flexible_Dimensions_Properties_NV(device->physical_device, &count, శూన్యసూచిక);

                Vk_Cooperative_మాతృక_Flexible_Dimensions_Properties_NV ఖాళీలు_prop {};
                ఖాళీలు_prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_మాతృక_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
                flexible_dimensions.resize(count, ఖాళీలు_prop);

                vk_Get_Physical_Device_Cooperative_మాతృక_Flexible_Dimensions_Properties_NV(device->physical_device, &count, flexible_dimensions.దత్తాంశం());

                తార్కిక దొరికింది_fp16_128 = అబద్ధం,
                     దొరికింది_fp16_256 = అబద్ధం,
                     దొరికింది_fp32_128 = అబద్ధం,
                     దొరికింది_fp32_256 = అబద్ధం;
                తార్కిక దొరికింది_bf16_128 = అబద్ధం,
                     దొరికింది_bf16_256 = అబద్ధం;
                // need to support fp16*fp16 with fp16/fp32 accumulator, for workgroupsize 128
                // with 32x16x16 and 256 with 32x32x16.
                కొరకు (స్వయంచాలక &prop : flexible_dimensions) {
                    ఒకవేళ (prop.saturatingAccumulation == VK_FALSE &&
                        prop.scope == VK_SCOPE_WORKGROUP_KHR) {

                        ఒకవేళ (prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR) {

                            ఒకవేళ (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                ఒకవేళ (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.ఫలితం_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    దొరికింది_fp16_128 = నిజం;
                                }
                                ఒకవేళ (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.ఫలితం_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    దొరికింది_fp32_128 = నిజం;
                                }
                            }
                            ఒకవేళ (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                ఒకవేళ (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.ఫలితం_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    దొరికింది_fp16_256 = నిజం;
                                }
                                ఒకవేళ (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.ఫలితం_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    దొరికింది_fp32_256 = నిజం;
                                }
                            }
                        }

#ఒకవేళ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                        ఒకవేళ (bfloat16_support &&
                            prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            prop.ఫలితం_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {

                            ఒకవేళ (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                దొరికింది_bf16_128 = నిజం;
                            }
                            ఒకవేళ (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                దొరికింది_bf16_256 = నిజం;
                            }
                        }
#అయితే_ముగింపు
                    }
                }
                ఒకవేళ (దొరికింది_fp16_128 && దొరికింది_fp16_256 &&
                    దొరికింది_fp32_128 && దొరికింది_fp32_256 &&
                    coopmat2_props.cooperative_మాతృక_Flexible_Dimensions_Max_Dimension >= 512) {
                    device->coopmat2 = నిజం;
                    device->coopmat2_bf16_support = దొరికింది_bf16_128 && దొరికింది_bf16_256;
                    device->coopmat2_decode_సదిశ = coopmat2_decode_సదిశ_support && coopmat2_decode_సదిశ_features.cooperative_మాతృక_Decode_సదిశ;
                }
            }
#అయితే_ముగింపు
        }

        ఒకవేళ (!vk11_features.storage_Buffer16_బిట్టు_Access) {
            std::cerr << "ggml_vulkan: device " << GGML_VK_NAME << idx << " does not support 16-bit storage." << std::పంక్తి_ముగింపు;
            విసురు std::runtime_error("Unsupported device");
        }

        device_extensions.push_back("VK_KHR_16bit_storage");

#నిర్వచితమైతే GGML_VULKAN_VALIDATE
        device_extensions.push_back("VK_KHR_shader_non_semantic_info");
#అయితే_ముగింపు

        ఒకవేళ (device->fp16) {
            device_extensions.push_back("VK_KHR_shader_float16_int8");
        }

#ఒకవేళ defined(VK_KHR_cooperative_మాతృక)
        ఒకవేళ (device->coopmat_support) {
            // Query supported shapes
            std::vector<Vk_Cooperative_మాతృక_Properties_KHR> cm_props;

            PFN_vk_Get_Physical_Device_Cooperative_మాతృక_Properties_KHR pfn_vk_Get_Physical_Device_Cooperative_మాతృక_Properties_KHR =
                (PFN_vk_Get_Physical_Device_Cooperative_మాతృక_Properties_KHR)vkGetInstanceProcAddr(vk_instance.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

            uint32_t cm_props_అంకె;

            pfn_vk_Get_Physical_Device_Cooperative_మాతృక_Properties_KHR(device->physical_device, &cm_props_అంకె, శూన్యసూచిక);

            cm_props.resize(cm_props_అంకె);

            కొరకు (స్వయంచాలక& prop : cm_props) {
                prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_మాతృక_PROPERTIES_KHR;
            }

            pfn_vk_Get_Physical_Device_Cooperative_మాతృక_Properties_KHR(device->physical_device, &cm_props_అంకె, cm_props.దత్తాంశం());

            VK_LOG_DEBUG("ggml_vulkan: Cooperative Matrix Shapes: " << cm_props.size());

            కొరకు (స్వయంచాలక& prop : cm_props) {
                VK_LOG_DEBUG("ggml_vulkan: M: " << prop.MSize << " N: " << prop.NSize << " K: " << prop.KSize << " A: " << vk::to_string((vk::ComponentTypeKHR)prop.AType) << " B: " << vk::to_string((vk::ComponentTypeKHR)prop.BType) << " C: " << vk::to_string((vk::ComponentTypeKHR)prop.CType) << " Result: " << vk::to_string((vk::ComponentTypeKHR)prop.ఫలితం_Type) << " saturatingAccumulation: " << prop.saturatingAccumulation << " scope: " << vk::to_string((vk::ScopeKHR)prop.scope));

                ఒకవేళ ((vk::ComponentTypeKHR)prop.AType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ComponentTypeKHR)prop.BType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    ఒకవేళ ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat32 &&
                        (vk::ComponentTypeKHR)prop.ఫలితం_Type == vk::ComponentTypeKHR::eFloat32) {
                        // coopmat sizes not set yet
                        ఒకవేళ (device->coopmat_m == 0) {
                            device->coopmat_acc_f32_support = నిజం;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } లేకపోతే ఒకవేళ (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f32_support = నిజం;
                        }
                        ఒకవేళ (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f32acc = నిజం;
                        }
                    } లేకపోతే ఒకవేళ ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat16 &&
                               (vk::ComponentTypeKHR)prop.ఫలితం_Type == vk::ComponentTypeKHR::eFloat16) {
                        // coopmat sizes not set yet
                        ఒకవేళ (device->coopmat_m == 0) {
                            device->coopmat_acc_f16_support = నిజం;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } లేకపోతే ఒకవేళ (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f16_support = నిజం;
                        }
                        ఒకవేళ (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f16acc = నిజం;
                        }
                    }
                } లేకపోతే ఒకవేళ ((vk::ComponentTypeKHR)prop.AType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.BType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.CType      == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ComponentTypeKHR)prop.ఫలితం_Type == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup &&
                           device->coopmat_int_m == 0
                ) {
                    device->coopmat_int_support = నిజం;
                    device->coopmat_int_m = prop.MSize;
                    device->coopmat_int_n = prop.NSize;
                    device->coopmat_int_k = prop.KSize;
                }
#ఒకవేళ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                ఒకవేళ (bfloat16_support &&
                    prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    prop.ఫలితం_Type == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    // coopmat sizes not set yet
                    ఒకవేళ (device->coopmat_m == 0) {
                        device->coopmat_bf16_support = నిజం;
                        device->coopmat_m = prop.MSize;
                        device->coopmat_n = prop.NSize;
                        device->coopmat_k = prop.KSize;
                    } లేకపోతే ఒకవేళ (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                        // Only enable if shape is identical
                        device->coopmat_bf16_support = నిజం;
                    }
                }
#అయితే_ముగింపు
            }

            ఒకవేళ (device->coopmat_m == 0 || !device->coopmat_acc_f32_support) {
                // No suitable matmul mode found
                GGML_LOG_DEBUG("ggml_vulkan: WARNING: No suitable matrix core mode found. Disabling matrix cores.\n");
                device->coopmat_support = అబద్ధం;
            }
        }

        ఒకవేళ (device->coopmat_support) {
            device_extensions.push_back("VK_KHR_cooperative_matrix");
        }
#అయితే_ముగింపు
        device->name = GGML_VK_NAME + std::to_string(idx);

        device_create_info
            .సమితి_Flags(vk::DeviceCreateFlags())
            .సమితి_వరుస_Create_Infos(device_వరుస_create_infos)
            .సమితి_PEnabled_Extension_పేర్లు(device_extensions);
        device_create_info.సమితి_PNext(&device_features2);
        device->device = device->physical_device.createDevice(device_create_info);

        ఒకవేళ (device->device_fault) {
            device->pfn_vkGetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)
                vkGetDeviceProcAddr(device->device, "vkGetDeviceFaultInfoEXT");
        }

        // Queues
        device->compute_వరుస = ggml_vk_create_వరుస(device, compute_వరుస_family_సూచీ, 0, { vk::Pipeline_Stage_Flag_బిట్లు::eComputeShader | vk::Pipeline_Stage_Flag_బిట్లు::eTransfer }, అబద్ధం);

        // Shaders
        // Disable matmul tile sizes early if performance low or not supported
        కొరకు (uint32_t i = 0; i < GGML_TYPE_లెక్క; ++i) {
            ఎంపిక (device->vendor_id) {
#నిర్వచితం_కాకపోతే GGML_VULKAN_నిరంతరం_TESTS
            స్థితి VK_VENDOR_ID_AMD:
                device->mul_mat_l[i]    = device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary;
                device->mul_mat_m[i]    = నిజం;
                device->mul_mat_s[i]    = నిజం;
                device->mul_mat_id_l[i] = అబద్ధం;
                device->mul_mat_id_m[i] = నిజం;
                device->mul_mat_id_s[i] = నిజం;
                ఆపు;
            స్థితి VK_VENDOR_ID_INTEL: {
                // Current Windows driver does not expose BF16 support.
                // We only want to use l_warptile if coopmat is available
                స్థిరరాశి తార్కిక use_l_warptile = (i == GGML_TYPE_BF16) ? (device->coopmat_bf16_support && device->coopmat_support) : device->coopmat_support;
                device->mul_mat_l[i] = use_l_warptile;
                device->mul_mat_id_l[i] = use_l_warptile;
                device->mul_mat_m[i] = నిజం;
                device->mul_mat_s[i] = నిజం;
                device->mul_mat_id_m[i] = నిజం;
                device->mul_mat_id_s[i] = నిజం;
                ఆపు;
            }
            స్థితి VK_VENDOR_ID_APPLE:
                device->mul_mat_l[i] = అబద్ధం;
                device->mul_mat_m[i] = నిజం;
                device->mul_mat_s[i] = అబద్ధం;
                device->mul_mat_id_l[i] = అబద్ధం;
                device->mul_mat_id_m[i] = నిజం;
                device->mul_mat_id_s[i] = అబద్ధం;
                ఆపు;
            స్థితి VK_VENDOR_ID_QUALCOMM:
                device->mul_mat_l[i] = అబద్ధం;
                device->mul_mat_m[i] = నిజం;
                device->mul_mat_s[i] = !device->coopmat_support;
                device->mul_mat_id_l[i] = అబద్ధం;
                device->mul_mat_id_m[i] = నిజం;
                device->mul_mat_id_s[i] = !device->coopmat_support;
                ఆపు;
            స్థితి VK_VENDOR_ID_SAMSUNG:
                device->mul_mat_l[i] = అబద్ధం;
                device->mul_mat_m[i] = నిజం;
                device->mul_mat_s[i] = నిజం;
                device->mul_mat_id_l[i] = అబద్ధం;
                device->mul_mat_id_m[i] = నిజం;
                device->mul_mat_id_s[i] = నిజం;
                ఆపు;
#అయితే_ముగింపు
            అప్రమేయ:
                device->mul_mat_l[i] = నిజం;
                device->mul_mat_m[i] = నిజం;
                device->mul_mat_s[i] = నిజం;
                device->mul_mat_id_l[i] = నిజం;
                device->mul_mat_id_m[i] = నిజం;
                device->mul_mat_id_s[i] = నిజం;
                ఆపు;
            }

#ఒకవేళ VK_శీర్షిక_సంస్కరణ >= 287
            // Honeykrisp driver for Asahi Linux doesn't report VK_VENDOR_ID_APPLE.
            // Check for Honeykrisp driver and force same configuration as the VK_VENDOR_ID_APPLE case.
            ఒకవేళ (device->driver_id == vk::DriverId::eMesaHoneykrisp) {
                device->mul_mat_l[i] = అబద్ధం;
                device->mul_mat_m[i] = నిజం;
                device->mul_mat_s[i] = అబద్ధం;
                device->mul_mat_id_l[i] = అబద్ధం;
                device->mul_mat_id_m[i] = నిజం;
                device->mul_mat_id_s[i] = అబద్ధం;
            }
#అయితే_ముగింపు

            device->mul_mat_l_int[i]    = device->mul_mat_l[i];
            device->mul_mat_m_int[i]    = device->mul_mat_m[i];
            device->mul_mat_s_int[i]    = device->mul_mat_s[i];
            device->mul_mat_id_l_int[i] = device->mul_mat_id_l[i];
            device->mul_mat_id_m_int[i] = device->mul_mat_id_m[i];
            device->mul_mat_id_s_int[i] = device->mul_mat_id_s[i];
        }


        std::vector<vk::Descriptor_సమితి_Layout_Binding> dsl_binding;
        std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
        కొరకు (uint32_t i = 0; i < MAX_PARAMETER_లెక్క; i++) {
            dsl_binding.push_back({i, vk::DescriptorType::e_Storage_బఫరు, 1, vk::Shader_Stage_Flag_బిట్లు::eCompute});
            dsl_binding_flags.push_back({});
        }

        vk::Descriptor_సమితి_Layout_Binding_Flags_Create_Info dslbfci = { dsl_binding_flags };

        vk::Descriptor_సమితి_Layout_Create_Info descriptor_సమితి_layout_create_info(
            {},
            dsl_binding);
        descriptor_సమితి_layout_create_info.సమితి_PNext(&dslbfci);
        device->dsl = device->device.create_Descriptor_సమితి_Layout(descriptor_సమితి_layout_create_info);

        ggml_vk_ఎక్కించు_shaders(device);

        // Prefer a dedicated transfer queue on AMD dGPUs (non-GCN) when graphics queue use is disabled.
        స్థిరరాశి తార్కిక prefers_transfer_వరుస =
            device->vendor_id == VK_VENDOR_ID_AMD &&
            device->architecture != AMD_GCN &&
            !device->uma &&
            !allow_graphics_వరుస;

        ఒకవేళ (!device->single_వరుస) {
            స్థిరరాశి uint32_t transfer_వరుస_సూచీ = compute_వరుస_family_సూచీ == transfer_వరుస_family_సూచీ ? 1 : 0;
            device->transfer_వరుస = ggml_vk_create_వరుస(device, transfer_వరుస_family_సూచీ, transfer_వరుస_సూచీ, { vk::Pipeline_Stage_Flag_బిట్లు::eTransfer }, నిజం);

            device->async_use_transfer_వరుస = prefers_transfer_వరుస || (getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") != శూన్యసూచిక);
        } లేకపోతే {
            device->transfer_వరుస = ggml_vk_create_aliased_వరుస(device, device->compute_వరుస);

            device->async_use_transfer_వరుస = అబద్ధం;
        }

        device->fence = device->device.createFence({});

        device->బఫరు_type = {
            /* .iface    = */ ggml_backend_vk_బఫరు_type_interface,
            /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), idx),
            /* .context  = */ కొత్త ggml_backend_vk_బఫరు_type_context{ device->name, device },
        };

        device->idx = idx;

        device->serialize_submissions = getenv("GGML_VK_SERIALIZE_SUBMISSIONS") != శూన్యసూచిక;

        device->disable_fusion = getenv("GGML_VK_DISABLE_FUSION") != శూన్యసూచిక;

        device->disable_descriptor_reuse = getenv("GGML_VK_DISABLE_DESCRIPTOR_REUSE") != శూన్యసూచిక;

        device->కూడు_rms_fusion = !device->disable_fusion &&
                                 device->subgroup_arithmetic &&
                                 device->vendor_id != VK_VENDOR_ID_INTEL;
        device->partials_binding_alignment =
            std::max(4u, (uint32_t)device->properties.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment);

        device->mmvq_బహుళకం = 0;
        ఒకవేళ (getenv("GGML_VK_DISABLE_MMVQ")) {
            device->mmvq_బహుళకం = -1;
        } లేకపోతే ఒకవేళ (getenv("GGML_VK_FORCE_MMVQ")) {
            device->mmvq_బహుళకం = 1;
        }

        // publish the device only when it is fully built
        vk_instance.devices[idx] = device;

        తిరిగి device;
    }

    తిరిగి vk_instance.devices[idx];
}

స్థిర శూన్య ggml_vk_print_gpu_info(size_t idx) {
    GGML_ASSERT(idx < vk_instance.device_indices.size());
    size_t dev_అంకె = vk_instance.device_indices[idx];
    VK_LOG_DEBUG("ggml_vk_print_gpu_info(" << dev_అంకె << ")");
    GGML_ASSERT(vk_instance_initialized);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    ఒకవేళ (dev_అంకె >= devices.size()) {
        std::cerr << "ggml_vulkan: Device with index " << dev_అంకె << " does not exist." << std::పంక్తి_ముగింపు;
        విసురు std::runtime_error("Device not found");
    }

    vk::PhysicalDevice physical_device = devices[dev_అంకె];
    std::vector<vk::ExtensionProperties> ext_props = physical_device.enumerateDeviceExtensionProperties();

    తార్కిక fp16_storage = అబద్ధం;
    తార్కిక fp16_compute = అబద్ధం;
    తార్కిక coopmat_support = అబద్ధం;
    తార్కిక coopmat2_support = అబద్ధం;
    తార్కిక coopmat2_decode_సదిశ_support = అబద్ధం;
    తార్కిక integer_dot_లబ్ధం = అబద్ధం;
    తార్కిక bfloat16_support = అబద్ధం;
    తార్కిక dot2_f16_support = అబద్ధం;
    తార్కిక ocp_microscaling_extension = అబద్ధం;
    తార్కిక shader_float8_extension = అబద్ధం;

    కొరకు (స్వయంచాలక properties : ext_props) {
        ఒకవేళ (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
            fp16_storage = నిజం;
        } లేకపోతే ఒకవేళ (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
            fp16_compute = నిజం;
#ఒకవేళ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
       } లేకపోతే ఒకవేళ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT")) {
            coopmat_support = నిజం;
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        } లేకపోతే ఒకవేళ (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2")) {
            coopmat2_support = నిజం;
#అయితే_ముగింపు
        } లేకపోతే ఒకవేళ (strcmp(VK_NV_COOPERATIVE_మాతృక_DECODE_సదిశ_EXTENSION_NAME, properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
            coopmat2_decode_సదిశ_support = నిజం;
#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        } లేకపోతే ఒకవేళ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
            integer_dot_లబ్ధం = నిజం;
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        } లేకపోతే ఒకవేళ (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_BFLOAT16")) {
            bfloat16_support = నిజం;
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
        } లేకపోతే ఒకవేళ (strcmp(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
            ocp_microscaling_extension = నిజం;
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        } లేకపోతే ఒకవేళ (strcmp(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
            shader_float8_extension = నిజం;
#అయితే_ముగింపు
        } లేకపోతే ఒకవేళ (strcmp("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_DOT2")) {
            dot2_f16_support = నిజం;
        }
    }

    స్థిరరాశి vk_device_architecture device_architecture = get_device_architecture(physical_device);

    స్థిరరాశి అక్షరం* GGML_VK_DISABLE_F16 = getenv("GGML_VK_DISABLE_F16");
    తార్కిక force_disable_f16 = GGML_VK_DISABLE_F16 != శూన్యసూచిక;

    తార్కిక fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceMaintenance3Properties props3;
    vk::PhysicalDeviceSubgroupProperties subgroup_props;
    vk::PhysicalDeviceDriverProperties driver_props;
    vk::Physical_Device_Shader_Integer_Dot_లబ్ధం_Properties_KHR shader_integer_dot_లబ్ధం_props;
    props2.p_తదుపరి = &props3;
    props3.p_తదుపరి = &subgroup_props;
    subgroup_props.p_తదుపరి = &driver_props;

    // Pointer to the last chain element
    Vk_పునాది_బహిర్గతం_Structure * చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&driver_props;

    ఒకవేళ (integer_dot_లబ్ధం) {
        చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&shader_integer_dot_లబ్ధం_props;
        చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&shader_integer_dot_లబ్ధం_props;
    }

    physical_device.getProperties2(&props2);

    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.p_తదుపరి = శూన్యసూచిక;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_తదుపరి = శూన్యసూచిక;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_తదుపరి = &vk11_features;

    VkPhysicalDeviceVulkan12Features vk12_features;
    vk12_features.p_తదుపరి = శూన్యసూచిక;
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11_features.p_తదుపరి = &vk12_features;

    // Pointer to the last chain element
    చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&vk12_features;

#ఒకవేళ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    Vk_Physical_Device_Cooperative_మాతృక_Features_KHR coopmat_features;
    coopmat_features.p_తదుపరి = శూన్యసూచిక;
    coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_మాతృక_FEATURES_KHR;
    coopmat_features.cooperative_మాతృక_2 = VK_FALSE;

    ఒకవేళ (coopmat_support) {
        చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat_features;
        చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat_features;
    }
#అయితే_ముగింపు

    Vk_Physical_Device_Shader_Integer_Dot_లబ్ధం_Features_KHR shader_integer_dot_లబ్ధం_features {};
    shader_integer_dot_లబ్ధం_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_లబ్ధం_FEATURES_KHR;
    ఒకవేళ (integer_dot_లబ్ధం) {
        చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&shader_integer_dot_లబ్ధం_features;
        చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&shader_integer_dot_లబ్ధం_features;
    }

#ఒకవేళ defined(VK_KHR_shader_bfloat16)
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
    bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    ఒకవేళ (bfloat16_support) {
        చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&bfloat16_features;
        చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&bfloat16_features;
    }
#అయితే_ముగింపు

#ఒకవేళ defined(VK_NV_cooperative_matrix2)
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
    coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_మాతృక_2_FEATURES_NV;
    ఒకవేళ (coopmat2_support) {
        చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat2_features;
        చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat2_features;
    }
#అయితే_ముగింపు

    Vk_Physical_Device_Cooperative_మాతృక_Decode_సదిశ_Features_NV coopmat2_decode_సదిశ_features {};
    coopmat2_decode_సదిశ_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_మాతృక_DECODE_సదిశ_FEATURES_NV;
    ఒకవేళ (coopmat2_decode_సదిశ_support) {
        చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat2_decode_సదిశ_features;
        చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&coopmat2_decode_సదిశ_features;
    }

    Vk_Physical_Device_Shader_Mixed_Float_Dot_లబ్ధం_Features_VALVE dot2_features {};
    dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_లబ్ధం_FEATURES_VALVE;
    ఒకవేళ (dot2_f16_support) {
        చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&dot2_features;
        చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&dot2_features;
    }

#ఒకవేళ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
    ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
    VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
    shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    ఒకవేళ (ocp_microscaling_extension) {
        చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&ocp_microscaling_features;
        చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&ocp_microscaling_features;
    }
    ఒకవేళ (shader_float8_extension) {
        చివరి_struct->p_తదుపరి = (Vk_పునాది_బహిర్గతం_Structure *)&shader_float8_features;
        చివరి_struct = (Vk_పునాది_బహిర్గతం_Structure *)&shader_float8_features;
    }
#అయితే_ముగింపు

    vkGetPhysicalDeviceFeatures2(physical_device, &device_features2);

    fp16 = fp16 && vk12_features.shaderFloat16;

#ఒకవేళ defined(VK_KHR_shader_bfloat16)
    తార్కిక bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#లేకపోతే
    తార్కిక bf16 = అబద్ధం;
#అయితే_ముగింపు

    uint32_t default_subgroup_కొలత = get_subgroup_కొలత("", device_architecture);
    స్థిరరాశి size_t subgroup_కొలత = (default_subgroup_కొలత != 0) ? default_subgroup_కొలత : subgroup_props.subgroup_కొలత_2;
    స్థిరరాశి తార్కిక uma = props2.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    integer_dot_లబ్ధం = integer_dot_లబ్ధం
                       && shader_integer_dot_లబ్ధం_props.integer_Dot_Product4x8_బిట్టు_Packed_Signed_Accelerated
                       && shader_integer_dot_లబ్ధం_features.shader_Integer_Dot_లబ్ధం;

    coopmat_support = coopmat_support
#ఒకవేళ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                   && coopmat_features.cooperative_మాతృక_2
#అయితే_ముగింపు
                   && ggml_vk_khr_cooperative_మాతృక_support(props2.properties, driver_props, device_architecture);

#ఒకవేళ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    coopmat2_support = coopmat2_support &&
                       coopmat2_features.cooperative_మాతృక_Workgroup_Scope &&
                       coopmat2_features.cooperative_మాతృక_Flexible_Dimensions &&
                       coopmat2_features.cooperative_మాతృక_Reductions &&
                       coopmat2_features.cooperative_మాతృక_Conversions &&
                       coopmat2_features.cooperative_మాతృక_చొప్పున_Element_Operations &&
                       coopmat2_features.cooperative_మాతృక_Tensor_Addressing &&
                       coopmat2_features.cooperative_మాతృక_ఖండం_Loads;
#లేకపోతే
    coopmat2_support = అబద్ధం;
#అయితే_ముగింపు

    coopmat2_decode_సదిశ_support = coopmat2_decode_సదిశ_support && coopmat2_decode_సదిశ_features.cooperative_మాతృక_Decode_సదిశ;
#ఒకవేళ !defined(GGML_VULKAN_COOPMAT2_DECODE_సదిశ_GLSLC_SUPPORT)
    coopmat2_decode_సదిశ_support = అబద్ధం;
#అయితే_ముగింపు

    std::string మాతృక_cores = coopmat2_support ? (coopmat2_decode_సదిశ_support ? "NV_coopmat2v" : "NV_coopmat2")
                             : coopmat_support  ? "KHR_coopmat"
                             : "none";

    తార్కిక dot2_f16 = dot2_f16_support && dot2_features.shader_Mixed_Float_Dot_లబ్ధం_Float16_Acc_Float32;
    స్థిరరాశి అక్షరం *fp16_str = fp16 ? (dot2_f16 ? "dot2" : "1") : "0";
#ఒకవేళ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    స్థిరరాశి తార్కిక fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                     shader_float8_extension && shader_float8_features.shaderFloat8 &&
                     !getenv("GGML_VK_DISABLE_OCP_FP4");
#లేకపోతే
    GGML_వాడని(ocp_microscaling_extension);
    GGML_వాడని(shader_float8_extension);
    స్థిరరాశి తార్కిక fp4 = అబద్ధం;
#అయితే_ముగింపు

    std::string device_name = props2.properties.deviceName.దత్తాంశం();
    GGML_LOG_DEBUG("ggml_vulkan: %zu = %s (%s) | uma: %d | fp16: %s | bf16: %d | fp4: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n",
              idx, device_name.c_str(), driver_props.driverName.దత్తాంశం(), uma, fp16_str, bf16, fp4, subgroup_కొలత,
              props2.properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత, integer_dot_లబ్ధం, మాతృక_cores.c_str());

    ఒకవేళ (props2.properties.deviceType == vk::PhysicalDeviceType::eCpu) {
        GGML_LOG_DEBUG("ggml_vulkan: Warning: Device type is CPU. This is probably not the device you want.\n");
    }
}

స్థిర DispatchLoaderDynamic ggml_vk_default_dispatcher_instance;

DispatchLoaderDynamic & ggml_vk_default_dispatcher() {
    తిరిగి ggml_vk_default_dispatcher_instance;
}

శూన్య ggml_vk_instance_init() {
    ఒకవేళ (vk_instance_initialized) {
        తిరిగి;
    }
    VK_LOG_DEBUG("ggml_vk_instance_init()");

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    ggml_vk_default_dispatcher_instance.init(vkGetInstanceProcAddr);

    // vkEnumerateInstanceVersion is Vulkan 1.1. A null value indicated Vulkan 1.0.
    ఒకవేళ (ggml_vk_default_dispatcher_instance.vk_Enumerate_Instance_సంస్కరణ == శూన్యసూచిక) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        విసురు vk::SystemError(vk::ఫలితం_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    uint32_t api_సంస్కరణ = vk::enumerate_Instance_సంస్కరణ();

    ఒకవేళ (api_సంస్కరణ < VK_API_సంస్కరణ_1_2) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        విసురు vk::SystemError(vk::ఫలితం_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, శూన్యసూచిక, 0, api_సంస్కరణ };

    స్థిరరాశి std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();
    స్థిరరాశి తార్కిక layer_settings = ggml_vk_instance_layer_settings_అందుబాటు();
#నిర్వచితమైతే __APPLE__
    స్థిరరాశి తార్కిక portability_enumeration_ext = ggml_vk_instance_portability_enumeration_ext_అందుబాటు(instance_extensions);
#అయితే_ముగింపు
    స్థిరరాశి తార్కిక debug_utils_ext = ggml_vk_instance_debug_utils_ext_అందుబాటు(instance_extensions) && getenv("GGML_VK_DEBUG_MARKERS") != శూన్యసూచిక;
    std::vector<స్థిరరాశి అక్షరం*> layers;

    ఒకవేళ (layer_settings) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    std::vector<స్థిరరాశి అక్షరం*> extensions;
    ఒకవేళ (layer_settings) {
        extensions.push_back("VK_EXT_layer_settings");
    }
#నిర్వచితమైతే __APPLE__
    ఒకవేళ (portability_enumeration_ext) {
        extensions.push_back("VK_KHR_portability_enumeration");
    }
#అయితే_ముగింపు
    ఒకవేళ (debug_utils_ext) {
        extensions.push_back("VK_EXT_debug_utils");
    }
    VkBool32 enable_ఉత్తమ_practice = layer_settings;
    std::vector<vk::LayerSettingEXT> settings = {
        {
            "VK_LAYER_KHRONOS_validation",
            "validate_best_practices",
            vk::LayerSettingTypeEXT::eBool32,
            1,
            &enable_ఉత్తమ_practice
        },
    };
    vk::LayerSettingsCreateInfoEXT layer_setting_info(settings);
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags{}, &app_info, layers, extensions, &layer_setting_info);
#నిర్వచితమైతే __APPLE__
    ఒకవేళ (portability_enumeration_ext) {
        instance_create_info.flags |= vk::Instance_Create_Flag_బిట్లు::eEnumeratePortabilityKHR;
    }
#అయితే_ముగింపు

    vk_instance.instance = vk::createInstance(instance_create_info);
    vk_instance_initialized = నిజం;

    ఒకవేళ (debug_utils_ext) {
        vk_instance.debug_utils_support              = నిజం;
        vk_instance.pfn_vk_సమితి_Debug_Utils_Object_Name_EXT = (PFN_vk_సమితి_Debug_Utils_Object_Name_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkSetDebugUtilsObjectNameEXT");
        vk_instance.pfn_vk_వరుస_Begin_Debug_Utils_Label_EXT = (PFN_vk_వరుస_Begin_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vk_వరుస_చివర_Debug_Utils_Label_EXT = (PFN_vk_వరుస_చివర_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueEndDebugUtilsLabelEXT");
        vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_చివర_Debug_Utils_Label_EXT =   (PFN_vk_Cmd_చివర_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdEndDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_చొప్పించు_Debug_Utils_Label_EXT = (PFN_vk_Cmd_చొప్పించు_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdInsertDebugUtilsLabelEXT");
    }

    vk_perf_logger_enabled = getenv("GGML_VK_PERF_LOGGER") != శూన్యసూచిక;
    vk_perf_logger_concurrent = getenv("GGML_VK_PERF_LOGGER_CONCURRENT") != శూన్యసూచిక;
    vk_enable_sync_logger = getenv("GGML_VK_SYNC_LOGGER") != శూన్యసూచిక;
    vk_జ్ఞాపకం_logger_enabled = getenv("GGML_VK_MEMORY_LOGGER") != శూన్యసూచిక;
    స్థిరరాశి అక్షరం* GGML_VK_PIPELINE_STATS = getenv("GGML_VK_PIPELINE_STATS");
    ఒకవేళ (GGML_VK_PIPELINE_STATS != శూన్యసూచిక) {
        vk_pipeline_stats_filter = GGML_VK_PIPELINE_STATS;
    }
    స్థిరరాశి అక్షరం* GGML_VK_PERF_LOGGER_FREQUENCY = getenv("GGML_VK_PERF_LOGGER_FREQUENCY");

    ఒకవేళ (GGML_VK_PERF_LOGGER_FREQUENCY != శూన్యసూచిక) {
        vk_perf_logger_frequency = std::stoul(GGML_VK_PERF_LOGGER_FREQUENCY);
    }

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance.instance);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    // Emulate behavior of CUDA_VISIBLE_DEVICES for Vulkan
    అక్షరం * devices_env = getenv("GGML_VK_VISIBLE_DEVICES");
    ఒకవేళ (devices_env != శూన్యసూచిక) {
        size_t అంకె_అందుబాటు_devices = devices.size();

        std::string devices(devices_env);
        std::replace(devices.begin(), devices.end(), ',', ' ');

        std::stringstream ss(devices);
        size_t tmp;
        అయితే (ss >> tmp) {
            ఒకవేళ(tmp >= అంకె_అందుబాటు_devices) {
                std::cerr << "ggml_vulkan: Invalid device index " << tmp << " in GGML_VK_VISIBLE_DEVICES." << std::పంక్తి_ముగింపు;
                విసురు std::runtime_error("Invalid Vulkan device index");
            }
            vk_instance.device_indices.push_back(tmp);
        }
    } లేకపోతే {
        // If no vulkan devices are found, return early
        ఒకవేళ (devices.ఖాళీలు()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            తిరిగి;
        }

        // Default to using all dedicated GPUs
        కొరకు (size_t i = 0; i < devices.size(); i++) {
            vk::PhysicalDeviceProperties2 new_props;
            vk::PhysicalDeviceDriverProperties new_driver;
            vk::PhysicalDeviceIDProperties new_id;
            new_props.p_తదుపరి = &new_driver;
            new_driver.p_తదుపరి = &new_id;
            devices[i].getProperties2(&new_props);

            ఒకవేళ ((new_props.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu || new_props.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) && ggml_vk_device_is_supported(devices[i])) {
                // Check if there are two physical devices corresponding to the same GPU
                // This handles the case where the same GPU appears with different drivers (e.g., RADV + AMDVLK on Linux),
                // see https://github.com/ggml-org/llama.cpp/pull/7582 for original deduplication.
                // MoltenVK on macOS may report the same UUID for distinct GPUs on multi-GPU cards,
                // see https://github.com/KhronosGroup/MoltenVK/issues/2683. Skip when both old/new
                // driver is MoltenVK
                స్వయంచాలక old_device = std::find_if(
                    vk_instance.device_indices.begin(),
                    vk_instance.device_indices.end(),
                    [&devices, &new_id, &new_driver](స్థిరరాశి size_t k){
                        vk::PhysicalDeviceProperties2 old_props;
                        vk::PhysicalDeviceDriverProperties old_driver;
                        vk::PhysicalDeviceIDProperties old_id;
                        old_props.p_తదుపరి = &old_driver;
                        old_driver.p_తదుపరి = &old_id;
                        devices[k].getProperties2(&old_props);

                        తార్కిక same_uuid = std::equal(std::begin(old_id.deviceUUID), std::end(old_id.deviceUUID), std::begin(new_id.deviceUUID));
                        same_uuid = same_uuid || (
                            old_id.deviceLUIDValid && new_id.deviceLUIDValid &&
                            std::equal(std::begin(old_id.deviceLUID), std::end(old_id.deviceLUID), std::begin(new_id.deviceLUID))
                        );
                        తార్కిక both_molten_vk = (new_driver.driverID == vk::DriverId::eMoltenvk && old_driver.driverID == vk::DriverId::eMoltenvk);

                        తిరిగి same_uuid && !both_molten_vk;
                    }
                );
                ఒకవేళ (old_device == vk_instance.device_indices.end()) {
                    vk_instance.device_indices.push_back(i);
                } లేకపోతే {
                    // There can be two physical devices corresponding to the same GPU if there are 2 different drivers
                    // This can cause error when splitting layers aross the devices, need to keep only 1
                    VK_LOG_DEBUG("Device " << i << " and device " << *old_device << " have the same deviceUUID");

                    vk::PhysicalDeviceProperties2 old_props;
                    vk::PhysicalDeviceDriverProperties old_driver;
                    old_props.p_తదుపరి = &old_driver;
                    devices[*old_device].getProperties2(&old_props);

                    std::map<vk::DriverId, పూర్ణాంకం> driver_priorities {};
                    పూర్ణాంకం old_ప్రాధాన్యత = std::numeric_limits<పూర్ణాంకం>::max();
                    పూర్ణాంకం new_ప్రాధాన్యత = std::numeric_limits<పూర్ణాంకం>::max();

                    // Check https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDriverId.html for the list of driver id
                    // Smaller number -> higher priority
                    ఎంపిక (old_props.properties.vendorID) {
                        స్థితి VK_VENDOR_ID_AMD:
                            driver_priorities[vk::DriverId::eMesaRadv] = 1;
                            driver_priorities[vk::DriverId::e_Amd_Open_ఆధారం] = 2;
                            driver_priorities[vk::DriverId::eAmdProprietary] = 3;
                            ఆపు;
                        స్థితి VK_VENDOR_ID_INTEL:
                            driver_priorities[vk::DriverId::e_Intel_Open_ఆధారం_MESA] = 1;
                            driver_priorities[vk::DriverId::eIntelProprietaryWindows] = 2;
                            ఆపు;
                        స్థితి VK_VENDOR_ID_NVIDIA:
                            driver_priorities[vk::DriverId::eNvidiaProprietary] = 1;
#ఒకవేళ defined(VK_API_సంస్కరణ_1_3) && VK_శీర్షిక_సంస్కరణ >= 235
                            driver_priorities[vk::DriverId::eMesaNvk] = 2;
#అయితే_ముగింపు
                            ఆపు;
                        స్థితి VK_VENDOR_ID_QUALCOMM:
                            driver_priorities[vk::DriverId::eQualcommProprietary] = 1;
                            driver_priorities[vk::DriverId::eMesaTurnip] = 2;
                            ఆపు;
                    }
                    driver_priorities[vk::DriverId::eMesaDozen] = 100;

                    ఒకవేళ (driver_priorities.count(old_driver.driverID)) {
                        old_ప్రాధాన్యత = driver_priorities[old_driver.driverID];
                    }
                    ఒకవేళ (driver_priorities.count(new_driver.driverID)) {
                        new_ప్రాధాన్యత = driver_priorities[new_driver.driverID];
                    }

                    ఒకవేళ (new_ప్రాధాన్యత < old_ప్రాధాన్యత) {
                        స్వయంచాలక r = std::remove(vk_instance.device_indices.begin(), vk_instance.device_indices.end(), *old_device);
                        vk_instance.device_indices.erase(r, vk_instance.device_indices.end());
                        vk_instance.device_indices.push_back(i);

                        VK_LOG_DEBUG("Prioritize device " << i << " driver " << new_driver.driverName << " over device " << *old_device << " driver " << old_driver.driverName);
                    }
                    లేకపోతే {
                        VK_LOG_DEBUG("Prioritize device " << *old_device << " driver " << old_driver.driverName << " over device " << i << " driver " << new_driver.driverName << std::పంక్తి_ముగింపు);
                    }
                }
            }
        }

        // If no GPUs found, fall back to the first non-CPU device.
        // If only CPU devices are available, return without devices.
        ఒకవేళ (vk_instance.device_indices.ఖాళీలు()) {
            కొరకు (size_t i = 0; i < devices.size(); i++) {
                ఒకవేళ (devices[i].getProperties().deviceType != vk::PhysicalDeviceType::eCpu) {
                    vk_instance.device_indices.push_back(i);
                    ఆపు;
                }
            }
        }

        ఒకవేళ (vk_instance.device_indices.ఖాళీలు()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            తిరిగి;
        }
    }
    GGML_LOG_DEBUG("ggml_vulkan: Found %zu Vulkan devices:\n", vk_instance.device_indices.size());

    size_t host_బఫరు_import_లెక్క = 0;
    కొరకు (size_t i = 0; i < vk_instance.device_indices.size(); i++) {
        vk::PhysicalDevice vkdev = devices[vk_instance.device_indices[i]];
        std::vector<vk::ExtensionProperties> extensionprops = vkdev.enumerateDeviceExtensionProperties();

        తార్కిక membudget_supported = అబద్ధం;
        తార్కిక external_జ్ఞాపకం_host_supported = అబద్ధం;
        కొరకు (స్థిరరాశి స్వయంచాలక & ext : extensionprops) {
            ఒకవేళ (strcmp(VK_EXT_జ్ఞాపకం_BUDGET_EXTENSION_NAME, ext.extensionName) == 0) {
                membudget_supported = నిజం;
            } లేకపోతే ఒకవేళ (strcmp("VK_EXT_external_memory_host", ext.extensionName) == 0) {
                external_జ్ఞాపకం_host_supported = నిజం;
            }
        }

        vk_instance.device_supports_membudget.push_back(membudget_supported);

        తార్కిక host_బఫరు_import = అబద్ధం;
        ఒకవేళ (external_జ్ఞాపకం_host_supported) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceDriverProperties driver_props;
            vk::Physical_Device_External_జ్ఞాపకం_Host_Properties_EXT ext_props;
            props2.p_తదుపరి = &driver_props;
            driver_props.p_తదుపరి = &ext_props;
            vkdev.getProperties2(&props2);
            host_బఫరు_import = ggml_vk_external_జ్ఞాపకం_host_usable(driver_props.driverID) &&
                                 ggml_vk_host_బఫరు_import_usable(props2.properties.vendorID);
            ఒకవేళ (host_బఫరు_import) {
                vk_instance.host_import_alignment = std::max<size_t>(vk_instance.host_import_alignment, ext_props.min_Imported_Host_సూచిక_Alignment);
                host_బఫరు_import_లెక్క++;
            }
        }
        vk_instance.device_host_బఫరు_import.push_back(host_బఫరు_import);

        ggml_vk_print_gpu_info(i);
    }

    vk_instance.multi_device = host_బఫరు_import_లెక్క > 1 && vk_instance.device_host_బఫరు_import[0];
}

శూన్య ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx) {
    VK_LOG_DEBUG("ggml_vk_init(" << ctx->name << ", " << idx << ")");
    ggml_vk_instance_init();
    GGML_ASSERT(idx < vk_instance.device_indices.size());

    ctx->name = GGML_VK_NAME + std::to_string(idx);

    ctx->device = ggml_vk_get_device(idx);

    ctx->semaphore_idx = 0;
    ctx->event_idx = 0;

    ctx->prealloc_కొలత_x = 0;
    ctx->prealloc_కొలత_y = 0;
    ctx->prealloc_కొలత_విభజించు_k = 0;
    // Fixed size of 1KB, for deterministic behavior
    ctx->prealloc_కొలత_కూడు_rms_partials = 1024;

    ctx->fence = ctx->device->device.createFence({});
    ctx->almost_ready_fence = ctx->device->device.createFence({});

    ctx->compute_cmd_pool.init(ctx->device, ctx->device->compute_వరుస.get());
    ఒకవేళ (ctx->device->async_use_transfer_వరుస) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.సమితి_PNext(&tci);
        ctx->transfer_semaphore.s = ctx->device->device.createSemaphore(ci);
        ctx->transfer_semaphore.value = 0;

        ctx->transfer_cmd_pool.init(ctx->device, ctx->device->transfer_వరుస.get());
    }

    ఒకవేళ (vk_perf_logger_enabled) {
        ctx->perf_logger = std::unique_ptr<vk_perf_logger>(కొత్త vk_perf_logger());
    }

#నిర్వచితమైతే GGML_VULKAN_CHECK_RESULTS
    స్థిరరాశి అక్షరం* skip_checks = getenv("GGML_VULKAN_SKIP_CHECKS");
    vk_skip_checks = (skip_checks == ఖాళీ ? 0 : atoi(skip_checks));
    స్థిరరాశి అక్షరం* output_tensor = getenv("GGML_VULKAN_OUTPUT_TENSOR");
    vk_output_tensor = (output_tensor == ఖాళీ ? 0 : atoi(output_tensor));
#అయితే_ముగింపు
}

vk_pipeline ggml_vk_get_గమ్యం_fp16(ggml_backend_vk_context * ctx, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_get_to_fp16()");
    ఎంపిక (type) {
        స్థితి GGML_TYPE_F32:
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ2_S:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ3_S:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_NVFP4:
        స్థితి GGML_TYPE_TQ1_0:
        స్థితి GGML_TYPE_TQ2_0:
            ఆపు;
        అప్రమేయ:
            తిరిగి శూన్యసూచిక;
    }

    తిరిగి ctx->device->pipeline_dequant[type];
}

స్థిర vk_pipeline ggml_vk_get_dequantize_mul_mat_vec(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t అంకె_నిలువుల_సంఖ్య, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_F16 || b_type == GGML_TYPE_Q8_1);
    GGML_ASSERT(అంకె_నిలువుల_సంఖ్య >= 1 && అంకె_నిలువుల_సంఖ్య <= mul_mat_vec_max_నిలువుల_సంఖ్య);

    ఒకవేళ (b_type == GGML_TYPE_Q8_1) {
        ఎంపిక (a_type) {
            స్థితి GGML_TYPE_Q2_0:
            స్థితి GGML_TYPE_Q4_0:
            స్థితి GGML_TYPE_Q4_1:
            స్థితి GGML_TYPE_Q5_0:
            స్థితి GGML_TYPE_Q5_1:
            స్థితి GGML_TYPE_Q8_0:
            స్థితి GGML_TYPE_MXFP4:
            స్థితి GGML_TYPE_Q2_K:
            స్థితి GGML_TYPE_Q3_K:
            స్థితి GGML_TYPE_Q4_K:
            స్థితి GGML_TYPE_Q5_K:
            స్థితి GGML_TYPE_Q6_K:
            స్థితి GGML_TYPE_IQ1_S:
            స్థితి GGML_TYPE_IQ1_M:
            స్థితి GGML_TYPE_IQ4_XS:
                ఆపు;
            అప్రమేయ:
                తిరిగి శూన్యసూచిక;
        }
    }

    ఎంపిక (a_type) {
        స్థితి GGML_TYPE_F32:
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_BF16:
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ2_S:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ3_S:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_NVFP4:
        స్థితి GGML_TYPE_TQ1_0:
        స్థితి GGML_TYPE_TQ2_0:
            ఆపు;
        అప్రమేయ:
            తిరిగి శూన్యసూచిక;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_కొలత_SUBGROUP;
    ఒకవేళ ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        ఒకవేళ (a_type == GGML_TYPE_Q6_K) {
            ఒకవేళ (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_కొలత_LARGE;
            }
        } లేకపోతే {
            ఒకవేళ (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_కొలత_LARGE;
            }
        }
    }

    ఒకవేళ (b_type == GGML_TYPE_Q8_1) {
        ఒకవేళ (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_కొలత_SUBGROUP;
        }
        తిరిగి ctx->device->pipeline_dequant_mul_mat_vec_q8_1_f32[dmmv_wg][a_type][అంకె_నిలువుల_సంఖ్య-1];
    }

    తిరిగి b_type == GGML_TYPE_F32 ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[dmmv_wg][a_type][అంకె_నిలువుల_సంఖ్య-1] : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[dmmv_wg][a_type][అంకె_నిలువుల_సంఖ్య-1];
}

స్థిర vk_pipeline ggml_vk_get_dequantize_mul_mat_vec_id(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec_id()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_Q8_1);

    ఒకవేళ (b_type == GGML_TYPE_Q8_1) {
        ఎంపిక (a_type) {
            స్థితి GGML_TYPE_Q2_0:
            స్థితి GGML_TYPE_Q4_0:
            స్థితి GGML_TYPE_Q4_1:
            స్థితి GGML_TYPE_Q5_0:
            స్థితి GGML_TYPE_Q5_1:
            స్థితి GGML_TYPE_Q8_0:
            స్థితి GGML_TYPE_MXFP4:
            స్థితి GGML_TYPE_Q2_K:
            స్థితి GGML_TYPE_Q3_K:
            స్థితి GGML_TYPE_Q4_K:
            స్థితి GGML_TYPE_Q5_K:
            స్థితి GGML_TYPE_Q6_K:
            స్థితి GGML_TYPE_IQ1_S:
            స్థితి GGML_TYPE_IQ1_M:
            స్థితి GGML_TYPE_IQ4_XS:
                ఆపు;
            అప్రమేయ:
                తిరిగి శూన్యసూచిక;
        }
    }

    ఎంపిక (a_type) {
        స్థితి GGML_TYPE_F32:
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_BF16:
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ2_S:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ3_S:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_NVFP4:
        స్థితి GGML_TYPE_TQ1_0:
        స్థితి GGML_TYPE_TQ2_0:
            ఆపు;
        అప్రమేయ:
            తిరిగి శూన్యసూచిక;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_కొలత_SUBGROUP;
    ఒకవేళ ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        ఒకవేళ (a_type == GGML_TYPE_Q6_K) {
            ఒకవేళ (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_కొలత_LARGE;
            }
        } లేకపోతే {
            ఒకవేళ (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_కొలత_LARGE;
            }
        }
    }

    ఒకవేళ (b_type == GGML_TYPE_Q8_1) {
        ఒకవేళ (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_కొలత_SUBGROUP;
        }
        తిరిగి ctx->device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[dmmv_wg][a_type];
    }

    తిరిగి ctx->device->pipeline_dequant_mul_mat_vec_id_f32[dmmv_wg][a_type];
}

vk_subbuffer ggml_vk_tensor_subbuffer(
    స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor * tensor, తార్కిక allow_misalign) {

    vk_బఫరు బఫరు = శూన్యసూచిక;
    size_t స్థానభ్రంశం = 0;
    ఒకవేళ (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, tensor->దత్తాంశం, బఫరు, స్థానభ్రంశం);
    }
    ఒకవేళ (!బఫరు) {
        స్వయంచాలక buf_ctx = (ggml_backend_vk_బఫరు_context *)tensor->బఫరు->context;
        బఫరు = buf_ctx->dev_బఫరు;
        స్థానభ్రంశం = vk_tensor_స్థానభ్రంశం(tensor) + tensor->view_offs;
    }
    GGML_ASSERT(బఫరు != శూన్యసూచిక);

    size_t size = ggml_nbytes(tensor);

    స్థిరరాశి size_t descriptor_స్థానభ్రంశం = ggml_vk_descriptor_స్థానభ్రంశం(
        స్థానభ్రంశం, ctx->device->properties.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment, ggml_type_కొలత(tensor->type));
    స్థిరరాశి size_t misalign_బైట్లు = స్థానభ్రంశం - descriptor_స్థానభ్రంశం;
    // The shader must support misaligned offsets when indexing into the buffer
    GGML_ASSERT(allow_misalign || misalign_బైట్లు == 0);
    స్థానభ్రంశం = descriptor_స్థానభ్రంశం;
    size += misalign_బైట్లు;

    తిరిగి vk_subbuffer{బఫరు, స్థానభ్రంశం, size};
}

స్థిర vk_command_బఫరు* ggml_vk_get_or_create_cmd_బఫరు(vk_device& device, vk_command_pool& pool) {
    కొరకు (స్వయంచాలక& cmd_బఫరు : pool.cmd_buffers) {
        ఒకవేళ (!cmd_బఫరు.లోపల_use) {
            cmd_బఫరు.use_counter++;
            cmd_బఫరు.లోపల_use = నిజం;
            తిరిగి &cmd_బఫరు;
        }
    }
    తిరిగి ggml_vk_create_cmd_బఫరు(device, pool);
}

స్థిర vk_submission ggml_vk_begin_submission(vk_device& device, vk_command_pool& p, తార్కిక one_సమయం = నిజం) {
    vk_submission s;
    s.బఫరు = ggml_vk_get_or_create_cmd_బఫరు(device, p);
    ఒకవేళ (one_సమయం) {
        s.బఫరు->buf.begin({ vk::Command_బఫరు_Usage_Flag_బిట్లు::e_One_సమయం_Submit });
    } లేకపోతే {
        s.బఫరు->buf.begin({ vk::Command_బఫరు_Usage_Flags{} });
    }

    తిరిగి s;
}

శూన్య ggml_vk_cmd_label_begin(vk::Command_బఫరు buf, స్థిరరాశి అక్షరం * name) {
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<దశాంశం, 4>{1.0f, 1.0f, 1.0f, 1.0f};
    vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT(buf, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

శూన్య ggml_vk_ctx_చివర(vk_context& ctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_end(" << ctx << ", " << ctx->seqs.size() << ")");
    ఒకవేళ (ctx->s == శూన్యసూచిక) {
        తిరిగి;
    }

    // close open labels so this buffer is balanced; reopened in ggml_vk_ctx_begin
    ఒకవేళ (vk_instance.debug_utils_support) {
        కొరకు (size_t i = 0; i < ctx->debug_labels.size(); i++) {
            vk_instance.pfn_vk_Cmd_చివర_Debug_Utils_Label_EXT(ctx->s->బఫరు->buf);
        }
        // the enclosing per-command-buffer region
        vk_instance.pfn_vk_Cmd_చివర_Debug_Utils_Label_EXT(ctx->s->బఫరు->buf);
    }

    ctx->s->బఫరు->buf.end();
    ctx->s = శూన్యసూచిక;
}

శూన్య ggml_vk_ctx_begin(vk_device& device, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_begin(" << device->name << ")");
    ఒకవేళ (subctx->s != శూన్యసూచిక) {
        ggml_vk_ctx_చివర(subctx);
    }

    subctx->seqs.push_back({ ggml_vk_begin_submission(device, *subctx->p) });
    subctx->s = subctx->seqs[subctx->seqs.size() - 1].దత్తాంశం();

    ఒకవేళ (vk_instance.debug_utils_support) {
        // outermost region, one per command buffer, so the gaps between submits stand out
        స్థిరరాశి std::string name = "submit " + std::to_string(device->debug_cmdbuf_idx++);
        ggml_vk_cmd_label_begin(subctx->s->బఫరు->buf, name.c_str());

        // reopen labels left open when the previous command buffer was submitted
        కొరకు (స్థిరరాశి std::string & label : subctx->debug_labels) {
            ggml_vk_cmd_label_begin(subctx->s->బఫరు->buf, label.c_str());
        }
    }
}

vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx) {
    vk_context ఫలితం;
    ఒకవేళ (!ctx->compute_ctx.expired()) {
        ఫలితం = ctx->compute_ctx.తాళం();
    } లేకపోతే {
        ఫలితం = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);

        ctx->compute_ctx = ఫలితం;
        ggml_vk_ctx_begin(ctx->device, ఫలితం);
    }

    ఒకవేళ (ctx->device->async_use_transfer_వరుస && ctx->transfer_semaphore_చివరి_submitted < ctx->transfer_semaphore.value) {
        ఫలితం->s->wait_semaphores.push_back(ctx->transfer_semaphore);
        ctx->transfer_semaphore_చివరి_submitted = ctx->transfer_semaphore.value;
    }

    తిరిగి ఫలితం;
}

vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx) {
    vk_context ఫలితం;
    ఒకవేళ (!ctx->transfer_ctx.expired()) {
        ఫలితం = ctx->transfer_ctx.తాళం();
    } లేకపోతే {
        ఫలితం = ggml_vk_create_context(ctx, ctx->transfer_cmd_pool);

        ctx->transfer_ctx = ఫలితం;
        ggml_vk_ctx_begin(ctx->device, ఫలితం);
    }

    తిరిగి ఫలితం;
}

తార్కిక ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx) {
    ఒకవేళ (!ctx->device->async_use_transfer_వరుస || ctx->transfer_ctx.expired()) {
        తిరిగి అబద్ధం;
    }

    vk_context cpy_ctx = ctx->transfer_ctx.తాళం();
    ggml_vk_ctx_చివర(cpy_ctx);

    కొరకు (స్వయంచాలక& cpy : cpy_ctx->లోపల_memcpys) {
        జ్ఞాపకం_నకలు(cpy.dst, cpy.src, cpy.n);
    }

    ctx->transfer_semaphore.value++;
    cpy_ctx->seqs.back().back().signal_semaphores.push_back(ctx->transfer_semaphore);

    ggml_vk_submit(cpy_ctx, {});
    ctx->transfer_ctx.reset();
    తిరిగి నిజం;
}

size_t ggml_vk_align_కొలత(size_t వెడల్పు, size_t align) {
    VK_LOG_DEBUG("ggml_vk_align_size(" << వెడల్పు << ", " << align << ")");
    తిరిగి CEIL_DIV(వెడల్పు, align) * align;
}

శూన్య deferred_memcpy(శూన్య * dst, స్థిరరాశి శూన్య * src, size_t size, std::vector<vk_staging_memcpy>* memcpys) {
    ఒకవేళ (memcpys == శూన్యసూచిక) {
        జ్ఞాపకం_నకలు(dst, src, size);
    } లేకపోతే {
        memcpys->emplace_back(dst, src, size);
    }
}

శూన్య deferred_memset(శూన్య * dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets) {
    ఒకవేళ (memsets == శూన్యసూచిక) {
        memset(dst, val, size);
    } లేకపోతే {
        memsets->emplace_back(dst, val, size);
    }
}

స్థిర uint32_t ggml_vk_guess_విభజించు_k(ggml_backend_vk_context * ctx, uint32_t m, uint32_t n, uint32_t k, తార్కిక disable_విభజించు_k, స్థిరరాశి vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_vk_guess_split_k(" << m << ", " << n << ", " << k << ", " << disable_విభజించు_k << ")");

    ఒకవేళ (disable_విభజించు_k) {
        తిరిగి 1;
    }

    uint32_t విభజించు_k = 1;
    ఒకవేళ (ctx->device->shader_core_లెక్క != 0 && n >= pipeline->wg_denoms[1]) {
        // If k is 'large' and the SMs will fill less than halfway, use split_k.
        uint32_t m_tiles = CEIL_DIV(m, pipeline->wg_denoms[0]);
        uint32_t n_tiles = CEIL_DIV(n, pipeline->wg_denoms[1]);

        ఒకవేళ (k >= 2048) {
            ఒకవేళ (m_tiles * n_tiles <= ctx->device->shader_core_లెక్క / 2) {
                విభజించు_k = ctx->device->shader_core_లెక్క / (m_tiles * n_tiles);
            } లేకపోతే ఒకవేళ (m_tiles * n_tiles <= ctx->device->shader_core_లెక్క * 2 / 3) {
                విభజించు_k = 3;
            }
            // Cap the split at 8x. Unless k is huge this is a lot of overhead.
            విభజించు_k = std::min(విభజించు_k, 8u);

            // ggml_vk_matmul will align the splits to be a multiple of 256.
            // If this rounded up size would cause the last split to be empty,
            // then reduce the split count.
            అయితే (నిజం) {
                ఒకవేళ (విభజించు_k == 1) {
                    ఆపు;
                }
                uint32_t k_విభజించు = CEIL_DIV(k, విభజించు_k);
                k_విభజించు = ROUNDUP_POW2(k_విభజించు, 256);
                ఒకవేళ (k_విభజించు * (విభజించు_k - 1) < k) {
                    ఆపు;
                }
                విభజించు_k--;
            }
        }
    }

    తిరిగి విభజించు_k;
}

శూన్య ggml_vk_matmul(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& విభజించు_k_బఫరు,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t విభజించు_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3,
        uint32_t padded_n, uint64_t b_నిలువు_బైట్లు, uint64_t d_నిలువు_బైట్లు) {
        VK_LOG_DEBUG("ggml_vk_matmul(a: (" << a.బఫరు->బఫరు << ", " << a.స్థానభ్రంశం << ", " << a.size << "), b: (" << b.బఫరు->బఫరు << ", " << b.స్థానభ్రంశం << ", " << b.size << "), d: (" << d.బఫరు->బఫరు << ", " << d.స్థానభ్రంశం << ", " << d.size << "), split_k: (" << (విభజించు_k_బఫరు.బఫరు != శూన్యసూచిక ? విభజించు_k_బఫరు.బఫరు->బఫరు : VK_NULL_HANDLE) << ", " << విభజించు_k_బఫరు.స్థానభ్రంశం << ", " << విభజించు_k_బఫరు.size << "), m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", split_k: " << విభజించు_k << ", batch: " << batch << ", ne02: " << ne02 << ", ne12: " << ne12 << ", broadcast2: " << broadcast2 << ", broadcast3: " << broadcast3 << ", padded_n: " << padded_n << ")");
    ఒకవేళ (విభజించు_k == 1) {
        // a small tile can still produce more workgroups than the device allows on the n axis:
        // split the dispatch into n-chunks, shifting the B and D column windows
        స్థిరరాశి uint64_t max_n = (uint64_t)ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1] * pipeline->wg_denoms[1];
        స్థిరరాశి uint64_t n_chunks = CEIL_DIV(n, max_n);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, n_chunks * CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]));

        కొరకు (uint64_t n_పునాది = 0; n_పునాది < n; n_పునాది += max_n) {
            స్థిరరాశి uint32_t n_chunk = (uint32_t) std::min(n - n_పునాది, max_n);
            GGML_ASSERT(n_పునాది * b_నిలువు_బైట్లు < b.size && n_పునాది * d_నిలువు_బైట్లు < d.size);
            GGML_ASSERT(b.size != VK_సంపూర్ణ_కొలత && d.size != VK_సంపూర్ణ_కొలత);
            vk_subbuffer b_chunk = b;
            vk_subbuffer d_chunk = d;
            b_chunk.స్థానభ్రంశం += n_పునాది * b_నిలువు_బైట్లు;
            b_chunk.size   -= n_పునాది * b_నిలువు_బైట్లు;
            d_chunk.స్థానభ్రంశం += n_పునాది * d_నిలువు_బైట్లు;
            d_chunk.size   -= n_పునాది * d_నిలువు_బైట్లు;

            uint32_t పునాది_work_సమూహం_z = 0;
            అయితే (పునాది_work_సమూహం_z < batch) {
                uint32_t groups_z = std::min(batch - పునాది_work_సమూహం_z, ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]);

                స్థిరరాశి vk_mat_mat_నెట్టు_constants pc = { m, n_chunk, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, పునాది_work_సమూహం_z, batch, k, ne02, ne12, broadcast2, broadcast3, uint32_t(padded_n - n_పునాది) };
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b_chunk, d_chunk }, pc, { m, n_chunk, groups_z });
                పునాది_work_సమూహం_z += groups_z;
            }
        }
        తిరిగి;
    }

    ఒకవేళ (ctx->prealloc_విభజించు_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    GGML_ASSERT(batch_stride_d == m * n);

    // Round the split size up to a multiple of 256 (k-quant alignment)
    uint32_t k_విభజించు = CEIL_DIV(k, విభజించు_k);
    k_విభజించు = ROUNDUP_POW2(k_విభజించు, 256);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]));

    uint32_t పునాది_work_సమూహం_z = 0;
    అయితే (పునాది_work_సమూహం_z < batch) {
        uint32_t groups_z = std::min(batch - పునాది_work_సమూహం_z, ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]);

        స్థిరరాశి vk_mat_mat_నెట్టు_constants pc1 = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, పునాది_work_సమూహం_z, batch, k_విభజించు, ne02, ne12, broadcast2, broadcast3, padded_n };
        // Make sure enough workgroups get assigned for split k to work
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, విభజించు_k_బఫరు }, pc1, { (CEIL_DIV(m, pipeline->wg_denoms[0]) * pipeline->wg_denoms[0]) * విభజించు_k, n, groups_z });
        పునాది_work_సమూహం_z += groups_z;
    }
    ggml_vk_sync_buffers(ctx, subctx);
    స్థిరరాశి std::array<uint32_t, 2> pc2 = { (uint32_t)(m * n * batch), విభజించు_k };
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_matmul_విభజించు_k_reduce, { విభజించు_k_బఫరు, d }, pc2, { m * n * batch, 1, 1 });
    ctx->prealloc_విభజించు_k_need_sync = నిజం;
}

స్థిర తార్కిక ggml_vk_get_mul_mat_mat_f16acc(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {
    ఒకవేళ (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16) తిరిగి అబద్ధం;
    ఒకవేళ (src1_type == GGML_TYPE_Q8_1) తిరిగి అబద్ధం;
    ఒకవేళ (src0_type == GGML_TYPE_F16) {
        తిరిగి prec == GGML_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support);
    }
    // quant types
    ఒకవేళ (ctx->device->coopmat2) {
        తిరిగి prec == GGML_PREC_DEFAULT;
    }
    ఒకవేళ (ctx->device->coopmat_support) {
        తిరిగి ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT;
    }
    తిరిగి ctx->device->fp16 && prec == GGML_PREC_DEFAULT;
}

స్థిర స్థిరరాశి std::vector<vk_matmul_pipeline_pair>* ggml_vk_get_mul_mat_mat_pipeline_map(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec, తార్కిక mul_mat_id = అబద్ధం) {
    తార్కిక f16acc = ggml_vk_get_mul_mat_mat_f16acc(ctx, src0_type, src1_type, prec);
    vk_matmul_pipeline_కీలకం కీలకం{src0_type, src1_type, mul_mat_id, f16acc};
    స్వయంచాలక it = ctx->device->pipeline_matmul.find(కీలకం);
    ఒకవేళ (it == ctx->device->pipeline_matmul.end() || it->second.ఖాళీలు()) {
        // Try without f16acc
        ఒకవేళ (f16acc) {
            కీలకం.f16acc = అబద్ధం;
            it = ctx->device->pipeline_matmul.find(కీలకం);
            ఒకవేళ (it != ctx->device->pipeline_matmul.end() && !it->second.ఖాళీలు()) తిరిగి &it->second;
        }
        తిరిగి శూన్యసూచిక;
    }
    తిరిగి &it->second;
}

స్థిర vk_pipeline ggml_vk_guess_matmul_pipeline_map(ggml_backend_vk_context * ctx,
        స్థిరరాశి std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, తార్కిక aligned, తార్కిక mul_mat_id) {
    స్వయంచాలక& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_లెక్క, configs);
    ఒకవేళ (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    తిరిగి (aligned && configs[idx].aligned) ? configs[idx].aligned : configs[idx].unaligned;
}

స్థిర uint32_t ggml_vk_guess_matmul_pipeline_align_map(ggml_backend_vk_context * ctx,
        స్థిరరాశి std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, తార్కిక mul_mat_id) {
    స్వయంచాలక& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_లెక్క, configs);
    ఒకవేళ (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    తిరిగి configs[idx].align;
}

స్థిర uint64_t ggml_vk_mul_mat_id_max_tiles(uint64_t n_experts, uint64_t n_అడ్డువరుసలు, uint32_t bn) {
    // Each nonempty expert can add at most one partial tile to the combined grid.
    తిరిగి std::min(n_అడ్డువరుసలు, CEIL_DIV(n_అడ్డువరుసలు, bn) + std::min(n_experts, n_అడ్డువరుసలు) - 1);
}

స్థిర శూన్య ggml_vk_matmul_id(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ids, స్థిరరాశి vk_subbuffer & expert_లెక్క_buf,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t n_as, uint32_t nei0, uint32_t nei1, uint32_t nbi1, uint32_t ne11,
        uint32_t అడ్డువరుస_ids_స్థానభ్రంశం, uint32_t max_tiles) {
    VK_LOG_DEBUG("ggml_vk_matmul_id(a: (" << a.బఫరు->బఫరు << ", " << a.స్థానభ్రంశం << ", " << a.size << "), b: (" << b.బఫరు->బఫరు << ", " << b.స్థానభ్రంశం << ", " << b.size << "), d: (" << d.బఫరు->బఫరు << ", " << d.స్థానభ్రంశం << ", " << d.size << "), ids: (" << ids.బఫరు->బఫరు << ", " << ids.స్థానభ్రంశం << ", " << ids.size << "), expert_count: (" << expert_లెక్క_buf.బఫరు->బఫరు << ", " << expert_లెక్క_buf.స్థానభ్రంశం << ", " << expert_లెక్క_buf.size << "), " <<
        "m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", " <<
        "batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", " <<
        "n_as: " << n_as << ", nei0: " << nei0 << ", nei1: " << nei1 << ", nbi1: " << nbi1 << ", ne11: " << ne11 << ")");
    స్థిరరాశి vk_mat_mat_id_నెట్టు_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d,
                                              nei0, nei1, nbi1, ne11, n_as, అడ్డువరుస_ids_స్థానభ్రంశం };
    స్థిరరాశి uint32_t tiles_y = std::min(max_tiles, ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]);
    స్థిరరాశి std::array<uint32_t, 3> elements = { m, tiles_y * pipeline->wg_denoms[1], uint32_t(CEIL_DIV(max_tiles, tiles_y)) };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d, ids, expert_లెక్క_buf }, pc, elements);
}

తార్కిక ggml_vk_dim01_contiguous(స్థిరరాశి ggml_tensor * tensor) {
    తిరిగి
        tensor->nb[0] == ggml_type_కొలత(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_కొలత(tensor->type) &&
        (tensor->ne[3] == 1 || tensor->nb[3] == tensor->nb[2]*tensor->ne[2]);
}

// Batch stride in elements of a tensor read in place.
స్థిర uint32_t ggml_vk_batch_stride(స్థిరరాశి ggml_tensor * tensor) {
    తిరిగి (uint32_t)(tensor->nb[2] / ggml_type_కొలత(tensor->type) * ggml_blck_కొలత(tensor->type));
}

vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor * src, స్థిరరాశి ggml_tensor * dst, ggml_type గమ్యం) {

    // Choose "contiguous copy" shader if src/dst are contiguous
    తార్కిక contig = ggml_is_contiguous(src) && (!dst || ggml_is_contiguous(dst));

    // Use optimized "transpose" shader if src dim1 is the innermost dimension.
    తార్కిక transpose = dst && src->nb[1] == ggml_type_కొలత(గమ్యం) && ggml_are_same_shape(dst, src);

    ఒకవేళ (transpose && src->type == గమ్యం) {
        ఒకవేళ (ggml_type_కొలత(గమ్యం) == 4) {
            తిరిగి ctx->device->pipeline_cpy_transpose_32;
        } లేకపోతే ఒకవేళ (ggml_type_కొలత(గమ్యం) == 2) {
            తిరిగి ctx->device->pipeline_cpy_transpose_16;
        }
    }

    // Same, for a 0<->2 swap: src dim2 is the innermost dimension.
    తార్కిక transpose02 = dst && !contig && src->nb[2] == ggml_type_కొలత(గమ్యం) &&
                       ggml_is_contiguous(dst) && ggml_are_same_shape(dst, src);

    ఒకవేళ (transpose02 && src->type == గమ్యం) {
        ఒకవేళ (ggml_type_కొలత(గమ్యం) == 4) {
            తిరిగి ctx->device->pipeline_cpy_transpose_02_32;
        } లేకపోతే ఒకవేళ (ggml_type_కొలత(గమ్యం) == 2) {
            తిరిగి ctx->device->pipeline_cpy_transpose_02_16;
        }
    }

    ఒకవేళ (src->type == GGML_TYPE_F32 && గమ్యం == GGML_TYPE_F16) {
        ఒకవేళ (contig) {
            తిరిగి ctx->device->pipeline_contig_cpy_f32_f16;
        } లేకపోతే {
            తిరిగి ctx->device->pipeline_cpy_f32_f16;
        }
    }
    ఒకవేళ (src->type == GGML_TYPE_F16 && గమ్యం == GGML_TYPE_F32) {
        ఒకవేళ (contig) {
            తిరిగి ctx->device->pipeline_contig_cpy_f16_f32;
        } లేకపోతే {
            తిరిగి ctx->device->pipeline_cpy_f16_f32;
        }
    }
    ఒకవేళ (src->type == GGML_TYPE_F32 && గమ్యం == GGML_TYPE_BF16) {
        ఒకవేళ (contig) {
            తిరిగి ctx->device->pipeline_contig_cpy_f32_bf16;
        } లేకపోతే {
            తిరిగి ctx->device->pipeline_cpy_f32_bf16;
        }
    }
    ఒకవేళ (src->type == GGML_TYPE_BF16 && గమ్యం == GGML_TYPE_F32) {
        ఒకవేళ (contig) {
            తిరిగి ctx->device->pipeline_contig_cpy_bf16_f32;
        } లేకపోతే {
            తిరిగి ctx->device->pipeline_cpy_bf16_f32;
        }
    }
    ఒకవేళ (src->type == GGML_TYPE_BF16 && గమ్యం == GGML_TYPE_F16) {
        ఒకవేళ (contig) {
            తిరిగి ctx->device->pipeline_contig_cpy_bf16_f16;
        } లేకపోతే {
            తిరిగి ctx->device->pipeline_cpy_bf16_f16;
        }
    }
    ఒకవేళ (src->type == GGML_TYPE_F32 && గమ్యం == GGML_TYPE_I32) {
        ఒకవేళ (contig) {
            తిరిగి ctx->device->pipeline_contig_cpy_f32_i32;
        } లేకపోతే {
            తిరిగి ctx->device->pipeline_cpy_f32_i32;
        }
    }
    ఒకవేళ (src->type == GGML_TYPE_I32 && గమ్యం == GGML_TYPE_F32) {
        ఒకవేళ (contig) {
            తిరిగి ctx->device->pipeline_contig_cpy_i32_f32;
        } లేకపోతే {
            తిరిగి ctx->device->pipeline_cpy_i32_f32;
        }
    }
    ఒకవేళ (src->type == GGML_TYPE_F32) {
        ఎంపిక (గమ్యం) {
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_IQ4_NL:
            తిరిగి ctx->device->pipeline_cpy_f32_quant[గమ్యం];
        అప్రమేయ:
            ఆపు;
        }
    }

    ఒకవేళ (గమ్యం == GGML_TYPE_F32) {
        ఎంపిక (src->type) {
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_IQ4_NL:
            తిరిగి ctx->device->pipeline_cpy_quant_f32[src->type];
        అప్రమేయ:
            ఆపు;
        }
    }

    ఒకవేళ (src->type == గమ్యం) {
        // Copy two or four bytes at a time, depending on block size.
        // For quantized types, we scale by block size/type size. But
        // this path is also used for bf16->bf16 for example, where the
        // type size must be exactly 2 or 4.
        GGML_ASSERT(ggml_is_quantized(గమ్యం) || ggml_type_కొలత(src->type) == 2 || ggml_type_కొలత(src->type) == 4);
        ఒకవేళ ((ggml_type_కొలత(src->type) % 4) == 0) {
            ఒకవేళ (contig) {
                తిరిగి ctx->device->pipeline_contig_cpy_32;
            } లేకపోతే {
                తిరిగి ctx->device->pipeline_cpy_32;
            }
        } లేకపోతే {
            ఒకవేళ (contig) {
                తిరిగి ctx->device->pipeline_contig_cpy_16;
            } లేకపోతే {
                తిరిగి ctx->device->pipeline_cpy_16;
            }
        }
    }

    std::cerr << "Missing CPY op for types: " << ggml_type_name(src->type) << " " << ggml_type_name(గమ్యం) << std::పంక్తి_ముగింపు;
    GGML_ABORT("fatal error");
}

శూన్య ggml_vk_cpy_గమ్యం_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి vk_subbuffer & లోపల, స్థిరరాశి vk_subbuffer & బహిర్గతం) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_contiguous((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "buffer in size=" << లోపల.బఫరు->size << ", buffer out size=" << బహిర్గతం.బఫరు->size << ")");

    స్థిరరాశి uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    ఒకవేళ (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } లేకపోతే ఒకవేళ (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } లేకపోతే {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_నెట్టు_constants pc = vk_op_unary_నెట్టు_constants_init(tensor, tensor, ne);
    pc.nb10 = 1;
    pc.nb11 = (uint32_t)tensor->ne[0];
    pc.nb12 = (uint32_t)(tensor->ne[0] * tensor->ne[1]);
    pc.nb13 = (uint32_t)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]);
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { లోపల, బహిర్గతం }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

స్థిర శూన్య ggml_vk_cpy_గమ్యం_strided(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, స్థిరరాశి ggml_tensor * tensor,
        స్థిరరాశి vk_subbuffer & లోపల, స్థిరరాశి vk_subbuffer & బహిర్గతం,
        uint32_t nb10, uint32_t nb11, uint32_t nb12, uint32_t nb13) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_strided((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "dst_nb=(" << nb10 << ", " << nb11 << ", " << nb12 << ", " << nb13 << "), buffer in size=" << లోపల.బఫరు->size << ", buffer out size=" << బహిర్గతం.బఫరు->size << ")");

    స్థిరరాశి uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    ఒకవేళ (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } లేకపోతే ఒకవేళ (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } లేకపోతే {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_నెట్టు_constants pc = vk_op_unary_నెట్టు_constants_init(tensor, tensor, ne);
    pc.nb10 = nb10;
    pc.nb11 = nb11;
    pc.nb12 = nb12;
    pc.nb13 = nb13;
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { లోపల, బహిర్గతం }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type) {
    ఎంపిక(type) {
        స్థితి GGML_TYPE_Q8_1:
            తిరిగి ctx->device->pipeline_quantize_q8_1_x4;
        అప్రమేయ:
            std::cerr << "Missing quantize pipeline for type: " << ggml_type_name(type) << std::పంక్తి_ముగింపు;
            GGML_ABORT("fatal error");
    }
}

శూన్య ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి vk_subbuffer & లోపల, స్థిరరాశి vk_subbuffer & బహిర్గతం, uint32_t ne) {
    VK_LOG_DEBUG("ggml_vk_quantize_q8_1(" << "buffer in size=" << లోపల.బఫరు->size << ", buffer out size=" << బహిర్గతం.బఫరు->size << ", " << ne << ")");

    vk_pipeline pipeline = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);

    స్థిరరాశి uint32_t అంకె_ఖండాలు = CEIL_DIV(ne, pipeline->wg_denoms[0]);
    // clamp the number of elements to the max workgroup count. The shader will iterate over the total number of blocks.
    స్థిరరాశి uint64_t max_elements = std::min<uint64_t>(uint64_t{ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[0]} * pipeline->wg_denoms[0], std::numeric_limits<uint32_t>::max());
    స్థిరరాశి uint32_t elements = std::min(ne, static_cast<uint32_t>(max_elements));

    స్థిరరాశి vk_quantize_q8_1_నెట్టు_constants pc = {
        ne,
        అంకె_ఖండాలు,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { లోపల, బహిర్గతం }, pc, { elements, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
}

స్థిర vk_pipeline ggml_vk_get_64b_indexing_pipeline(ggml_backend_vk_context * ctx, vk_pipeline &pipeline) {
    GGML_వాడని(ctx);
#ఒకవేళ defined(VK_EXT_shader_64bit_indexing)
    vk_pipeline *ptr = &pipeline;
    అయితే (*ptr) {
        ఒకవేళ ((*ptr)->is_64b_indexing) {
            తిరిగి *ptr;
        }
        ptr = &(*ptr)->తదుపరి;
    }
#అయితే_ముగింపు
    తిరిగి pipeline;
}

// perf-heuristic for choosing mul_mat b type, based on shader availability
స్థిర ggml_type ggml_vk_mul_mat_b_type(
        ggml_backend_vk_context * ctx, ggml_type src0_type, స్థిరరాశి ggml_tensor * src1,
        ggml_type f16_type, ggml_prec prec, తార్కిక mul_mat_id, తార్కిక can_quantize_y, తార్కిక prefer_f16_b) {
    స్వయంచాలక అందుబాటు = [&](ggml_type b) {
        తిరిగి ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0_type, b, prec, mul_mat_id) != శూన్యసూచిక;
    };
    ఒకవేళ (can_quantize_y && అందుబాటు(GGML_TYPE_Q8_1))             తిరిగి GGML_TYPE_Q8_1;
    ఒకవేళ (prefer_f16_b   && అందుబాటు(f16_type))                   తిరిగి f16_type;
    ఒకవేళ (ggml_vk_dim01_contiguous(src1) && అందుబాటు(src1->type)) తిరిగి src1->type;
    ఒకవేళ (src1->type == GGML_TYPE_BF16 && అందుబాటు(GGML_TYPE_F32)) తిరిగి GGML_TYPE_F32;
    తిరిగి f16_type;
}

స్థిర శూన్య ggml_vk_mul_mat_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst, తార్కిక disable_విభజించు_k) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_q_f16((" << src0 << ", name=" << src0->name << ", type=" << ggml_type_name(src0->type) << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << ggml_type_name(src1->type) << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << ggml_type_name(dst->type) << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    స్థిరరాశి uint64_t ne00 = src0->ne[0];
    స్థిరరాశి uint64_t ne01 = src0->ne[1];
    స్థిరరాశి uint64_t ne02 = src0->ne[2];
    స్థిరరాశి uint64_t ne03 = src0->ne[3];

    స్థిరరాశి uint64_t ne10 = src1->ne[0];
    స్థిరరాశి uint64_t ne11 = src1->ne[1];
    స్థిరరాశి uint64_t ne12 = src1->ne[2];
    స్థిరరాశి uint64_t ne13 = src1->ne[3];

    స్థిరరాశి uint64_t ne21 = dst->ne[1];
    స్థిరరాశి uint32_t stride_d = dst->nb[1] / ggml_type_కొలత(dst->type);
    స్థిరరాశి uint32_t stride_batch_d = stride_d*ne21;

    స్థిరరాశి uint64_t r2 = ne12 / ne02;
    స్థిరరాశి uint64_t r3 = ne13 / ne03;

    ggml_backend_vk_బఫరు_context * dst_buf_ctx = (ggml_backend_vk_బఫరు_context *)dst->బఫరు->context;
    ggml_backend_vk_బఫరు_context * src0_buf_ctx = (ggml_backend_vk_బఫరు_context *)src0->బఫరు->context;
    ggml_backend_vk_బఫరు_context * src1_buf_ctx = (ggml_backend_vk_బఫరు_context *)src1->బఫరు->context;

    vk_బఫరు d_Qx = శూన్యసూచిక;
    size_t qx_buf_స్థానభ్రంశం = 0;
    vk_బఫరు d_Qy = శూన్యసూచిక;
    size_t qy_buf_స్థానభ్రంశం = 0;

    తార్కిక src0_uma = అబద్ధం;
    తార్కిక src1_uma = అబద్ధం;

    ఒకవేళ (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->దత్తాంశం, d_Qx, qx_buf_స్థానభ్రంశం);
        ggml_vk_host_get(ctx->device, src1->దత్తాంశం, d_Qy, qy_buf_స్థానభ్రంశం);
        src0_uma = d_Qx != శూన్యసూచిక;
        src1_uma = d_Qy != శూన్యసూచిక;
    }

    స్థిరరాశి తార్కిక x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;

    స్థిరరాశి తార్కిక is_coopmat = ctx->device->coopmat_support || ctx->device->coopmat2;
    స్థిరరాశి తార్కిక can_quantize_y = (ctx->device->integer_dot_లబ్ధం || ctx->device->coopmat_int_support) &&
                                src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // coopmat shaders are fast enough that memory bandwidth reduction from f16 activations helps
    స్థిరరాశి తార్కిక prefer_f16_b = is_coopmat && ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    స్థిరరాశి ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], అబద్ధం, can_quantize_y, prefer_f16_b);

    స్థిరరాశి తార్కిక quantize_y       = y_kernel_type == GGML_TYPE_Q8_1;
    స్థిరరాశి తార్కిక y_f32_kernel     = y_kernel_type == GGML_TYPE_F32;
    స్థిరరాశి తార్కిక y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type);

    స్థిరరాశి std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0]);

    స్థిరరాశి తార్కిక qx_needs_dequant = mmp_map == శూన్యసూచిక || x_non_contig;
    స్థిరరాశి తార్కిక qy_needs_dequant = y_needs_reformat;

    ఒకవేళ (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0]);
    }

    GGML_ASSERT(mmp_map != శూన్యసూచిక);

    స్థిరరాశి uint32_t kpad = quantize_y ? 0 : ggml_vk_align_కొలత(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, ne11, అబద్ధం));
    స్థిరరాశి తార్కిక aligned = !quantize_y && ne10 == kpad && ne01 > 8 && ne11 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, ne11, aligned, అబద్ధం);

    ఒకవేళ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_బఫరు_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
    uint32_t padded_n = qy_needs_dequant ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1]) : ne11;
    స్థిరరాశి uint64_t x_ne = ggml_nelements(src0);
    // 128 elements per Q8_1 x4 block
    స్థిరరాశి uint64_t y_ne = padded_n * ne10 * ne12 * ne13;
    స్థిరరాశి uint64_t d_ne = ggml_nelements(dst);

    స్థిరరాశి uint32_t విభజించు_k = ggml_vk_guess_విభజించు_k(ctx, ne01, ne11, ne10, disable_విభజించు_k, pipeline);

    స్థిరరాశి uint64_t qx_sz = ggml_type_కొలత(src0->type) * x_ne / ggml_blck_కొలత(src0->type);
    స్థిరరాశి uint64_t qy_sz = ggml_type_కొలత(src1->type) * y_ne / ggml_blck_కొలత(src1->type);
    స్థిరరాశి uint64_t x_sz = !qx_needs_dequant ? qx_sz : పరిమాణం(ggml_fp16_t) * x_ne;
    స్థిరరాశి uint64_t y_sz = quantize_y ? (ggml_vk_align_కొలత(y_ne, 128) * ggml_type_కొలత(GGML_TYPE_Q8_1) / ggml_blck_కొలత(GGML_TYPE_Q8_1)) : (y_f32_kernel ? పరిమాణం(దశాంశం) * y_ne : పరిమాణం(ggml_fp16_t) * y_ne);
    స్థిరరాశి uint64_t d_sz = పరిమాణం(దశాంశం) * d_ne;

    vk_pipeline గమ్యం_fp16_vk_0 = శూన్యసూచిక;
    vk_pipeline గమ్యం_fp16_vk_1 = శూన్యసూచిక;
    vk_pipeline గమ్యం_q8_1 = శూన్యసూచిక;

    ఒకవేళ (x_non_contig) {
        గమ్యం_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, శూన్యసూచిక, f16_type);
    } లేకపోతే {
        గమ్యం_fp16_vk_0 = ggml_vk_get_గమ్యం_fp16(ctx, src0->type);
    }
    ఒకవేళ (y_needs_reformat) {
        గమ్యం_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, శూన్యసూచిక, y_kernel_type);
    } లేకపోతే {
        గమ్యం_fp16_vk_1 = ggml_vk_get_గమ్యం_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || గమ్యం_fp16_vk_0 != శూన్యసూచిక);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || గమ్యం_fp16_vk_1 != శూన్యసూచిక);  // NOLINT

    ఒకవేళ (quantize_y) {
        గమ్యం_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    {
        స్థిరరాశి uint64_t విభజించు_k_కొలత = విభజించు_k > 1 ? d_sz * విభజించు_k : 0;
        ఒకవేళ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_బఫరు_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_బఫరు_Range) ||
                (విభజించు_k > 1 && విభజించు_k_కొలత > ctx->device->properties.limits.max_Storage_బఫరు_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ఒకవేళ (qx_needs_dequant && ctx->prealloc_కొలత_x < x_sz) {
            ctx->prealloc_కొలత_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ఒకవేళ ((qy_needs_dequant || quantize_y) && ctx->prealloc_కొలత_y < y_sz) {
            ctx->prealloc_కొలత_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ఒకవేళ (విభజించు_k > 1 && ctx->prealloc_కొలత_విభజించు_k < విభజించు_k_కొలత) {
            ctx->prealloc_కొలత_విభజించు_k = విభజించు_k_కొలత;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ఒకవేళ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, గమ్యం_fp16_vk_0, 1);
        }
        ఒకవేళ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, గమ్యం_fp16_vk_1, 1);
        }
        ఒకవేళ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, గమ్యం_q8_1, 1);
        }
        ఒకవేళ (విభజించు_k > 1) {
            ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_విభజించు_k_reduce, 1);
        }
    }

    vk_బఫరు d_D = dst_buf_ctx->dev_బఫరు;
    స్థిరరాశి uint64_t d_buf_స్థానభ్రంశం = vk_tensor_స్థానభ్రంశం(dst) + dst->view_offs;
    GGML_ASSERT(d_D != శూన్యసూచిక);
    GGML_ASSERT(d_D->size >= d_buf_స్థానభ్రంశం + d_sz);
    vk_బఫరు d_X;
    uint64_t x_buf_స్థానభ్రంశం = 0;
    vk_బఫరు d_Y;
    uint64_t y_buf_స్థానభ్రంశం = 0;
    ఒకవేళ (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_బఫరు;
        qx_buf_స్థానభ్రంశం = vk_tensor_స్థానభ్రంశం(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != శూన్యసూచిక);
    }
    ఒకవేళ (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_బఫరు;
        qy_buf_స్థానభ్రంశం = vk_tensor_స్థానభ్రంశం(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != శూన్యసూచిక);
    }
    ఒకవేళ (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } లేకపోతే {
        d_X = d_Qx;
        x_buf_స్థానభ్రంశం = qx_buf_స్థానభ్రంశం;
        GGML_ASSERT(qx_sz == x_sz);
    }
    ఒకవేళ (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } లేకపోతే ఒకవేళ (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } లేకపోతే {
        d_Y = d_Qy;
        y_buf_స్థానభ్రంశం = qy_buf_స్థానభ్రంశం;
        GGML_ASSERT(qy_sz == y_sz);
    }

    ఒకవేళ (x_non_contig || qx_needs_dequant) {
        ఒకవేళ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ఒకవేళ (x_non_contig) {
        ggml_vk_cpy_గమ్యం_contiguous(ctx, subctx, గమ్యం_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_స్థానభ్రంశం), ggml_vk_subbuffer(ctx, d_X, 0));
    } లేకపోతే ఒకవేళ (qx_needs_dequant) {
        స్థిరరాశి std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, గమ్యం_fp16_vk_0, { vk_subbuffer{ d_Qx, qx_buf_స్థానభ్రంశం, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)(x_ne), 1, 1});
        ggml_vk_sync_buffers(ctx, subctx);
    }
    ఒకవేళ (y_needs_reformat) {
        ఒకవేళ (ctx->prealloc_y_చివరి_pipeline_వాడిన != గమ్యం_fp16_vk_1.get() ||
            ctx->prealloc_y_చివరి_tensor_వాడిన != src1 ||
            ctx->prealloc_y_చివరి_k_padded) {
            ఒకవేళ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_గమ్యం_contiguous(ctx, subctx, గమ్యం_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_స్థానభ్రంశం), ggml_vk_subbuffer(ctx, d_Y, 0));
            ctx->prealloc_y_చివరి_pipeline_వాడిన = గమ్యం_fp16_vk_1.get();
            ctx->prealloc_y_చివరి_tensor_వాడిన = src1;
            ctx->prealloc_y_చివరి_k_padded = అబద్ధం;
        }
    }
    ఒకవేళ (quantize_y) {
        ఒకవేళ (ctx->prealloc_y_చివరి_pipeline_వాడిన != గమ్యం_q8_1.get() ||
            ctx->prealloc_y_చివరి_tensor_వాడిన != src1 ||
            ctx->prealloc_y_చివరి_k_padded) {
            ఒకవేళ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_స్థానభ్రంశం), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_చివరి_pipeline_వాడిన = గమ్యం_q8_1.get();
            ctx->prealloc_y_చివరి_tensor_వాడిన = src1;
            ctx->prealloc_y_చివరి_k_padded = అబద్ధం;
        }
    }

    // The partial tiles of a quantized A rely on the bound range to read zeros past the last row,
    // so the range stays exact: the strided extent of a tensor read in place, the staged size otherwise.
    స్థిరరాశి uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    స్థిరరాశి uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    ఒకవేళ (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_కొలత(src0->type);
    }

    ఒకవేళ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_కొలత(src1->type);
    }

    // column stride in bytes of the B buffer: raw src1 keeps its own stride, staging buffers are tightly packed
    స్థిరరాశి uint64_t b_నిలువు_బైట్లు = (y_needs_reformat || qy_needs_dequant || quantize_y)
        ? (quantize_y ? (uint64_t)(ne10 / ggml_blck_కొలత(GGML_TYPE_Q8_1)) * ggml_type_కొలత(GGML_TYPE_Q8_1) : (uint64_t)ne10 * పరిమాణం(ggml_fp16_t))
        : (uint64_t)src1->nb[1];

    // compute
    ggml_vk_matmul(
        ctx, subctx, pipeline,
        { d_X, x_buf_స్థానభ్రంశం, x_range }, { d_Y, y_buf_స్థానభ్రంశం, y_range },
        ggml_vk_subbuffer(ctx, d_D, d_buf_స్థానభ్రంశం), { ctx->prealloc_విభజించు_k, 0, d_sz * విభజించు_k },
        ne01, ne11, ne10,
        ne10, ne10, stride_d, stride_batch_x, stride_batch_y, stride_batch_d,
        విభజించు_k, ne12*ne13, ne02, ne12, r2, r3, padded_n,
        b_నిలువు_బైట్లు, (uint64_t)dst->nb[1]
    );  // NOLINT

    ఒకవేళ (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = నిజం;
    }
    ఒకవేళ (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = నిజం;
    }
}

స్థిర తార్కిక ggml_vk_should_use_mmvq(స్థిరరాశి vk_device& device, uint32_t m, uint32_t n, uint32_t k, ggml_type src0_type) {
    ఒకవేళ (device->mmvq_బహుళకం == 1) {
        తిరిగి నిజం;
    } లేకపోతే ఒకవేళ (device->mmvq_బహుళకం == -1) {
        తిరిగి అబద్ధం;
    }

    // q6_k only has 2-byte alignment which makes it somewhat problematic,
    // using MMVQ is only a win on Intel.
    తార్కిక mmvq_q6 = device->vendor_id == VK_VENDOR_ID_INTEL;
    ఒకవేళ (src0_type == GGML_TYPE_Q6_K && !mmvq_q6) {
        తిరిగి అబద్ధం;
    }

    // MMVQ is generally good for batches
    ఒకవేళ (n > 1) {
        తిరిగి నిజం;
    }

    // Quantization overhead is not worth it for small k
    ఎంపిక (device->vendor_id) {
    స్థితి VK_VENDOR_ID_NVIDIA:
        ఒకవేళ (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_IQ1_S || src0_type == GGML_TYPE_IQ1_M) {
            తిరిగి నిజం;
        }

        ఒకవేళ (k <= 4096) {
            తిరిగి అబద్ధం;
        }

        ఎంపిక (src0_type) {
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_Q8_0:
            తిరిగి device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        అప్రమేయ:
            తిరిగి నిజం;
        }
    స్థితి VK_VENDOR_ID_AMD:
        ఒకవేళ (k < 2048) {
            తిరిగి అబద్ధం;
        }

        ఎంపిక (src0_type) {
        స్థితి GGML_TYPE_Q8_0:
            తిరిగి device->architecture == vk_device_architecture::AMD_GCN;
        అప్రమేయ:
            తిరిగి నిజం;
        }
    స్థితి VK_VENDOR_ID_INTEL:
        ఒకవేళ (device->architecture == vk_device_architecture::INTEL_XE2) {
            ఒకవేళ (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_Q6_K) {
                తిరిగి నిజం;
            }
        }

        ఒకవేళ (device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            // Intel Windows proprietary driver MMVQ performance for !Q2/Q3/Q6 is worse than fp16,
            // see https://github.com/ggml-org/llama.cpp/issues/17628 and
            // https://github.com/ggml-org/llama.cpp/pull/23056
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (k < 2048) {
            తిరిగి అబద్ధం;
        }

        ఎంపిక (src0_type) {
        // From tests on A770 Linux, may need more tuning
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_IQ4_XS:
            తిరిగి అబద్ధం;
        అప్రమేయ:
            తిరిగి నిజం;
        }
    స్థితి VK_VENDOR_ID_QUALCOMM:
        తిరిగి అబద్ధం;
    అప్రమేయ:
        తిరిగి నిజం;
    }

    GGML_వాడని(m);
}

స్థిర శూన్య ggml_vk_mul_mat_vec_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, తార్కిక మార్పిడి_inputs = అబద్ధం) {
    ggml_tensor * dst = cgraph->nodes[కణుపు_idx];
    స్థిరరాశి ggml_tensor * src0 = dst->src[మార్పిడి_inputs ? 1 : 0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[మార్పిడి_inputs ? 0 : 1];

    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << ")),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    స్థిరరాశి uint64_t ne00 = src0->ne[0];
    స్థిరరాశి uint64_t ne01 = src0->ne[1];
    స్థిరరాశి uint64_t ne02 = src0->ne[2];
    స్థిరరాశి uint64_t ne03 = src0->ne[3];

    స్థిరరాశి uint64_t ne10 = src1->ne[0];
    స్థిరరాశి uint64_t ne11 = src1->ne[1];
    స్థిరరాశి uint64_t ne12 = src1->ne[2];
    స్థిరరాశి uint64_t ne13 = src1->ne[3];

    స్థిరరాశి uint64_t ne20 = dst->ne[మార్పిడి_inputs ? 1 : 0];
    స్థిరరాశి uint64_t ne21 = dst->ne[మార్పిడి_inputs ? 0 : 1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    స్థిరరాశి uint64_t r2 = ne12 / ne02;
    స్థిరరాశి uint64_t r3 = ne13 / ne03;

    // batch_n indicates that we need to compute a few vector results, and this assumes
    // ne12 and ne13 are 1. It overloads the batch_strides to hold the row strides.
    GGML_ASSERT(ne11 == 1 || ne12 * ne13 == 1);
    తార్కిక batch_n = ne11 > 1;

    స్థిరరాశి తార్కిక x_non_contig = !ggml_vk_dim01_contiguous(src0);

    // BF16 src1 has no mul_mat_vec B-type; widen to F32
    స్థిరరాశి తార్కిక widen_bf16 = src1->type == GGML_TYPE_BF16;
    స్థిరరాశి తార్కిక y_non_contig = !ggml_vk_dim01_contiguous(src1) || widen_bf16;

    స్థిరరాశి ggml_type y_kernel_type = widen_bf16 ? GGML_TYPE_F32 : src1->type;
    స్థిరరాశి తార్కిక f16_f32_kernel = y_kernel_type == GGML_TYPE_F32;
    తార్కిక quantize_y = ctx->device->integer_dot_లబ్ధం && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);

    vk_pipeline గమ్యం_fp16_vk_0 = శూన్యసూచిక;
    vk_pipeline గమ్యం_fp16_vk_1 = శూన్యసూచిక;
    ఒకవేళ (x_non_contig) {
        గమ్యం_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, శూన్యసూచిక, src0->type);
    }
    ఒకవేళ (y_non_contig) {
        గమ్యం_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, శూన్యసూచిక, y_kernel_type);
    } లేకపోతే {
        గమ్యం_fp16_vk_1 = ggml_vk_get_గమ్యం_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, GGML_TYPE_Q8_1, ne11, ne20, ne00) : శూన్యసూచిక;
    vk_pipeline గమ్యం_q8_1 = శూన్యసూచిక;

    ఒకవేళ (dmmv == శూన్యసూచిక) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, y_kernel_type, ne11, ne20, ne00);
        quantize_y = అబద్ధం;
    }

    ఒకవేళ (quantize_y) {
        గమ్యం_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    ఒకవేళ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_బఫరు_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    స్థిరరాశి తార్కిక qx_needs_dequant = x_non_contig;
    స్థిరరాశి తార్కిక qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    GGML_ASSERT(!qx_needs_dequant || గమ్యం_fp16_vk_0 != శూన్యసూచిక);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || గమ్యం_fp16_vk_1 != శూన్యసూచిక);  // NOLINT
    GGML_ASSERT(dmmv != శూన్యసూచిక);

    స్థిరరాశి uint64_t x_ne = ggml_nelements(src0);
    స్థిరరాశి uint64_t y_ne = ggml_nelements(src1);

    స్థిరరాశి uint64_t qx_sz = ggml_vk_align_కొలత(ggml_type_కొలత(src0->type) * x_ne / ggml_blck_కొలత(src0->type), ctx->device->properties.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment);
    స్థిరరాశి uint64_t x_sz = x_non_contig ? ggml_vk_align_కొలత(ggml_type_కొలత(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment) : qx_sz;
    స్థిరరాశి uint64_t y_sz = quantize_y ? (ggml_vk_align_కొలత(y_ne, 128) * ggml_type_కొలత(GGML_TYPE_Q8_1) / ggml_blck_కొలత(GGML_TYPE_Q8_1)) :
                         (f16_f32_kernel ? పరిమాణం(దశాంశం) * y_ne : పరిమాణం(ggml_fp16_t) * y_ne);

    {
        ఒకవేళ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_బఫరు_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_బఫరు_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ఒకవేళ (qx_needs_dequant && ctx->prealloc_కొలత_x < x_sz) {
            ctx->prealloc_కొలత_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ఒకవేళ ((qy_needs_dequant || quantize_y) && ctx->prealloc_కొలత_y < y_sz) {
            ctx->prealloc_కొలత_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ఒకవేళ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, గమ్యం_fp16_vk_0, 1);
        }
        ఒకవేళ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, గమ్యం_fp16_vk_1, 1);
        }
        ఒకవేళ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, గమ్యం_q8_1, 1);
        }
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[కణుపు_idx + ctx->అంకె_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_X, d_Y;

    ఒకవేళ (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } లేకపోతే {
        d_X = d_Qx;
        GGML_ASSERT(qx_sz == x_sz);
    }
    ఒకవేళ (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } లేకపోతే {
        d_Y = d_Qy;
    }

    ఒకవేళ (x_non_contig) {
        ఒకవేళ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        GGML_ASSERT(x_sz == ggml_vk_align_కొలత(ggml_type_కొలత(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment));
        ggml_vk_cpy_గమ్యం_contiguous(ctx, subctx, గమ్యం_fp16_vk_0, src0, d_Qx, d_X);
    }
    ఒకవేళ (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_కొలత(y_kernel_type) * y_ne);
        ఒకవేళ (ctx->prealloc_y_చివరి_pipeline_వాడిన != గమ్యం_fp16_vk_1.get() ||
            ctx->prealloc_y_చివరి_tensor_వాడిన != src1 ||
            ctx->prealloc_y_చివరి_k_padded) {
            ఒకవేళ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_గమ్యం_contiguous(ctx, subctx, గమ్యం_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_చివరి_pipeline_వాడిన = గమ్యం_fp16_vk_1.get();
            ctx->prealloc_y_చివరి_tensor_వాడిన = src1;
            ctx->prealloc_y_చివరి_k_padded = అబద్ధం;
        }
    }
    ఒకవేళ (quantize_y) {
        ఒకవేళ (ctx->prealloc_y_చివరి_pipeline_వాడిన != గమ్యం_q8_1.get() ||
            ctx->prealloc_y_చివరి_tensor_వాడిన != src1 ||
            ctx->prealloc_y_చివరి_k_padded) {
            ఒకవేళ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_చివరి_pipeline_వాడిన = గమ్యం_q8_1.get();
            ctx->prealloc_y_చివరి_tensor_వాడిన = src1;
            ctx->prealloc_y_చివరి_k_padded = అబద్ధం;
        }
    }

    // For batch_n, the A matrix is the same for each batch, and B/D use the row stride as the batch stride
    uint32_t stride_batch_x = batch_n ? 0 : (qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0));
    uint32_t stride_batch_y = batch_n ? ne10 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));
    uint32_t stride_batch_d = batch_n ? ne20 : (ne20*ne21);

    ఒకవేళ (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_కొలత(src0->type);
    }

    ఒకవేళ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_కొలత(src1->type);
    }

    స్థిరరాశి uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    ఒకవేళ (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    vk_subbuffer d_F0 = d_D;
    ఒకవేళ (ctx->అంకె_additional_fused_ops > 0) {
        స్థిరరాశి ggml_tensor * కూడు_2 = cgraph->nodes[కణుపు_idx + 1];
        స్థిరరాశి ggml_tensor * bias = కూడు_2->src[0] == dst ? కూడు_2->src[1] : కూడు_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    ఒకవేళ (ctx->అంకె_additional_fused_ops == 2) {
        స్థిరరాశి ggml_tensor * కూడు_2 = cgraph->nodes[కణుపు_idx + 2];
        స్థిరరాశి ggml_tensor * bias = కూడు_2->src[0] == cgraph->nodes[కణుపు_idx + 1] ? కూడు_2->src[1] : కూడు_2->src[0];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    ggml_pipeline_request_descriptor_sets(ctx, dmmv, CEIL_DIV(ne12 * ne13, ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]));

    uint32_t పునాది_work_సమూహం_y = 0;
    అయితే (పునాది_work_సమూహం_y < ne12 * ne13) {

        uint32_t groups_y = std::min((uint32_t)(ne12 * ne13) - పునాది_work_సమూహం_y, ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]);
        స్థిరరాశి vk_mat_vec_నెట్టు_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            stride_batch_x, stride_batch_y, stride_batch_d,
            fusion_flags, పునాది_work_సమూహం_y,
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
        పునాది_work_సమూహం_y += groups_y;
    }

    ఒకవేళ (x_non_contig) {
        ctx->prealloc_x_need_sync = నిజం;
    }
    ఒకవేళ (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = నిజం;
    }
}

స్థిర శూన్య ggml_vk_mul_mat_vec_p021_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    ggml_tensor * dst = cgraph->nodes[కణుపు_idx];
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_p021_f16_f32(" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]);  // NOLINT
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]);  // NOLINT
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    స్థిరరాశి uint64_t ne00 = src0->ne[0];
    స్థిరరాశి uint64_t ne01 = src0->ne[1];
    స్థిరరాశి uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    //const uint64_t ne10 = src1->ne[0];
    స్థిరరాశి uint64_t ne11 = src1->ne[1];
    స్థిరరాశి uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);

    // With grouped query attention there are > 1 Q matrices per K, V matrix.
    uint32_t gqa_నిష్పత్తి = (uint32_t)ne12 / (uint32_t)ne02;
    ఒకవేళ (gqa_నిష్పత్తి > 8 || gqa_నిష్పత్తి == 0 || ne12 != ne02 * gqa_నిష్పత్తి) {
        gqa_నిష్పత్తి = 1;
    }

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_నిష్పత్తి - 1];

    ఒకవేళ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_బఫరు_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[కణుపు_idx + ctx->అంకె_additional_fused_ops], నిజం);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, నిజం);

    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    ఒకవేళ (ctx->అంకె_additional_fused_ops > 0) {
        స్థిరరాశి ggml_tensor * కూడు_2 = cgraph->nodes[కణుపు_idx + 1];
        స్థిరరాశి ggml_tensor * bias = కూడు_2->src[0] == dst ? కూడు_2->src[1] : కూడు_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    ఒకవేళ (ctx->అంకె_additional_fused_ops > 1) {
        స్థిరరాశి ggml_tensor * bias = cgraph->nodes[కణుపు_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute

    vk_mat_vec_p021_నెట్టు_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne12,
        0, 0, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, శూన్యసూచిక, శూన్యసూచిక, cgraph->nodes[కణుపు_idx + ctx->అంకె_additional_fused_ops]);

    uint32_t workgroups_z = (uint32_t)ne12;
    // When gqa_ratio > 1, each invocation does multiple rows and we can launch fewer workgroups
    ఒకవేళ (gqa_నిష్పత్తి > 1) {
        workgroups_z /= gqa_నిష్పత్తి;
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

స్థిర శూన్య ggml_vk_mul_mat_vec_nc_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    ggml_tensor * dst = cgraph->nodes[కణుపు_idx];
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_nc_f16_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    స్థిరరాశి uint64_t ne00 = src0->ne[0];
    స్థిరరాశి uint64_t ne01 = src0->ne[1];
    స్థిరరాశి uint64_t ne02 = src0->ne[2];
    స్థిరరాశి uint64_t ne03 = src0->ne[3];

    స్థిరరాశి uint64_t nb01 = src0->nb[1];
    స్థిరరాశి uint64_t nb02 = src0->nb[2];

    స్థిరరాశి uint64_t nb12 = src1->nb[2];

    // const uint64_t ne10 = src1->ne[0];
    స్థిరరాశి uint64_t ne11 = src1->ne[1];
    స్థిరరాశి uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    స్థిరరాశి uint32_t nb03 = (uint32_t)(src0->nb[3] / పరిమాణం(ggml_fp16_t));
    స్థిరరాశి uint32_t nb13 = (uint32_t)(src1->nb[3] / పరిమాణం(దశాంశం));
    స్థిరరాశి uint32_t nb23 = (uint32_t)(dst->nb[3] / పరిమాణం(దశాంశం));

    GGML_ASSERT(ne11 == 1);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]); // checked in supports_op

    స్థిరరాశి uint32_t అడ్డువరుస_stride_x = nb01 / పరిమాణం(ggml_fp16_t);
    స్థిరరాశి uint32_t channel_stride_x = nb02 / పరిమాణం(ggml_fp16_t);
    స్థిరరాశి uint32_t channel_stride_y = nb12 / పరిమాణం(దశాంశం);

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_nc_f16_f32;
    ఒకవేళ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_బఫరు_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[కణుపు_idx + ctx->అంకె_additional_fused_ops], నిజం);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, నిజం);
    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    ఒకవేళ (ctx->అంకె_additional_fused_ops > 0) {
        స్థిరరాశి ggml_tensor * కూడు_2 = cgraph->nodes[కణుపు_idx + 1];
        స్థిరరాశి ggml_tensor * bias = కూడు_2->src[0] == dst ? కూడు_2->src[1] : కూడు_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    ఒకవేళ (ctx->అంకె_additional_fused_ops > 1) {
        స్థిరరాశి ggml_tensor * bias = cgraph->nodes[కణుపు_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute
    vk_mat_vec_nc_నెట్టు_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01,
        అడ్డువరుస_stride_x, channel_stride_x, channel_stride_y,
        (uint32_t)(ne12 / ne02), (uint32_t)ne12,
        0, 0,
        nb03, nb13, nb23, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, శూన్యసూచిక, శూన్యసూచిక, cgraph->nodes[కణుపు_idx + ctx->అంకె_additional_fused_ops]);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { (uint32_t)ne03, (uint32_t)ne01, (uint32_t)ne12 });
}

స్థిర పూర్ణాంకం ggml_vk_fwht_pipeline_idx(int64_t n) {
    ఎంపిక (n) {
        స్థితి 64:  తిరిగి 0;
        స్థితి 128: తిరిగి 1;
        స్థితి 256: తిరిగి 2;
        స్థితి 512: తిరిగి 3;
        అప్రమేయ:  తిరిగి -1;
    }
}

తార్కిక ggml_vk_can_use_fwht(స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * dst) {
    ఒకవేళ (ctx->అంకె_additional_fused_ops != 0) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి పూర్ణాంకం idx = ggml_vk_fwht_pipeline_idx(src1->ne[0]);
    ఒకవేళ (idx < 0 || ctx->device->pipeline_fwht_f32[idx] == శూన్యసూచిక) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(src1)) {
        తిరిగి అబద్ధం;
    }
    GGML_ASSERT(ggml_is_contiguous(dst));

    తిరిగి నిజం;
}

శూన్య ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst) {
    స్థిరరాశి పూర్ణాంకం idx = ggml_vk_fwht_pipeline_idx(src->ne[0]);
    vk_pipeline pipeline = ctx->device->pipeline_fwht_f32[idx];

    స్థిరరాశి uint32_t అడ్డువరుసలు_చొప్పున_workgroup = 4;
    స్థిరరాశి uint32_t n_అడ్డువరుసలు = (uint32_t)ggml_nrows(src);
    స్థిరరాశి uint32_t max_workgroups_x = ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[0];

    స్థిరరాశి uint32_t మొత్తం_workgroups = CEIL_DIV(n_అడ్డువరుసలు, అడ్డువరుసలు_చొప్పున_workgroup);
    స్థిరరాశి uint32_t workgroups_x = std::min(మొత్తం_workgroups, max_workgroups_x);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    స్థిరరాశి vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src, నిజం);
    స్థిరరాశి vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, నిజం);

    vk_op_fwht_నెట్టు_constants pc = {
        n_అడ్డువరుసలు,
        0,
        0,
        1.0f / std::వర్గమూలం((దశాంశం)src->ne[0]),
    };
    init_pushconst_tensor_offsets(ctx, pc, src, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc, { workgroups_x, 1, 1 });
}

స్థిర uint32_t ggml_vk_nb_elem(స్థిరరాశి ggml_tensor * t, పూర్ణాంకం i) {
    తిరిగి (uint32_t)(t->nb[i] / ggml_type_కొలత(t->type));
}

శూన్య ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * mixes, స్థిరరాశి ggml_tensor * scale, స్థిరరాశి ggml_tensor * పునాది, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_comb(" << mixes << ", " << scale << ", " << పునాది << ", " << dst << ")");

    vk_pipeline pipeline = ctx->device->pipeline_dsv4_hc_comb_f32;
    GGML_ASSERT(pipeline != శూన్యసూచిక);

    స్థిరరాశి uint32_t n_టోకెన్లు = (uint32_t)mixes->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    స్థిరరాశి vk_subbuffer mixes_buf = ggml_vk_tensor_subbuffer(ctx, mixes, నిజం);
    స్థిరరాశి vk_subbuffer scale_buf = ggml_vk_tensor_subbuffer(ctx, scale, నిజం);
    స్థిరరాశి vk_subbuffer పునాది_buf  = ggml_vk_tensor_subbuffer(ctx, పునాది,  నిజం);
    స్థిరరాశి vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, dst,   నిజం);

    vk_op_dsv4_hc_comb_నెట్టు_constants pc = {
        n_టోకెన్లు,
        ggml_vk_nb_elem(mixes, 0), ggml_vk_nb_elem(mixes, 1),
        ggml_vk_nb_elem(scale, 0),
        ggml_vk_nb_elem(పునాది,  0),
        ggml_vk_nb_elem(dst,   0), ggml_vk_nb_elem(dst, 1), ggml_vk_nb_elem(dst, 2),
        0, 0, 0, 0,
        ggml_get_op_params_f32(dst, 0),
        (uint32_t)ggml_get_op_params_i32(dst, 1),
    };
    init_pushconst_tensor_offsets(ctx, pc, mixes, scale, పునాది, శూన్యసూచిక, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { mixes_buf, scale_buf, పునాది_buf, dst_buf }, pc, { n_టోకెన్లు, 1, 1 });
}

శూన్య ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * x, స్థిరరాశి ggml_tensor * weights, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_pre(" << x << ", " << weights << ", " << dst << ")");

    స్థిరరాశి దశాంశం scale = ggml_get_op_params_f32(dst, 0);
    స్థిరరాశి తార్కిక  gated = ggml_get_op_params_i32(dst, 1) != 0;

    vk_pipeline pipeline = gated ? ctx->device->pipeline_dsv4_hc_pre_gated_f32 : ctx->device->pipeline_dsv4_hc_pre_f32;
    GGML_ASSERT(pipeline != శూన్యసూచిక);

    స్థిరరాశి uint32_t n_embd   = (uint32_t)x->ne[0];
    స్థిరరాశి uint32_t n_టోకెన్లు = (uint32_t)x->ne[2];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    స్థిరరాశి vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,       నిజం);
    స్థిరరాశి vk_subbuffer w_buf = ggml_vk_tensor_subbuffer(ctx, weights, నిజం);
    స్థిరరాశి vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,     నిజం);

    vk_op_dsv4_hc_pre_నెట్టు_constants pc = {
        n_embd, n_టోకెన్లు,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1), ggml_vk_nb_elem(x, 2),
        ggml_vk_nb_elem(weights, 0), ggml_vk_nb_elem(weights, 1), ggml_vk_nb_elem(weights, 2),
        ggml_vk_nb_elem(dst, 0), ggml_vk_nb_elem(dst, 1),
        0, 0, 0,
        scale,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, weights, శూన్యసూచిక, శూన్యసూచిక, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, w_buf, d_buf }, pc, { n_embd, n_టోకెన్లు, 1 });
}

శూన్య ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * x, స్థిరరాశి ggml_tensor * residual, స్థిరరాశి ggml_tensor * post, స్థిరరాశి ggml_tensor * comb, ggml_tensor * dst, స్థిరరాశి ggml_tensor * gate_scale_లోపల) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_post(" << x << ", " << residual << ", " << post << ", " << comb << ", " << dst << ")");

    vk_pipeline pipeline = comb ? ctx->device->pipeline_dsv4_hc_post_f32 : ctx->device->pipeline_dsv4_hc_post_nocomb_f32;
    GGML_ASSERT(pipeline != శూన్యసూచిక);

    స్థిరరాశి uint32_t n_embd   = (uint32_t)x->ne[0];
    స్థిరరాశి uint32_t n_టోకెన్లు = (uint32_t)x->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    స్థిరరాశి vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,        నిజం);
    స్థిరరాశి vk_subbuffer r_buf = ggml_vk_tensor_subbuffer(ctx, residual, నిజం);
    // with a fused gate, post is scale(sigmoid(scale(p_src))) and the shader applies it to p_src
    స్థిరరాశి ggml_tensor * p_src = gate_scale_లోపల ? gate_scale_లోపల->src[0] : post;
    స్థిరరాశి vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p_src,    నిజం);
    స్థిరరాశి vk_subbuffer c_buf = comb ? ggml_vk_tensor_subbuffer(ctx, comb, నిజం) : x_buf;
    స్థిరరాశి vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,      నిజం);

    vk_op_dsv4_hc_post_నెట్టు_constants pc = {
        n_embd, n_టోకెన్లు,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1),
        ggml_vk_nb_elem(residual, 0), ggml_vk_nb_elem(residual, 1), ggml_vk_nb_elem(residual, 2),
        ggml_vk_nb_elem(p_src, 0), ggml_vk_nb_elem(p_src, 1),
        comb ? ggml_vk_nb_elem(comb, 0) : 0, comb ? ggml_vk_nb_elem(comb, 1) : 0, comb ? ggml_vk_nb_elem(comb, 2) : 0,
        ggml_vk_nb_elem(dst,  0), ggml_vk_nb_elem(dst,  1), ggml_vk_nb_elem(dst,  2),
        0, 0, 0, 0, 0,
        gate_scale_లోపల ? 1u : 0u,
        gate_scale_లోపల ? ggml_get_op_params_f32(gate_scale_లోపల, 0) : 1.0f,
        gate_scale_లోపల ? ggml_get_op_params_f32(post, 0) : 1.0f,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, residual, p_src, comb, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, r_buf, p_buf, c_buf, d_buf }, pc, { n_embd, n_టోకెన్లు, 1 });
}

శూన్య ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    ggml_tensor * dst = cgraph->nodes[కణుపు_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")");

    // Handle huge A matrix by splitting the M dimensions. This works well for convolution use cases
    // where the M dimension is very large.
    // Split_k doesn't work with M splitting.
    // This only supports batchsize == 1.
    స్థిరరాశి size_t nbytes = ggml_nbytes(src0);
    స్థిరరాశి తార్కిక needs_విభజించు = dst->ne[2] == 1 && dst->ne[3] == 1 && nbytes > ctx->device->properties.limits.max_Storage_బఫరు_Range;
    ఒకవేళ (needs_విభజించు) {
        // Choose the number of rows that can fit (and divide by two, to allow for any additional offsets)
        స్థిరరాశి uint32_t M_విభజించు = ctx->device->properties.limits.max_Storage_బఫరు_Range / (2 * src0->nb[1]);
        uint32_t m_స్థానభ్రంశం = 0;
        అయితే (m_స్థానభ్రంశం < dst->ne[0]) {
            స్థిరరాశి uint32_t cur_M_కొలత = std::min(M_విభజించు, (uint32_t)(dst->ne[0] - m_స్థానభ్రంశం));
            ggml_tensor dst2 = *dst;
            ggml_tensor src02 = *src0;

            dst2.view_src = dst->view_src ? dst->view_src : dst;
            src02.view_src = src0->view_src ? src0->view_src : src0;

            dst2.view_offs += m_స్థానభ్రంశం * dst->nb[0];
            src02.view_offs += m_స్థానభ్రంశం * src0->nb[1];
            dst2.ne[0] = cur_M_కొలత;
            src02.ne[1] = cur_M_కొలత;

            ggml_vk_mul_mat_q_f16(ctx, subctx, &src02, src1, &dst2, నిజం);

            m_స్థానభ్రంశం += cur_M_కొలత;
        }
    } లేకపోతే ఒకవేళ (ggml_vk_can_use_fwht(ctx, src1, dst)) {
        ggml_vk_fwht(ctx, subctx, src1, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && dst->ne[1] == 1 &&
        // detect 0213 permutation, and batch size of 1
        src0->nb[0] <= src0->nb[2] &&
        src0->nb[2] <= src0->nb[1] &&
        src0->nb[1] <= src0->nb[3] &&
        src1->nb[0] <= src1->nb[2] &&
        src1->nb[2] <= src1->nb[1] &&
        src1->nb[1] <= src1->nb[3] &&
        src0->ne[3] == 1 &&
        src1->ne[3] == 1 &&
        src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1] &&
        src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]) {
        ggml_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, cgraph, కణుపు_idx);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && dst->ne[1] == 1 &&
               !ggml_is_permuted(src0) && !ggml_is_permuted(src1) &&
               src0->ne[3] <= ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[0] &&
               src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1] &&
               src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]) {
        ggml_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, cgraph, కణుపు_idx);
    // With one output row, B^T*A has the same flat output as A^T*B.
    } లేకపోతే ఒకవేళ (ctx->అంకె_additional_fused_ops == 0 &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
               (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16 || ggml_is_quantized(src1->type)) &&
               dst->ne[0] == 1 && dst->ne[1] > mul_mat_vec_max_నిలువుల_సంఖ్య &&
               src0->ne[2] == 1 && src0->ne[3] == 1 &&
               src1->ne[2] == 1 && src1->ne[3] == 1 &&
               ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
               get_misalign_బైట్లు(ctx, src0) == 0 && get_misalign_బైట్లు(ctx, src1) == 0 && get_misalign_బైట్లు(ctx, dst) == 0) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, కణుపు_idx, నిజం);
    // mul_mat_vec supports batching ne12*ne13 when ne11==1, or treating ne11 as the batch size (up to four)
    // when ne12 and ne13 are one.
    } లేకపోతే ఒకవేళ ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_నిలువుల_సంఖ్య && src1->ne[2] * src1->ne[3] == 1)) &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, కణుపు_idx);
    } లేకపోతే {
        ggml_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst, అబద్ధం);
    }
}

స్థిర శూన్య ggml_vk_mul_mat_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * ids, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    స్థిరరాశి uint64_t ne00 = src0->ne[0];
    స్థిరరాశి uint64_t ne01 = src0->ne[1];
    స్థిరరాశి uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    స్థిరరాశి uint64_t ne10 = src1->ne[0];
    స్థిరరాశి uint64_t ne11 = src1->ne[1];
    స్థిరరాశి uint64_t ne12 = src1->ne[2];
    స్థిరరాశి uint64_t ne13 = src1->ne[3];

    స్థిరరాశి uint64_t nei0 = ids->ne[0];
    స్థిరరాశి uint64_t nei1 = ids->ne[1];

    స్థిరరాశి uint32_t nbi0 = ids->nb[0];
    స్థిరరాశి uint32_t nbi1 = ids->nb[1];

    స్థిరరాశి uint64_t ne20 = dst->ne[0];
    స్థిరరాశి uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    స్థిరరాశి uint64_t n_as = ne02;

    ggml_backend_vk_బఫరు_context * dst_buf_ctx = (ggml_backend_vk_బఫరు_context *)dst->బఫరు->context;
    ggml_backend_vk_బఫరు_context * src0_buf_ctx = (ggml_backend_vk_బఫరు_context *)src0->బఫరు->context;
    ggml_backend_vk_బఫరు_context * src1_buf_ctx = (ggml_backend_vk_బఫరు_context *)src1->బఫరు->context;

    vk_బఫరు d_Qx = శూన్యసూచిక;
    size_t qx_buf_స్థానభ్రంశం = 0;
    vk_బఫరు d_Qy = శూన్యసూచిక;
    size_t qy_buf_స్థానభ్రంశం = 0;

    తార్కిక src0_uma = అబద్ధం;
    తార్కిక src1_uma = అబద్ధం;

    ఒకవేళ (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->దత్తాంశం, d_Qx, qx_buf_స్థానభ్రంశం);
        ggml_vk_host_get(ctx->device, src1->దత్తాంశం, d_Qy, qy_buf_స్థానభ్రంశం);
        src0_uma = d_Qx != శూన్యసూచిక;
        src1_uma = d_Qy != శూన్యసూచిక;
    }

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    స్థిరరాశి తార్కిక x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
#ఒకవేళ defined(GGML_VULKAN_COOPMAT2_DECODE_సదిశ_GLSLC_SUPPORT)
    // B must already be, or be convertible to, the matmul B type used by this path.
    స్థిరరాశి తార్కిక y_decode_సదిశ_supported = ctx->device->coopmat2_decode_సదిశ &&
                                           (f16_type != GGML_TYPE_BF16 || ctx->device->coopmat2_bf16_support) &&
                                           (src1->type == GGML_TYPE_F32 || src1->type == f16_type);
    // If B is copied to prealloc_y, we can choose a 4-element-aligned row stride.
    స్థిరరాశి తార్కిక y_decode_సదిశ_ఉపయోగిస్తుంది_prealloc = !ggml_vk_dim01_contiguous(src1) || src1->type != f16_type;
    // Direct B reads are safe only if row starts and the original buffer offset are 4-element aligned.
    స్థిరరాశి తార్కిక y_decode_సదిశ_aligned =
        (ne10 % 4 == 0) &&
        (y_decode_సదిశ_ఉపయోగిస్తుంది_prealloc || get_misalign_బైట్లు(ctx, src1) % (4 * ggml_type_కొలత(src1->type)) == 0);
    // Stage B only when decode-vector is available and direct B reads would be misaligned.
    స్థిరరాశి తార్కిక y_decode_సదిశ_staging = y_decode_సదిశ_supported && !y_decode_సదిశ_aligned;
#లేకపోతే
    స్థిరరాశి తార్కిక y_decode_సదిశ_staging = అబద్ధం;
#అయితే_ముగింపు
    స్థిరరాశి తార్కిక can_quantize_y = ctx->device->integer_dot_లబ్ధం && src1->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // Intel coopmat1 matrix cores prefer f16 activations for quantized weights
    స్థిరరాశి తార్కిక prefer_f16_b = ctx->device->coopmat_support && !ctx->device->coopmat2 &&
                              ctx->device->vendor_id == VK_VENDOR_ID_INTEL &&
                              ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    స్థిరరాశి ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], నిజం, can_quantize_y, prefer_f16_b);

    స్థిరరాశి తార్కిక quantize_y   = y_kernel_type == GGML_TYPE_Q8_1;
    స్థిరరాశి తార్కిక y_f32_kernel = y_kernel_type == GGML_TYPE_F32;

    స్థిరరాశి std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0], నిజం);

    స్థిరరాశి తార్కిక qx_needs_dequant = mmp_map == శూన్యసూచిక || x_non_contig;

    ఒకవేళ (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0], నిజం);
    }

    GGML_ASSERT(mmp_map != శూన్యసూచిక);

    // Coopmat2 MUL_MAT_ID BK specialization constants in ggml_vk_load_shaders are at most 64.
    స్థిరరాశి uint32_t y_staged_అడ్డువరుస_stride = ctx->device->coopmat2 && !quantize_y ? ggml_vk_align_కొలత(ne10, 64) : ne10;
    స్థిరరాశి తార్కిక y_needs_k_padding = ne10 != y_staged_అడ్డువరుస_stride;

    స్థిరరాశి తార్కిక y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type ||
                                                  y_decode_సదిశ_staging || y_needs_k_padding);
    స్థిరరాశి తార్కిక qy_needs_dequant = y_needs_reformat;

    స్థిరరాశి uint32_t kpad = quantize_y ? 0 : ggml_vk_align_కొలత(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, nei1, నిజం));
    స్థిరరాశి తార్కిక aligned = !quantize_y && ne10 == kpad && ne01 > 8 && nei1 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, nei1, aligned, నిజం);

    స్థిరరాశి uint32_t bn = pipeline->wg_denoms[1];
    స్థిరరాశి uint64_t max_tiles = ggml_vk_mul_mat_id_max_tiles(n_as, nei0 * nei1, bn);
    స్థిరరాశి uint64_t అడ్డువరుస_ids_స్థానభ్రంశం = 1 + 2 * max_tiles;
    స్థిరరాశి uint64_t అడ్డువరుస_map_పదాలు = అడ్డువరుస_ids_స్థానభ్రంశం + max_tiles * bn;

    ఒకవేళ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_బఫరు_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }
    స్థిరరాశి uint64_t x_ne = ggml_nelements(src0);
    స్థిరరాశి uint64_t y_ne = (uint64_t)y_staged_అడ్డువరుస_stride * ne11 * ne12 * ne13;
    స్థిరరాశి uint64_t d_ne = ggml_nelements(dst);

    స్థిరరాశి uint64_t qx_sz = ggml_type_కొలత(src0->type) * x_ne / ggml_blck_కొలత(src0->type);
    స్థిరరాశి uint64_t qy_sz = ggml_type_కొలత(src1->type) * ggml_nelements(src1) / ggml_blck_కొలత(src1->type);
    స్థిరరాశి uint64_t x_sz = !qx_needs_dequant ? qx_sz : పరిమాణం(ggml_fp16_t) * x_ne;
    స్థిరరాశి uint64_t y_sz = quantize_y ? (ggml_vk_align_కొలత(y_ne, 128) * ggml_type_కొలత(GGML_TYPE_Q8_1) / ggml_blck_కొలత(GGML_TYPE_Q8_1)) : (y_f32_kernel ? పరిమాణం(దశాంశం) * y_ne : పరిమాణం(ggml_fp16_t) * y_ne);
    స్థిరరాశి uint64_t d_sz = పరిమాణం(దశాంశం) * d_ne;

    vk_pipeline గమ్యం_fp16_vk_0 = శూన్యసూచిక;
    vk_pipeline గమ్యం_fp16_vk_1 = శూన్యసూచిక;
    vk_pipeline గమ్యం_q8_1 = శూన్యసూచిక;

    స్వయంచాలక make_y_staged_dst = [&]() {
        ggml_tensor y_staged_dst = *src1;
        y_staged_dst.type = f16_type;
        y_staged_dst.nb[0] = ggml_type_కొలత(f16_type);
        y_staged_dst.nb[1] = y_staged_dst.nb[0] * y_staged_అడ్డువరుస_stride;
        y_staged_dst.nb[2] = y_staged_dst.nb[1] * ne11;
        y_staged_dst.nb[3] = y_staged_dst.nb[2] * y_staged_dst.ne[2];
        తిరిగి y_staged_dst;
    };

    ఒకవేళ (x_non_contig) {
        గమ్యం_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, శూన్యసూచిక, f16_type);
    } లేకపోతే {
        గమ్యం_fp16_vk_0 = ggml_vk_get_గమ్యం_fp16(ctx, src0->type);
    }
    ఒకవేళ (y_needs_reformat) {
        ggml_tensor y_staged_dst;
        స్థిరరాశి ggml_tensor * y_staged_dst_ptr = శూన్యసూచిక;
        ఒకవేళ (y_needs_k_padding) {
            y_staged_dst = make_y_staged_dst();
            y_staged_dst_ptr = &y_staged_dst;
        }

        గమ్యం_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, y_staged_dst_ptr, f16_type);
    } లేకపోతే {
        గమ్యం_fp16_vk_1 = ggml_vk_get_గమ్యం_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || గమ్యం_fp16_vk_0 != శూన్యసూచిక);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || గమ్యం_fp16_vk_1 != శూన్యసూచిక);  // NOLINT

    ఒకవేళ (quantize_y) {
        గమ్యం_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }
    vk_pipeline లెక్క_experts = ctx->device->pipeline_లెక్క_experts;

    స్థిరరాశి size_t expert_దత్తాంశం_కొలత = పరిమాణం(uint32_t) * అడ్డువరుస_map_పదాలు;

    {
        ఒకవేళ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_బఫరు_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_బఫరు_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ఒకవేళ (qx_needs_dequant && ctx->prealloc_కొలత_x < x_sz) {
            ctx->prealloc_కొలత_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ఒకవేళ ((qy_needs_dequant || quantize_y) && ctx->prealloc_కొలత_y < y_sz) {
            ctx->prealloc_కొలత_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ఒకవేళ (ctx->prealloc_కొలత_విభజించు_k < expert_దత్తాంశం_కొలత) {
            ctx->prealloc_కొలత_విభజించు_k = expert_దత్తాంశం_కొలత;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ఒకవేళ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, గమ్యం_fp16_vk_0, 1);
        }
        ఒకవేళ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, గమ్యం_fp16_vk_1, 1);
        }
        ఒకవేళ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, గమ్యం_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, లెక్క_experts, 1);
    }

    vk_బఫరు d_D = dst_buf_ctx->dev_బఫరు;
    స్థిరరాశి uint64_t d_buf_స్థానభ్రంశం = vk_tensor_స్థానభ్రంశం(dst) + dst->view_offs;
    GGML_ASSERT(d_D != శూన్యసూచిక);
    vk_బఫరు d_X;
    uint64_t x_buf_స్థానభ్రంశం = 0;
    vk_బఫరు d_Y;
    uint64_t y_buf_స్థానభ్రంశం = 0;
    ఒకవేళ (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_బఫరు;
        qx_buf_స్థానభ్రంశం = vk_tensor_స్థానభ్రంశం(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != శూన్యసూచిక);
    }
    ఒకవేళ (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_బఫరు;
        qy_buf_స్థానభ్రంశం = vk_tensor_స్థానభ్రంశం(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != శూన్యసూచిక);
    }
    ఒకవేళ (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } లేకపోతే {
        d_X = d_Qx;
        x_buf_స్థానభ్రంశం = qx_buf_స్థానభ్రంశం;
        GGML_ASSERT(qx_sz == x_sz);
    }
    ఒకవేళ (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } లేకపోతే ఒకవేళ (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } లేకపోతే {
        d_Y = d_Qy;
        y_buf_స్థానభ్రంశం = qy_buf_స్థానభ్రంశం;
        GGML_ASSERT(qy_sz == y_sz);
    }

    ఒకవేళ (x_non_contig || qx_needs_dequant) {
        ఒకవేళ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids, నిజం);

    // Count how many times each expert is used
    vk_subbuffer expert_లెక్క_buf = { ctx->prealloc_విభజించు_k, 0, expert_దత్తాంశం_కొలత };
    ఒకవేళ (ctx->prealloc_విభజించు_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }
    {
        vk_op_లెక్క_experts_నెట్టు_constants pc = { (uint32_t)nei0,
                                           (uint32_t)nei1,
                                           (uint32_t)(nbi0 / ggml_type_కొలత(ids->type)),
                                           (uint32_t)(nbi1 / ggml_type_కొలత(ids->type)),
                                           (uint32_t)(get_misalign_బైట్లు(ctx, ids) / ggml_type_కొలత(ids->type)),
                                           (uint32_t)n_as,
                                           0, 0, bn, (uint32_t)అడ్డువరుస_ids_స్థానభ్రంశం };
        init_pushconst_fastdiv(pc);
        ggml_vk_dispatch_pipeline(ctx, subctx, లెక్క_experts,
            { d_ids, expert_లెక్క_buf }, pc,
            { 1, 1, 1 });
    }

    ఒకవేళ (x_non_contig) {
        ggml_vk_cpy_గమ్యం_contiguous(ctx, subctx, గమ్యం_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_స్థానభ్రంశం), ggml_vk_subbuffer(ctx, d_X, 0));
    } లేకపోతే ఒకవేళ (qx_needs_dequant) {
        స్థిరరాశి std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, గమ్యం_fp16_vk_0,
            { vk_subbuffer{ d_Qx, qx_buf_స్థానభ్రంశం, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)x_ne, 1, 1});
    }
    ఒకవేళ (y_needs_reformat) {
        ఒకవేళ (ctx->prealloc_y_చివరి_pipeline_వాడిన != గమ్యం_fp16_vk_1.get() ||
            ctx->prealloc_y_చివరి_tensor_వాడిన != src1 ||
            ctx->prealloc_y_చివరి_k_padded != y_needs_k_padding) {
            ఒకవేళ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ఒకవేళ (y_needs_k_padding) {
                GGML_ASSERT(y_sz % 4 == 0);
                // Zero B padding because clamping only A can produce 0 * Inf or NaN.
                subctx->s->బఫరు->buf.fill_బఫరు(d_Y->బఫరు, 0, y_sz, 0);
                ggml_vk_sync_buffers(ctx, subctx);
                స్థిరరాశి ggml_tensor y_staged_dst = make_y_staged_dst();
                స్థిరరాశి uint32_t y_staged_dst_type_కొలత = ggml_type_కొలత(y_staged_dst.type);
                ggml_vk_cpy_గమ్యం_strided(
                    ctx, subctx, గమ్యం_fp16_vk_1, src1,
                    ggml_vk_subbuffer(ctx, d_Qy, qy_buf_స్థానభ్రంశం), ggml_vk_subbuffer(ctx, d_Y, 0),
                    (uint32_t)(y_staged_dst.nb[0] / y_staged_dst_type_కొలత),
                    (uint32_t)(y_staged_dst.nb[1] / y_staged_dst_type_కొలత),
                    (uint32_t)(y_staged_dst.nb[2] / y_staged_dst_type_కొలత),
                    (uint32_t)(y_staged_dst.nb[3] / y_staged_dst_type_కొలత));
            } లేకపోతే {
                ggml_vk_cpy_గమ్యం_contiguous(ctx, subctx, గమ్యం_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_స్థానభ్రంశం), ggml_vk_subbuffer(ctx, d_Y, 0));
            }
            ctx->prealloc_y_చివరి_pipeline_వాడిన = గమ్యం_fp16_vk_1.get();
            ctx->prealloc_y_చివరి_tensor_వాడిన = src1;
            ctx->prealloc_y_చివరి_k_padded = y_needs_k_padding;
        }
    }
    ఒకవేళ (quantize_y) {
        ఒకవేళ (ctx->prealloc_y_చివరి_pipeline_వాడిన != గమ్యం_q8_1.get() ||
            ctx->prealloc_y_చివరి_tensor_వాడిన != src1 ||
            ctx->prealloc_y_చివరి_k_padded) {
            ఒకవేళ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_స్థానభ్రంశం), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_చివరి_pipeline_వాడిన = గమ్యం_q8_1.get();
            ctx->prealloc_y_చివరి_tensor_వాడిన = src1;
            ctx->prealloc_y_చివరి_k_padded = అబద్ధం;
        }
    }
    ggml_vk_sync_buffers(ctx, subctx);

    స్థిరరాశి uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    స్థిరరాశి uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_b_y = y_needs_k_padding ? y_staged_అడ్డువరుస_stride : ne10;
    uint32_t stride_batch_y = y_needs_k_padding ? y_staged_అడ్డువరుస_stride * ne11 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));

    ఒకవేళ (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_కొలత(src0->type);
    }

    ఒకవేళ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_కొలత(src1->type);
    }

    // compute
    ggml_vk_matmul_id(
        ctx, subctx, pipeline,
        { d_X, x_buf_స్థానభ్రంశం, x_range }, { d_Y, y_buf_స్థానభ్రంశం, y_range },
        { d_D, d_buf_స్థానభ్రంశం, d_sz }, std::move(d_ids), expert_లెక్క_buf,
        ne01, ne21, ne10, ne10, stride_b_y, ne01,
        stride_batch_x, stride_batch_y, ne20*ne21,
        n_as, nei0, nei1, nbi1 / ggml_type_కొలత(ids->type), ne11, అడ్డువరుస_ids_స్థానభ్రంశం, max_tiles
    );  // NOLINT

    ఒకవేళ (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = నిజం;
    }
    ఒకవేళ (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = నిజం;
    }
    ctx->prealloc_విభజించు_k_need_sync = నిజం;
}

స్థిర శూన్య ggml_vk_mul_mat_vec_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    ggml_tensor * dst = cgraph->nodes[కణుపు_idx];
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

    స్థిరరాశి uint64_t ne00 = src0->ne[0];
    స్థిరరాశి uint64_t ne01 = src0->ne[1];
    // const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    స్థిరరాశి uint64_t ne10 = src1->ne[0];
    స్థిరరాశి uint64_t ne11 = src1->ne[1];
    స్థిరరాశి uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    స్థిరరాశి uint64_t nei0 = ids->ne[0];
    స్థిరరాశి uint64_t nei1 = ids->ne[1];
    స్థిరరాశి uint32_t nbi1 = (uint32_t)(ids->nb[1] / పరిమాణం(పూర్ణాంకం));

    స్థిరరాశి uint64_t ne20 = dst->ne[0];
    స్థిరరాశి uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    స్థిరరాశి తార్కిక x_non_contig = !ggml_vk_dim01_contiguous(src0);
    స్థిరరాశి తార్కిక y_non_contig = !ggml_vk_dim01_contiguous(src1);

    స్థిరరాశి తార్కిక f16_f32_kernel = src1->type == GGML_TYPE_F32;
    తార్కిక quantize_y = ctx->device->integer_dot_లబ్ధం && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne12, ne10, src0->type);

    vk_pipeline గమ్యం_fp16_vk_0 = శూన్యసూచిక;
    vk_pipeline గమ్యం_fp16_vk_1 = శూన్యసూచిక;
    ఒకవేళ (x_non_contig) {
        గమ్యం_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, శూన్యసూచిక, src0->type);
    }
    ఒకవేళ (y_non_contig) {
        గమ్యం_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, శూన్యసూచిక, src1->type);
    } లేకపోతే {
        గమ్యం_fp16_vk_1 = ggml_vk_get_గమ్యం_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, GGML_TYPE_Q8_1, ne20, ne00) : శూన్యసూచిక;
    vk_pipeline గమ్యం_q8_1 = శూన్యసూచిక;

    ఒకవేళ (dmmv == శూన్యసూచిక) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, src1->type, ne20, ne00);
        quantize_y = అబద్ధం;
    }

    ఒకవేళ (quantize_y) {
        గమ్యం_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    స్థిరరాశి తార్కిక qx_needs_dequant = x_non_contig;
    స్థిరరాశి తార్కిక qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    ఒకవేళ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_బఫరు_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT
    GGML_ASSERT(!qx_needs_dequant || గమ్యం_fp16_vk_0 != శూన్యసూచిక);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || గమ్యం_fp16_vk_1 != శూన్యసూచిక);  // NOLINT
    GGML_ASSERT(dmmv != శూన్యసూచిక);

    స్థిరరాశి uint64_t x_ne = ggml_nelements(src0);
    స్థిరరాశి uint64_t y_ne = ggml_nelements(src1);

    స్థిరరాశి uint64_t qx_sz = ggml_vk_align_కొలత(ggml_type_కొలత(src0->type) * x_ne / ggml_blck_కొలత(src0->type), ctx->device->properties.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment);
    స్థిరరాశి uint64_t x_sz = x_non_contig ? ggml_vk_align_కొలత(ggml_type_కొలత(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment) : qx_sz;
    స్థిరరాశి uint64_t y_sz = quantize_y ? (ggml_vk_align_కొలత(y_ne, 128) * ggml_type_కొలత(GGML_TYPE_Q8_1) / ggml_blck_కొలత(GGML_TYPE_Q8_1)) :
                                       (f16_f32_kernel ? పరిమాణం(దశాంశం) * y_ne : పరిమాణం(ggml_fp16_t) * y_ne);

    {
        ఒకవేళ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_బఫరు_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_బఫరు_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ఒకవేళ (qx_needs_dequant && ctx->prealloc_కొలత_x < x_sz) {
            ctx->prealloc_కొలత_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ఒకవేళ ((qy_needs_dequant || quantize_y) && ctx->prealloc_కొలత_y < y_sz) {
            ctx->prealloc_కొలత_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ఒకవేళ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, గమ్యం_fp16_vk_0, 1);
        }
        ఒకవేళ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, గమ్యం_fp16_vk_1, 1);
        }
        ఒకవేళ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, గమ్యం_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, dmmv, nei1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[కణుపు_idx + ctx->అంకె_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids);
    vk_subbuffer d_F0 = d_D;
    vk_subbuffer d_X, d_Y;

    ఒకవేళ (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } లేకపోతే {
        d_X = d_Qx;
    }
    ఒకవేళ (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } లేకపోతే {
        d_Y = d_Qy;
    }

    ఒకవేళ (x_non_contig) {
        ఒకవేళ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ఒకవేళ (x_non_contig) {
        GGML_ASSERT(x_sz == ggml_vk_align_కొలత(ggml_type_కొలత(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment));
        ggml_vk_cpy_గమ్యం_contiguous(ctx, subctx, గమ్యం_fp16_vk_0, src0, d_Qx, d_X);
    }
    ఒకవేళ (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_కొలత(src1->type) * y_ne);
        ఒకవేళ (ctx->prealloc_y_చివరి_pipeline_వాడిన != గమ్యం_fp16_vk_1.get() ||
            ctx->prealloc_y_చివరి_tensor_వాడిన != src1 ||
            ctx->prealloc_y_చివరి_k_padded) {
            ఒకవేళ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_గమ్యం_contiguous(ctx, subctx, గమ్యం_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_చివరి_pipeline_వాడిన = గమ్యం_fp16_vk_1.get();
            ctx->prealloc_y_చివరి_tensor_వాడిన = src1;
            ctx->prealloc_y_చివరి_k_padded = అబద్ధం;
        }
    }
    ఒకవేళ (quantize_y) {
        ఒకవేళ (ctx->prealloc_y_చివరి_pipeline_వాడిన != గమ్యం_q8_1.get() ||
            ctx->prealloc_y_చివరి_tensor_వాడిన != src1 ||
            ctx->prealloc_y_చివరి_k_padded) {
            ఒకవేళ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_చివరి_pipeline_వాడిన = గమ్యం_q8_1.get();
            ctx->prealloc_y_చివరి_tensor_వాడిన = src1;
            ctx->prealloc_y_చివరి_k_padded = అబద్ధం;
        }
    }

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    ఒకవేళ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[2] / ggml_type_కొలత(src1->type);
    }

    స్థిరరాశి uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    ఒకవేళ (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    ఒకవేళ (ctx->అంకె_additional_fused_ops > 0) {
        స్థిరరాశి ggml_tensor * bias = cgraph->nodes[కణుపు_idx + 1]->src[1];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);

        ఒకవేళ (cgraph->nodes[కణుపు_idx + 1]->op == GGML_OP_MUL) {
            fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE0;
        } లేకపోతే {
            GGML_ASSERT(cgraph->nodes[కణుపు_idx + 1]->op == GGML_OP_కూడు_ID);
            fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
        }
    }

    vk_subbuffer d_F1 = d_D;
    ఒకవేళ (ctx->అంకె_additional_fused_ops > 1) {
        స్థిరరాశి ggml_tensor * scale = cgraph->nodes[కణుపు_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, scale);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE1;
    }

    // Loop over the batch dimension
    కొరకు (uint32_t expert_i1 = 0; expert_i1 < nei1; ++expert_i1) {
        స్థిరరాశి vk_mat_vec_id_నెట్టు_constants pc = {
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

    ఒకవేళ (x_non_contig) {
        ctx->prealloc_x_need_sync = నిజం;
    }
    ఒకవేళ (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = నిజం;
    }
}

స్థిర తార్కిక ggml_vk_use_mul_mat_vec_id(స్థిరరాశి ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * ids = dst->src[2];
    తిరిగి ids->ne[1] <= 8 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));
}

తార్కిక ggml_vk_use_mul_mat_vec_id(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    తిరిగి ggml_vk_use_mul_mat_vec_id(cgraph->nodes[కణుపు_idx]);
}

శూన్య ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    ggml_tensor * dst = cgraph->nodes[కణుపు_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * src2 = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_id(" << src0 << ", " << src1 << ", " << src2 << ", " << dst << ")");
    ఒకవేళ (ggml_vk_use_mul_mat_vec_id(cgraph, కణుపు_idx)) {
        ggml_vk_mul_mat_vec_id_q_f16(ctx, subctx, cgraph, కణుపు_idx);
    } లేకపోతే {
        ggml_vk_mul_mat_id_q_f16(ctx, subctx, src0, src1, src2, dst);
    }
}

తార్కిక ggml_vk_flash_attn_scalar_shmem_support(స్థిరరాశి vk_device& device, స్థిరరాశి vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, తార్కిక f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_వాడని(f32acc);
    // Needs to be kept up to date on shader changes
    స్థిరరాశి uint32_t wg_కొలత = params.workgroup_కొలత;
    స్థిరరాశి uint32_t Br = params.ఖండం_అడ్డువరుసలు;
    స్థిరరాశి uint32_t Bc = params.ఖండం_నిలువుల_సంఖ్య;

    // BF16 uses the fp32 shader (FLOAT_TYPE=float)
    స్థిరరాశి uint32_t float_type_కొలత = (device->fp16 && k_type != GGML_TYPE_BF16) ? పరిమాణం(ggml_fp16_t) : పరిమాణం(దశాంశం);

    స్థిరరాశి తార్కిక mmq = ggml_vk_fa_scalar_ఉపయోగిస్తుంది_mmq(device, k_type, v_type);

    // tmpsh is overestimated slightly
    స్థిరరాశి uint32_t tmpsh = wg_కొలత * పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t tmpshv4 = wg_కొలత * 4 * float_type_కొలత;

    స్థిరరాశి uint32_t masksh = Bc * (Br + 1) * float_type_కొలత;
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    స్థిరరాశి uint32_t iq_shmem = 16 * float_type_కొలత;

    uint32_t Qf, kvsh, kblocksh_కొలత;
    ఒకవేళ (mmq) {
        // block_b_cache: int32_t qs[8] + FLOAT_TYPEV2 ds
        స్థిరరాశి uint32_t ఖండం_b_కొలత = 8 * పరిమాణం(int32_t) + 2 * float_type_కొలత;
        Qf = Br * (hsk / 32) * ఖండం_b_కొలత;

        // kvsh uses D = HSV (K goes through kblocksh instead)
        kvsh = params.shmem_staging ? Bc * (hsv / 4 + 1) * 4 * float_type_కొలత : 4 * float_type_కొలత;

        // The mixed MMQ shader uses a superset block_a_cache that fits every
        // FA-supported quant: int32_t qs[8] + uint32_t qh + FLOAT_TYPEV2 dm.
        // Single-scale types leave dm.y unused; non-Q5_* leave qh unused.
        స్థిరరాశి uint32_t ఖండం_a_కొలత = 8 * పరిమాణం(int32_t) + పరిమాణం(uint32_t) + 2 * float_type_కొలత;
        kblocksh_కొలత = params.shmem_staging ? Bc * (hsk / 32) * ఖండం_a_కొలత : ఖండం_a_కొలత;
    } లేకపోతే {
        Qf = Br * (hsk / 4 + 1) * 4 * float_type_కొలత;

        స్థిరరాశి uint32_t D = std::max(hsk, hsv);
        kvsh = params.shmem_staging ? Bc * (D / 4 + 1) * 4 * float_type_కొలత : 4 * float_type_కొలత;

        kblocksh_కొలత = 0;
    }

    స్థిరరాశి uint32_t మొత్తం_కొలత = tmpsh + tmpshv4 + masksh + iq_shmem + Qf + kvsh + kblocksh_కొలత;
    స్థిరరాశి తార్కిక supported = మొత్తం_కొలత <= device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత;

    VK_LOG_DEBUG("ggml_vk_flash_attn_scalar_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", mmq=" << mmq << ", total_size=" << మొత్తం_కొలత << ", supported=" << supported);

    తిరిగి supported;
}

తార్కిక ggml_vk_flash_attn_coopmat_shmem_support(స్థిరరాశి vk_device& device, స్థిరరాశి vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, తార్కిక f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_వాడని(v_type);
    // Needs to be kept up to date on shader changes
    స్థిరరాశి uint32_t Br = params.ఖండం_అడ్డువరుసలు;
    స్థిరరాశి uint32_t Bc = params.ఖండం_నిలువుల_సంఖ్య;

    స్థిరరాశి uint32_t MatBr = 16, MatBc = 16;

    స్థిరరాశి uint32_t అడ్డువరుస_విభజించు = Bc / MatBc;

    స్థిరరాశి uint32_t hsk_pad = ROUNDUP_POW2(hsk, 16);
    స్థిరరాశి uint32_t hsv_pad = ROUNDUP_POW2(hsv, 16);

    స్థిరరాశి uint32_t acctype = f32acc ? 4 : 2;
    స్థిరరాశి uint32_t f16vec4 = 8;

    స్థిరరాశి uint32_t tmpsh = (Bc / MatBc) * పరిమాణం(దశాంశం);
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    స్థిరరాశి uint32_t iq_shmem = 16 * పరిమాణం(ggml_fp16_t);

    స్థిరరాశి uint32_t qstride = hsk_pad / 4 + 2;
    స్థిరరాశి uint32_t Qf = Br * qstride * f16vec4;

    స్థిరరాశి uint32_t psh_stride = Br / 4 + 2;
    స్థిరరాశి uint32_t Psh = Bc * psh_stride * f16vec4;

    స్థిరరాశి uint32_t sfshstride = (hsk <= 128) ? (Br + 8) : Br;
    స్థిరరాశి uint32_t sfsh = Bc * sfshstride * acctype;

    స్థిరరాశి uint32_t kvshstride = (params.shmem_staging ? std::max(hsk_pad, hsv_pad) : MatBr) / 4 + 2;
    స్థిరరాశి uint32_t vsh_stride = MatBc / 4 * అడ్డువరుస_విభజించు;
    స్థిరరాశి uint32_t ksh = ((kvshstride >= vsh_stride) ? (Bc * kvshstride) : (Bc * vsh_stride)) * f16vec4;

    // BF16 PVMat accumulator is f32 (no bf16 accumulator support), so pvsh is vec4 (16 bytes)
    స్థిరరాశి uint32_t pvsh_elem_కొలత = (k_type == GGML_TYPE_BF16) ? 16u : f16vec4;
    స్థిరరాశి uint32_t osh_stride = params.అడ్డువరుస_విభజించు * MatBr / 4;
    స్థిరరాశి uint32_t pvsh = MatBc * osh_stride * pvsh_elem_కొలత;

    స్థిరరాశి uint32_t వాలు = Br * acctype;

    స్థిరరాశి uint32_t మొత్తం_కొలత = tmpsh + iq_shmem + Qf + Psh + sfsh + ksh + pvsh + వాలు;
    స్థిరరాశి తార్కిక supported = మొత్తం_కొలత <= device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత;

    VK_LOG_DEBUG("ggml_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", f32acc=" << f32acc << ", total_size=" << మొత్తం_కొలత << ", supported=" << supported);

    తిరిగి supported;
}

శూన్య ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * q, స్థిరరాశి ggml_tensor * k, స్థిరరాశి ggml_tensor * v, స్థిరరాశి ggml_tensor * ముసుగు, స్థిరరాశి ggml_tensor * sinks, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_flash_attn((" << q << ", name=" << q->name << ", type=" << q->type << ", ne0=" << q->ne[0] << ", ne1=" << q->ne[1] << ", ne2=" << q->ne[2] << ", ne3=" << q->ne[3] << ", nb0=" << q->nb[0] << ", nb1=" << q->nb[1] << ", nb2=" << q->nb[2] << ", nb3=" << q->nb[3];
    std::cerr << "), (" << k << ", name=" << k->name << ", type=" << k->type << ", ne0=" << k->ne[0] << ", ne1=" << k->ne[1] << ", ne2=" << k->ne[2] << ", ne3=" << k->ne[3] << ", nb0=" << k->nb[0] << ", nb1=" << k->nb[1] << ", nb2=" << k->nb[2] << ", nb3=" << k->nb[3];
    std::cerr << "), (" << v << ", name=" << v->name << ", type=" << v->type << ", ne0=" << v->ne[0] << ", ne1=" << v->ne[1] << ", ne2=" << v->ne[2] << ", ne3=" << v->ne[3] << ", nb0=" << v->nb[0] << ", nb1=" << v->nb[1] << ", nb2=" << v->nb[2] << ", nb3=" << v->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    ఒకవేళ (sinks) {
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

    స్థిరరాశి uint32_t nem0 = ముసుగు ? ముసుగు->ne[0] : 0;
    స్థిరరాశి uint32_t nem1 = ముసుగు ? ముసుగు->ne[1] : 0;
    స్థిరరాశి uint32_t nem2 = ముసుగు ? ముసుగు->ne[2] : 0;
    స్థిరరాశి uint32_t nem3 = ముసుగు ? ముసుగు->ne[3] : 0;

    స్థిరరాశి uint32_t HSK = nek0;
    స్థిరరాశి uint32_t HSV = nev0;
    uint32_t N = neq1;
    స్థిరరాశి uint32_t KV = nek1;

    GGML_ASSERT(ne0 == HSV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_కొలత(q->type));
    GGML_ASSERT(nbk0 == ggml_type_కొలత(k->type));
    GGML_ASSERT(nbv0 == ggml_type_కొలత(v->type));

    GGML_ASSERT(neq0 == HSK);

    GGML_ASSERT(neq1 == N);

    GGML_ASSERT(nev1 == nek1);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ప్రతిజ్ఞా(dst->type == GGML_TYPE_F32);
    ప్రతిజ్ఞా(q->type == GGML_TYPE_F32);
    uint32_t gqa_నిష్పత్తి = 1;
    uint32_t qk_నిష్పత్తి = neq2 / nek2;
    uint32_t workgroups_x = (uint32_t)neq1;
    uint32_t workgroups_y = (uint32_t)neq2;
    uint32_t workgroups_z = (uint32_t)neq3;

    స్థిరరాశి తార్కిక f32acc = !ctx->device->fp16 || dst->op_params[3] == GGML_PREC_F32 || k->type == GGML_TYPE_BF16;

    // dequant K/V once into an f16 scratch, reordered KV layout so FA can read without a stride
    స్వయంచాలక is_dense_kv_cache = [](స్థిరరాశి ggml_tensor * t) {
        తిరిగి t->nb[0] == ggml_type_కొలత(t->type) &&
               t->nb[2] == ggml_అడ్డువరుస_కొలత(t->type, t->ne[0]) &&
               t->nb[1] == t->nb[2] * t->ne[2] &&
               (t->ne[3] == 1 || t->nb[3] == t->nb[1] * t->ne[1]);
    };
    స్థిరరాశి తార్కిక k_quant = k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16 && k->type != GGML_TYPE_F32;
    స్థిరరాశి తార్కిక v_quant = v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_BF16 && v->type != GGML_TYPE_F32;
    స్థిరరాశి తార్కిక kv_raw = (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) && k->type == v->type;
    // gated to AMD, except RDNA4, which regressed in benchmarks
    స్థిరరాశి తార్కిక kv_raw_strided = ctx->device->vendor_id == VK_VENDOR_ID_AMD &&
                                ctx->device->architecture != vk_device_architecture::AMD_RDNA4 &&
                                neq1 >= 256 &&
                                kv_raw &&
                                (k->nb[1] != (uint64_t)HSK * ggml_type_కొలత(k->type) ||
                                 v->nb[1] != (uint64_t)HSV * ggml_type_కొలత(v->type)) &&
                                (HSK % 8) == 0 && (HSV % 8) == 0;
    స్థిరరాశి size_t kv_scratch_ts = kv_raw_strided ? ggml_type_కొలత(k->type) : పరిమాణం(ggml_fp16_t);
    స్థిరరాశి తార్కిక use_dequant_kv = ((k_quant && v_quant) || kv_raw_strided) && neq1 >= 64 &&
                                is_dense_kv_cache(k) && is_dense_kv_cache(v) &&
                                (uint64_t)ggml_nelements(k) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_బఫరు_Range &&
                                (uint64_t)ggml_nelements(v) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_బఫరు_Range &&
                                (kv_raw_strided ||
                                 (ctx->device->pipeline_dequant_transpose[k->type] != శూన్యసూచిక &&
                                  ctx->device->pipeline_dequant_transpose[v->type] != శూన్యసూచిక)) &&
                                // coopmat2 path does not benefit from the f16 scratch
                                !ctx->device->coopmat2 &&
                                // Intel Xe1 regresses, see PR 25494
                                (ctx->device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 (ctx->device->coopmat_support && ctx->device->architecture != vk_device_architecture::INTEL_XE1));
    స్థిరరాశి ggml_type k_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : k->type;
    స్థిరరాశి ggml_type v_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : v->type;

    // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
    // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
    vk_fa_tuning_params tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, 512, KV, k_type_eff, v_type_eff, f32acc);
    స్థిరరాశి uint32_t max_gqa = std::min(tuning_params.ఖండం_అడ్డువరుసలు, 32u);

    ఒకవేళ (N <= 8 && qk_నిష్పత్తి > 1 && qk_నిష్పత్తి <= max_gqa &&
        qk_నిష్పత్తి * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
        // grouped query attention - make the N dimension equal to gqa_ratio, reduce
        // workgroups proportionally in y dimension. The shader will detect gqa_ratio > 1
        // and change addressing calculations to index Q's dimension 2.
        gqa_నిష్పత్తి = qk_నిష్పత్తి;
        N = gqa_నిష్పత్తి;
        workgroups_y /= gqa_నిష్పత్తి;
    }

    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);

    దశాంశం scale         = 1.0f;
    దశాంశం max_bias      = 0.0f;
    దశాంశం logit_softcap = 0.0f;

    జ్ఞాపకం_నకలు(&scale,         (స్థిరరాశి దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max_bias,      (స్థిరరాశి దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&logit_softcap, (స్థిరరాశి దశాంశం *) dst->op_params + 2, పరిమాణం(దశాంశం));

    ఒకవేళ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // Sparse mask hint (op_params[4]): compact the <= n_kv_max finite positions and gather only those.
    స్థిరరాశి int32_t n_kv_max = ముసుగు ? ggml_get_op_params_i32(dst, 4) : 0;
    స్థిర స్థిరరాశి తార్కిక disable_sparse = getenv("GGML_VK_FA_SPARSE_DISABLE") != శూన్యసూచిక;
    స్థిరరాశి తార్కిక kv_f16 = k_type_eff == GGML_TYPE_F16 && v_type_eff == GGML_TYPE_F16;
    // cm2 dense is fast, so it needs a larger reduction to win.
    // With quantized K/V, sparse only breaks even around 16x (measured on RDNA3/RDNA4).
    స్థిరరాశి int64_t min_నిష్పత్తి = tuning_params.path == FA_COOPMAT2 ? 4 : (kv_f16 ? 2 : 16);
    // coopmat2 vector decode requires 8B strides.
    స్వయంచాలక sparse_gather_aligned = [](స్థిరరాశి ggml_tensor * t) {
        తిరిగి (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) ||
               (t->nb[1] | t->nb[2] | t->nb[3]) % (4 * పరిమాణం(ggml_fp16_t)) == 0;
    };
    స్థిరరాశి తార్కిక use_sparse = !disable_sparse && n_kv_max > 0 && ముసుగు &&
                            max_bias == 0.0f && logit_softcap == 0.0f &&
                            (tuning_params.path != FA_COOPMAT2 || (sparse_gather_aligned(k) && sparse_gather_aligned(v))) &&
                            nem0 == KV &&
                            (int64_t)KV >= std::max<int64_t>(4096, min_నిష్పత్తి * (int64_t)n_kv_max) &&
                            (gqa_నిష్పత్తి > 1 || (tuning_params.path == FA_SCALAR && N == 1));

    స్థిరరాశి uint32_t q_stride = (uint32_t)(nbq1 / ggml_type_కొలత(q->type));
    uint32_t k_stride = (uint32_t)(nbk1 / ggml_type_కొలత(k->type));
    uint32_t v_stride = (uint32_t)(nbv1 / ggml_type_కొలత(v->type));

    // For F32, the shader treats it as a block of size 4 (for vec4 loads)
    ఒకవేళ (k->type == GGML_TYPE_F32) {
        k_stride /= 4;
    }
    ఒకవేళ (v->type == GGML_TYPE_F32) {
        v_stride /= 4;
    }

    uint32_t nbk2_eff = (uint32_t)nbk2, nbk3_eff = (uint32_t)nbk3;
    uint32_t nbv2_eff = (uint32_t)nbv2, nbv3_eff = (uint32_t)nbv3;
    ఒకవేళ (use_dequant_kv) {
        k_stride = HSK;
        v_stride = HSV;
        nbk2_eff = (uint32_t)((uint64_t)HSK * KV * పరిమాణం(ggml_fp16_t));
        nbk3_eff = (uint32_t)((uint64_t)HSK * KV * nek2 * పరిమాణం(ggml_fp16_t));
        nbv2_eff = (uint32_t)((uint64_t)HSV * KV * పరిమాణం(ggml_fp16_t));
        nbv3_eff = (uint32_t)((uint64_t)HSV * KV * nev2 * పరిమాణం(ggml_fp16_t));
    }
    స్థిరరాశి uint32_t alignment = tuning_params.ఖండం_నిలువుల_సంఖ్య;
    తార్కిక aligned = (KV % alignment) == 0 &&
                   // the "aligned" shader variant will forcibly align strides, for performance
                   (q_stride & 7) == 0 && (k_stride & 7) == 0 && (v_stride & 7) == 0;

    // Need to use the coopmat2 variant that clamps loads when HSK/HSV aren't sufficiently aligned.
    ఒకవేళ (((HSK | HSV) % 16) != 0 && tuning_params.path == FA_COOPMAT2) {
        aligned = అబద్ధం;
    }

    // Only use mask opt when the mask is fairly large. This hasn't been tuned extensively.
    తార్కిక use_ముసుగు_opt = ముసుగు && !use_sparse && nem1 >= 32 && nem0 * nem1 > 32768 && nem0 >= tuning_params.ఖండం_నిలువుల_సంఖ్య * 16
                        && (ctx->device->architecture != vk_device_architecture::AMD_GCN || HSK > 256 || HSV > 256);
    vk_fa_pipeline_state fa_pipeline_state = get_fa_pipeline_state(ctx->device, tuning_params, HSK, HSV, aligned, f32acc,
                                                                   ముసుగు != శూన్యసూచిక, use_ముసుగు_opt, logit_softcap != 0, use_sparse, k_type_eff, v_type_eff);

    vk_pipeline pipeline = శూన్యసూచిక;

    తార్కిక xe_fa_opt = అబద్ధం;
    తార్కిక fa_copy_qstate = అబద్ధం;
    తార్కిక xe_fa_supported_platform =
        (ctx->device.get()->architecture == INTEL_XE2 && ctx->device.get()->properties.deviceID != 0xFD80 && ctx->device.get()->properties.deviceID != 0xFD81) ||
        (ctx->device.get()->architecture == INTEL_XE1 && ctx->device.get()->coopmat_support && ctx->device.get()->uma);
    తార్కిక xe_fa_supported_usage = neq0 % 32 == 0 && nev0 % 16 == 0 && q->nb[1] > q->nb[2] && k->nb[1] > k->nb[2] && v->nb[1] > v->nb[2] && ముసుగు != శూన్యసూచిక;
    తార్కిక xe_fa_supported_dtype = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && (ముసుగు != శూన్యసూచిక && ముసుగు->type == GGML_TYPE_F16);
    std::pair<vk_pipeline, vk_pipeline> xe_fa_pipeline_dual_phases = { శూన్యసూచిక , శూన్యసూచిక };
    vk_pipeline xe_fa_pipeline = శూన్యసూచిక;
    size_t కొలత_p = 0;
    size_t కొలత_సమూహం_max = 0;

    {
        std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
        స్వయంచాలక &pipelines = ctx->device->pipeline_flash_attn_f32_f16;
        స్వయంచాలక it = pipelines.find(fa_pipeline_state);
        ఒకవేళ (it != pipelines.end()) {
            pipeline = it->second;
        } లేకపోతే {
            pipelines[fa_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
        }
    }

    ప్రతిజ్ఞా(pipeline);
    // Compile early to initialize wg_denoms.
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    uint32_t విభజించు_kv = KV;
    uint32_t విభజించు_k = 1;

    // Intel Alchemist prefers more workgroups
    స్థిరరాశి uint32_t shader_core_లెక్క_multiplier = (ctx->device->vendor_id == VK_VENDOR_ID_INTEL && ctx->device->architecture != INTEL_XE2) ? 2 : 1;

    // Use a placeholder core count if one isn't available. split_k is a big help for perf.
    స్థిరరాశి uint32_t shader_core_లెక్క = ctx->device->shader_core_లెక్క ? ctx->device->shader_core_లెక్క * shader_core_లెక్క_multiplier : 16;

    స్థిరరాశి uint32_t Br = fa_pipeline_state.Br;
    స్థిరరాశి uint32_t Bc = fa_pipeline_state.Bc;

    GGML_ASSERT(Br == pipeline->wg_denoms[0]);
    స్థిరరాశి uint32_t Tr = CEIL_DIV(N, Br);

    // Try to use split_k when KV is large enough to be worth the overhead.
    // Sparse: split_kv carries n_kv_max, split_k partitions its blocks for occupancy.
    ఒకవేళ (use_sparse) {
        విభజించు_kv = (uint32_t)n_kv_max;
        స్థిరరాశి uint32_t మొత్తం_ఖండాలు = CEIL_DIV((uint32_t)n_kv_max, Bc);
        స్థిరరాశి uint32_t పునాది_wgs = (gqa_నిష్పత్తి > 1 ? workgroups_x : Tr) * workgroups_y * workgroups_z;
        ఒకవేళ (పునాది_wgs < shader_core_లెక్క * 2) {
            విభజించు_k = shader_core_లెక్క * 2 / పునాది_wgs;
        }
        విభజించు_k = std::max(1u, std::min(విభజించు_k, మొత్తం_ఖండాలు));
        // Match the shader's per-split block count so no split is empty.
        స్థిరరాశి uint32_t చొప్పున_ఖండాలు = CEIL_DIV(మొత్తం_ఖండాలు, విభజించు_k);
        విభజించు_k = CEIL_DIV(మొత్తం_ఖండాలు, చొప్పున_ఖండాలు);
    } లేకపోతే ఒకవేళ (gqa_నిష్పత్తి > 1 && workgroups_x <= Br) {
        విభజించు_k = shader_core_లెక్క * 2 / (workgroups_x * workgroups_y * workgroups_z);
    } లేకపోతే ఒకవేళ (gqa_నిష్పత్తి <= 1) {
        uint32_t మొత్తం_wgs_no_విభజించు = Tr * workgroups_y * workgroups_z;
        ఒకవేళ (మొత్తం_wgs_no_విభజించు < shader_core_లెక్క * 2) {
            విభజించు_k = shader_core_లెక్క * 2 / మొత్తం_wgs_no_విభజించు;
        }
    }

    ఒకవేళ (!use_sparse && విభజించు_k > 1) {
        // Try to evenly split KV into split_k chunks, but it needs to be a multiple
        // of "align", so recompute split_k based on that.
        విభజించు_kv = ROUNDUP_POW2(std::max(1u, KV / విభజించు_k), alignment);
        విభజించు_k = CEIL_DIV(KV, విభజించు_kv);
        xe_fa_opt = xe_fa_supported_platform && xe_fa_supported_usage && xe_fa_supported_dtype;
        ఒకవేళ (xe_fa_opt) {
            స్థిరరాశి uint32_t విభజించు_p_కొలత = 32;
            స్థిరరాశి size_t max_dim = (nek1 + విభజించు_p_కొలత - 1) / విభజించు_p_కొలత;
            స్థిరరాశి size_t p_dim = max_dim * విభజించు_p_కొలత;
#ఒకవేళ defined(VK_KHR_cooperative_మాతృక) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            స్వయంచాలక& pipelines = ctx->device->pipeline_xe_fa_decode_dual_phases;
            స్వయంచాలక it = pipelines.find({ (uint32_t)neq0, (uint32_t)nev0, qk_నిష్పత్తి, (uint32_t)neq1 });
            ఒకవేళ (it != pipelines.end()) {
                xe_fa_pipeline_dual_phases = it->second;
            } లేకపోతే {
                pipelines[{(uint32_t)neq0, (uint32_t)nev0, qk_నిష్పత్తి, (uint32_t)neq1}] = xe_fa_pipeline_dual_phases = std::make_pair(std::make_shared<vk_pipeline_struct>(), std::make_shared<vk_pipeline_struct>());
            }
#అయితే_ముగింపు
            ఒకవేళ (xe_fa_pipeline_dual_phases.మొదటి == శూన్యసూచిక || xe_fa_pipeline_dual_phases.second == శూన్యసూచిక) {
                xe_fa_opt = అబద్ధం;
                fa_copy_qstate = అబద్ధం;
            } లేకపోతే {
                కొలత_p = neq1 * neq2 * p_dim * neq3 * పరిమాణం(ggml_fp16_t);
                కొలత_సమూహం_max = neq1 * neq2 * max_dim * neq3 * పరిమాణం(దశాంశం);
                size_t తాత్కాలిక_కొలత = ggml_nelements(q) * పరిమాణం(ggml_fp16_t) + కొలత_p + కొలత_సమూహం_max;
                fa_copy_qstate = నిజం;
                ఒకవేళ (ctx->prealloc_కొలత_x < తాత్కాలిక_కొలత) {
                    ctx->prealloc_కొలత_x = తాత్కాలిక_కొలత;
                    ggml_vk_preallocate_buffers(ctx, subctx);
                }

                ఒకవేళ (ctx->prealloc_x_need_sync) {
                    ggml_vk_sync_buffers(ctx, subctx);
                }
            }
        }
    }

    ఒకవేళ (xe_fa_opt == నిజం) {
        use_ముసుగు_opt = అబద్ధం;
    }

    // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)
    // and the per-row m and L values (ne1 rows). We store all the matrices first, followed by the rows.
    // For matrices, the order is (inner to outer) [HSV, ne1, k, ne2, ne3].
    // For L/M, the order is (inner to outer) [ne1, k, ne2, ne3].
    స్థిరరాశి uint64_t విభజించు_k_కొలత = విభజించు_k > 1 ? (HSV * ne1 * పరిమాణం(దశాంశం) + ne1 * పరిమాణం(దశాంశం) * 2) * విభజించు_k * ne2 * ne3 : 0;
    ఒకవేళ (విభజించు_k_కొలత > ctx->device->properties.limits.max_Storage_బఫరు_Range) {
        GGML_ABORT("Requested preallocation size is too large");
    }
    ఒకవేళ (ctx->prealloc_కొలత_విభజించు_k < విభజించు_k_కొలత) {
        ctx->prealloc_కొలత_విభజించు_k = విభజించు_k_కొలత;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    స్థిరరాశి uint32_t ముసుగు_opt_అంకె_dwords = CEIL_DIV(nem0, 16 * Bc);
    స్థిరరాశి uint64_t ముసుగు_opt_కొలత = పరిమాణం(uint32_t) * ముసుగు_opt_అంకె_dwords * CEIL_DIV(nem1, Br) * nem2 * nem3;

    vk_pipeline pipeline_fa_ముసుగు_opt = శూన్యసూచిక;
    ఒకవేళ (use_ముసుగు_opt) {
        {
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            స్వయంచాలక &pipelines = ctx->device->pipeline_fa_ముసుగు_opt;
            స్వయంచాలక it = pipelines.find({Br, Bc});
            ఒకవేళ (it != pipelines.end()) {
                pipeline_fa_ముసుగు_opt = it->second;
            } లేకపోతే {
                pipelines[{Br, Bc}] = pipeline_fa_ముసుగు_opt = std::make_shared<vk_pipeline_struct>();
            }
        }
        ప్రతిజ్ఞా(pipeline_fa_ముసుగు_opt);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_fa_ముసుగు_opt, 1);

        ఒకవేళ (ctx->prealloc_కొలత_y < ముసుగు_opt_కొలత) {
            ctx->prealloc_కొలత_y = ముసుగు_opt_కొలత;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ఒకవేళ (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    // Sparse index scratch reuses prealloc_y (mutually exclusive with mask opt).
    స్థిరరాశి uint64_t sparse_idx_కొలత = use_sparse
        ? పరిమాణం(int32_t) * (uint64_t)n_kv_max * nem1 * nem2 * nem3
        : 0;
    vk_pipeline sparse_compact_pipeline = ctx->device->fa_sparse_compact_use_subgroups
        ? ctx->device->pipeline_fa_sparse_compact_subgroup
        : ctx->device->pipeline_fa_sparse_compact;
    ఒకవేళ (use_sparse) {
        ggml_pipeline_request_descriptor_sets(ctx, sparse_compact_pipeline, 1);
        ఒకవేళ (ctx->prealloc_కొలత_y < sparse_idx_కొలత) {
            ctx->prealloc_కొలత_y = sparse_idx_కొలత;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ఒకవేళ (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    స్థిరరాశి uint32_t n_తల_kv   = neq2;
    స్థిరరాశి uint32_t n_తల_log2 = 1u << (uint32_t) floorf(log2f((దశాంశం) n_తల_kv));
    స్థిరరాశి దశాంశం m0 = powf(2.0f, -(max_bias       ) / n_తల_log2);
    స్థిరరాశి దశాంశం m1 = powf(2.0f, -(max_bias / 2.0f) / n_తల_log2);

    vk_subbuffer q_buf = ggml_vk_tensor_subbuffer(ctx, q);
    vk_subbuffer k_buf = ggml_vk_tensor_subbuffer(ctx, k);
    vk_subbuffer v_buf = ggml_vk_tensor_subbuffer(ctx, v);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer ముసుగు_buf = ముసుగు ? ggml_vk_tensor_subbuffer(ctx, ముసుగు) : q_buf;
    vk_subbuffer sinks_buf = sinks ? ggml_vk_tensor_subbuffer(ctx, sinks) : q_buf;
    vk_subbuffer ముసుగు_opt_buf = use_ముసుగు_opt ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    vk_subbuffer sparse_buf = use_sparse ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    ఒకవేళ (use_ముసుగు_opt || use_sparse) {
        // the mask opt bits and the sparse index list overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_చివరి_pipeline_వాడిన = శూన్యసూచిక;
        ctx->prealloc_y_చివరి_tensor_వాడిన = శూన్యసూచిక;
    }

    ఒకవేళ (use_dequant_kv) {
        స్థిరరాశి uint64_t k_sz = (uint64_t)ggml_nelements(k) * kv_scratch_ts;
        స్థిరరాశి uint64_t v_sz = (uint64_t)ggml_nelements(v) * kv_scratch_ts;
        ఒకవేళ (ctx->prealloc_కొలత_x < k_sz + v_sz) {
            ctx->prealloc_కొలత_x = k_sz + v_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        vk_subbuffer k_dst = vk_subbuffer{ ctx->prealloc_x, 0,    k_sz };
        vk_subbuffer v_dst = vk_subbuffer{ ctx->prealloc_x, k_sz, v_sz };
        స్థిరరాశి uint32_t k_nel = (uint32_t)ggml_nelements(k);
        స్థిరరాశి uint32_t v_nel = (uint32_t)ggml_nelements(v);
        ఒకవేళ (kv_raw_strided) {
            vk_pipeline cp_k = ggml_vk_get_cpy_pipeline(ctx, k, శూన్యసూచిక, k->type);
            vk_pipeline cp_v = ggml_vk_get_cpy_pipeline(ctx, v, శూన్యసూచిక, v->type);
            ggml_pipeline_request_descriptor_sets(ctx, cp_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, cp_v, 1);
            ఒకవేళ (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            స్వయంచాలక make_pc = [](uint32_t hs, uint32_t nh, uint32_t kv, uint32_t nel) {
                vk_op_unary_నెట్టు_constants pc{};
                pc.ne = nel;
                pc.ne00 = hs; pc.ne01 = nh;      pc.ne02 = kv; pc.ne03 = nel / (hs * nh * kv);
                pc.nb00 = 1;  pc.nb01 = hs;      pc.nb02 = hs * nh; pc.nb03 = hs * nh * kv;
                pc.ne10 = hs; pc.ne11 = nh;      pc.ne12 = kv; pc.ne13 = pc.ne03;
                pc.nb10 = 1;  pc.nb11 = hs * kv; pc.nb12 = hs; pc.nb13 = hs * kv * nh;
                init_pushconst_fastdiv(pc);
                తిరిగి pc;
            };
            స్వయంచాలక cpy_elems = [](uint32_t ne) -> std::array<uint32_t, 3> {
                ఒకవేళ (ne > 262144) { తిరిగి { 512, 512, CEIL_DIV(ne, 262144) }; }
                ఒకవేళ (ne > 512)    { తిరిగి { 512, CEIL_DIV(ne, 512), 1 }; }
                తిరిగి { ne, 1, 1 };
            };
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_k, { k_buf, k_dst },
                make_pc((uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, k_nel), cpy_elems(k_nel));
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_v, { v_buf, v_dst },
                make_pc((uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, v_nel), cpy_elems(v_nel));
        } లేకపోతే {
            vk_pipeline tr_k = ctx->device->pipeline_dequant_transpose[k->type];
            vk_pipeline tr_v = ctx->device->pipeline_dequant_transpose[v->type];
            ggml_pipeline_request_descriptor_sets(ctx, tr_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, tr_v, 1);
            ఒకవేళ (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            { స్థిరరాశి std::vector<uint32_t> pc = { (uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, 0, k_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_k, { k_buf, k_dst }, pc, { k_nel, 1, 1 }); }
            { స్థిరరాశి std::vector<uint32_t> pc = { (uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, 0, v_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_v, { v_buf, v_dst }, pc, { v_nel, 1, 1 }); }
        }
        ggml_vk_sync_buffers(ctx, subctx);
        k_buf = k_dst;
        v_buf = v_dst;
    }

    uint32_t ముసుగు_n_తల_log2 = ((sinks != శూన్యసూచిక) << 24) | n_తల_log2;

    ఒకవేళ (use_ముసుగు_opt)
    {
        స్థిరరాశి vk_op_flash_attn_ముసుగు_opt_నెట్టు_constants opt_pc = {
            nem0,
            nem1,
            nem2,
            (uint32_t)(ముసుగు->nb[1] / పరిమాణం(ggml_fp16_t)),
            (uint32_t)(ముసుగు->nb[2] / పరిమాణం(ggml_fp16_t)),
            (uint32_t)(ముసుగు->nb[3] / పరిమాణం(ggml_fp16_t)),
            ముసుగు_opt_అంకె_dwords,
            ముసుగు_opt_అంకె_dwords * CEIL_DIV(nem1, Br),
            ముసుగు_opt_అంకె_dwords * CEIL_DIV(nem1, Br) * nem2,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline_fa_ముసుగు_opt,
                                  { ముసుగు_buf, ముసుగు_opt_buf }, opt_pc,
                                  { ముసుగు_opt_అంకె_dwords, CEIL_DIV(nem1, Br), nem2 * nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ఒకవేళ (use_sparse)
    {
        స్థిరరాశి vk_op_flash_attn_sparse_compact_నెట్టు_constants sc_pc = {
            KV,
            nem1,
            nem2,
            (uint32_t)(ముసుగు->nb[1] / పరిమాణం(ggml_fp16_t)),
            (uint32_t)(ముసుగు->nb[2] / పరిమాణం(ggml_fp16_t)),
            (uint32_t)(ముసుగు->nb[3] / పరిమాణం(ggml_fp16_t)),
            (uint32_t)n_kv_max,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, sparse_compact_pipeline,
                                  { ముసుగు_buf, sparse_buf }, sc_pc,
                                  { nem1, nem2, nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    స్థిరరాశి vk_flash_attn_నెట్టు_constants pc = { N, KV,
                                              (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                                              (uint32_t)neq2, (uint32_t)neq3,
                                              (uint32_t)nek2, (uint32_t)nek3,
                                              (uint32_t)nev2, (uint32_t)nev3,
                                              nem1, nem2, nem3,
                                              q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
                                              k_stride, nbk2_eff, nbk3_eff,
                                              v_stride, nbv2_eff, nbv3_eff,
                                              scale, max_bias, logit_softcap,
                                              ముసుగు_n_తల_log2, m0, m1,
                                              gqa_నిష్పత్తి, విభజించు_kv, విభజించు_k };

    ఒకవేళ (xe_fa_opt && విభజించు_k > 1) {
        స్వయంచాలక పై_power_యొక్క_2 = [&](uint32_t లోపల) {
            GGML_ASSERT(లోపల != 0);
            ఒకవేళ (లోపల <= 1) తిరిగి 1u;
            uint32_t ret = లోపల - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            తిరిగి ret + 1;
        };
        స్వయంచాలక గమ్యం_fp16_vk_0 = ggml_vk_get_గమ్యం_fp16(ctx, q->type);
        స్థిరరాశి uint32_t బహిర్గతం_dim_చొప్పున_wg = qk_నిష్పత్తి > 16 ? 8 : 16;
        size_t x_ne = ggml_nelements(q);
        size_t తాత్కాలిక_buf_స్థానభ్రంశం = 0;
        uint32_t తల_stride_k = uint32_t(nbk2 / ggml_type_కొలత(k->type));
        uint32_t తల_stride_v = uint32_t(nbv2 / ggml_type_కొలత(v->type));
        uint32_t batch_stride_q = uint32_t(nbq3 / ggml_type_కొలత(q->type));
        uint32_t batch_stride_k = uint32_t(nbk3 / ggml_type_కొలత(k->type));
        uint32_t batch_stride_v = uint32_t(nbv3 / ggml_type_కొలత(v->type));
        uint32_t batch_stride_m = ముసుగు ? uint32_t(ముసుగు->nb[3] / ggml_type_కొలత(ముసుగు->type)) : 0u;
        uint32_t batch_stride_o = uint32_t(nb3 / ggml_type_కొలత(dst->type));
        vk_fa_xe_opt_నెట్టు_constants pc_ph1 = { (uint32_t)nek1, (uint32_t)neq1, (uint32_t)neq2, (uint32_t)nek2, qk_నిష్పత్తి, 1, (sinks != శూన్యసూచిక) ? 1u : 0u, (uint32_t)k_stride, తల_stride_k,
            batch_stride_q, batch_stride_k, batch_stride_v, batch_stride_m, batch_stride_o, scale };
        vk_fa_xe_opt_నెట్టు_constants pc_ph2 = pc_ph1;
        pc_ph2.nbkv_tok = v_stride;
        pc_ph2.nbkv_తల = తల_stride_v;
        vk_subbuffer q_తాత్కాలిక_buf = fa_copy_qstate ? ggml_vk_subbuffer(ctx, ctx->prealloc_x, తాత్కాలిక_buf_స్థానభ్రంశం) : q_buf;
        తాత్కాలిక_buf_స్థానభ్రంశం += fa_copy_qstate ? x_ne * పరిమాణం(ggml_fp16_t) : 0;
        vk_subbuffer p_తాత్కాలిక_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, తాత్కాలిక_buf_స్థానభ్రంశం);
        తాత్కాలిక_buf_స్థానభ్రంశం += కొలత_p;
        vk_subbuffer max_తాత్కాలిక_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, తాత్కాలిక_buf_స్థానభ్రంశం);
        తాత్కాలిక_buf_స్థానభ్రంశం += కొలత_సమూహం_max;
        uint32_t xe_native_sub_సమూహం_కొలత = ctx->device.get()->architecture == INTEL_XE1 ? 8 : 16;
        uint32_t aligned_gqa_నిష్పత్తి = పై_power_యొక్క_2(qk_నిష్పత్తి);
        uint32_t బహిర్గతం_చొప్పున_wg_ph1 = std::min(256u / xe_native_sub_సమూహం_కొలత, (uint32_t)neq1);
        uint32_t బహిర్గతం_చొప్పున_wg_ph2 = std::min(std::max(16u / aligned_gqa_నిష్పత్తి, 1u), (uint32_t)neq1);
        uint32_t ph1_wg = ((neq1 + బహిర్గతం_చొప్పున_wg_ph1 - 1) / బహిర్గతం_చొప్పున_wg_ph1) * nek2;
        uint32_t ph2_wg = ((neq1 + బహిర్గతం_చొప్పున_wg_ph2 - 1) / బహిర్గతం_చొప్పున_wg_ph2) * ne0 / బహిర్గతం_dim_చొప్పున_wg;
        ఒకవేళ (fa_copy_qstate) {
            స్థిరరాశి std::vector<uint32_t> pc_cpy_fp16 =
            { (uint32_t)q->ne[0], (uint32_t)q->ne[1], (uint32_t)q->ne[2], (uint32_t)q->ne[3], (uint32_t)(x_ne) };
            ggml_vk_sync_buffers(ctx, subctx);
            ggml_pipeline_request_descriptor_sets(ctx, గమ్యం_fp16_vk_0, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, గమ్యం_fp16_vk_0, { q_buf, q_తాత్కాలిక_buf }, pc_cpy_fp16, { (uint32_t)(x_ne), 1, 1 });
        }

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.మొదటి, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.మొదటి,
            { q_తాత్కాలిక_buf, k_buf, ముసుగు_buf, p_తాత్కాలిక_buf, max_తాత్కాలిక_buf },
            pc_ph1, { (uint32_t)ph1_wg, (uint32_t)nek1, (uint32_t)neq3 });

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.second, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.second,
            { p_తాత్కాలిక_buf, v_buf, max_తాత్కాలిక_buf, sinks_buf, dst_buf },
            pc_ph2, { (uint32_t)ph2_wg, (uint32_t)nev2, (uint32_t)neq3 });

        ctx->prealloc_x_need_sync = నిజం;
    } లేకపోతే ఒకవేళ (విభజించు_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_విభజించు_k_reduce, 1);

        ఒకవేళ (ctx->prealloc_విభజించు_k_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        // We reuse workgroups_x to mean the number of splits, so we need to
        // cancel out the divide by wg_denoms[0].
        uint32_t dispatch_x;
        ఒకవేళ (gqa_నిష్పత్తి > 1) {
            workgroups_x *= pipeline->wg_denoms[0];
            dispatch_x = విభజించు_k * workgroups_x;
        } లేకపోతే {
            dispatch_x = Tr * విభజించు_k * pipeline->wg_denoms[0];
        }

        vk_subbuffer విభజించు_k_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_విభజించు_k, 0);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, ముసుగు_buf, sinks_buf, విభజించు_k_buf, ముసుగు_opt_buf, sparse_buf},
                                    pc, { dispatch_x, workgroups_y, workgroups_z });

        ggml_vk_sync_buffers(ctx, subctx);
        స్థిరరాశి vk_op_flash_attn_విభజించు_k_reduce_నెట్టు_constants pc2 = { HSV, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3, విభజించు_k, (sinks != శూన్యసూచిక) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_flash_attn_విభజించు_k_reduce,
                                    {విభజించు_k_buf, sinks_buf, dst_buf},
                                    pc2, { (uint32_t)ne1, HSV, (uint32_t)(ne2 * ne3) });
        ctx->prealloc_విభజించు_k_need_sync = నిజం;
    } లేకపోతే {
        ఒకవేళ (gqa_నిష్పత్తి > 1) {
            // When using gqa, we want one actual workgroup per batch, so cancel out wg_denoms
            workgroups_x *= pipeline->wg_denoms[0];
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, ముసుగు_buf, sinks_buf, dst_buf, ముసుగు_opt_buf, sparse_buf},
                                    pc, { workgroups_x, workgroups_y, workgroups_z });
    }

    ఒకవేళ (use_dequant_kv) {
        ctx->prealloc_x_need_sync = నిజం;
    }
    ఒకవేళ (use_ముసుగు_opt || use_sparse) {
        ctx->prealloc_y_need_sync = నిజం;
    }
}

స్థిర vk_conv_shapes ggml_vk_conv_select_shape(ggml_backend_vk_context * ctx, uint32_t K, uint32_t NPQ) {
    స్వయంచాలక n_tiles = [&](vk_conv_shapes s) {
        తిరిగి CEIL_DIV(K, vk_conv_ఖండం_sizes[s].K)
            * CEIL_DIV(NPQ, vk_conv_ఖండం_sizes[s].NPQ);
    };

    // We can't query number of shader cores on Intel, use 32 as a placeholder
    // so small convolutions will still choose a smaller tile.
    స్థిరరాశి uint32_t shader_core_లెక్క = ctx->device->shader_core_లెక్క > 0 ? ctx->device->shader_core_లెక్క : 32;

    // 128x128 isn't used with cm1 due to shared memory size; fall through to a smaller tile.
    తార్కిక allow_128x128 = నిజం;
#ఒకవేళ defined(VK_KHR_cooperative_మాతృక) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ఒకవేళ (!ctx->device->coopmat2 && ctx->device->coopmat_support && ctx->device->coopmat_support_16x16x16_f16acc) {
        allow_128x128 = అబద్ధం;
    }
#అయితే_ముగింపు

    ఒకవేళ (allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_128x128) >= shader_core_లెక్క * 2) {
        తిరిగి CONV_SHAPE_128x128;
    } లేకపోతే ఒకవేళ (K <= 32 && n_tiles(CONV_SHAPE_32x256) >= shader_core_లెక్క * 2) {
        తిరిగి CONV_SHAPE_32x256;
    } లేకపోతే ఒకవేళ (K <= 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_లెక్క * 2) {
        తిరిగి CONV_SHAPE_64x128;
    } లేకపోతే ఒకవేళ (!allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_లెక్క * 2) {
        // cm1 fallback for large K when 128x128 isn't available
        తిరిగి CONV_SHAPE_64x128;
    } లేకపోతే {
        తిరిగి CONV_SHAPE_64x32;
    }
}

స్థిర vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * dst, ggml_op op) {
    ఎంపిక (op) {
    స్థితి GGML_OP_GET_అడ్డువరుసలు:
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        ఒకవేళ (src0->type == GGML_TYPE_I32) {
            // i32 src only supports i32 result
            GGML_ASSERT(dst->type == GGML_TYPE_I32);
            తిరిగి ctx->device->pipeline_get_అడ్డువరుసలు[src0->type];
        }
        ఒకవేళ (dst->type == GGML_TYPE_F16) {
            తిరిగి ctx->device->pipeline_get_అడ్డువరుసలు[src0->type];
        }
        ఒకవేళ (dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_get_అడ్డువరుసలు_f32[src0->type];
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_GET_అడ్డువరుసలు_వెనుక:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_get_అడ్డువరుసలు_వెనుక_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_ACC:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_acc_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_సమితి:
        ఒకవేళ (src0->type == src1->type && src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32)) {
            తిరిగి ctx->device->pipeline_సమితి_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_కూడు:
    స్థితి GGML_OP_SUB:
    స్థితి GGML_OP_MUL:
    స్థితి GGML_OP_DIV:
        ఒకవేళ ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)) {
            తిరిగి శూన్యసూచిక;
        }
        ఎంపిక (op) {
        స్థితి GGML_OP_కూడు:
        {
            ఒకవేళ (ctx->అంకె_additional_fused_ops > 0) {
                ఒకవేళ (ctx->do_కూడు_rms_partials) {
                    తిరిగి ctx->device->pipeline_multi_కూడు_rms[ctx->అంకె_additional_fused_ops];
                } లేకపోతే {
                    తిరిగి ctx->device->pipeline_multi_కూడు[ctx->అంకె_additional_fused_ops];
                }
            }
            ఒకవేళ (ctx->do_కూడు_rms_partials) {
                స్వయంచాలక pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_కూడు_rms_norepeat : ctx->device->pipeline_కూడు_rms;
                తిరిగి pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            } లేకపోతే {
                స్వయంచాలక pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_కూడు_norepeat : ctx->device->pipeline_కూడు;
                తిరిగి pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            }
        }
        స్థితి GGML_OP_SUB:
        {
            స్వయంచాలక pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_sub_norepeat : ctx->device->pipeline_sub;
            తిరిగి pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        స్థితి GGML_OP_MUL:
        {
            స్వయంచాలక pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_mul_norepeat : ctx->device->pipeline_mul;
            తిరిగి pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        స్థితి GGML_OP_DIV:
        {
            స్వయంచాలక pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_div_norepeat : ctx->device->pipeline_div;
            తిరిగి pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        అప్రమేయ:
            ఆపు;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_కూడు_ID:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_కూడు_id_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_బహిర్గతం_PROD:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_బహిర్గతం_prod_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_CONCAT: {
        ఒకవేళ (!ggml_vk_concat_supported(src0, src1, dst)) {
            తిరిగి శూన్యసూచిక;
        }
        ఎంపిక (ggml_vk_concat_unit_కొలత(src0->type)) {
        స్థితి 1:
            తిరిగి ctx->device->pipeline_concat_i8;
        స్థితి 2:
            తిరిగి ctx->device->pipeline_concat_i16;
        స్థితి 4:
            తిరిగి ctx->device->pipeline_concat_i32;
        స్థితి 8:
            తిరిగి ctx->device->pipeline_concat_i64;
        అప్రమేయ:
            తిరిగి శూన్యసూచిక;
        }
    }
    స్థితి GGML_OP_UPSCALE:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t mode = (ggml_get_op_params_i32(dst, 0) & (0xFF | GGML_SCALE_FLAG_ANTIALIAS));
            ఎంపిక (mode) {
                స్థితి GGML_SCALE_బహుళకం_సమీప:
                    తిరిగి ctx->device->pipeline_upscale_సమీప_f32;
                స్థితి GGML_SCALE_బహుళకం_BILINEAR:
                    తిరిగి ctx->device->pipeline_upscale_bilinear_f32;
                స్థితి GGML_SCALE_బహుళకం_BICUBIC:
                    తిరిగి ctx->device->pipeline_upscale_bicubic_f32;
                స్థితి GGML_SCALE_బహుళకం_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS:
                    తిరిగి ctx->device->pipeline_upscale_bilinear_antialias_f32;
                అప్రమేయ:
                    తిరిగి శూన్యసూచిక;
            }
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_SCALE:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_scale_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_SQR:
        ఒకవేళ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            తిరిగి ctx->device->pipeline_sqr[dst->type == GGML_TYPE_F16];
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_SQRT:
        ఒకవేళ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            తిరిగి ctx->device->pipeline_sqrt[dst->type == GGML_TYPE_F16];
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_SIN:
        ఒకవేళ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            తిరిగి ctx->device->pipeline_sin[dst->type == GGML_TYPE_F16];
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_COS:
        ఒకవేళ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            తిరిగి ctx->device->pipeline_cos[dst->type == GGML_TYPE_F16];
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_LOG:
        ఒకవేళ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            తిరిగి ctx->device->pipeline_log[dst->type == GGML_TYPE_F16];
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_TRI:
        ఒకవేళ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            తిరిగి ctx->device->pipeline_tri[dst->type == GGML_TYPE_F16];
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_DIAG:
        ఒకవేళ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            తిరిగి ctx->device->pipeline_diag[dst->type == GGML_TYPE_F16];
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_CLAMP:
        ఒకవేళ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            తిరిగి ctx->device->pipeline_clamp[dst->type == GGML_TYPE_F16];
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_PAD:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_pad_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_PAD_REFLECT_1D:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_pad_reflect_1d_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_ROLL:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_roll_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_REPEAT:
        ఒకవేళ (ggml_type_కొలత(src0->type) == పరిమాణం(దశాంశం) && ggml_type_కొలత(dst->type) == పరిమాణం(దశాంశం)) {
            తిరిగి ctx->device->pipeline_repeat_i32;
        }
        ఒకవేళ (ggml_type_కొలత(src0->type) == 2 && ggml_type_కొలత(dst->type) == 2) {
            తిరిగి ctx->device->pipeline_repeat_i16;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_REPEAT_వెనుక:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_repeat_వెనుక_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_CPY:
    స్థితి GGML_OP_CONT:
    స్థితి GGML_OP_DUP:
        తిరిగి ggml_vk_get_cpy_pipeline(ctx, src0, dst, dst->type);
    స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
        {
            ఒకవేళ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                తిరిగి శూన్యసూచిక;
            }
            స్థిరరాశి పూర్ణాంకం src_idx = src0->type == GGML_TYPE_F16;
            ఒకవేళ (src1->type == GGML_TYPE_I64) {
                తిరిగి ctx->device->pipeline_సమితి_అడ్డువరుసలు_i64[src_idx][dst->type];
            } లేకపోతే ఒకవేళ (src1->type == GGML_TYPE_I32) {
                తిరిగి ctx->device->pipeline_సమితి_అడ్డువరుసలు_i32[src_idx][dst->type];
            }
            తిరిగి శూన్యసూచిక;
        }
    స్థితి GGML_OP_SILU_వెనుక:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_silu_వెనుక_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_NORM:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_norm_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_సమూహం_NORM:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_సమూహం_norm_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_RMS_NORM:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ఒకవేళ (ctx->do_కూడు_rms_partials) {
                తిరిగి ctx->fused_rms_norm_బహుళకం == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_partials_f32 : ctx->device->pipeline_rms_norm_partials_f32;
            }
            తిరిగి ctx->fused_rms_norm_బహుళకం == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_f32 : ctx->device->pipeline_rms_norm_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_RMS_NORM_వెనుక:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_rms_norm_వెనుక_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_L2_NORM:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_l2_norm_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_UNARY:
        ఒకవేళ ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            తిరిగి శూన్యసూచిక;
        }

        ఎంపిక (ggml_get_unary_op(dst)) {
            స్థితి GGML_UNARY_OP_EXP:
                తిరిగి ctx->device->pipeline_exp[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_EXPM1:
                తిరిగి ctx->device->pipeline_expm1[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_ELU:
                తిరిగి ctx->device->pipeline_elu[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_SILU:
                తిరిగి ctx->device->pipeline_silu[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_GELU:
                తిరిగి ctx->device->pipeline_gelu[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_GELU_ERF:
                తిరిగి ctx->device->pipeline_gelu_erf[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:
                తిరిగి ctx->device->pipeline_gelu_శీఘ్ర[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_RELU:
                తిరిగి ctx->device->pipeline_relu[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_XIELU:
                తిరిగి ctx->device->pipeline_xielu[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_NEG:
                తిరిగి ctx->device->pipeline_neg[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_TANH:
                తిరిగి ctx->device->pipeline_tanh[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_SIGMOID:
                తిరిగి ctx->device->pipeline_sigmoid[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_HARDSIGMOID:
                తిరిగి ctx->device->pipeline_hardsigmoid[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_HARDSWISH:
                తిరిగి ctx->device->pipeline_hardswish[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_ABS:
                తిరిగి ctx->device->pipeline_abs[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_SOFTPLUS:
                తిరిగి ctx->device->pipeline_softplus[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_అడుగు:
                తిరిగి ctx->device->pipeline_అడుగు[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_విడత:
                తిరిగి ctx->device->pipeline_విడత[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_CEIL:
                తిరిగి ctx->device->pipeline_ceil[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_FLOOR:
                తిరిగి ctx->device->pipeline_floor[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_TRUNC:
                తిరిగి ctx->device->pipeline_trunc[dst->type == GGML_TYPE_F16];
            స్థితి GGML_UNARY_OP_SGN:
                తిరిగి ctx->device->pipeline_sgn[dst->type == GGML_TYPE_F16];
            అప్రమేయ:
                ఆపు;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_GLU:
        ఒకవేళ ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            తిరిగి శూన్యసూచిక;
        }

        ఎంపిక (ggml_get_glu_op(dst)) {
            స్థితి GGML_GLU_OP_GEGLU:
                తిరిగి ctx->device->pipeline_geglu[dst->type == GGML_TYPE_F16];
            స్థితి GGML_GLU_OP_REGLU:
                తిరిగి ctx->device->pipeline_reglu[dst->type == GGML_TYPE_F16];
            స్థితి GGML_GLU_OP_SWIGLU:
                తిరిగి ctx->device->pipeline_swiglu[dst->type == GGML_TYPE_F16];
            స్థితి GGML_GLU_OP_SWIGLU_OAI:
                తిరిగి ctx->device->pipeline_swiglu_oai[dst->type == GGML_TYPE_F16];
            స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
                తిరిగి ctx->device->pipeline_swiglu_clamp[dst->type == GGML_TYPE_F16];
            స్థితి GGML_GLU_OP_GEGLU_ERF:
                తిరిగి ctx->device->pipeline_geglu_erf[dst->type == GGML_TYPE_F16];
            స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
                తిరిగి ctx->device->pipeline_geglu_శీఘ్ర[dst->type == GGML_TYPE_F16];
            అప్రమేయ:
                ఆపు;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_DIAG_ముసుగు_INF:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_diag_ముసుగు_inf_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_SOFT_MAX:
        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        GGML_ASSERT(!src2 || src2->type == GGML_TYPE_F32);

        ఒకవేళ (ctx->అంకె_additional_fused_ops) {
            uint32_t idx = (uint32_t)ceilf(log2f(దశాంశం(dst->ne[0])));
            GGML_ASSERT(idx < అంకె_topk_moe_pipelines);
            // use n_experts from push constant if it's not equal to the power of two spec constant
            తార్కిక use_నెట్టు = dst->ne[0] != (1u << idx);
            తిరిగి ctx->device->pipeline_topk_moe[idx][use_నెట్టు];
        }

        ఒకవేళ (src0->type == GGML_TYPE_F32 && (src1 == శూన్యసూచిక || src1->type == GGML_TYPE_F32) && dst->type == GGML_TYPE_F32) {
            తిరిగి src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_wg512 : ctx->device->pipeline_soft_max_f32;
        }
        ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            తిరిగి src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_f16_wg512 : ctx->device->pipeline_soft_max_f32_f16;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_SOFT_MAX_వెనుక:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_soft_max_వెనుక_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_ROPE:
    స్థితి GGML_OP_ROPE_వెనుక:
        {
            స్థిరరాశి ggml_tensor *rope = ctx->అంకె_additional_fused_ops == 2 ? dst->src[0]->src[0] : dst;
            స్థిరరాశి పూర్ణాంకం mode = ((స్థిరరాశి int32_t *) rope->op_params)[2];
            స్థిరరాశి తార్కిక is_neox = mode & GGML_ROPE_TYPE_NEOX;
            స్థిరరాశి తార్కిక is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            స్థిరరాశి తార్కిక is_vision = mode == GGML_ROPE_TYPE_VISION;

            ఒకవేళ (is_neox) {
                ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    తిరిగి ctx->device->pipeline_rope_neox_f32;
                }
                ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    తిరిగి ctx->device->pipeline_rope_neox_f32_f16;
                }
                ఒకవేళ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    తిరిగి ctx->device->pipeline_rope_neox_f16;
                }
            } లేకపోతే ఒకవేళ (is_mrope && !is_vision) {
                ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    తిరిగి ctx->device->pipeline_rope_multi_f32;
                }
                ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    తిరిగి ctx->device->pipeline_rope_multi_f32_f16;
                }
                ఒకవేళ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    తిరిగి ctx->device->pipeline_rope_multi_f16;
                }
            } లేకపోతే ఒకవేళ (is_vision) {
                ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    తిరిగి ctx->device->pipeline_rope_vision_f32;
                }
                ఒకవేళ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    తిరిగి ctx->device->pipeline_rope_vision_f16;
                }
            } లేకపోతే {
                ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    తిరిగి ctx->device->pipeline_rope_norm_f32;
                }
                ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    తిరిగి ctx->device->pipeline_rope_norm_f32_f16;
                }
                ఒకవేళ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    తిరిగి ctx->device->pipeline_rope_norm_f16;
                }
            }
            తిరిగి శూన్యసూచిక;
        }
    స్థితి GGML_OP_కూడిక:
    స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
    స్థితి GGML_OP_సగటు:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_కూడిక_అడ్డువరుసలు_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_CROSS_ENTROPY_LOSS:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి src0->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_CROSS_ENTROPY_LOSS_వెనుక:
        // src0 is the scalar grad; src1 is logits
        ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2 && src2->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి src1->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_వెనుక_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_వెనుక_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_CUMSUM:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ఒకవేళ (src0->ne[0] <= 512) {
                తిరిగి ctx->device->pipeline_cumsum_small_f32;
            } లేకపోతే {
                తిరిగి ctx->device->pipeline_cumsum_f32;
            }
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_SOLVE_TRI:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {

            vk_solve_tri_pipeline_state solve_tri_pipeline_state(src0->ne[0], src1->ne[0]);

            vk_pipeline pipeline = శూన్యసూచిక;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                స్వయంచాలక it = ctx->device->pipeline_solve_tri_f32.find(solve_tri_pipeline_state);
                ఒకవేళ (it != ctx->device->pipeline_solve_tri_f32.end()) {
                    pipeline = it->second;
                } లేకపోతే {
                    ctx->device->pipeline_solve_tri_f32[solve_tri_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            తిరిగి pipeline;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_ARGMAX:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
            తిరిగి ctx->device->pipeline_argmax_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_లెక్క_EQUAL:
        ఒకవేళ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I64) {
            తిరిగి ctx->device->pipeline_లెక్క_equal_i32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_IM2_నిలువు:
        ఒకవేళ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_im2col_f32;
        }
        ఒకవేళ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            తిరిగి ctx->device->pipeline_im2col_f32_f16;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_IM2_నిలువు_3_D:
        ఒకవేళ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_im2col_3d_f32;
        }
        ఒకవేళ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            తిరిగి ctx->device->pipeline_im2col_3d_f32_f16;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_TIMESTEP_EMBEDDING:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_timestep_embedding_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_CONV_TRANSPOSE_1D:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_conv_transpose_1d_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_COL2IM_1D:
        ఎంపిక (src0->type) {
            స్థితి GGML_TYPE_F32:  తిరిగి ctx->device->pipeline_col2im_1d_f32;
            స్థితి GGML_TYPE_F16:  తిరిగి ctx->device->pipeline_col2im_1d_f16;
            స్థితి GGML_TYPE_BF16: తిరిగి ctx->device->pipeline_col2im_1d_bf16;
            అప్రమేయ:             తిరిగి శూన్యసూచిక;
        }
    స్థితి GGML_OP_POOL_1D:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_pool1d_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_POOL_2D:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            // Wide windows use neighboring subgroup lanes to read neighboring inputs.
            ఒకవేళ (ctx->device->pipeline_pool2d_subgroup_f32 && dst->op_params[1] >= int32_t(ctx->device->subgroup_కొలత)) {
                తిరిగి ctx->device->pipeline_pool2d_subgroup_f32;
            }
            తిరిగి ctx->device->pipeline_pool2d_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_RWKV_WKV6:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_rwkv_wkv6_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_RWKV_WKV7:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_rwkv_wkv7_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_GATED_రేఖీయ_ATTN:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_gated_రేఖీయ_attn_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_LIGHTNING_INDEXER:
        // only the k type selects a pipeline, the other types are fixed by ggml_lightning_indexer()
        ఒకవేళ (ggml_vk_lightning_indexer_k_type_supported(src1->type)) {
            తిరిగి ctx->device->pipeline_lightning_indexer_f32[src1->type];
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_GATED_DELTA_NET:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            స్థిరరాశి uint32_t S_v = dst->src[2]->ne[0];
            స్థిరరాశి uint32_t kda = (dst->src[3]->ne[0] == (int64_t)S_v) ? 1 : 0;
            uint32_t si;
            ఎంపిక (S_v) {
                స్థితి 16:  si = 0; ఆపు;
                స్థితి 32:  si = 1; ఆపు;
                స్థితి 64:  si = 2; ఆపు;
                స్థితి 128: si = 3; ఆపు;
                అప్రమేయ: తిరిగి శూన్యసూచిక;
            }
            తిరిగి ctx->device->pipeline_gated_delta_net[si][kda];
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_SSM_పరిశీలన:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            స్థిరరాశి uint32_t d_state = src0->ne[0];
            ఒకవేళ (d_state == 128) {
                తిరిగి ctx->device->pipeline_ssm_పరిశీలన_f32_d128;
            } లేకపోతే ఒకవేళ (d_state == 256) {
                తిరిగి ctx->device->pipeline_ssm_పరిశీలన_f32_d256;
            }
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_SSM_CONV:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ఎంపిక (ctx->అంకె_additional_fused_ops) {
                స్థితి 0:  తిరిగి ctx->device->pipeline_ssm_conv_f32;
                స్థితి 1:  తిరిగి ctx->device->pipeline_ssm_conv_silu_f32;
                స్థితి 2:  తిరిగి ctx->device->pipeline_ssm_conv_bias_silu_f32;
                అప్రమేయ: తిరిగి శూన్యసూచిక;
            }
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_OPT_అడుగు_ADAMW:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_opt_అడుగు_adamw_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_OPT_అడుగు_SGD:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_opt_అడుగు_sgd_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_LEAKY_RELU:
        ఒకవేళ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            తిరిగి ctx->device->pipeline_leaky_relu[dst->type == GGML_TYPE_F16];
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_CONV_2D:
    స్థితి GGML_OP_CONV_TRANSPOSE_2D:
        ఒకవేళ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t K = dst->ne[2]; // Cout
            uint32_t NPQ = dst->ne[3] * dst->ne[1] * dst->ne[0]; // N * OH * OW
            vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, K, NPQ);

            తార్కిక transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;
            uint32_t KW = (uint32_t)src0->ne[0];
            uint32_t KH = (uint32_t)src0->ne[1];
            uint32_t s0 = (uint32_t)(ggml_get_op_params_i32(dst, 0));
            uint32_t s1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 1) : s0;
            uint32_t p0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 2) : 0;
            uint32_t p1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 3) : 0;
            uint32_t d0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 4) : 1;
            uint32_t d1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 5) : 1;

            // tile-aligned shapes let the shader skip bounds checks
            స్థిరరాశి uint32_t Cin = (uint32_t)src1->ne[2];
            స్థిరరాశి uint32_t CRS = Cin * KW * KH;
            స్థిరరాశి uint32_t BS_K   = vk_conv_ఖండం_sizes[shape].K;
            స్థిరరాశి uint32_t BS_CRS = vk_conv_ఖండం_sizes[shape].CRS;
            స్థిరరాశి uint32_t BS_NPQ = vk_conv_ఖండం_sizes[shape].NPQ;
            స్థిరరాశి uint32_t aligned = ((K   % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv2d_pipeline_state conv2d_pipeline_state(s0, s1, p0, p1, d0, d1, KW, KH, aligned);

            std::map<vk_conv2d_pipeline_state, vk_pipeline> *pipelines = శూన్యసూచిక;
            ఒకవేళ (op == GGML_OP_CONV_2D) {
                ఒకవేళ (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv2d_f32[shape];
                } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv2d_f16_f32[shape];
                }
            } లేకపోతే ఒకవేళ (op == GGML_OP_CONV_TRANSPOSE_2D) {
                ఒకవేళ (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f32[shape];
                } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f16_f32[shape];
                }
            }

            vk_pipeline pipeline = శూన్యసూచిక;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                స్వయంచాలక it = pipelines->find(conv2d_pipeline_state);
                ఒకవేళ (it != pipelines->end()) {
                    pipeline = it->second;
                } లేకపోతే {
                    (*pipelines)[conv2d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            తిరిగి pipeline;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_CONV_2D_DW:
        ఒకవేళ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ఒకవేళ (ggml_is_contiguous(src1)) {
                తిరిగి ctx->device->pipeline_conv2d_dw_whcn_f32;
            } లేకపోతే ఒకవేళ (ggml_is_contiguous_channels(src1)) {
                తిరిగి ctx->device->pipeline_conv2d_dw_cwhn_f32;
            }
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            ఒకవేళ (ggml_is_contiguous(src1)) {
                తిరిగి ctx->device->pipeline_conv2d_dw_whcn_f16_f32;
            } లేకపోతే ఒకవేళ (ggml_is_contiguous_channels(src1)) {
                తిరిగి ctx->device->pipeline_conv2d_dw_cwhn_f16_f32;
            }
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_CONV_3D:
        ఒకవేళ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            స్థిరరాశి uint32_t OC = (uint32_t)ggml_get_op_params_i32(dst, 11);
            స్థిరరాశి uint32_t IC = (uint32_t)ggml_get_op_params_i32(dst, 9);
            స్థిరరాశి uint32_t N  = (uint32_t)ggml_get_op_params_i32(dst, 10);
            స్థిరరాశి uint32_t NPQ = N * dst->ne[2] * dst->ne[1] * dst->ne[0];
            స్థిరరాశి vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, OC, NPQ);

            స్థిరరాశి uint32_t KW = (uint32_t)src0->ne[0];
            స్థిరరాశి uint32_t KH = (uint32_t)src0->ne[1];
            స్థిరరాశి uint32_t KD = (uint32_t)src0->ne[2];
            స్థిరరాశి uint32_t s0 = (uint32_t)ggml_get_op_params_i32(dst, 0);
            స్థిరరాశి uint32_t s1 = (uint32_t)ggml_get_op_params_i32(dst, 1);
            స్థిరరాశి uint32_t s2 = (uint32_t)ggml_get_op_params_i32(dst, 2);
            స్థిరరాశి uint32_t p0 = (uint32_t)ggml_get_op_params_i32(dst, 3);
            స్థిరరాశి uint32_t p1 = (uint32_t)ggml_get_op_params_i32(dst, 4);
            స్థిరరాశి uint32_t p2 = (uint32_t)ggml_get_op_params_i32(dst, 5);
            స్థిరరాశి uint32_t d0 = (uint32_t)ggml_get_op_params_i32(dst, 6);
            స్థిరరాశి uint32_t d1 = (uint32_t)ggml_get_op_params_i32(dst, 7);
            స్థిరరాశి uint32_t d2 = (uint32_t)ggml_get_op_params_i32(dst, 8);

            స్థిరరాశి uint32_t CRS = IC * KW * KH * KD;
            స్థిరరాశి uint32_t BS_K   = vk_conv_ఖండం_sizes[shape].K;
            స్థిరరాశి uint32_t BS_CRS = vk_conv_ఖండం_sizes[shape].CRS;
            స్థిరరాశి uint32_t BS_NPQ = vk_conv_ఖండం_sizes[shape].NPQ;
            స్థిరరాశి uint32_t aligned = ((OC  % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv3d_pipeline_state conv3d_pipeline_state(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned);

            std::map<vk_conv3d_pipeline_state, vk_pipeline> *pipelines = శూన్యసూచిక;
            ఒకవేళ (src0->type == GGML_TYPE_F32) {
                pipelines = &ctx->device->pipeline_conv3d_f32[shape];
            } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16) {
                pipelines = &ctx->device->pipeline_conv3d_f16_f32[shape];
            } లేకపోతే {
                తిరిగి శూన్యసూచిక;
            }

            vk_pipeline pipeline = శూన్యసూచిక;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                స్వయంచాలక it = pipelines->find(conv3d_pipeline_state);
                ఒకవేళ (it != pipelines->end()) {
                    pipeline = it->second;
                } లేకపోతే {
                    (*pipelines)[conv3d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            తిరిగి pipeline;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_ADD1:
        ఒకవేళ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            తిరిగి ctx->device->pipeline_add1_f16_f16;
        }
        ఒకవేళ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            తిరిగి ctx->device->pipeline_add1_f16_f32;
        }
        ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_add1_f32_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_ARANGE:
        ఒకవేళ (dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_arange_f32;
        }
        తిరిగి శూన్యసూచిక;
    స్థితి GGML_OP_FILL:
        ఒకవేళ (dst->type == GGML_TYPE_F32) {
            తిరిగి ctx->device->pipeline_fill_f32;
        }
        ఒకవేళ (dst->type == GGML_TYPE_F16) {
            తిరిగి ctx->device->pipeline_fill_f16;
        }
        తిరిగి శూన్యసూచిక;
    అప్రమేయ:
        తిరిగి శూన్యసూచిక;
    }

    GGML_వాడని(src2);
}

మూస<రకం_పేరు PC>
స్థిర శూన్య ggml_vk_op_f32(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, స్థిరరాశి ggml_tensor * src3, ggml_tensor * dst, ggml_op op, PC&& pc, vk_pipeline pipeline_override = శూన్యసూచిక) {
    VK_LOG_DEBUG("ggml_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    ఒకవేళ (src1 != శూన్యసూచిక) {
        std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    ఒకవేళ (src2 != శూన్యసూచిక) {
        std::cerr << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] << ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1=" << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    ఒకవేళ (src3 != శూన్యసూచిక) {
        std::cerr << "), (" << src3 << ", name=" << src3->name << ", type=" << src3->type << ", ne0=" << src3->ne[0] << ", ne1=" << src3->ne[1] << ", ne2=" << src3->ne[2] << ", ne3=" << src3->ne[3] << ", nb0=" << src3->nb[0] << ", nb1=" << src3->nb[1] << ", nb2=" << src3->nb[2] << ", nb3=" << src3->nb[3];
    }
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << ggml_op_name(op) << ")");
    GGML_ASSERT(op == GGML_OP_GET_అడ్డువరుసలు || op == GGML_OP_CPY || op == GGML_OP_CONCAT || (!ggml_is_quantized(src0->type) && (src1 == శూన్యసూచిక || !ggml_is_quantized(src1->type))));  // NOLINT
    GGML_ASSERT(dst->బఫరు != శూన్యసూచిక);
    స్థిరరాశి uint64_t ne00 = src0->ne[0];
    స్థిరరాశి uint64_t ne01 = src0->ne[1];
    స్థిరరాశి uint64_t ne02 = src0->ne[2];
    స్థిరరాశి uint64_t ne03 = src0->ne[3];

    స్థిరరాశి తార్కిక use_src1 = src1 != శూన్యసూచిక;
    స్థిరరాశి uint64_t ne10 = use_src1 ? src1->ne[0] : 0;
    స్థిరరాశి uint64_t ne11 = use_src1 ? src1->ne[1] : 0;
    స్థిరరాశి uint64_t ne12 = use_src1 ? src1->ne[2] : 0;
    స్థిరరాశి uint64_t ne13 = use_src1 ? src1->ne[3] : 0;

    స్థిరరాశి తార్కిక use_src2 = src2 != శూన్యసూచిక;
    స్థిరరాశి తార్కిక use_src3 = src3 != శూన్యసూచిక;

    init_pushconst_fastdiv(pc);

    vk_pipeline pipeline;
    ఒకవేళ (pipeline_override) {
        pipeline = pipeline_override;
    } లేకపోతే {
        pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);
    }

    ఒకవేళ (pipeline == శూన్యసూచిక) {
        std::cerr << "ggml_vulkan: Error: Missing op: " << ggml_op_name(op) << " for " << ggml_type_name(src0->type);
        ఒకవేళ (src1 != శూన్యసూచిక) {
            std::cerr << " and " << ggml_type_name(src1->type);
        }
        std::cerr << " to " << ggml_type_name(dst->type) << std::పంక్తి_ముగింపు;
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0, నిజం);
    vk_subbuffer src1_buf = use_src1 ? ggml_vk_tensor_subbuffer(ctx, src1, నిజం) : vk_subbuffer{};
    vk_subbuffer src2_buf = use_src2 ? ggml_vk_tensor_subbuffer(ctx, src2, నిజం) : vk_subbuffer{};
    vk_subbuffer src3_buf = use_src3 ? ggml_vk_tensor_subbuffer(ctx, src3, నిజం) : vk_subbuffer{};
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, నిజం);

    // Compute misalignment offset for descriptors and store it in in push constants.
    init_pushconst_tensor_offsets(ctx, pc, src0, src1, src2, src3, dst);

    std::array<uint32_t, 3> elements;

    ఎంపిక (op) {
    స్థితి GGML_OP_NORM:
    స్థితి GGML_OP_RMS_NORM_వెనుక:
    స్థితి GGML_OP_L2_NORM:
    స్థితి GGML_OP_SOFT_MAX:
    స్థితి GGML_OP_SOFT_MAX_వెనుక:
    స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
    స్థితి GGML_OP_CUMSUM:
    స్థితి GGML_OP_సగటు:
    స్థితి GGML_OP_ARGMAX:
        {
            స్థిరరాశి uint32_t nr = ggml_nrows(src0);
            ఒకవేళ (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } లేకపోతే ఒకవేళ (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } లేకపోతే {
                elements = { nr, 1, 1 };
            }
        } ఆపు;
    స్థితి GGML_OP_SOLVE_TRI:
        {
            uint32_t nr = (uint32_t)(ne02 * ne03);
            ఒకవేళ (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } లేకపోతే ఒకవేళ (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } లేకపోతే {
                elements = { nr, 1, 1 };
            }
        }
        ఆపు;
    స్థితి GGML_OP_RMS_NORM:
        ఒకవేళ (ctx->do_కూడు_rms_partials) {
            // Run one element per thread, 128 threads per workgroup
            elements = { (uint32_t)CEIL_DIV(ne00, 128), 1, 1 };
        } లేకపోతే {
            elements = { (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne03 };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]);
        }
        ఆపు;

    స్థితి GGML_OP_కూడిక:
        // We use GGML_OP_SUM_ROWS with 1 row.
        elements = { 1, 1, 1 };
        ఆపు;
    స్థితి GGML_OP_సమూహం_NORM:
        {
            స్థిరరాశి uint32_t అంకె_groups = dst->op_params[0];
            elements = { అంకె_groups * (uint32_t)src0->ne[3], 1, 1 };
        } ఆపు;
    స్థితి GGML_OP_DIAG_ముసుగు_INF:
        elements = { (uint32_t)ggml_nrows(src0), (uint32_t)ne00, 1 };
        ఆపు;
    స్థితి GGML_OP_ROPE:
    స్థితి GGML_OP_ROPE_వెనుక:
        {
            uint32_t nrows = (uint32_t)ggml_nrows(src0);
            uint32_t z = 1;
            ఒకవేళ (nrows > ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[0]) {
                z = CEIL_DIV(nrows, 32768);
                nrows = 32768;
            }
            elements = { nrows, (uint32_t)ne00, z };

        } ఆపు;
    స్థితి GGML_OP_GET_అడ్డువరుసలు:
        elements = { (uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12) };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]);
        ఆపు;
    స్థితి GGML_OP_GET_అడ్డువరుసలు_వెనుక:
        elements = { (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], 1 };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]);
        ఆపు;
    స్థితి GGML_OP_ARGSORT:
        GGML_ASSERT(0);
        ఆపు;
    స్థితి GGML_OP_IM2_నిలువు:
        {
            స్థిరరాశి తార్కిక is_2D = dst->op_params[6] == 1;

            స్థిరరాశి uint32_t IC = src1->ne[is_2D ? 2 : 1];

            స్థిరరాశి uint32_t KH = is_2D ? src0->ne[1] : 1;
            స్థిరరాశి uint32_t KW =         src0->ne[0];

            స్థిరరాశి uint32_t OH = is_2D ? dst->ne[2] : 1;
            స్థిరరాశి uint32_t OW =         dst->ne[1];

            స్థిరరాశి uint32_t batch = src1->ne[is_2D ? 3 : 2];

            స్థిరరాశి uint32_t CHW = IC * KH * KW;
            // Cap X workgroups to limit concurrent IC channel reads.
            // The shader loops over X to cover the full CHW dimension.
            // AMD prefers a lower limit
            స్థిరరాశి uint32_t min_cap = ctx->device->vendor_id == VK_VENDOR_ID_AMD ? 512u : 4096u;
            స్థిరరాశి uint32_t x_elements = std::min(CHW, std::max(min_cap, OW * KH * KW));
            elements = { x_elements, OW, OH * batch };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]);
        } ఆపు;
    స్థితి GGML_OP_IM2_నిలువు_3_D:
        {
            స్థిరరాశి uint32_t IC = ((స్థిరరాశి uint32_t *)(dst->op_params))[9];

            స్థిరరాశి uint32_t N  = ne13 / IC;

            స్థిరరాశి uint32_t KD = ne02;
            స్థిరరాశి uint32_t KH = ne01;
            స్థిరరాశి uint32_t KW = ne00;

            స్థిరరాశి uint32_t OD = dst->ne[3] / N;
            స్థిరరాశి uint32_t OH = dst->ne[2];
            స్థిరరాశి uint32_t OW = dst->ne[1];

            స్థిరరాశి uint32_t IC_KD_KH_KW = IC*KD*KH*KW;
            స్థిరరాశి uint32_t N_OD_OH = N*OD*OH;

            elements = { IC_KD_KH_KW, OW, N_OD_OH };
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]);
        } ఆపు;
    స్థితి GGML_OP_TIMESTEP_EMBEDDING:
        {
            స్థిరరాశి uint32_t dim = dst->op_params[0];
            uint32_t half_ceil = (dim + 1) / 2;
            elements = { half_ceil, (uint32_t)src0->ne[0], 1 };
        } ఆపు;
    స్థితి GGML_OP_CONV_TRANSPOSE_1D:
        {
            elements = {uint32_t(src0->ne[1]), 1, 1}; // parallelize in {Cout, 1, 1}
        } ఆపు;
    స్థితి GGML_OP_COL2IM_1D:
        {
            elements = { uint32_t(dst->ne[0]), uint32_t(dst->ne[1]), 1 };
        } ఆపు;
    స్థితి GGML_OP_POOL_1D:
        {
            స్థిరరాశి uint32_t N = dst->ne[3] * dst->ne[2];
            స్థిరరాశి uint32_t OC = dst->ne[1];
            స్థిరరాశి uint32_t OL = dst->ne[0];
            elements = { N * OC * OL, 1, 1};
        } ఆపు;
    స్థితి GGML_OP_POOL_2D:
        {
            స్థిరరాశి uint32_t outputs = ggml_nelements(dst);
            స్థిరరాశి uint32_t outputs_చొప్పున_workgroup = pipeline == ctx->device->pipeline_pool2d_subgroup_f32 ? 128 / ctx->device->subgroup_కొలత : 512;
            స్థిరరాశి uint32_t workgroups = CEIL_DIV(outputs, outputs_చొప్పున_workgroup);
            స్థిరరాశి uint32_t groups_y = CEIL_DIV(workgroups, ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[0]);
            స్థిరరాశి uint32_t groups_x = CEIL_DIV(workgroups, groups_y);
            elements = { groups_x, groups_y, 1 };
        } ఆపు;
    స్థితి GGML_OP_CONV_2D:
    స్థితి GGML_OP_CONV_TRANSPOSE_2D:
        ఒకవేళ constexpr (std::is_same_v<PC, vk_op_conv2d_నెట్టు_constants>) {
            స్థిరరాశి uint32_t NPQ = pc.N * pc.OH * pc.OW;
            స్థిరరాశి vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.Cout, NPQ);
            స్థిరరాశి uint32_t NPQ_ఖండాలు = CEIL_DIV(NPQ, vk_conv_ఖండం_sizes[shape].NPQ);

            elements = { pc.Cout, NPQ_ఖండాలు, 1 };
            ఒకవేళ (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } లేకపోతే {
            GGML_ABORT("invalid push constant type for CONV_2D");
        }
        ఆపు;
    స్థితి GGML_OP_CONV_3D:
        ఒకవేళ constexpr (std::is_same_v<PC, vk_op_conv3d_నెట్టు_constants>) {
            స్థిరరాశి uint32_t NPQ = pc.N * pc.OD * pc.OH * pc.OW;
            స్థిరరాశి vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.OC, NPQ);
            స్థిరరాశి uint32_t NPQ_ఖండాలు = CEIL_DIV(NPQ, vk_conv_ఖండం_sizes[shape].NPQ);

            elements = { pc.OC, NPQ_ఖండాలు, 1 };
            ఒకవేళ (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } లేకపోతే {
            GGML_ABORT("invalid push constant type for CONV_3D");
        }
        ఆపు;
    స్థితి GGML_OP_కూడు:
    స్థితి GGML_OP_SUB:
    స్థితి GGML_OP_DIV:
    స్థితి GGML_OP_MUL:
    స్థితి GGML_OP_ADD1:
    స్థితి GGML_OP_బహిర్గతం_PROD:
    స్థితి GGML_OP_ARANGE:
    స్థితి GGML_OP_FILL:
    స్థితి GGML_OP_SCALE:
    స్థితి GGML_OP_SQR:
    స్థితి GGML_OP_SQRT:
    స్థితి GGML_OP_SIN:
    స్థితి GGML_OP_COS:
    స్థితి GGML_OP_LOG:
    స్థితి GGML_OP_TRI:
    స్థితి GGML_OP_DIAG:
    స్థితి GGML_OP_CLAMP:
    స్థితి GGML_OP_LEAKY_RELU:
    స్థితి GGML_OP_PAD:
    స్థితి GGML_OP_PAD_REFLECT_1D:
    స్థితి GGML_OP_ROLL:
    స్థితి GGML_OP_REPEAT:
    స్థితి GGML_OP_REPEAT_వెనుక:
    స్థితి GGML_OP_CPY:
    స్థితి GGML_OP_CONCAT:
    స్థితి GGML_OP_UPSCALE:
    స్థితి GGML_OP_UNARY:
    స్థితి GGML_OP_GLU:
    స్థితి GGML_OP_CONV_2D_DW:
        {
            uint32_t ne = ggml_nelements(dst);
            ఒకవేళ (op == GGML_OP_CPY && ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                // Convert from number of logical elements to 2- or 4-byte units.
                ne /= ggml_blck_కొలత(src0->type);
                ఒకవేళ ((ggml_type_కొలత(src0->type) % 4) == 0) {
                    ne *= ggml_type_కొలత(src0->type) / 4;
                } లేకపోతే {
                    ne *= ggml_type_కొలత(src0->type) / 2;
                }
            }
            ఒకవేళ (op == GGML_OP_CONCAT && ggml_is_quantized(dst->type)) {
                ne = ne / ggml_blck_కొలత(dst->type) * ggml_type_కొలత(dst->type) / ggml_vk_concat_unit_కొలత(dst->type);
            }
            // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
            // Splitting into 512x512xZ wouldn't work well since each workgroup does 1024 elements.
            // So divide by block size here before splitting into 512x512 groups.
            ఒకవేళ (op == GGML_OP_CPY && !ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                ne = CEIL_DIV(ne, ggml_blck_కొలత(dst->type));
            }
            ఒకవేళ (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } లేకపోతే ఒకవేళ (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } లేకపోతే {
                elements = { ne, 1, 1 };
            }

            ఒకవేళ (pipeline == ctx->device->pipeline_cpy_transpose_02_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_02_16) {
                // 32x32 tiles over dims 0 and 2; dim1 and dim3 are the batch
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[2], 32);
                elements[2] = (uint32_t)(dst->ne[1]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]);
            } లేకపోతే ఒకవేళ (pipeline == ctx->device->pipeline_cpy_transpose_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_16) {
                // 32x32 tiles
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[1], 32);
                elements[2] = (uint32_t)(dst->ne[2]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]);
            }
        } ఆపు;
    స్థితి GGML_OP_కూడు_ID:
        {
            elements = { (uint32_t)ne01, (uint32_t)ne02, 1 };
        } ఆపు;
    స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
        {
            uint32_t ne = ggml_nelements(src0);
            ఒకవేళ (ggml_is_quantized(dst->type)) {
                // quants run 32 threads each doing QUANT_K elements
                ne = CEIL_DIV(ne, 32 * ggml_blck_కొలత(dst->type));
            } లేకపోతే {
                // scalar types do one element per thread, running 512 threads
                ne = CEIL_DIV(ne, 512);
            }
            ఒకవేళ (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } లేకపోతే ఒకవేళ (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } లేకపోతే {
                elements = { ne, 1, 1 };
            }
        }
        ఆపు;
    స్థితి GGML_OP_SSM_CONV:
        {
            స్థిరరాశి uint32_t nr  = src0->ne[1];
            స్థిరరాశి uint32_t n_t = dst->ne[1];
            స్థిరరాశి uint32_t n_s = dst->ne[2];
            elements = { nr, n_t, n_s };
        }
        ఆపు;
    అప్రమేయ:
        elements = { (uint32_t)ggml_nelements(src0), 1, 1 };
        ఆపు;
    }

    ఒకవేళ (op == GGML_OP_కూడు || op == GGML_OP_RMS_NORM) {
        vk_subbuffer a_buf = src0_buf;
        ఒకవేళ (ctx->do_కూడు_rms_partials) {
            a_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_కూడు_rms_partials, ctx->prealloc_కొలత_కూడు_rms_partials_స్థానభ్రంశం);
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, src1_buf, dst_buf, a_buf }, pc, elements);
    } లేకపోతే ఒకవేళ (op == GGML_OP_GLU) {
        // Empty src1 is possible in glu, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc, elements);
    } లేకపోతే ఒకవేళ (op == GGML_OP_SOFT_MAX) {
        // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, subbuf2, dst_buf }, pc, elements);
    } లేకపోతే ఒకవేళ (op == GGML_OP_ROPE || op == GGML_OP_ROPE_వెనుక) {
        // Empty src2 and src3 is possible in rope, but the shader needs a buffer
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        vk_subbuffer subbuf3 = use_src3 ? src3_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, subbuf2, dst_buf, subbuf3 }, pc, elements);
    } లేకపోతే ఒకవేళ (op == GGML_OP_IM2_నిలువు || op == GGML_OP_IM2_నిలువు_3_D) {
        ఒకవేళ (ctx->device->shader_int64 && ctx->device->బఫరు_device_చిరునామా) {
            // buffer device address path doesn't use dst buffer
            dst_buf.size = 1;
        }
        // im2col uses only src1 and dst buffers
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src1_buf, dst_buf }, pc, elements);
    } లేకపోతే ఒకవేళ (op == GGML_OP_లెక్క_EQUAL) {
        // count_equal assumes that destination buffer is initialized with zeroes
        ggml_vk_బఫరు_memset_async(subctx, dst_buf.బఫరు, dst_buf.స్థానభ్రంశం, 0, dst_buf.size);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } లేకపోతే ఒకవేళ (op == GGML_OP_OPT_అడుగు_SGD) {
        // OPT_STEP_SGD works on src0, it does not need dst
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf }, pc, elements);
    } లేకపోతే ఒకవేళ (use_src3) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, src3_buf, dst_buf }, pc, elements);
    } లేకపోతే ఒకవేళ (use_src2) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, dst_buf }, pc, elements);
    } లేకపోతే ఒకవేళ (use_src1) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } లేకపోతే {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, dst_buf }, pc, elements);
    }
}

శూన్య ggml_vk_get_అడ్డువరుసలు(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(dst->type);

    ggml_vk_op_f32<vk_op_ద్విమాన_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_GET_అడ్డువరుసలు, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_కొలత, (uint32_t)src0->nb[1] / src0_type_కొలత, (uint32_t)src0->nb[2] / src0_type_కొలత, (uint32_t)src0->nb[3] / src0_type_కొలత,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_కొలత, (uint32_t)src1->nb[1] / src1_type_కొలత, (uint32_t)src1->nb[2] / src1_type_కొలత, (uint32_t)src1->nb[3] / src1_type_కొలత,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_కొలత, (uint32_t) dst->nb[1] /  dst_type_కొలత, (uint32_t) dst->nb[2] /  dst_type_కొలత, (uint32_t) dst->nb[3] /  dst_type_కొలత,
        0,
        0.0f, 0.0f, 0,
    });
}

శూన్య ggml_vk_get_అడ్డువరుసలు_వెనుక(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(dst->type);

    ggml_vk_op_f32<vk_op_ద్విమాన_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_GET_అడ్డువరుసలు_వెనుక, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_కొలత, (uint32_t)src0->nb[1] / src0_type_కొలత, (uint32_t)src0->nb[2] / src0_type_కొలత, (uint32_t)src0->nb[3] / src0_type_కొలత,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2], (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_కొలత, (uint32_t)src1->nb[1] / src1_type_కొలత, (uint32_t)src1->nb[2] / src1_type_కొలత, (uint32_t)src1->nb[3] / src1_type_కొలత,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_కొలత, (uint32_t) dst->nb[1] /  dst_type_కొలత, (uint32_t) dst->nb[2] /  dst_type_కొలత, (uint32_t) dst->nb[3] /  dst_type_కొలత,
        0,
        0.0f, 0.0f, 0,
    });
}

శూన్య ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(dst->type);

    పూర్ణాంకం nb1 = dst->op_params[0] / src0_type_కొలత; // 4 bytes of float32
    పూర్ణాంకం nb2 = dst->op_params[1] / src0_type_కొలత; // 4 bytes of float32
    పూర్ణాంకం nb3 = dst->op_params[2] / src0_type_కొలత; // 4 bytes of float32
    పూర్ణాంకం స్థానభ్రంశం = dst->op_params[3] / src0_type_కొలత; // offset in bytes

    ggml_vk_op_f32<vk_op_ద్విమాన_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, dst->op, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_కొలత, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_కొలత, (uint32_t)src1->nb[1] / src1_type_కొలత, (uint32_t)src1->nb[2] / src1_type_కొలత, (uint32_t)src1->nb[3] / src1_type_కొలత,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_కొలత, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        0,
        0.0f, 0.0f, స్థానభ్రంశం,
    });
}

శూన్య ggml_vk_multi_కూడు(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    స్థిరరాశి ggml_tensor *మొదటి_కణుపు = cgraph->nodes[కణుపు_idx];
    స్థిరరాశి ggml_tensor *dst = cgraph->nodes[కణుపు_idx + ctx->అంకె_additional_fused_ops];

    // Make a list of all the tensors used by the op.
    // Last element of the list is the dest tensor.
    స్థిరరాశి ggml_tensor *tensors[MAX_PARAMETER_లెక్క];
    uint32_t అంకె_srcs = ctx->అంకె_additional_fused_ops + 2;
    uint32_t అంకె_tensors = అంకె_srcs + 1;
    GGML_ASSERT(అంకె_tensors + ctx->do_కూడు_rms_partials <= MAX_PARAMETER_లెక్క);

    tensors[0] = మొదటి_కణుపు->src[0];
    tensors[1] = మొదటి_కణుపు->src[1];
    కొరకు (int32_t i = 0; i < ctx->అంకె_additional_fused_ops; ++i) {
        // check whether the previous result is src[0] or src[1]
        ఒకవేళ (cgraph->nodes[కణుపు_idx + i] == cgraph->nodes[కణుపు_idx + i + 1]->src[0]) {
            tensors[i+2] = cgraph->nodes[కణుపు_idx + i + 1]->src[1];
        } లేకపోతే {
            tensors[i+2] = cgraph->nodes[కణుపు_idx + i + 1]->src[0];
        }
    }
    tensors[అంకె_srcs] = dst;

    vk_op_multi_కూడు_నెట్టు_constants pc;
    pc.ne20 = (uint32_t)dst->ne[0];
    pc.ne21 = (uint32_t)dst->ne[1];
    pc.ne22 = (uint32_t)dst->ne[2];
    pc.ne23 = (uint32_t)dst->ne[3];

    కొరకు (uint32_t i = 0; i < అంకె_tensors; ++i) {
        స్థిరరాశి ggml_tensor *t = tensors[i];
        pc.nb[i][0] = (uint32_t)t->nb[0] / పరిమాణం(దశాంశం);
        pc.nb[i][1] = (uint32_t)t->nb[1] / పరిమాణం(దశాంశం);
        pc.nb[i][2] = (uint32_t)t->nb[2] / పరిమాణం(దశాంశం);
        pc.nb[i][3] = (uint32_t)t->nb[3] / పరిమాణం(దశాంశం);
    }
    pc.rms_partials = ctx->do_కూడు_rms_partials;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, tensors[0], tensors[1], శూన్యసూచిక, dst, dst->op);

    ఒకవేళ (pipeline == శూన్యసూచిక) {
        std::cerr << "ggml_vulkan: Error: Missing multi_add";
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ggml_backend_vk_బఫరు_context * buf_ctx[MAX_PARAMETER_లెక్క];
    vk_బఫరు buf[MAX_PARAMETER_లెక్క];
    size_t స్థానభ్రంశం[MAX_PARAMETER_లెక్క];
    తార్కిక uma[MAX_PARAMETER_లెక్క];

    కొరకు (uint32_t i = 0; i < అంకె_tensors; ++i) {
        buf_ctx[i] = (ggml_backend_vk_బఫరు_context *)tensors[i]->బఫరు->context;
        buf[i] = శూన్యసూచిక;
        స్థానభ్రంశం[i] = 0;
        uma[i] = అబద్ధం;

        ఒకవేళ (ctx->device->uma) {
            ggml_vk_host_get(ctx->device, tensors[i]->దత్తాంశం, buf[i], స్థానభ్రంశం[i]);
            uma[i] = buf[i] != శూన్యసూచిక;
        }
        ఒకవేళ (!uma[i]) {
            buf[i] = buf_ctx[i]->dev_బఫరు;
            స్థానభ్రంశం[i] = vk_tensor_స్థానభ్రంశం(tensors[i]) + tensors[i]->view_offs;
        }
        GGML_ASSERT(buf[i] != శూన్యసూచిక);
    }
    // If any remaining descriptors are unused, just point them at src[0]
    కొరకు (uint32_t i = అంకె_tensors; i < MAX_PARAMETER_లెక్క; ++i) {
        buf[i] = buf[0];
        స్థానభ్రంశం[i] = 0;
    }
    ఒకవేళ (ctx->do_కూడు_rms_partials) {
        buf[అంకె_tensors] = ctx->prealloc_కూడు_rms_partials;
        స్థానభ్రంశం[అంకె_tensors] = ctx->prealloc_కొలత_కూడు_rms_partials_స్థానభ్రంశం;
    }

    std::array<uint32_t, 3> elements;

    uint32_t ne = ggml_nelements(dst);
    ఒకవేళ (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } లేకపోతే ఒకవేళ (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } లేకపోతే {
        elements = { ne, 1, 1 };
    }

    static_assert(MAX_PARAMETER_లెక్క == 12);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            ggml_vk_subbuffer(ctx, buf[0], స్థానభ్రంశం[0]),
            ggml_vk_subbuffer(ctx, buf[1], స్థానభ్రంశం[1]),
            ggml_vk_subbuffer(ctx, buf[2], స్థానభ్రంశం[2]),
            ggml_vk_subbuffer(ctx, buf[3], స్థానభ్రంశం[3]),
            ggml_vk_subbuffer(ctx, buf[4], స్థానభ్రంశం[4]),
            ggml_vk_subbuffer(ctx, buf[5], స్థానభ్రంశం[5]),
            ggml_vk_subbuffer(ctx, buf[6], స్థానభ్రంశం[6]),
            ggml_vk_subbuffer(ctx, buf[7], స్థానభ్రంశం[7]),
            ggml_vk_subbuffer(ctx, buf[8], స్థానభ్రంశం[8]),
            ggml_vk_subbuffer(ctx, buf[9], స్థానభ్రంశం[9]),
            ggml_vk_subbuffer(ctx, buf[10], స్థానభ్రంశం[10]),
            ggml_vk_subbuffer(ctx, buf[11], స్థానభ్రంశం[11]),
        }, pc, elements);
}

శూన్య ggml_vk_కూడు(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(dst->type);

    ggml_vk_op_f32<vk_op_ద్విమాన_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_కూడు, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_కొలత, (uint32_t)src0->nb[1] / src0_type_కొలత, (uint32_t)src0->nb[2] / src0_type_కొలత, (uint32_t)src0->nb[3] / src0_type_కొలత,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_కొలత, (uint32_t)src1->nb[1] / src1_type_కొలత, (uint32_t)src1->nb[2] / src1_type_కొలత, (uint32_t)src1->nb[3] / src1_type_కొలత,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_కొలత, (uint32_t) dst->nb[1] /  dst_type_కొలత, (uint32_t) dst->nb[2] /  dst_type_కొలత, (uint32_t) dst->nb[3] /  dst_type_కొలత,
        0,
        0.0f, 0.0f, ctx->do_కూడు_rms_partials,
    });
}

శూన్య ggml_vk_బహిర్గతం_prod(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(dst->type);

    ggml_vk_op_f32<vk_op_ద్విమాన_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_బహిర్గతం_PROD, {
        (uint32_t)ggml_nelements(dst),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3],
        (uint32_t)src0->nb[0] / src0_type_కొలత, (uint32_t)src0->nb[1] / src0_type_కొలత, (uint32_t)src0->nb[2] / src0_type_కొలత, (uint32_t)src0->nb[3] / src0_type_కొలత,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3],
        (uint32_t)src1->nb[0] / src1_type_కొలత, (uint32_t)src1->nb[1] / src1_type_కొలత, (uint32_t)src1->nb[2] / src1_type_కొలత, (uint32_t)src1->nb[3] / src1_type_కొలత,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3],
        (uint32_t) dst->nb[0] /  dst_type_కొలత, (uint32_t) dst->nb[1] /  dst_type_కొలత, (uint32_t) dst->nb[2] /  dst_type_కొలత, (uint32_t) dst->nb[3] /  dst_type_కొలత,
        0,
        0.0f, 0.0f, 0,
    });
}

శూన్య ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(dst->type);

    ggml_vk_op_f32<vk_op_ద్విమాన_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_SUB, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_కొలత, (uint32_t)src0->nb[1] / src0_type_కొలత, (uint32_t)src0->nb[2] / src0_type_కొలత, (uint32_t)src0->nb[3] / src0_type_కొలత,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_కొలత, (uint32_t)src1->nb[1] / src1_type_కొలత, (uint32_t)src1->nb[2] / src1_type_కొలత, (uint32_t)src1->nb[3] / src1_type_కొలత,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_కొలత, (uint32_t) dst->nb[1] /  dst_type_కొలత, (uint32_t) dst->nb[2] /  dst_type_కొలత, (uint32_t) dst->nb[3] /  dst_type_కొలత,
        0,
        0.0f, 0.0f, 0,
    });
}

శూన్య ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(dst->type);

    ggml_vk_op_f32<vk_op_ద్విమాన_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_MUL, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_కొలత, (uint32_t)src0->nb[1] / src0_type_కొలత, (uint32_t)src0->nb[2] / src0_type_కొలత, (uint32_t)src0->nb[3] / src0_type_కొలత,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_కొలత, (uint32_t)src1->nb[1] / src1_type_కొలత, (uint32_t)src1->nb[2] / src1_type_కొలత, (uint32_t)src1->nb[3] / src1_type_కొలత,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_కొలత, (uint32_t) dst->nb[1] /  dst_type_కొలత, (uint32_t) dst->nb[2] /  dst_type_కొలత, (uint32_t) dst->nb[3] /  dst_type_కొలత,
        0,
        0.0f, 0.0f, 0,
    });
}

పూర్ణాంకం ggml_vk_unary_mul_op_సూచీ(ggml_unary_op op) {
    ఎంపిక (op) {
        స్థితి GGML_UNARY_OP_GELU:     తిరిగి 0;
        స్థితి GGML_UNARY_OP_SIGMOID:  తిరిగి 1;
        స్థితి GGML_UNARY_OP_SILU:     తిరిగి 2;
        స్థితి GGML_UNARY_OP_SOFTPLUS: తిరిగి 3;
        అప్రమేయ:                     తిరిగి -1;
    }
}

శూన్య ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    స్థిరరాశి ggml_tensor * unary = cgraph->nodes[కణుపు_idx];
    ggml_tensor * mul = cgraph->nodes[కణుపు_idx + 1];

    // unary on src1 that tiles into src0
    స్థిరరాశి తార్కిక op_on_b = mul->src[1] == unary &&
                         !ggml_are_same_shape(unary->src[0], mul->src[0]) &&
                         ggml_can_repeat(unary, mul->src[0]);

    స్థిరరాశి ggml_tensor * src0 = op_on_b ? mul->src[0] : unary->src[0];
    స్థిరరాశి ggml_tensor * src1 = op_on_b ? unary->src[0] :
        ((mul->src[0] == unary) ? mul->src[1] : mul->src[0]);

    స్థిరరాశి తార్కిక f16 = src0->type == GGML_TYPE_F16;
    స్థిరరాశి తార్కిక norepeat = ggml_are_same_shape(src0, src1);
    స్థిరరాశి పూర్ణాంకం oi = ggml_vk_unary_mul_op_సూచీ(ggml_get_unary_op(unary));
    ఒకవేళ (oi < 0) {
        GGML_ABORT("fatal error");
    }
    vk_pipeline pipeline = ctx->device->pipeline_unary_mul[oi][f16][norepeat][op_on_b];

    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(mul->type);

    ggml_vk_op_f32<vk_op_ద్విమాన_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, mul, GGML_OP_UNARY, {
        (uint32_t)ggml_nelements(op_on_b ? mul : src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_కొలత, (uint32_t)src0->nb[1] / src0_type_కొలత, (uint32_t)src0->nb[2] / src0_type_కొలత, (uint32_t)src0->nb[3] / src0_type_కొలత,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_కొలత, (uint32_t)src1->nb[1] / src1_type_కొలత, (uint32_t)src1->nb[2] / src1_type_కొలత, (uint32_t)src1->nb[3] / src1_type_కొలత,
        (uint32_t) mul->ne[0], (uint32_t) mul->ne[1], (uint32_t) mul->ne[2],(uint32_t) mul->ne[3], (uint32_t) mul->nb[0] /  dst_type_కొలత, (uint32_t) mul->nb[1] /  dst_type_కొలత, (uint32_t) mul->nb[2] /  dst_type_కొలత, (uint32_t) mul->nb[3] /  dst_type_కొలత,
        0,
        0.0f, 0.0f, 0,
    }, pipeline);
}

శూన్య ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(dst->type);

    ggml_vk_op_f32<vk_op_ద్విమాన_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_DIV, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_కొలత, (uint32_t)src0->nb[1] / src0_type_కొలత, (uint32_t)src0->nb[2] / src0_type_కొలత, (uint32_t)src0->nb[3] / src0_type_కొలత,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_కొలత, (uint32_t)src1->nb[1] / src1_type_కొలత, (uint32_t)src1->nb[2] / src1_type_కొలత, (uint32_t)src1->nb[3] / src1_type_కొలత,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_కొలత, (uint32_t) dst->nb[1] /  dst_type_కొలత, (uint32_t) dst->nb[2] /  dst_type_కొలత, (uint32_t) dst->nb[3] /  dst_type_కొలత,
        0,
        0.0f, 0.0f, 0,
    });
}

శూన్య ggml_vk_కూడు_id(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, ggml_tensor * dst) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t src2_type_కొలత = ggml_type_కొలత(src2->type);

    ggml_vk_op_f32<vk_op_కూడు_id_నెట్టు_constants>(ctx, subctx, src0, src1, src2, శూన్యసూచిక, dst, GGML_OP_కూడు_ID, {
        (uint32_t)dst->ne[0],
        (uint32_t)dst->ne[1],
        (uint32_t)src0->nb[1] / src0_type_కొలత,
        (uint32_t)src0->nb[2] / src0_type_కొలత,
        (uint32_t)src1->nb[1] / src1_type_కొలత,
        (uint32_t)src2->nb[1] / src2_type_కొలత,
    });
}

స్థిర శూన్య ggml_vk_op_f32_wkv(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, స్థిరరాశి vk_op_rwkv_wkv6_నెట్టు_constants&& pc, పూర్ణాంకం సంస్కరణ_2) {
    GGML_ASSERT(సంస్కరణ_2 == 6 || సంస్కరణ_2 == 7);
    పూర్ణాంకం అంకె_srcs = సంస్కరణ_2 == 6 ? 6 : 7;

    కొరకు (పూర్ణాంకం i = 0; i < అంకె_srcs; i++) {
        GGML_ASSERT(!ggml_is_quantized(dst->src[i]->type));
    }

    GGML_ASSERT(dst->బఫరు != శూన్యసూచిక);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != శూన్యసూచిక);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    కొరకు (పూర్ణాంకం i = 0; i < అంకె_srcs; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements = {
        (uint32_t)(pc.B * pc.H),
        1,
        1
    };

    ఒకవేళ (సంస్కరణ_2 == 6) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
            pc, elements);
    } లేకపోతే ఒకవేళ (సంస్కరణ_2 == 7) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
            pc, elements);
    } లేకపోతే {
        // shouldn't happen
        GGML_ASSERT(అబద్ధం);
    }
}

శూన్య ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    స్థిరరాశి size_t seq_నిడివి = dst->src[0]->ne[2];
    స్థిరరాశి size_t n_embed = dst->ne[0];
    స్థిరరాశి size_t n_heads = dst->src[0]->ne[1];
    స్థిరరాశి size_t n_seqs = dst->src[5]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_నిడివి,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        6
    );
}

శూన్య ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    స్థిరరాశి size_t seq_నిడివి = dst->src[0]->ne[2];
    స్థిరరాశి size_t n_embed = dst->ne[0];
    స్థిరరాశి size_t n_heads = dst->src[0]->ne[1];
    స్థిరరాశి size_t n_seqs = dst->src[6]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_నిడివి,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        7
    );
}

శూన్య ggml_vk_gated_రేఖీయ_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    స్థిరరాశి size_t seq_నిడివి = dst->src[0]->ne[2];
    స్థిరరాశి size_t n_embed    = dst->ne[0];
    స్థిరరాశి size_t n_heads    = dst->src[0]->ne[1];
    స్థిరరాశి size_t n_seqs     = dst->src[4]->ne[1];

    దశాంశం scale;
    జ్ఞాపకం_నకలు(&scale, dst->op_params, పరిమాణం(దశాంశం));

    GGML_ASSERT(dst->బఫరు != శూన్యసూచిక);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != శూన్యసూచిక);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[5] = {};
    కొరకు (పూర్ణాంకం i = 0; i < 5; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    స్థిరరాశి vk_op_gated_రేఖీయ_attn_నెట్టు_constants pc = {
        (uint32_t)n_seqs,
        (uint32_t)seq_నిడివి,
        (uint32_t)n_embed,
        (uint32_t)n_heads,
        scale,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], dst_buf},
        pc, { (uint32_t)(n_seqs * n_heads), 1, 1 });
}

శూన్య ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * q = dst->src[0];
    స్థిరరాశి ggml_tensor * k = dst->src[1];
    స్థిరరాశి ggml_tensor * w = dst->src[2];
    స్థిరరాశి ggml_tensor * m = dst->src[3];

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, q, k, w, dst, dst->op);
    GGML_ASSERT(pipeline != శూన్యసూచిక);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    స్థిరరాశి uint32_t n_kv      = k->ne[2];
    స్థిరరాశి uint32_t n_heads   = q->ne[1];
    స్థిరరాశి uint32_t n_టోకెన్లు  = q->ne[2];
    స్థిరరాశి uint32_t n_streams = q->ne[3];
    స్థిరరాశి uint32_t n_masks   = m->ne[3];

    // one workgroup per tile of 64 keys and 8 tokens, see lightning_indexer.comp
    స్థిరరాశి uint32_t n_tiles_kv = CEIL_DIV(n_kv, 64);
    స్థిరరాశి uint32_t n_tiles_t  = CEIL_DIV(n_టోకెన్లు, 8);

    // q, w and dst are f32 and m is f16, so their strides are passed in elements;
    // k may be quantized, so its strides stay in bytes
    స్థిరరాశి uint32_t q_nb1 = q->nb[1] / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t q_nb2 = q->nb[2] / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t q_nb3 = q->nb[3] / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t k_nb2 = k->nb[2];
    స్థిరరాశి uint32_t k_nb3 = k->nb[3];
    స్థిరరాశి uint32_t w_nb1 = w->nb[1] / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t w_nb3 = w->nb[3] / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t m_nb1 = m->nb[1] / పరిమాణం(ggml_fp16_t);
    స్థిరరాశి uint32_t m_nb3 = m->nb[3] / పరిమాణం(ggml_fp16_t);
    స్థిరరాశి uint32_t d_nb1 = dst->nb[1] / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t d_nb3 = dst->nb[3] / పరిమాణం(దశాంశం);

    స్థిరరాశి vk_op_lightning_indexer_నెట్టు_constants pc = {
        n_kv, n_heads, n_టోకెన్లు, n_masks,
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

శూన్య ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src_q     = dst->src[0];
    స్థిరరాశి ggml_tensor * src_v     = dst->src[2];
    స్థిరరాశి ggml_tensor * src_beta  = dst->src[4];

    GGML_ASSERT(dst->బఫరు != శూన్యసూచిక);

    స్థిరరాశి uint32_t S_v      = (uint32_t)src_v->ne[0];
    స్థిరరాశి uint32_t H        = (uint32_t)src_v->ne[1];
    స్థిరరాశి uint32_t n_టోకెన్లు = (uint32_t)src_v->ne[2];
    స్థిరరాశి uint32_t n_seqs   = (uint32_t)src_v->ne[3];

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    స్థిరరాశి uint32_t K = (uint32_t)ggml_get_op_params_i32(dst, 0);

    స్థిరరాశి uint32_t s_off = S_v * H * n_టోకెన్లు * n_seqs;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != శూన్యసూచిక);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[6] = {};
    కొరకు (పూర్ణాంకం i = 0; i < 6; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    స్థిరరాశి uint32_t sq1 = (uint32_t)(src_q->nb[1] / పరిమాణం(దశాంశం));
    స్థిరరాశి uint32_t sq2 = (uint32_t)(src_q->nb[2] / పరిమాణం(దశాంశం));
    స్థిరరాశి uint32_t sq3 = (uint32_t)(src_q->nb[3] / పరిమాణం(దశాంశం));
    స్థిరరాశి uint32_t sv1 = (uint32_t)(src_v->nb[1] / పరిమాణం(దశాంశం));
    స్థిరరాశి uint32_t sv2 = (uint32_t)(src_v->nb[2] / పరిమాణం(దశాంశం));
    స్థిరరాశి uint32_t sv3 = (uint32_t)(src_v->nb[3] / పరిమాణం(దశాంశం));
    స్థిరరాశి uint32_t sb1 = (uint32_t)(src_beta->nb[1] / పరిమాణం(దశాంశం));
    స్థిరరాశి uint32_t sb2 = (uint32_t)(src_beta->nb[2] / పరిమాణం(దశాంశం));
    స్థిరరాశి uint32_t sb3 = (uint32_t)(src_beta->nb[3] / పరిమాణం(దశాంశం));

    స్థిరరాశి uint32_t neq1 = (uint32_t)src_q->ne[1];
    స్థిరరాశి uint32_t rq3  = (uint32_t)(src_v->ne[3] / src_q->ne[3]);

    స్థిరరాశి దశాంశం scale = 1.0f / sqrtf((దశాంశం)S_v);
    స్థిరరాశి vk_op_gated_delta_net_నెట్టు_constants pc = {
        H, n_టోకెన్లు, n_seqs, s_off,
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

శూన్య ggml_vk_ssm_పరిశీలన(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    స్థిరరాశి ggml_tensor * src2 = dst->src[2];
    స్థిరరాశి ggml_tensor * src3 = dst->src[3];
    స్థిరరాశి ggml_tensor * src4 = dst->src[4];
    స్థిరరాశి ggml_tensor * src5 = dst->src[5];

    GGML_ASSERT(dst->బఫరు != శూన్యసూచిక);

    స్థిరరాశి uint32_t తల_dim = src0->ne[1];
    స్థిరరాశి uint32_t n_తల = src1->ne[1];
    స్థిరరాశి uint32_t n_సమూహం = src4->ne[1];
    స్థిరరాశి uint32_t n_tok = src1->ne[2];
    స్థిరరాశి uint32_t n_seq = src1->ne[3];

    తార్కిక is_mamba2 = (src3->nb[1] == పరిమాణం(దశాంశం));
    GGML_ASSERT(is_mamba2);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, dst->op);
    GGML_ASSERT(pipeline != శూన్యసూచిక);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    స్థిరరాశి int64_t s_off = ggml_nelements(src1) * పరిమాణం(దశాంశం);

    స్థిరరాశి vk_op_ssm_పరిశీలన_నెట్టు_constants pc = {
        (uint32_t)src0->nb[2], (uint32_t)src0->nb[3],
        (uint32_t)src1->nb[2], (uint32_t)src1->nb[3],
        (uint32_t)src2->nb[1], (uint32_t)src2->nb[2],
        (uint32_t)src3->nb[1],
        (uint32_t)src4->nb[2], (uint32_t)src4->nb[3],
        (uint32_t)src5->nb[2], (uint32_t)src5->nb[3],
        (uint32_t)s_off,
        n_తల, తల_dim, n_సమూహం, n_tok,
        n_seq, (uint32_t) ggml_get_op_params_i32(dst, 0)
    };

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    కొరకు (పూర్ణాంకం i = 0; i < 7 && dst->src[i] != శూన్యసూచిక; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements;

    స్థిరరాశి uint32_t d_state = src0->ne[0];
    uint32_t అంకె_subgroups = d_state / ctx->device->subgroup_కొలత;
    స్థిరరాశి uint32_t అంకె_workgroups_x = CEIL_DIV(n_తల * తల_dim, అంకె_subgroups);
    స్థిరరాశి uint32_t అంకె_workgroups_y = n_seq;
    elements = { అంకె_workgroups_x, అంకె_workgroups_y, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
        pc, elements);
}

శూన్య ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    ggml_tensor * conv = cgraph->nodes[కణుపు_idx];
    స్థిరరాశి ggml_tensor * src0 = conv->src[0];
    స్థిరరాశి ggml_tensor * src1 = conv->src[1];

    // Pick the destination tensor (last node in the fused chain) and the optional bias.
    // Fusion modes: 0 = ssm_conv, 1 = ssm_conv+silu, 2 = ssm_conv+add(bias)+silu.
    ggml_tensor * dst = conv;
    స్థిరరాశి ggml_tensor * bias = శూన్యసూచిక;

    ఒకవేళ (ctx->అంకె_additional_fused_ops == 1) {
        dst = cgraph->nodes[కణుపు_idx + 1]; // silu
    } లేకపోతే ఒకవేళ (ctx->అంకె_additional_fused_ops == 2) {
        ggml_tensor * కూడు_2 = cgraph->nodes[కణుపు_idx + 1];
        bias = (కూడు_2->src[0] == conv) ? కూడు_2->src[1] : కూడు_2->src[0];
        dst = cgraph->nodes[కణుపు_idx + 2]; // silu
    }

    // The shader always declares 4 bindings; bind src0 as a dummy when bias isn't fused.
    స్థిరరాశి ggml_tensor * src2 = bias ? bias : src0;

    ggml_vk_op_f32<vk_op_ssm_conv_నెట్టు_constants>(ctx, subctx, src0, src1, src2, శూన్యసూచిక, dst, GGML_OP_SSM_CONV, {
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

స్థిర శూన్య ggml_vk_op_f32_opt_అడుగు_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, స్థిరరాశి vk_op_నెట్టు_constants&& pc) {
    స్థిరరాశి ggml_tensor * x = dst->src[0];
    స్థిరరాశి ggml_tensor * g = dst->src[1];
    స్థిరరాశి ggml_tensor * gm = dst->src[2];
    స్థిరరాశి ggml_tensor * gv = dst->src[3];
    స్థిరరాశి ggml_tensor * p = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(gm->type == GGML_TYPE_F32);
    GGML_ASSERT(gv->type == GGML_TYPE_F32);
    GGML_ASSERT(p->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->బఫరు != శూన్యసూచిక);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(gm));
    GGML_ASSERT(ggml_is_contiguous(gv));
    GGML_ASSERT(ggml_is_contiguous(p));
    GGML_ASSERT(ggml_are_same_shape(x, g));
    GGML_ASSERT(ggml_are_same_shape(x, gm));
    GGML_ASSERT(ggml_are_same_shape(x, gv));
    GGML_ASSERT(ggml_nelements(p) == 7);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, g, gm, gv, dst, GGML_OP_OPT_అడుగు_ADAMW);
    GGML_ASSERT(pipeline != శూన్యసూచిక);

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

శూన్య ggml_vk_opt_అడుగు_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    స్థిరరాశి size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32_opt_అడుగు_adamw(
        ctx, subctx, dst,
        { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f }
    );
}

శూన్య ggml_vk_opt_అడుగు_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, ggml_tensor * dst) {
    స్థిరరాశి size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32<vk_op_నెట్టు_constants>(ctx, subctx, src0, src1, src2, శూన్యసూచిక, dst, GGML_OP_OPT_అడుగు_SGD, { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

శూన్య ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    పూర్ణాంకం * op_params = (పూర్ణాంకం *)dst->op_params;

    స్థిరరాశి uint32_t unit_కొలత = ggml_vk_concat_unit_కొలత(dst->type);
    స్థిరరాశి uint32_t units_చొప్పున_ఖండం = ggml_type_కొలత(dst->type) / unit_కొలత;
    స్థిరరాశి uint32_t ఖండం_కొలత_3 = ggml_blck_కొలత(dst->type);
    స్థిరరాశి తార్కిక quantized = ggml_is_quantized(dst->type);

    // Address dimension 0 in packed storage units; higher strides may be noncontiguous.
    స్థిరరాశి uint32_t ne00 = src0->ne[0] / ఖండం_కొలత_3 * units_చొప్పున_ఖండం;
    స్థిరరాశి uint32_t ne10 = src1->ne[0] / ఖండం_కొలత_3 * units_చొప్పున_ఖండం;
    స్థిరరాశి uint32_t ne20 =  dst->ne[0] / ఖండం_కొలత_3 * units_చొప్పున_ఖండం;
    స్థిరరాశి uint32_t nb00 = quantized ? 1 : src0->nb[0] / unit_కొలత;
    స్థిరరాశి uint32_t nb10 = quantized ? 1 : src1->nb[0] / unit_కొలత;
    స్థిరరాశి uint32_t nb20 = quantized ? 1 :  dst->nb[0] / unit_కొలత;

    vk_op_concat_నెట్టు_constants pc {{
        ne20 * (uint32_t)dst->ne[1] * (uint32_t)dst->ne[2] * (uint32_t)dst->ne[3],
        ne00, (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], nb00, (uint32_t)src0->nb[1] / unit_కొలత, (uint32_t)src0->nb[2] / unit_కొలత, (uint32_t)src0->nb[3] / unit_కొలత,
        ne10, (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], nb10, (uint32_t)src1->nb[1] / unit_కొలత, (uint32_t)src1->nb[2] / unit_కొలత, (uint32_t)src1->nb[3] / unit_కొలత,
        ne20, (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], nb20, (uint32_t) dst->nb[1] / unit_కొలత, (uint32_t) dst->nb[2] / unit_కొలత, (uint32_t) dst->nb[3] / unit_కొలత,
        0,
        0.0f, 0.0f, op_params[0],
    }};
    ggml_vk_op_f32<vk_op_concat_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_CONCAT, std::move(pc));
}

శూన్య ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t mode = (uint32_t)ggml_get_op_params_i32(dst, 0);

    GGML_TENSOR_UNARY_OP_LOCALS

    దశాంశం sf0 = (దశాంశం)ne0 / ne00;
    దశాంశం sf1 = (దశాంశం)ne1 / ne01;
    దశాంశం sf2 = (దశాంశం)ne2 / ne02;
    దశాంశం sf3 = (దశాంశం)ne3 / ne03;
    దశాంశం pixel_స్థానభ్రంశం = 0.5f;

    ఒకవేళ (mode & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = ne0 > 1 && ne00 > 1 ? (దశాంశం)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (దశాంశం)(ne1 - 1) / (ne01 - 1) : sf1;
        pixel_స్థానభ్రంశం = 0.0f;
    }

    ggml_vk_op_f32<vk_op_upscale_నెట్టు_constants>(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_UPSCALE, {
        (uint32_t)ggml_nelements(dst), 0, 0,
        (uint32_t)ne00, (uint32_t)ne01,
        (uint32_t)nb00 / src0_type_కొలత, (uint32_t)nb01 / src0_type_కొలత, (uint32_t)nb02 / src0_type_కొలత, (uint32_t)nb03 / src0_type_కొలత,
        (uint32_t)ne0, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
        sf0, sf1, sf2, sf3, pixel_స్థానభ్రంశం
    });
}

శూన్య ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_నెట్టు_constants p = vk_op_unary_నెట్టు_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_SCALE, std::move(p));
}

శూన్య ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_SQR, vk_op_unary_నెట్టు_constants_init(src0, dst));
}

శూన్య ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_SQRT, vk_op_unary_నెట్టు_constants_init(src0, dst));
}

శూన్య ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(dst->type);

    ggml_vk_op_f32<vk_op_ద్విమాన_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_ADD1, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_కొలత, (uint32_t)src0->nb[1] / src0_type_కొలత, (uint32_t)src0->nb[2] / src0_type_కొలత, (uint32_t)src0->nb[3] / src0_type_కొలత,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_కొలత, (uint32_t)src1->nb[1] / src1_type_కొలత, (uint32_t)src1->nb[2] / src1_type_కొలత, (uint32_t)src1->nb[3] / src1_type_కొలత,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_కొలత, (uint32_t) dst->nb[1] /  dst_type_కొలత, (uint32_t) dst->nb[2] /  dst_type_కొలత, (uint32_t) dst->nb[3] /  dst_type_కొలత,
        0,
        0.0f, 0.0f, 0,
    });
}

శూన్య ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_arange(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");

    vk_op_నెట్టు_constants pc = {
        (uint32_t)ggml_nelements(dst),
        1,
        ggml_get_op_params_f32(dst, 0),
        ggml_get_op_params_f32(dst, 2),
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_ARANGE);
    GGML_ASSERT(pipeline != శూన్యసూచిక);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, అబద్ధం);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(dst), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

శూన్య ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_fill(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");
    స్థిరరాశి uint64_t n = ggml_nelements(dst);
    GGML_ASSERT(n > 0);

    vk_op_నెట్టు_constants pc = {
        (uint32_t)n,
        1,
        ggml_get_op_params_f32(dst, 0),
        0.0f,
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_FILL);
    GGML_ASSERT(pipeline != శూన్యసూచిక);

    // Split the task distribution to 2D to avoid exceeding maxComputeWorkGroupCount
    స్థిరరాశి uint32_t మొత్తం_wg = CEIL_DIV(n, pipeline->wg_denoms[0]);
    స్థిరరాశి uint32_t wg_x = std::min(మొత్తం_wg, ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[0]);
    స్థిరరాశి uint32_t wg_y = CEIL_DIV(మొత్తం_wg, wg_x);
    GGML_ASSERT(wg_y <= ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, అబద్ధం);

    std::array<uint32_t, 3> elements = { wg_x * pipeline->wg_denoms[0], wg_y * pipeline->wg_denoms[1], 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

శూన్య ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_SIN, vk_op_unary_నెట్టు_constants_init(src0, dst));
}

శూన్య ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_COS, vk_op_unary_నెట్టు_constants_init(src0, dst));
}

శూన్య ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_LOG, vk_op_unary_నెట్టు_constants_init(src0, dst));
}

శూన్య ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_నెట్టు_constants p = vk_op_unary_నెట్టు_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);

    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_TRI, std::move(p));
}

శూన్య ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_నెట్టు_constants p = vk_op_unary_నెట్టు_constants_init(src0, dst, ggml_nelements(dst));

    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_DIAG, std::move(p));
}

శూన్య ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_నెట్టు_constants p = vk_op_unary_నెట్టు_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_CLAMP, std::move(p));
}

శూన్య ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_pad_నెట్టు_constants p = vk_op_pad_నెట్టు_constants_init(src0, dst);
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_PAD, std::move(p));
}

శూన్య ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    స్థిరరాశి uint32_t p0 = (uint32_t)dst->op_params[0];
    స్థిరరాశి uint32_t p1 = (uint32_t)dst->op_params[1];

    vk_op_unary_నెట్టు_constants p = vk_op_unary_నెట్టు_constants_init(src0, dst, ggml_nelements(dst));
    జ్ఞాపకం_నకలు(&p.param1, &p0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&p.param2, &p1, పరిమాణం(దశాంశం));

    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_PAD_REFLECT_1D, std::move(p));
}

శూన్య ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    స్థిరరాశి int32_t s0 = ggml_get_op_params_i32(dst, 0);
    స్థిరరాశి int32_t s1 = ggml_get_op_params_i32(dst, 1);
    స్థిరరాశి int32_t s2 = ggml_get_op_params_i32(dst, 2);
    స్థిరరాశి int32_t s3 = ggml_get_op_params_i32(dst, 3);
    స్థిరరాశి uint32_t s01_packed = ((s0 + 0x8000) << 16) | (s1 + 0x8000);
    స్థిరరాశి uint32_t s23_packed = ((s2 + 0x8000) << 16) | (s3 + 0x8000);

    vk_op_unary_నెట్టు_constants p = vk_op_unary_నెట్టు_constants_init(src0, dst);
    జ్ఞాపకం_నకలు(&p.param1, &s01_packed, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&p.param2, &s23_packed, పరిమాణం(దశాంశం));

    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_ROLL, std::move(p));
}

శూన్య ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_నెట్టు_constants p = vk_op_unary_నెట్టు_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_REPEAT, std::move(p));
}

శూన్య ggml_vk_repeat_వెనుక(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_నెట్టు_constants p = vk_op_unary_నెట్టు_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_REPEAT_వెనుక, std::move(p));
}

శూన్య ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t)ggml_nelements(src0);
    ఒకవేళ (ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
        // Convert from number of logical elements to 2- or 4-byte units.
        ne /= ggml_blck_కొలత(src0->type);
        ఒకవేళ ((ggml_type_కొలత(src0->type) % 4) == 0) {
            ne *= ggml_type_కొలత(src0->type) / 4;
        } లేకపోతే {
            ne *= ggml_type_కొలత(src0->type) / 2;
        }
    }

    vk_op_unary_నెట్టు_constants p = vk_op_unary_నెట్టు_constants_init(src0, dst, ne);
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_CPY, std::move(p));
}

శూన్య ggml_vk_సమితి_అడ్డువరుసలు(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(dst->type);

    // Skip empty skip_rows operations. For most ops the empty check at the start
    // of ggml_vk_build_graph is sufficient, but set_rows can have a nonempty dst
    // with empty srcs.
    ఒకవేళ (ggml_is_ఖాళీలు(src0) || ggml_is_ఖాళీలు(src1)) {
        తిరిగి;
    }

    ggml_vk_op_f32<vk_op_ద్విమాన_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_సమితి_అడ్డువరుసలు, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_కొలత, (uint32_t)src0->nb[1] / src0_type_కొలత, (uint32_t)src0->nb[2] / src0_type_కొలత, (uint32_t)src0->nb[3] / src0_type_కొలత,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_కొలత, (uint32_t)src1->nb[1] / src1_type_కొలత, (uint32_t)src1->nb[2] / src1_type_కొలత, (uint32_t)src1->nb[3] / src1_type_కొలత,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_కొలత, (uint32_t) dst->nb[1] /  dst_type_కొలత, (uint32_t) dst->nb[2] /  dst_type_కొలత, (uint32_t) dst->nb[3] /  dst_type_కొలత,
        0,
        0.0f, 0.0f, 0,
    });
}

శూన్య ggml_vk_silu_వెనుక(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_SILU_వెనుక, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

శూన్య ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    దశాంశం * op_params = (దశాంశం *)dst->op_params;
    vk_op_unary_నెట్టు_constants p = vk_op_unary_నెట్టు_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_NORM, std::move(p));
}

శూన్య ggml_vk_సమూహం_norm(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    స్థిరరాశి పూర్ణాంకం * int_op_params = (స్థిరరాశి పూర్ణాంకం *)dst->op_params;
    స్థిరరాశి దశాంశం * float_op_params = (స్థిరరాశి దశాంశం *)dst->op_params;

    స్థిరరాశి uint32_t అంకె_groups = int_op_params[0];
    స్థిరరాశి దశాంశం eps = float_op_params[1];
    స్థిరరాశి uint32_t సమూహం_కొలత = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + అంకె_groups - 1) / అంకె_groups);

    ggml_vk_op_f32<vk_op_నెట్టు_constants>(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_సమూహం_NORM, { సమూహం_కొలత, 0, eps, 0.0f, 0.0f, 0.0f });
}

స్థిర uint32_t ggml_vk_rms_అంకె_partials(ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor *కణుపు) {
    స్థిరరాశి uint32_t ne = (uint32_t)కణుపు->ne[0];
    స్థిరరాశి uint32_t denom = ctx->device->pipeline_కూడు_rms[0][0][0]->wg_denoms[0];
    స్థిరరాశి uint32_t అంకె_partials = CEIL_DIV(ne, denom);
    తిరిగి అంకె_partials;
}

uint32_t ggml_vk_rms_partials_కొలత(ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor *కణుపు) {
    స్థిరరాశి uint32_t అంకె_partials = ggml_vk_rms_అంకె_partials(ctx, కణుపు);
    స్థిరరాశి uint32_t అంకె_బైట్లు = ROUNDUP_POW2(అంకె_partials * పరిమాణం(uint32_t), ctx->device->partials_binding_alignment);
    తిరిగి అంకె_బైట్లు;
}

స్థిర vk_op_rope_నెట్టు_constants ggml_vk_make_rope_constants(స్థిరరాశి ggml_tensor *dst, స్థిరరాశి ggml_tensor *src0, స్థిరరాశి తార్కిక has_ff, తార్కిక backprop, స్థిరరాశి uint32_t సమితి_అడ్డువరుసలు_stride) {
    స్థిరరాశి పూర్ణాంకం n_dims        = ((స్థిరరాశి int32_t *) dst->op_params)[1];
    స్థిరరాశి పూర్ణాంకం mode          = ((స్థిరరాశి int32_t *) dst->op_params)[2];
    స్థిరరాశి పూర్ణాంకం n_offs        = ((స్థిరరాశి int32_t *) dst->op_params)[15];
    // const int n_ctx         = ((const int32_t *) dst->op_params)[3];
    స్థిరరాశి పూర్ణాంకం n_ctx_orig    = ((స్థిరరాశి int32_t *) dst->op_params)[4];
    స్థిరరాశి దశాంశం freq_పునాది   = ((స్థిరరాశి దశాంశం *)   dst->op_params)[5];
    స్థిరరాశి దశాంశం freq_scale  = ((స్థిరరాశి దశాంశం *)   dst->op_params)[6];
    స్థిరరాశి దశాంశం ext_గుణకం  = ((స్థిరరాశి దశాంశం *)   dst->op_params)[7];
    స్థిరరాశి దశాంశం attn_గుణకం = ((స్థిరరాశి దశాంశం *)   dst->op_params)[8];
    స్థిరరాశి దశాంశం beta_fast   = ((స్థిరరాశి దశాంశం *)   dst->op_params)[9];
    స్థిరరాశి దశాంశం beta_slow   = ((స్థిరరాశి దశాంశం *)   dst->op_params)[10];
    పూర్ణాంకం sections[4] {};
    ఒకవేళ (mode & GGML_ROPE_TYPE_MROPE) {
        జ్ఞాపకం_నకలు(sections, (స్థిరరాశి int32_t *) dst->op_params + 11, పరిమాణం(పూర్ణాంకం)*4);
    }

    స్థిరరాశి తార్కిక is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    దశాంశం corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_పునాది, beta_fast, beta_slow, corr_dims);

    స్థిరరాశి దశాంశం theta_scale = powf(freq_పునాది, -2.0f/n_dims);

    uint32_t nb01 = src0->nb[1] / ggml_type_కొలత(src0->type);
    uint32_t nb02 = src0->nb[2] / ggml_type_కొలత(src0->type);
    uint32_t nb03 = src0->nb[3] / ggml_type_కొలత(src0->type);

    uint32_t nb11 = dst->nb[1] / ggml_type_కొలత(dst->type);
    uint32_t nb12 = dst->nb[2] / ggml_type_కొలత(dst->type);
    uint32_t nb13 = dst->nb[3] / ggml_type_కొలత(dst->type);

    vk_op_rope_నెట్టు_constants rope {
        (uint32_t)mode, (uint32_t)ggml_nrows(src0), (uint32_t)n_dims, (uint32_t)n_offs, freq_scale,
        freq_పునాది, ext_గుణకం, attn_గుణకం, {corr_dims[0], corr_dims[1]}, theta_scale, has_ff,
        { sections[0], sections[1], sections[2], sections[3] }, is_imrope, backprop, సమితి_అడ్డువరుసలు_stride,

        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)src0->ne[2],
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        0, 0, // a_offset, d_offset filled in by init_pushconst_tensor_offsets
    };

    తిరిగి rope;
}

స్థిర శూన్య ggml_vk_rms_norm_ముగింపు(ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor * src0) {
    ఒకవేళ (ctx->do_కూడు_rms_partials_స్థానభ్రంశం_calculation) {
        ctx->prealloc_కొలత_కూడు_rms_partials_స్థానభ్రంశం += ggml_vk_rms_partials_కొలత(ctx, src0);
        ctx->do_కూడు_rms_partials = అబద్ధం;
        ctx->do_కూడు_rms_partials_స్థానభ్రంశం_calculation = అబద్ధం;
    }
}

శూన్య ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, దశాంశం * op_params) {
    ggml_tensor * rms = cgraph->nodes[కణుపు_idx];
    స్థిరరాశి ggml_tensor * src0 = rms->src[0];

    ఒకవేళ (ctx->fused_rms_norm_బహుళకం == RMS_NORM_VIEW_సమితి_అడ్డువరుసలు) {
        GGML_ASSERT(ctx->అంకె_additional_fused_ops == 2);
        ggml_tensor * సమితి_అడ్డువరుసలు = cgraph->nodes[కణుపు_idx + 2];
        స్థిరరాశి ggml_tensor * indices = సమితి_అడ్డువరుసలు->src[1];
        vk_op_ద్విమాన_నెట్టు_constants pc = ggml_vk_rms_norm_నెట్టు_constants(src0, src0, సమితి_అడ్డువరుసలు, op_params[0], 0);
        init_pushconst_tensor_offsets(ctx, pc, src0, src0, శూన్యసూచిక, శూన్యసూచిక, సమితి_అడ్డువరుసలు);

        vk_pipeline pipeline = సమితి_అడ్డువరుసలు->type == GGML_TYPE_F16 ?
            ctx->device->pipeline_rms_norm_సమితి_అడ్డువరుసలు_f32_f16 : ctx->device->pipeline_rms_norm_సమితి_అడ్డువరుసలు_f32_f32;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_tensor_subbuffer(ctx, src0, నిజం),
                ggml_vk_tensor_subbuffer(ctx, src0, నిజం),
                ggml_vk_tensor_subbuffer(ctx, సమితి_అడ్డువరుసలు, నిజం),
                ggml_vk_tensor_subbuffer(ctx, indices),
            }, pc, {
                (uint32_t)src0->ne[1],
                std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]),
                std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]),
            });
        ggml_vk_rms_norm_ముగింపు(ctx, src0);
        తిరిగి;
    }

    ఒకవేళ (ctx->fused_rms_norm_బహుళకం == RMS_NORM_MUL_కూడు || ctx->fused_rms_norm_బహుళకం == RMS_NORM_MUL_కూడు_MUL) {
        ggml_tensor * mul = cgraph->nodes[కణుపు_idx + 1];
        ggml_tensor * కూడు_2 = cgraph->nodes[కణుపు_idx + 2];
        స్థిరరాశి ggml_tensor * బరువు = mul->src[0] == rms ? mul->src[1] : mul->src[0];
        స్థిరరాశి ggml_tensor * residual = కూడు_2->src[0] == mul ? కూడు_2->src[1] : కూడు_2->src[0];
        స్థిరరాశి తార్కిక do_post_multiply = ctx->fused_rms_norm_బహుళకం == RMS_NORM_MUL_కూడు_MUL;
        GGML_ASSERT(ctx->అంకె_additional_fused_ops == (do_post_multiply ? 3 : 2));
        ggml_tensor * dst = do_post_multiply ? cgraph->nodes[కణుపు_idx + 3] : కూడు_2;
        స్థిరరాశి ggml_tensor * post_scale = do_post_multiply ?
            (dst->src[0] == కూడు_2 ? dst->src[1] : dst->src[0]) : src0;

        స్థిరరాశి uint32_t అంకె_partials = ctx->do_కూడు_rms_partials ? ggml_vk_rms_అంకె_partials(ctx, dst) : 0;
        vk_op_ద్విమాన_నెట్టు_constants pc = ggml_vk_rms_norm_నెట్టు_constants(src0, బరువు, dst, op_params[0], అంకె_partials);
        init_pushconst_tensor_offsets(ctx, pc, src0, బరువు, residual, post_scale, dst);

        vk_pipeline pipeline;
        ఒకవేళ (ctx->do_కూడు_rms_partials) {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_కూడు_mul_partials_f32 : ctx->device->pipeline_rms_norm_mul_కూడు_partials_f32;
        } లేకపోతే {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_కూడు_mul_f32 : ctx->device->pipeline_rms_norm_mul_కూడు_f32;
        }
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ఒకవేళ (ctx->do_కూడు_rms_partials) {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, నిజం),
                    ggml_vk_tensor_subbuffer(ctx, బరువు, నిజం),
                    ggml_vk_tensor_subbuffer(ctx, dst, నిజం),
                    ggml_vk_subbuffer(ctx, ctx->prealloc_కూడు_rms_partials, ctx->prealloc_కొలత_కూడు_rms_partials_స్థానభ్రంశం),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, { (uint32_t)CEIL_DIV(src0->ne[0], 128), 1, 1 });
        } లేకపోతే {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, నిజం),
                    ggml_vk_tensor_subbuffer(ctx, బరువు, నిజం),
                    ggml_vk_tensor_subbuffer(ctx, dst, నిజం),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, {
                    (uint32_t)src0->ne[1],
                    std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]),
                    std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]),
                });
        }
        ggml_vk_rms_norm_ముగింపు(ctx, src0);
        తిరిగి;
    }

    ggml_tensor * dst;
    స్థిరరాశి ggml_tensor * src1;

    ఒకవేళ (ctx->fused_rms_norm_బహుళకం != RMS_NORM_లెక్క) {
        ggml_tensor * mul = cgraph->nodes[కణుపు_idx + 1];
        dst = mul;
        src1 = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    } లేకపోతే {
        dst = rms;
        src1 = src0;
    }

    స్థిరరాశి uint32_t అంకె_partials = ctx->do_కూడు_rms_partials ? ggml_vk_rms_అంకె_partials(ctx, dst) : 0;
    vk_op_ద్విమాన_నెట్టు_constants bin = ggml_vk_rms_norm_నెట్టు_constants(src0, src1, dst, op_params[0], అంకె_partials);

    ఒకవేళ (ctx->fused_rms_norm_బహుళకం == RMS_NORM_MUL_ROPE ||
        ctx->fused_rms_norm_బహుళకం == RMS_NORM_MUL_ROPE_VIEW_సమితి_అడ్డువరుసలు) {
        స్థిర constexpr uint32_t max_tensors = 7;
        స్థిరరాశి ggml_tensor *tensors[max_tensors] {};

        ggml_tensor *rms = cgraph->nodes[కణుపు_idx + 0];
        ggml_tensor *mul = cgraph->nodes[కణుపు_idx + 1];
        ggml_tensor *rope = cgraph->nodes[కణుపు_idx + 2];

        ggml_tensor *other_src = mul->src[0] == rms ? mul->src[1] : mul->src[0];

        తార్కిక do_సమితి_అడ్డువరుసలు = ctx->fused_rms_norm_బహుళకం == RMS_NORM_MUL_ROPE_VIEW_సమితి_అడ్డువరుసలు;
        GGML_ASSERT(ctx->అంకె_additional_fused_ops == (do_సమితి_అడ్డువరుసలు ? 4 : 2));

        tensors[0] = rms->src[0];
        tensors[1] = other_src;
        tensors[2] = mul;
        tensors[3] = rope->src[1]; // pos
        tensors[4] = rope->src[2]; // ff
        tensors[5] = cgraph->nodes[కణుపు_idx + ctx->అంకె_additional_fused_ops]; // dst
        tensors[6] = do_సమితి_అడ్డువరుసలు ? tensors[5]->src[1] : శూన్యసూచిక;
        స్థిరరాశి uint32_t సమితి_అడ్డువరుసలు_stride = do_సమితి_అడ్డువరుసలు ? tensors[5]->nb[1] / ggml_type_కొలత(tensors[5]->type) : 0;

        vk_op_rms_norm_mul_rope_నెట్టు_constants pc;
        pc.bin = bin;
        pc.rope = ggml_vk_make_rope_constants(rope, rope->src[0], tensors[4] != శూన్యసూచిక, అబద్ధం, సమితి_అడ్డువరుసలు_stride);

        vk_pipeline pipeline = tensors[5]->type == GGML_TYPE_F16 ? ctx->device->pipeline_rms_norm_mul_rope_f32_f16 : ctx->device->pipeline_rms_norm_mul_rope_f32_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

        ggml_backend_vk_బఫరు_context * buf_ctx[max_tensors];
        vk_బఫరు buf[max_tensors];
        size_t స్థానభ్రంశం[max_tensors];
        తార్కిక uma[max_tensors];

        కొరకు (uint32_t i = 0; i < max_tensors; ++i) {
            ఒకవేళ (!tensors[i]) {
                // If any remaining descriptors are unused, just point them at src[0]
                buf[i] = buf[0];
                స్థానభ్రంశం[i] = 0;
                కొనసాగు;
            }
            buf_ctx[i] = (ggml_backend_vk_బఫరు_context *)tensors[i]->బఫరు->context;
            buf[i] = శూన్యసూచిక;
            స్థానభ్రంశం[i] = 0;
            uma[i] = అబద్ధం;

            ఒకవేళ (ctx->device->uma) {
                ggml_vk_host_get(ctx->device, tensors[i]->దత్తాంశం, buf[i], స్థానభ్రంశం[i]);
                uma[i] = buf[i] != శూన్యసూచిక;
            }
            ఒకవేళ (!uma[i]) {
                buf[i] = buf_ctx[i]->dev_బఫరు;
                స్థానభ్రంశం[i] = vk_tensor_స్థానభ్రంశం(tensors[i]) + tensors[i]->view_offs;
            }
            GGML_ASSERT(buf[i] != శూన్యసూచిక);
        }

        // a_offset is unused (the fused path reads from shared memory), but the rope/set_rows dst can be misaligned.
        // Round the binding offset down to the storage buffer alignment; the in-element shift goes in pc.rope.d_offset.
        pc.rope.d_స్థానభ్రంశం = get_misalign_బైట్లు(ctx, tensors[5]) / ggml_type_కొలత(tensors[5]->type);
        స్థానభ్రంశం[5] &= ~(size_t(ctx->device->properties.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment) - 1);

        std::array<uint32_t, 3> elements;
        elements = { (uint32_t)rms->src[0]->ne[1], (uint32_t)rms->src[0]->ne[2], (uint32_t)rms->src[0]->ne[3] };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]);

        static_assert(max_tensors == 7);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_subbuffer(ctx, buf[0], స్థానభ్రంశం[0]),
                ggml_vk_subbuffer(ctx, buf[1], స్థానభ్రంశం[1]),
                ggml_vk_subbuffer(ctx, buf[2], స్థానభ్రంశం[2]),
                ggml_vk_subbuffer(ctx, buf[3], స్థానభ్రంశం[3]),
                ggml_vk_subbuffer(ctx, buf[4], స్థానభ్రంశం[4]),
                ggml_vk_subbuffer(ctx, buf[5], స్థానభ్రంశం[5]),
                ggml_vk_subbuffer(ctx, buf[6], స్థానభ్రంశం[6]),
            }, pc, elements);
    } లేకపోతే {
        GGML_ASSERT(ctx->fused_rms_norm_బహుళకం == RMS_NORM_MUL || ctx->fused_rms_norm_బహుళకం == RMS_NORM_లెక్క);
        ggml_vk_op_f32<vk_op_ద్విమాన_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_RMS_NORM, std::move(bin));
    }

    ggml_vk_rms_norm_ముగింపు(ctx, src0);
}

శూన్య ggml_vk_rms_norm_వెనుక(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    దశాంశం * op_params = (దశాంశం *)dst->op_params;
    ggml_vk_op_f32<vk_op_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_RMS_NORM_వెనుక, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f, 0.0f, 0.0f });
}

శూన్య ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    స్థిరరాశి దశాంశం * op_params = (స్థిరరాశి దశాంశం *)dst->op_params;
    vk_op_unary_నెట్టు_constants p = vk_op_unary_నెట్టు_constants_init(src0, dst);
    p.param1 = op_params[0];
    ggml_vk_op_f32<vk_op_unary_నెట్టు_constants>(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_L2_NORM, std::move(p));
}

శూన్య ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_UNARY, vk_op_unary_నెట్టు_constants_init(src0, dst));
}

శూన్య ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    దశాంశం * op_params = (దశాంశం *)dst->op_params;
    vk_op_unary_నెట్టు_constants p = vk_op_unary_నెట్టు_constants_init(src0, dst);
    p.param1 = op_params[1];
    p.param2 = op_params[2];
    p.param3 = op_params[3];
    p.param4 = op_params[4];
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_UNARY, std::move(p));
}

శూన్య ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి దశాంశం * op_params_f = (స్థిరరాశి దశాంశం *)dst->op_params;

    స్థిరరాశి తార్కిక swapped = (తార్కిక)dst->op_params[1];
    స్థిరరాశి తార్కిక విభజించు = src1 != శూన్యసూచిక;
    స్థిరరాశి దశాంశం alpha = op_params_f[2];
    స్థిరరాశి దశాంశం హద్దు = op_params_f[3];

    ఒకవేళ (!విభజించు) {
        GGML_ASSERT(src0->ne[0] / 2 == dst->ne[0]);
    } లేకపోతే {
        GGML_ASSERT(src0->ne[0] == src1->ne[0]);
        GGML_ASSERT(src0->ne[0] == dst->ne[0]);
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి uint32_t mode = విభజించు ? 2 : (swapped ? 1 : 0);
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = విభజించు ? ggml_type_కొలత(src1->type) : src0_type_కొలత;
    స్థిరరాశి uint32_t dst_type_కొలత  = ggml_type_కొలత(dst->type);

    ggml_vk_op_f32<vk_op_glu_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_GLU,
        {
            (uint32_t)ggml_nelements(dst),
            (uint32_t)src0->ne[0],
            (uint32_t)dst->ne[0],
            mode,
            alpha,
            హద్దు,
            (uint32_t)(src0->nb[0] / src0_type_కొలత),
            (uint32_t)(src0->nb[1] / src0_type_కొలత),
            (uint32_t)(src0->nb[2] / src0_type_కొలత),
            (uint32_t)(src0->nb[3] / src0_type_కొలత),
            (uint32_t)((విభజించు ? src1->nb[0] : src0->nb[0]) / src1_type_కొలత),
            (uint32_t)((విభజించు ? src1->nb[1] : src0->nb[1]) / src1_type_కొలత),
            (uint32_t)((విభజించు ? src1->nb[2] : src0->nb[2]) / src1_type_కొలత),
            (uint32_t)((విభజించు ? src1->nb[3] : src0->nb[3]) / src1_type_కొలత),
            (uint32_t)(dst->nb[0] / dst_type_కొలత),
            (uint32_t)(dst->nb[1] / dst_type_కొలత),
            (uint32_t)(dst->nb[2] / dst_type_కొలత),
            (uint32_t)(dst->nb[3] / dst_type_కొలత),
            (uint32_t)dst->ne[1],
            (uint32_t)dst->ne[2],
            0,
            0, 0, 0, 0, 0, 0,
        });
}

శూన్య ggml_vk_diag_ముసుగు_inf(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    int32_t * op_params = (int32_t *)dst->op_params;
    ggml_vk_op_f32<vk_op_diag_ముసుగు_నెట్టు_constants>(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_DIAG_ముసుగు_INF, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0] });
}

శూన్య ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, ggml_tensor * dst) {
    దశాంశం * op_params = (దశాంశం *)dst->op_params;

    దశాంశం scale = op_params[0];
    దశాంశం max_bias = op_params[1];

    స్థిరరాశి uint32_t ncols =   (uint32_t)src0->ne[0];
    స్థిరరాశి uint32_t nrows_x = (uint32_t)ggml_nrows(src0);
    స్థిరరాశి uint32_t nrows_y = (uint32_t)src0->ne[1];

    స్థిరరాశి uint32_t ne12 = src1 ? (uint32_t)(src1->ne[2]) : 0u;
    స్థిరరాశి uint32_t ne13 = src1 ? (uint32_t)(src1->ne[3]) : 0u;
    స్థిరరాశి uint32_t nb11 = src1 ? (uint32_t)(src1->nb[1] / src1->nb[0]) : 0u;
    స్థిరరాశి uint32_t nb12 = src1 ? (uint32_t)(src1->nb[2] / src1->nb[0]) : 0u;
    స్థిరరాశి uint32_t nb13 = src1 ? (uint32_t)(src1->nb[3] / src1->nb[0]) : 0u;

    స్థిరరాశి uint32_t n_తల_kv   = src0->ne[2];
    స్థిరరాశి uint32_t n_తల_log2 = 1u << (uint32_t) floorf(log2f((దశాంశం) n_తల_kv));

    స్థిరరాశి దశాంశం m0 = powf(2.0f, -(max_bias       ) / n_తల_log2);
    స్థిరరాశి దశాంశం m1 = powf(2.0f, -(max_bias / 2.0f) / n_తల_log2);

    vk_op_soft_max_నెట్టు_constants pc {
        ncols,
        src1 != శూన్యసూచిక ? nrows_y : (uint32_t)0,
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
        ne12, ne13,
        nb11, nb12, nb13,
        scale, max_bias,
        m0, m1,
        n_తల_log2,
        nrows_x,
        src2 != శూన్యసూచిక
    };

    ఒకవేళ (ncols <= 16384) {
        ggml_vk_op_f32<vk_op_soft_max_నెట్టు_constants>(ctx, subctx, src0, src1, src2, శూన్యసూచిక, dst, GGML_OP_SOFT_MAX, std::move(pc));
    } లేకపోతే {

        vk_subbuffer buf_a = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer buf_b = src1 ? ggml_vk_tensor_subbuffer(ctx, src1) : buf_a;
        vk_subbuffer buf_c = src2 ? ggml_vk_tensor_subbuffer(ctx, src2) : buf_a;
        vk_subbuffer buf_d = ggml_vk_tensor_subbuffer(ctx, dst);

        uint32_t elems_చొప్పున_wg = 128 * 4;
        uint32_t అంకె_wgs = CEIL_DIV(ncols, elems_చొప్పున_wg);
        size_t tmp_కొలత = అంకె_wgs * nrows_x * పరిమాణం(దశాంశం);

        ఒకవేళ (ctx->prealloc_కొలత_x < tmp_కొలత) {
            ctx->prealloc_కొలత_x = tmp_కొలత;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ఒకవేళ (ctx->prealloc_కొలత_y < tmp_కొలత) {
            ctx->prealloc_కొలత_y = tmp_కొలత;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ఒకవేళ (ctx->prealloc_x_need_sync || ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_subbuffer buf_x = { ctx->prealloc_x, 0, tmp_కొలత };
        vk_subbuffer buf_y = { ctx->prealloc_y, 0, tmp_కొలత };

        // the partial results overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_చివరి_pipeline_వాడిన = శూన్యసూచిక;
        ctx->prealloc_y_చివరి_tensor_వాడిన = శూన్యసూచిక;

        std::array<uint32_t, 3> elements = { అంకె_wgs, nrows_x, 1 };

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

        ctx->prealloc_x_need_sync = నిజం;
        ctx->prealloc_y_need_sync = నిజం;
    }
}

శూన్య ggml_vk_soft_max_వెనుక(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    దశాంశం * op_params = (దశాంశం *)dst->op_params;
    ggml_vk_op_f32<vk_op_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_SOFT_MAX_వెనుక, { (uint32_t)src0->ne[0], (uint32_t)ggml_nrows(src0), op_params[0], op_params[1], 0.0f, 0.0f });
}

శూన్య ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    topk_moe_బహుళకం mode = ctx->fused_topk_moe_బహుళకం;
    స్థిరరాశి తార్కిక has_bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS || mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
    ggml_tensor * logits = cgraph->nodes[కణుపు_idx + 0]->src[0];
    ggml_tensor * bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[కణుపు_idx + 2]->src[1] :
                         mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[కణుపు_idx + 3]->src[1] :
                                                                   logits;
    ggml_tensor * weights = cgraph->nodes[కణుపు_idx + ctx->అంకె_additional_fused_ops];
    ggml_tensor * ids = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[కణుపు_idx + 4] :
                        mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[కణుపు_idx + 5] :
                        mode == TOPK_MOE_LATE_SOFTMAX             ? cgraph->nodes[కణుపు_idx + 1] :
                                                                   cgraph->nodes[కణుపు_idx + 3];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    స్థిరరాశి పూర్ణాంకం n_experts = logits->ne[0];
    స్థిరరాశి పూర్ణాంకం n_అడ్డువరుసలు    = logits->ne[1];
    స్థిరరాశి పూర్ణాంకం n_expert_వాడిన = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_కొలత(ids->type) == (size_t) n_experts);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, cgraph->nodes[కణుపు_idx], GGML_OP_SOFT_MAX);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer bias_buf = ggml_vk_tensor_subbuffer(ctx, bias);
    vk_subbuffer weights_buf = ggml_vk_tensor_subbuffer(ctx, weights);
    vk_subbuffer ids_buf = ggml_vk_tensor_subbuffer(ctx, ids);

    vk_op_topk_moe_నెట్టు_constants pc {};
    pc.n_అడ్డువరుసలు = n_అడ్డువరుసలు;
    pc.n_experts_నెట్టు = n_experts;
    pc.n_expert_వాడిన = n_expert_వాడిన;
    pc.clamp_min = -std::numeric_limits<దశాంశం>::infinity();
    pc.clamp_max = std::numeric_limits<దశాంశం>::infinity();
    ఒకవేళ (mode == TOPK_MOE_EARLY_SOFTMAX_NORM) {
        ggml_tensor * clamp = cgraph->nodes[కణుపు_idx + 7];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    ఒకవేళ (mode == TOPK_MOE_SIGMOID_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[కణుపు_idx + 8];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    ఒకవేళ (mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[కణుపు_idx + 9];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }

#నిర్వచించు GATING_FUNC_SOFTMAX 0
#నిర్వచించు GATING_FUNC_SIGMOID 1
#నిర్వచించు GATING_FUNC_SOFTMAX_బరువు 2
#నిర్వచించు GATING_FUNC_SQRT_SOFTPLUS 3

    pc.gating_func = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? GATING_FUNC_SIGMOID :
                     mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? GATING_FUNC_SQRT_SOFTPLUS :
                     mode == TOPK_MOE_LATE_SOFTMAX             ? GATING_FUNC_SOFTMAX_బరువు :
                                                                 GATING_FUNC_SOFTMAX;
    pc.has_bias = has_bias;
    pc.with_norm = mode == TOPK_MOE_EARLY_SOFTMAX_NORM || has_bias;
    ఒకవేళ (ctx->fused_topk_moe_scale) {
        GGML_ASSERT(weights->op == GGML_OP_SCALE);
        pc.output_scale = ggml_get_op_params_f32(weights, 0);
        pc.output_bias = ggml_get_op_params_f32(weights, 1);
    } లేకపోతే {
        pc.output_scale = 1.0f;
        pc.output_bias = 0.0f;
    }

    GGML_ASSERT(n_expert_వాడిన <= n_experts);

    స్థిరరాశి uint32_t అడ్డువరుసలు_చొప్పున_ఖండం = 4;
    std::array<uint32_t, 3> elements = { CEIL_DIV(n_అడ్డువరుసలు, అడ్డువరుసలు_చొప్పున_ఖండం), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {logits_buf, bias_buf, weights_buf, ids_buf}, pc, elements);
}

శూన్య ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, తార్కిక backprop) {
    ggml_tensor * dst = cgraph->nodes[కణుపు_idx];
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    స్థిరరాశి ggml_tensor * src2 = dst->src[2];
    స్థిరరాశి ggml_tensor * src3 = శూన్యసూచిక;
    స్థిరరాశి పూర్ణాంకం n_dims        = ((int32_t *) dst->op_params)[1];
    స్థిరరాశి పూర్ణాంకం mode          = ((int32_t *) dst->op_params)[2];
    // const int n_ctx         = ((int32_t *) dst->op_params)[3];
    స్థిరరాశి పూర్ణాంకం n_ctx_orig    = ((int32_t *) dst->op_params)[4];
    స్థిరరాశి దశాంశం freq_పునాది   = ((దశాంశం *)   dst->op_params)[5];
    స్థిరరాశి దశాంశం beta_fast   = ((దశాంశం *)   dst->op_params)[9];
    స్థిరరాశి దశాంశం beta_slow   = ((దశాంశం *)   dst->op_params)[10];
    పూర్ణాంకం sections[4] {};
    ఒకవేళ (mode & GGML_ROPE_TYPE_MROPE) {
        జ్ఞాపకం_నకలు(sections, (int32_t *) dst->op_params + 11, పరిమాణం(పూర్ణాంకం)*4);
    }

    దశాంశం corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_పునాది, beta_fast, beta_slow, corr_dims);

    uint32_t సమితి_అడ్డువరుసలు_stride = 0;
    // Fused rope + view + set_rows passes the set_rows destination stride in set_rows_stride
    // and overrides the dst and sets src3=row_indices
    ఒకవేళ (ctx->అంకె_additional_fused_ops > 0) {
        సమితి_అడ్డువరుసలు_stride = cgraph->nodes[కణుపు_idx + 2]->nb[1] / ggml_type_కొలత(cgraph->nodes[కణుపు_idx + 2]->type);
        src3 = cgraph->nodes[కణుపు_idx + 2]->src[1];
        dst = cgraph->nodes[కణుపు_idx + 2];
    }

    ggml_vk_op_f32<vk_op_rope_నెట్టు_constants>(ctx, subctx, src0, src1, src2, src3, dst, GGML_OP_ROPE,
        ggml_vk_make_rope_constants(cgraph->nodes[కణుపు_idx], src0, src2 != శూన్యసూచిక, backprop, సమితి_అడ్డువరుసలు_stride));
}

శూన్య ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    స్థిరరాశి uint32_t * op_params = (స్థిరరాశి uint32_t *)dst->op_params;

    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);

    uint32_t ncols_pad_log2 = (uint32_t)ceilf(log2f(దశాంశం(ncols)));
    uint32_t ncolsp2 = 1 << ncols_pad_log2;

    vk_op_argsort_నెట్టు_constants pc { ncols, ncolsp2, ncols_pad_log2, nrows, op_params[0], 0, 0, 0, 0, };

    // Pick the largest workgroup size <= ncolsp2
    uint32_t pipeline_idx = std::min(ncols_pad_log2, అంకె_argsort_pipelines - 1);

    uint32_t max_wg_log2 = std::min(ctx->device->max_workgroup_కొలత_log2, అంకె_argsort_pipelines - 1);

    // Use the "small" argsort shader if the whole sort can be done by a single workgroup.
    తార్కిక use_small = ncols_pad_log2 <= max_wg_log2 &&
                     ctx->device->pipeline_argsort_f32[pipeline_idx] != శూన్యసూచిక;

    vk_pipeline pipeline = use_small ? ctx->device->pipeline_argsort_f32[pipeline_idx]
                                     : ctx->device->pipeline_argsort_large_f32[pipeline_idx];

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer subbuf1 = dst_buf;

    // Reserve space for ivec2 per element, with rows padded to a power of two
    ఒకవేళ (!use_small) {
        స్థిరరాశి size_t x_sz = size_t{ncolsp2} * nrows * 2 * పరిమాణం(పూర్ణాంకం);

        ఒకవేళ (ctx->prealloc_కొలత_x < x_sz) {
            ctx->prealloc_కొలత_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ఒకవేళ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        subbuf1 = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    }

    std::array<uint32_t, 3> elements;

    elements[0] = ncolsp2;
    elements[1] = std::min((uint32_t)ggml_nrows(src0), ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]);
    elements[2] = 1;

    // First dispatch initializes tmp_idx and does the first N passes where
    // there is only communication between threads in the same workgroup.
    {
        vk_op_argsort_నెట్టు_constants pc2 = pc;
        pc2.outer_ప్రారంభం = 0;
        pc2.outer_చివర = std::min(ncols_pad_log2, max_wg_log2);
        pc2.inner_ప్రారంభం = 0;
        pc2.inner_చివర = 100;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
    }
    ఒకవేళ (!use_small) {
        ggml_vk_sync_buffers(ctx, subctx);
        // Loop over outer/inner passes, synchronizing between each pass.
        కొరకు (uint32_t outer = max_wg_log2; outer < ncols_pad_log2; ++outer) {
            కొరకు (uint32_t inner = 0; inner < outer + 1; ++inner) {
                vk_op_argsort_నెట్టు_constants pc2 = pc;
                pc2.outer_ప్రారంభం = outer;
                pc2.outer_చివర = outer + 1;
                pc2.inner_ప్రారంభం = inner;
                pc2.inner_చివర = inner + 1;
                // When the inner idx is large enough, there's only communication
                // within a workgroup. So the remaining inner iterations can all
                // run in the same dispatch.
                ఒకవేళ (outer - inner < pipeline_idx) {
                    pc2.inner_చివర = 100;
                    inner = outer;
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx];
                } లేకపోతే {
                    // Smaller workgroup empirically seems to perform better
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx - 2];
                }
                ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
                ggml_vk_sync_buffers(ctx, subctx);
            }
        }
        ctx->prealloc_x_need_sync = నిజం;
    }
}

శూన్య ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);
    uint32_t k = dst->ne[0];

    // tournament path is faster where it fits; use radix-select only past its k limit
    స్థిరరాశి uint32_t k_min_pipeline = std::max((uint32_t) log2f(దశాంశం(k)) + 1, ctx->device->subgroup_కొలత_log2);
    ఒకవేళ (k_min_pipeline >= అంకె_topk_pipelines || ctx->device->pipeline_topk_f32[k_min_pipeline] == శూన్యసూచిక) {
        vk_pipeline pipeline = ctx->device->pipeline_topk_radix_f32;
        GGML_ASSERT(pipeline != శూన్యసూచిక);

        ఒకవేళ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_op_topk_radix_నెట్టు_constants pc { ncols, k, nrows, 0, 0, 0 };
        std::array<uint32_t, 3> elements {
            pipeline->wg_denoms[0],
            std::min(nrows, ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]),
            1,
        };
        // the non-QSA path only uses bindings 0/1; bind valid buffers for the unused QSA slots
        vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, dst_buf, src0_buf, src0_buf, src0_buf }, pc, elements);
        తిరిగి;
    }

    vk_op_topk_నెట్టు_constants pc { ncols, ncols, ncols, k, nrows, 0, 0 };

    ఒకవేళ (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    std::array<uint32_t, 3> elements;
    elements[1] = std::min(nrows, ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]);
    elements[2] = 1;

    uint32_t అంకె_elements = ncols;

    // Each iteration reduces a workgroup's worth of elements down to the K
    // largest elements. Repeat until we have the top K elements.
    // Need to do at least one iteration to write out the results.
    తార్కిక పూర్తి_one_iter = అబద్ధం;
    uint32_t dbl_buf_సూచీ = 0;
    size_t dbl_buf_కొలత;
    అయితే (అంకె_elements > k || !పూర్తి_one_iter) {

        // Prefer going as small as num_topk_pipelines - 3 for perf reasons.
        // But if K is larger, then we need a larger workgroup
        uint32_t max_pipeline = అంకె_topk_pipelines - 1;
        uint32_t preferred_pipeline = std::max(అంకె_topk_pipelines - 3, (uint32_t)log2f(దశాంశం(k)) + 2);
        max_pipeline = std::min(preferred_pipeline, max_pipeline);
        uint32_t min_pipeline = (uint32_t)log2f(దశాంశం(k)) + 1;
        // require full subgroup
        min_pipeline = std::max(min_pipeline, ctx->device->subgroup_కొలత_log2);

        uint32_t pipeline_idx = (uint32_t)ceilf(log2f(దశాంశం(అంకె_elements)));
        pipeline_idx = std::min(pipeline_idx, max_pipeline);
        pipeline_idx = std::max(pipeline_idx, min_pipeline);

        ఒకవేళ (అంకె_elements > (1u << pipeline_idx)) {
            // If we could finish on this loop iteration (i.e. a single workgroup)
            // then do so. It's better than the overhead of another pass.
            కొరకు (uint32_t i = pipeline_idx; i < అంకె_topk_pipelines; ++i) {
                ఒకవేళ (అంకె_elements <= (1u << i)) {
                    pipeline_idx = i;
                    ఆపు;
                }
            }
        }

        vk_pipeline pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        // If the device doesn't support a pipeline this large, use smaller
        అయితే (!pipeline) {
            pipeline_idx--;
            GGML_ASSERT(pipeline_idx >= min_pipeline);
            pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        }

        vk_op_topk_నెట్టు_constants pc2 = pc;
        pc2.ncols_ఆదానం = అంకె_elements;

        // Number of elements remaining after this pass
        uint32_t అంకె_dst_elements = (అంకె_elements / pipeline->wg_denoms[0]) * k + std::min(k, అంకె_elements % pipeline->wg_denoms[0]);

        pc2.ncols_output = అంకె_dst_elements;

        ఒకవేళ (!పూర్తి_one_iter) {
            // Reserve space for ivec2 per element, double buffered
            // K per workgroup per row
            dbl_buf_కొలత = అంకె_dst_elements * nrows * 2 * పరిమాణం(పూర్ణాంకం);
            dbl_buf_కొలత = ROUNDUP_POW2(dbl_buf_కొలత, ctx->device->properties.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment);
            స్థిరరాశి size_t x_sz = dbl_buf_కొలత * 2;

            ఒకవేళ (ctx->prealloc_కొలత_x < x_sz) {
                ctx->prealloc_కొలత_x = x_sz;
                ggml_vk_preallocate_buffers(ctx, subctx);
            }
        }

        vk_subbuffer src_buf;
        vk_subbuffer dst_buf;

        ఒకవేళ (అంకె_elements == ncols) {
            pc2.మొదటి_pass = 1;
            src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        } లేకపోతే {
            src_buf = { ctx->prealloc_x, dbl_buf_సూచీ * dbl_buf_కొలత, dbl_buf_కొలత };
        }
        ఒకవేళ (అంకె_dst_elements == k) {
            pc2.చివరి_pass = 1;
            dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
        } లేకపోతే {
            dst_buf = { ctx->prealloc_x, (dbl_buf_సూచీ ^ 1) * dbl_buf_కొలత, dbl_buf_కొలత };
        }

        elements[0] = అంకె_elements;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc2, elements);
        అంకె_elements = అంకె_dst_elements;
        dbl_buf_సూచీ ^= 1;
        ఒకవేళ (అంకె_elements > k) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        పూర్తి_one_iter = నిజం;
    }
    ctx->prealloc_x_need_sync = నిజం;
}

శూన్య ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    స్థిరరాశి ggml_tensor * get_అడ్డువరుసలు = cgraph->nodes[కణుపు_idx + 0];
    స్థిరరాశి ggml_tensor * కూడు_2      = cgraph->nodes[కణుపు_idx + ctx->అంకె_additional_fused_ops - 1];
    ggml_tensor *       శిఖరం_k    = cgraph->nodes[కణుపు_idx + ctx->అంకె_additional_fused_ops];

    స్థిరరాశి ggml_tensor * scores   = get_అడ్డువరుసలు->src[0]; // [n_tps, n_blocks, n_stream]
    స్థిరరాశి ggml_tensor * cell_blk = get_అడ్డువరుసలు->src[1]; // [n_kv, n_stream]

    // raw f16 mask: follow the reshape/cpy chain back to the materialized input
    స్థిరరాశి ggml_tensor * ముసుగు = కూడు_2->src[1];
    అయితే (ముసుగు->op == GGML_OP_RESHAPE || ముసుగు->op == GGML_OP_CPY) {
        ముసుగు = ముసుగు->src[0];
    }

    స్థిరరాశి uint32_t n_tps    = scores->ne[0];
    స్థిరరాశి uint32_t n_ఖండాలు = scores->ne[1];
    స్థిరరాశి uint32_t n_ప్రవాహం = scores->ne[2];
    స్థిరరాశి uint32_t n_kv     = cell_blk->ne[0];
    స్థిరరాశి uint32_t వెడల్పు    = శిఖరం_k->ne[0];
    స్థిరరాశి uint32_t nrows    = n_tps * n_ప్రవాహం;

    vk_pipeline pipeline = ctx->device->pipeline_topk_radix_qsa;
    GGML_ASSERT(pipeline != శూన్యసూచిక);

    // scratch holds the gathered+masked input, materialized once and reused across passes
    స్థిరరాశి size_t scratch_కొలత = size_t{ n_kv } * nrows * పరిమాణం(దశాంశం);
    ఒకవేళ (ctx->prealloc_కొలత_x < scratch_కొలత) {
        ctx->prealloc_కొలత_x = scratch_కొలత;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    ఒకవేళ (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_op_topk_radix_నెట్టు_constants pc { n_kv, వెడల్పు, nrows, n_tps, n_ఖండాలు, n_ప్రవాహం };
    std::array<uint32_t, 3> elements {
        pipeline->wg_denoms[0],
        std::min(nrows, ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1]),
        1,
    };
    vk_subbuffer scratch_buf { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ggml_vk_tensor_subbuffer(ctx, scores), ggml_vk_tensor_subbuffer(ctx, శిఖరం_k),
          ggml_vk_tensor_subbuffer(ctx, cell_blk), ggml_vk_tensor_subbuffer(ctx, ముసుగు),
          scratch_buf }, pc, elements);
    ctx->prealloc_x_need_sync = నిజం;
}

శూన్య ggml_vk_కూడిక(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants p = vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants_init(src0, dst, ggml_nelements(src0));
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_కూడిక, p);
}

శూన్య ggml_vk_కూడిక_అడ్డువరుసలు(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants p = vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants_init(src0, dst, src0->ne[0]);
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_కూడిక_అడ్డువరుసలు, p);
}

శూన్య ggml_vk_సగటు(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants p = vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants_init(src0, dst, src0->ne[0]);
    p.బరువు = 1.0f / (దశాంశం)src0->ne[0];
    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_సగటు, p);
}

శూన్య ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants pc = vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants_init(src0, dst, src0->ne[0]);
    // Use the single pass shader when the rows are small or there are enough rows to fill the GPU.
    // For fewer, larger rows, use the multipass shader to spread each row across SMs.
    ఒకవేళ (dst->ne[0] <= 4096 || ggml_nrows(dst) >= ctx->device->shader_core_లెక్క) {
        ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_CUMSUM, pc);
        తిరిగి;
    }

    // First pass computes partial sums within a block, and stores the last partial
    // to the temp buffer. Second pass sums the block partials from the temp buffer
    // and adds that to the result of the first pass.
    vk_pipeline pipeline1 = ctx->device->pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline2 = ctx->device->pipeline_cumsum_multipass2_f32;
    GGML_ASSERT(pipeline1 != శూన్యసూచిక && pipeline2 != శూన్యసూచిక);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline1, 1);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline2, 1);

    std::array<uint32_t, 3> elements;

    elements[0] = dst->ne[0];
    elements[1] = (uint32_t)ggml_nrows(dst);
    elements[2] = 1;

    size_t తాత్కాలిక_కొలత = పరిమాణం(దశాంశం) * elements[0] * ggml_nrows(dst);

    ఒకవేళ (ctx->prealloc_కొలత_విభజించు_k < తాత్కాలిక_కొలత) {
        ctx->prealloc_కొలత_విభజించు_k = తాత్కాలిక_కొలత;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer తాత్కాలిక_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_విభజించు_k, 0);

    ఒకవేళ (ctx->prealloc_విభజించు_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline1, {src_buf, dst_buf, తాత్కాలిక_buf}, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline2, {src_buf, dst_buf, తాత్కాలిక_buf}, pc, elements);

    ctx->prealloc_విభజించు_k_need_sync = నిజం;
}

స్థిర std::array<uint32_t, 3> ggml_vk_nrows_elements(uint32_t nr) {
    ఒకవేళ (nr > 262144) {
        తిరిగి { 512, 512, CEIL_DIV(nr, 262144) };
    }
    ఒకవేళ (nr > 512) {
        తిరిగి { 512, CEIL_DIV(nr, 512), 1 };
    }
    తిరిగి { nr, 1, 1 };
}

శూన్య ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    స్థిరరాశి uint32_t nclasses = (uint32_t)src0->ne[0];
    స్థిరరాశి uint32_t nrows    = (uint32_t)ggml_nrows(src0);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, శూన్యసూచిక, dst, GGML_OP_CROSS_ENTROPY_LOSS);
    GGML_ASSERT(pipeline != శూన్యసూచిక);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_కూడిక_అడ్డువరుసలు_f32, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer src1_buf = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst, నిజం);

    స్థిరరాశి vk_op_నెట్టు_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };

    స్థిరరాశి size_t tmp_కొలత = (size_t)nrows * పరిమాణం(దశాంశం);
    ఒకవేళ (ctx->prealloc_కొలత_x < tmp_కొలత) {
        ctx->prealloc_కొలత_x = tmp_కొలత;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    ఒకవేళ (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_subbuffer tmp_buf = { ctx->prealloc_x, 0, tmp_కొలత };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, tmp_buf }, pc, ggml_vk_nrows_elements(nrows));
    ggml_vk_sync_buffers(ctx, subctx);

    vk_op_కూడిక_అడ్డువరుసలు_నెట్టు_constants sp = {};
    sp.n_నిలువుల_సంఖ్య = nrows;
    sp.ne01 = 1;
    sp.ne02 = 1;
    sp.బరువు = 1.0f;
    init_pushconst_fastdiv(sp);
    sp.misalign_offsets = get_misalign_బైట్లు(ctx, dst) / ggml_type_కొలత(dst->type);

    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_కూడిక_అడ్డువరుసలు_f32, { tmp_buf, dst_buf }, sp, { 1, 1, 1 });
    ctx->prealloc_x_need_sync = నిజం;
}

శూన్య ggml_vk_cross_entropy_loss_వెనుక(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * grad   = dst->src[0];
    స్థిరరాశి ggml_tensor * logits = dst->src[1];
    స్థిరరాశి ggml_tensor * labels = dst->src[2];

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

    స్థిరరాశి uint32_t nclasses = (uint32_t)logits->ne[0];
    స్థిరరాశి uint32_t nrows    = (uint32_t)ggml_nrows(logits);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, grad, logits, labels, dst, GGML_OP_CROSS_ENTROPY_LOSS_వెనుక);
    GGML_ASSERT(pipeline != శూన్యసూచిక);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer grad_buf   = ggml_vk_tensor_subbuffer(ctx, grad);
    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer labels_buf = ggml_vk_tensor_subbuffer(ctx, labels);
    vk_subbuffer dst_buf    = ggml_vk_tensor_subbuffer(ctx, dst);

    స్థిరరాశి vk_op_నెట్టు_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { grad_buf, logits_buf, labels_buf, dst_buf }, pc, ggml_vk_nrows_elements(nrows));
}

శూన్య ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_నెట్టు_constants>(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_ARGMAX, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], 0.0f, 0.0f, 0.0f, 0.0f });
}

శూన్య ggml_vk_లెక్క_equal(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_లెక్క_EQUAL, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

శూన్య ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి uint32_t src0_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t src1_type_కొలత = ggml_type_కొలత(src1->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(dst->type);

    ggml_vk_op_f32<vk_op_ద్విమాన_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_SOLVE_TRI, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_కొలత, (uint32_t)src0->nb[1] / src0_type_కొలత, (uint32_t)src0->nb[2] / src0_type_కొలత, (uint32_t)src0->nb[3] / src0_type_కొలత,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_కొలత, (uint32_t)src1->nb[1] / src1_type_కొలత, (uint32_t)src1->nb[2] / src1_type_కొలత, (uint32_t)src1->nb[3] / src1_type_కొలత,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_కొలత, (uint32_t) dst->nb[1] /  dst_type_కొలత, (uint32_t) dst->nb[2] /  dst_type_కొలత, (uint32_t) dst->nb[3] /  dst_type_కొలత,
        0,
        0.0f, 0.0f, 0,
    });
}

శూన్య ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి int32_t s0 = dst->op_params[0];
    స్థిరరాశి int32_t s1 = dst->op_params[1];
    స్థిరరాశి int32_t p0 = dst->op_params[2];
    స్థిరరాశి int32_t p1 = dst->op_params[3];
    స్థిరరాశి int32_t d0 = dst->op_params[4];
    స్థిరరాశి int32_t d1 = dst->op_params[5];

    స్థిరరాశి తార్కిక is_2D = dst->op_params[6] == 1;

    స్థిరరాశి uint32_t IC = src1->ne[is_2D ? 2 : 1];
    స్థిరరాశి uint32_t IH = is_2D ? src1->ne[1] : 1;
    స్థిరరాశి uint32_t IW =         src1->ne[0];

    స్థిరరాశి uint32_t KH = is_2D ? src0->ne[1] : 1;
    స్థిరరాశి uint32_t KW =         src0->ne[0];

    స్థిరరాశి uint32_t OH = is_2D ? dst->ne[2] : 1;
    స్థిరరాశి uint32_t OW =         dst->ne[1];

    స్థిరరాశి uint32_t స్థానభ్రంశం_delta = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    స్థిరరాశి uint32_t batch_స్థానభ్రంశం = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    స్థిరరాశి uint32_t batch = src1->ne[is_2D ? 3 : 2];

    స్థిరరాశి ggml_backend_vk_బఫరు_context * d_buf_ctx = (ggml_backend_vk_బఫరు_context *)dst->బఫరు->context;
    స్థిరరాశి vk_బఫరు d_buf = d_buf_ctx->dev_బఫరు;

    స్థిరరాశి vk::Device_చిరునామా dst_addr = d_buf->bda_addr + vk_tensor_స్థానభ్రంశం(dst) + dst->view_offs;

    ggml_vk_op_f32<vk_op_im2col_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_IM2_నిలువు, {
        dst_addr,
        batch_స్థానభ్రంశం, స్థానభ్రంశం_delta,
        IC, IW, IH, OW, OH, KW, KH,
        OH * batch,
        IC * KH * KW,
        s0, s1, p0, p1, d0, d1, batch * IC
    });
}

శూన్య ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి int32_t s0 = ((స్థిరరాశి int32_t *)(dst->op_params))[0];
    స్థిరరాశి int32_t s1 = ((స్థిరరాశి int32_t *)(dst->op_params))[1];
    స్థిరరాశి int32_t s2 = ((స్థిరరాశి int32_t *)(dst->op_params))[2];
    స్థిరరాశి int32_t p0 = ((స్థిరరాశి int32_t *)(dst->op_params))[3];
    స్థిరరాశి int32_t p1 = ((స్థిరరాశి int32_t *)(dst->op_params))[4];
    స్థిరరాశి int32_t p2 = ((స్థిరరాశి int32_t *)(dst->op_params))[5];
    స్థిరరాశి int32_t d0 = ((స్థిరరాశి int32_t *)(dst->op_params))[6];
    స్థిరరాశి int32_t d1 = ((స్థిరరాశి int32_t *)(dst->op_params))[7];
    స్థిరరాశి int32_t d2 = ((స్థిరరాశి int32_t *)(dst->op_params))[8];
    స్థిరరాశి int32_t IC = ((స్థిరరాశి int32_t *)(dst->op_params))[9];

    స్థిరరాశి int64_t N  = ne13 / IC;
    స్థిరరాశి int64_t ID = ne12;
    స్థిరరాశి int64_t IH = ne11;
    స్థిరరాశి int64_t IW = ne10;

    స్థిరరాశి int64_t KD = ne02;
    స్థిరరాశి int64_t KH = ne01;
    స్థిరరాశి int64_t KW = ne00;

    స్థిరరాశి int64_t OD = ne3 / N;
    స్థిరరాశి int64_t OH = ne2;
    స్థిరరాశి int64_t OW = ne1;

    స్థిరరాశి ggml_backend_vk_బఫరు_context * d_buf_ctx = (ggml_backend_vk_బఫరు_context *)dst->బఫరు->context;
    స్థిరరాశి vk_బఫరు d_buf = d_buf_ctx->dev_బఫరు;

    స్థిరరాశి vk::Device_చిరునామా dst_addr = d_buf->bda_addr + vk_tensor_స్థానభ్రంశం(dst) + dst->view_offs;

    vk_op_im2col_3d_నెట్టు_constants pc {};

    pc.dst_addr = dst_addr;
    pc.nb10 = nb10 / ggml_type_కొలత(src1->type);
    pc.nb11 = nb11 / ggml_type_కొలత(src1->type);
    pc.nb12 = nb12 / ggml_type_కొలత(src1->type);
    pc.nb13 = nb13 / ggml_type_కొలత(src1->type);
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

    ggml_vk_op_f32<vk_op_im2col_3d_నెట్టు_constants>(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_IM2_నిలువు_3_D, std::move(pc));
}

శూన్య ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    స్థిరరాశి uint32_t dim = dst->op_params[0];
    స్థిరరాశి uint32_t max_period = dst->op_params[1];
    స్థిరరాశి uint32_t nb1 = dst->nb[1] / ggml_type_కొలత(dst->type);

    ggml_vk_op_f32<vk_op_timestep_embedding_నెట్టు_constants>(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_TIMESTEP_EMBEDDING, {
        nb1, dim, max_period,
    });
}

శూన్య ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    // src0: (K, Cout, Cin, 1) -- kernel
    // src1: (L, Cin, 1, 1) -- input
    // dst: (*, Cout, 1, 1)

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    GGML_ASSERT(nb00 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));

    స్థిరరాశి int32_t s0 = dst->op_params[0];

    vk_op_conv_transpose_1d_నెట్టు_constants p{};
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

    ggml_vk_op_f32(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_CONV_TRANSPOSE_1D, std::move(p));
}

శూన్య ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    // src0: [K_OC, T_in] columns from matmul
    // dst:  [T_out, OC]

    స్థిరరాశి int32_t stride = dst->op_params[0];
    స్థిరరాశి int32_t oc     = dst->op_params[1];
    స్థిరరాశి int32_t p0     = dst->op_params[2];

    స్థిరరాశి uint32_t K_OC  = static_cast<uint32_t>(src0->ne[0]);
    స్థిరరాశి uint32_t T_లోపల  = static_cast<uint32_t>(src0->ne[1]);
    స్థిరరాశి uint32_t T_బహిర్గతం = static_cast<uint32_t>(dst->ne[0]);
    స్థిరరాశి uint32_t OC    = static_cast<uint32_t>(oc);
    స్థిరరాశి uint32_t K     = K_OC / OC;

    vk_op_col2im_1d_నెట్టు_constants p{};
    p.T_బహిర్గతం  = T_బహిర్గతం;
    p.OC     = OC;
    p.K_OC   = K_OC;
    p.T_లోపల   = T_లోపల;
    p.K      = K;
    p.stride = stride;
    p.p0     = p0;

    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_COL2IM_1D, std::move(p));
}

శూన్య ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    స్థిరరాశి ggml_tensor * mul0 = cgraph->nodes[కణుపు_idx + 0];
    స్థిరరాశి ggml_tensor * sqr  = cgraph->nodes[కణుపు_idx + 2];
    స్థిరరాశి ggml_tensor * mul1 = cgraph->nodes[కణుపు_idx + 3];
    ggml_tensor *       కూడు_2  = cgraph->nodes[కణుపు_idx + 4];

    // x carries the full activation shape, a is the broadcast operand
    స్థిరరాశి ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    స్థిరరాశి ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    స్థిరరాశి ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    vk_pipeline pipeline = శూన్యసూచిక;
    ఎంపిక (x->type) {
        స్థితి GGML_TYPE_F32:  pipeline = ctx->device->pipeline_snake_f32;  ఆపు;
        స్థితి GGML_TYPE_F16:  pipeline = ctx->device->pipeline_snake_f16;  ఆపు;
        స్థితి GGML_TYPE_BF16: pipeline = ctx->device->pipeline_snake_bf16; ఆపు;
        అప్రమేయ:             GGML_ABORT("unsupported type");
    }
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf     = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer a_buf     = ggml_vk_tensor_subbuffer(ctx, a);
    vk_subbuffer inv_b_buf = ggml_vk_tensor_subbuffer(ctx, inv_b);
    vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, కూడు_2);

    vk_op_snake_నెట్టు_constants pc{};
    pc.ne0 = static_cast<uint32_t>(x->ne[0]);
    pc.ne1 = static_cast<uint32_t>(x->ne[1]);

    std::array<uint32_t, 3> elements = { pc.ne0, pc.ne1, 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, a_buf, inv_b_buf, dst_buf }, pc, elements);
}

శూన్య ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    స్థిరరాశి int32_t k0 = dst->op_params[1];
    స్థిరరాశి int32_t s0 = dst->op_params[2];
    స్థిరరాశి int32_t p0 = dst->op_params[3];

    స్థిరరాశి uint32_t IL = src0->ne[0];

    స్థిరరాశి uint32_t N = dst->ne[3] * dst->ne[2];

    స్థిరరాశి uint32_t OC = dst->ne[1];
    స్థిరరాశి uint32_t OL = dst->ne[0];

    స్థిరరాశి uint32_t parallel_elements = N * OC * OL;

    ggml_vk_op_f32<vk_op_pool1d_నెట్టు_constants>(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_POOL_1D, {
        IL, OL, OC,
        parallel_elements,
        op,
        k0, s0, p0,
    });
}

శూన్య ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    స్థిరరాశి int32_t k1 = dst->op_params[1];
    స్థిరరాశి int32_t k0 = dst->op_params[2];
    స్థిరరాశి int32_t s1 = dst->op_params[3];
    స్థిరరాశి int32_t s0 = dst->op_params[4];
    స్థిరరాశి int32_t p1 = dst->op_params[5];
    స్థిరరాశి int32_t p0 = dst->op_params[6];

    స్థిరరాశి uint32_t IH = src0->ne[1];
    స్థిరరాశి uint32_t IW = src0->ne[0];

    స్థిరరాశి uint32_t N = dst->ne[3];

    స్థిరరాశి uint32_t OC = dst->ne[2];
    స్థిరరాశి uint32_t OH = dst->ne[1];
    స్థిరరాశి uint32_t OW = dst->ne[0];

    స్థిరరాశి uint32_t parallel_elements = N * OC * OH * OW;

    ggml_vk_op_f32<vk_op_pool2d_నెట్టు_constants>(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_POOL_2D, {
        IW, IH, OW, OH, OC,
        parallel_elements,
        op,
        k0, k1, s0, s1, p0, p1,
    });
}

శూన్య ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, స్థిరరాశి ggml_tensor * src0,
                            స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ద్విమాన_OP_LOCALS
    GGML_ASSERT(nb00 == పరిమాణం(దశాంశం) || nb00 == పరిమాణం(ggml_fp16_t));
    GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));

    తార్కిక transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;

    vk_op_conv2d_నెట్టు_constants p{};
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

    ggml_vk_op_f32(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, dst->op, std::move(p));
}

శూన్య ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, స్థిరరాశి ggml_tensor * src0,
                            స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ద్విమాన_OP_LOCALS
    GGML_ASSERT(nb00 == పరిమాణం(దశాంశం) || nb00 == పరిమాణం(ggml_fp16_t));
    GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));

    vk_op_conv3d_నెట్టు_constants p{};
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

    ggml_vk_op_f32(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_CONV_3D, std::move(p));
}

శూన్య ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    vk_op_conv2d_dw_నెట్టు_constants p{};
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

    ggml_vk_op_f32(ctx, subctx, src0, src1, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_CONV_2D_DW, std::move(p));
}

శూన్య ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst) {
    స్థిరరాశి దశాంశం * op_params = (స్థిరరాశి దశాంశం *)dst->op_params;
    vk_op_unary_నెట్టు_constants p = vk_op_unary_నెట్టు_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, dst, GGML_OP_LEAKY_RELU, std::move(p));
}

శూన్య ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {
#ఒకవేళ defined(GGML_VULKAN_నిరంతరం_TESTS)
    స్థిరరాశి std::vector<size_t> vals {
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
    స్థిరరాశి size_t అంకె_it = 100;

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, అంకె_it, 1, 0, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, అంకె_it, 1, 1, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, అంకె_it, 1, 2, GGML_TYPE_Q4_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, అంకె_it, 1, 0, GGML_TYPE_Q4_0, నిజం);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, అంకె_it, 1, 1, GGML_TYPE_Q4_0, నిజం);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, అంకె_it, 1, 2, GGML_TYPE_Q4_0, నిజం);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, అంకె_it, 1, 0, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, అంకె_it, 1, 1, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, అంకె_it, 1, 2, GGML_TYPE_Q8_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, అంకె_it, 1, 0, GGML_TYPE_Q8_0, నిజం);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, అంకె_it, 1, 1, GGML_TYPE_Q8_0, నిజం);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, అంకె_it, 1, 2, GGML_TYPE_Q8_0, నిజం);

    abort();

    కొరకు (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul<ggml_fp16_t, దశాంశం>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 1, 0);
        ggml_vk_test_matmul<ggml_fp16_t, దశాంశం>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 1, 1);
        ggml_vk_test_matmul<ggml_fp16_t, దశాంశం>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 1, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, దశాంశం>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 2, 0);
        ggml_vk_test_matmul<ggml_fp16_t, దశాంశం>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 2, 1);
        ggml_vk_test_matmul<ggml_fp16_t, దశాంశం>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 2, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, దశాంశం>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 4, 0);
        ggml_vk_test_matmul<ggml_fp16_t, దశాంశం>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 4, 1);
        ggml_vk_test_matmul<ggml_fp16_t, దశాంశం>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 4, 2);
        std::cerr << '\n' << std::పంక్తి_ముగింపు;

        ఒకవేళ (vals[i + 2] % 32 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 1, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 1, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 1, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 2, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 2, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 2, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 4, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 4, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 4, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n' << std::పంక్తి_ముగింపు;
        }

        ఒకవేళ (vals[i + 2] % 256 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 1, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 1, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 1, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 2, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 2, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 2, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 4, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 4, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, అంకె_it, 4, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n' << std::పంక్తి_ముగింపు;
        }
    }

    GGML_ABORT("fatal error");
#అయితే_ముగింపు

    ఒకవేళ (subctx) {
        // Submit and wait for any pending work before reallocating the buffers
        ggml_vk_ctx_చివర(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = నిజం;
        ggml_vk_synchronize(ctx);
        GGML_ASSERT(ctx->compute_ctx.expired());
        ggml_vk_ctx_begin(ctx->device, subctx);
        ctx->compute_ctx = subctx;
    }

    ఒకవేళ (ctx->prealloc_x == శూన్యసూచిక || (ctx->prealloc_కొలత_x > 0 && ctx->prealloc_x->size < ctx->prealloc_కొలత_x)) {
        VK_LOG_జ్ఞాపకం("ggml_vk_preallocate_buffers(x_size: " << ctx->prealloc_కొలత_x << ")");
        // Resize buffer
        ఒకవేళ (ctx->prealloc_x != శూన్యసూచిక) {
            ggml_vk_destroy_బఫరు(ctx->prealloc_x);
        }
        ctx->prealloc_x = ggml_vk_create_బఫరు_device(ctx->device, ctx->prealloc_కొలత_x);
    }
    ఒకవేళ (ctx->prealloc_y == శూన్యసూచిక || (ctx->prealloc_కొలత_y > 0 && ctx->prealloc_y->size < ctx->prealloc_కొలత_y)) {
        VK_LOG_జ్ఞాపకం("ggml_vk_preallocate_buffers(y_size: " << ctx->prealloc_కొలత_y << ")");
        // Resize buffer
        ఒకవేళ (ctx->prealloc_y != శూన్యసూచిక) {
            ggml_vk_destroy_బఫరు(ctx->prealloc_y);
        }
        ctx->prealloc_y = ggml_vk_create_బఫరు_device(ctx->device, ctx->prealloc_కొలత_y);
        ctx->prealloc_y_చివరి_pipeline_వాడిన = శూన్యసూచిక;
        ctx->prealloc_y_చివరి_tensor_వాడిన = శూన్యసూచిక;
        ctx->prealloc_y_చివరి_k_padded = అబద్ధం;
    }
    ఒకవేళ (ctx->prealloc_విభజించు_k == శూన్యసూచిక || (ctx->prealloc_కొలత_విభజించు_k > 0 && ctx->prealloc_విభజించు_k->size < ctx->prealloc_కొలత_విభజించు_k)) {
        VK_LOG_జ్ఞాపకం("ggml_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_కొలత_విభజించు_k << ")");
        // Resize buffer
        ఒకవేళ (ctx->prealloc_విభజించు_k != శూన్యసూచిక) {
            ggml_vk_destroy_బఫరు(ctx->prealloc_విభజించు_k);
        }
        ctx->prealloc_విభజించు_k = ggml_vk_create_బఫరు_device(ctx->device, ctx->prealloc_కొలత_విభజించు_k);
    }
    ఒకవేళ (ctx->prealloc_కూడు_rms_partials == శూన్యసూచిక || (ctx->prealloc_కొలత_కూడు_rms_partials > 0 && ctx->prealloc_కూడు_rms_partials->size < ctx->prealloc_కొలత_కూడు_rms_partials)) {
        VK_LOG_జ్ఞాపకం("ggml_vk_preallocate_buffers(add_partials_size: " << ctx->prealloc_కూడు_rms_partials << ")");
        // Resize buffer
        ఒకవేళ (ctx->prealloc_కూడు_rms_partials != శూన్యసూచిక) {
            ggml_vk_destroy_బఫరు(ctx->prealloc_కూడు_rms_partials);
        }
        ctx->prealloc_కూడు_rms_partials = ggml_vk_create_బఫరు_device(ctx->device, ctx->prealloc_కొలత_కూడు_rms_partials);
    }
}

తార్కిక ggml_vk_build_లేఖాచిత్రం(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, ggml_tensor *కణుపు_begin, పూర్ణాంకం కణుపు_idx_begin, తార్కిక చివరి_కణుపు, తార్కిక almost_ready, తార్కిక submit){
    ggml_tensor * కణుపు = cgraph->nodes[కణుపు_idx];
    ఒకవేళ (ggml_is_ఖాళీలు(కణుపు) || ggml_op_is_ఖాళీలు(కణుపు->op) || !కణుపు->బఫరు) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ ((కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        తిరిగి అబద్ధం;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << కణుపు << ", " << ggml_op_name(కణుపు->op) << ")");
    ctx->semaphore_idx = 0;

    ggml_tensor * src0 = కణుపు->src[0];
    ggml_tensor * src1 = కణుపు->src[1];
    ggml_tensor * src2 = కణుపు->src[2];
    ggml_tensor * src3 = కణుపు->src[3];

    ఒకవేళ (కణుపు->op == GGML_OP_కూడు) {
        పూర్ణాంకం తదుపరి_కణుపు_idx = కణుపు_idx + 1 + ctx->అంకె_additional_fused_ops;
        ఒకవేళ (తదుపరి_కణుపు_idx < cgraph->n_nodes &&
            cgraph->nodes[తదుపరి_కణుపు_idx]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[తదుపరి_కణుపు_idx]->src[0] == cgraph->nodes[తదుపరి_కణుపు_idx - 1] &&
            ggml_nrows(cgraph->nodes[తదుపరి_కణుపు_idx]) == 1 &&
            ctx->device->కూడు_rms_fusion) {
            uint32_t size = ggml_vk_rms_partials_కొలత(ctx, cgraph->nodes[కణుపు_idx]);
            ctx->do_కూడు_rms_partials_స్థానభ్రంశం_calculation = నిజం;
            ఒకవేళ (ctx->prealloc_కొలత_కూడు_rms_partials_స్థానభ్రంశం + size <= ctx->prealloc_కొలత_కూడు_rms_partials) {
                ctx->do_కూడు_rms_partials = నిజం;
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
        తార్కిక need_sync = అబద్ధం;

        // Check whether "node" requires synchronization. The node requires synchronization if it
        // overlaps in memory with another unsynchronized node and at least one of them is a write.
        // Destination nodes are checked against both the written/read lists. Source nodes are only
        // checked against the written list. Two nodes overlap in memory if they come from the same
        // buffer and the tensor or view ranges overlap.
        స్వయంచాలక స్థిరరాశి &overlaps_unsynced = [&](స్థిరరాశి ggml_tensor *కణుపు, స్థిరరాశి std::vector<స్థిరరాశి ggml_tensor *> &unsynced_nodes) -> తార్కిక {
            ఒకవేళ (unsynced_nodes.size() == 0) {
                తిరిగి అబద్ధం;
            }
            స్వయంచాలక n_పునాది = vk_tensor_స్థానభ్రంశం(కణుపు) + కణుపు->view_offs;
            స్వయంచాలక n_కొలత = ggml_nbytes(కణుపు);
            ggml_backend_vk_బఫరు_context * a_buf_ctx = (ggml_backend_vk_బఫరు_context *)కణుపు->బఫరు->context;
            vk_బఫరు a_buf = a_buf_ctx->dev_బఫరు;
            కొరకు (స్వయంచాలక &other : unsynced_nodes) {
                ggml_backend_vk_బఫరు_context * o_buf_ctx = (ggml_backend_vk_బఫరు_context *)other->బఫరు->context;
                vk_బఫరు o_buf = o_buf_ctx->dev_బఫరు;
                ఒకవేళ (a_buf == o_buf) {
                    స్వయంచాలక o_పునాది = vk_tensor_స్థానభ్రంశం(other) + other->view_offs;
                    స్వయంచాలక o_కొలత = ggml_nbytes(other);

                    ఒకవేళ ((o_పునాది <= n_పునాది && n_పునాది < o_పునాది + o_కొలత) ||
                        (n_పునాది <= o_పునాది && o_పునాది < n_పునాది + n_కొలత)) {
                        తిరిగి నిజం;
                    }
                }
            }
            తిరిగి అబద్ధం;
        };

        // For all fused ops, check if the destination node or any of the source
        // nodes require synchronization.
        కొరకు (int32_t i = 0; i < ctx->అంకె_additional_fused_ops + 1 && !need_sync; ++i) {
            స్థిరరాశి ggml_tensor *cur_కణుపు = cgraph->nodes[కణుపు_idx + i];
            // If the node actually writes to memory, then check if it needs to sync
            ఒకవేళ (ctx->fused_ops_write_ముసుగు & (1 << i)) {
                ఒకవేళ (overlaps_unsynced(cur_కణుపు, ctx->unsynced_nodes_read) || overlaps_unsynced(cur_కణుపు, ctx->unsynced_nodes_written)) {
                    need_sync = నిజం;
                    ఆపు;
                }
            }
            కొరకు (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                ఒకవేళ (!cur_కణుపు->src[j]) {
                    కొనసాగు;
                }
                ఒకవేళ (overlaps_unsynced(cur_కణుపు->src[j], ctx->unsynced_nodes_written)) {
                    need_sync = నిజం;
                    ఆపు;
                }
            }
        }

        ఒకవేళ (need_sync) {
            ఒకవేళ (vk_enable_sync_logger) {
                std::cerr <<  "sync" << std::పంక్తి_ముగింపు;
            }
            ctx->unsynced_nodes_written.clear();
            ctx->unsynced_nodes_read.clear();
            ggml_vk_sync_buffers(ctx, compute_ctx);

            ఒకవేళ (vk_perf_logger_enabled && vk_perf_logger_concurrent) {
                ctx->query_కణుపు_idx[ctx->query_idx] = కణుపు_idx;
                compute_ctx->s->బఫరు->buf.writeTimestamp(vk::Pipeline_Stage_Flag_బిట్లు::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            }
        }
        // Add all fused nodes to the unsynchronized lists.
        కొరకు (int32_t i = 0; i < ctx->అంకె_additional_fused_ops + 1; ++i) {
            స్థిరరాశి ggml_tensor *cur_కణుపు = cgraph->nodes[కణుపు_idx + i];
            // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
            ఒకవేళ (ctx->fused_ops_write_ముసుగు & (1 << i)) {
                ctx->unsynced_nodes_written.push_back(cur_కణుపు);
            }
            కొరకు (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                ఒకవేళ (!cur_కణుపు->src[j]) {
                    కొనసాగు;
                }
                ctx->unsynced_nodes_read.push_back(cur_కణుపు->src[j]);
            }
        }
    }
    ఒకవేళ (vk_enable_sync_logger) {
        కొరకు (పూర్ణాంకం i = 0; i < ctx->అంకె_additional_fused_ops + 1; ++i) {
            స్వయంచాలక *n = cgraph->nodes[కణుపు_idx + i];
            std::cerr << కణుపు_idx + i << " " << ggml_op_name(n->op) << " " <<  n->name;
            ఒకవేళ (n->op == GGML_OP_GLU) {
                std::cerr << " " << ggml_glu_op_name(ggml_get_glu_op(n)) << " " << (n->src[1] ? "split" : "single") << " ";
            }
            ఒకవేళ (n->op == GGML_OP_ROPE) {
                స్థిరరాశి పూర్ణాంకం mode = ((స్థిరరాశి int32_t *) n->op_params)[2];
                std::cerr << " rope mode: " << mode;
            }
            std::cerr << std::పంక్తి_ముగింపు;
        }
    }

    // closed explicitly below, and by the destructor on the early returns
    ggml_vk_debug_label dbg(compute_ctx, cgraph, కణుపు_idx, ctx->అంకె_additional_fused_ops);

    ఎంపిక (కణుపు->op) {
    స్థితి GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_REPEAT_వెనుక:
        ggml_vk_repeat_వెనుక(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_ACC:
    స్థితి GGML_OP_సమితి:
        ggml_vk_acc(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_GET_అడ్డువరుసలు:
        ఒకవేళ (ctx->fused_topk_qsa) {
            ggml_vk_topk_qsa(ctx, compute_ctx, cgraph, కణుపు_idx);
        } లేకపోతే {
            ggml_vk_get_అడ్డువరుసలు(ctx, compute_ctx, src0, src1, కణుపు);
        }

        ఆపు;
    స్థితి GGML_OP_GET_అడ్డువరుసలు_వెనుక:
        ggml_vk_get_అడ్డువరుసలు_వెనుక(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_కూడు:
        ఒకవేళ (ctx->అంకె_additional_fused_ops) {
            ggml_vk_multi_కూడు(ctx, compute_ctx, cgraph, కణుపు_idx);
        } లేకపోతే {
            ggml_vk_కూడు(ctx, compute_ctx, src0, src1, కణుపు);
        }
        ఆపు;
    స్థితి GGML_OP_బహిర్గతం_PROD:
        ggml_vk_బహిర్గతం_prod(ctx, compute_ctx, src0, src1, కణుపు);
        ఆపు;
    స్థితి GGML_OP_SUB:
        ggml_vk_sub(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_MUL:
        ఒకవేళ (ctx->అంకె_additional_fused_ops) {
            ggml_vk_snake_dispatch_fused(ctx, compute_ctx, cgraph, కణుపు_idx);
        } లేకపోతే {
            ggml_vk_mul(ctx, compute_ctx, src0, src1, కణుపు);
        }

        ఆపు;
    స్థితి GGML_OP_DIV:
        ggml_vk_div(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_కూడు_ID:
        ggml_vk_కూడు_id(ctx, compute_ctx, src0, src1, src2, కణుపు);

        ఆపు;
    స్థితి GGML_OP_CONCAT:
        ggml_vk_concat(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_UPSCALE:
        ggml_vk_upscale(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_ADD1:
        ggml_vk_add1(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_ARANGE:
        ggml_vk_arange(ctx, compute_ctx, కణుపు);

        ఆపు;
    స్థితి GGML_OP_FILL:
        ggml_vk_fill(ctx, compute_ctx, కణుపు);

        ఆపు;
    స్థితి GGML_OP_SCALE:
        ఒకవేళ (ctx->fused_hc_post_gate) {
            ggml_tensor * hc_post = cgraph->nodes[కణుపు_idx + ctx->అంకె_additional_fused_ops];
            ggml_vk_dsv4_hc_post(ctx, compute_ctx, hc_post->src[0], hc_post->src[1], hc_post->src[2], hc_post->src[3], hc_post, కణుపు);
        } లేకపోతే {
            ggml_vk_scale(ctx, compute_ctx, src0, కణుపు);
        }

        ఆపు;
    స్థితి GGML_OP_SQR:
        ggml_vk_sqr(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_SQRT:
        ggml_vk_sqrt(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_SIN:
        ggml_vk_sin(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_COS:
        ggml_vk_cos(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_LOG:
        ggml_vk_log(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_TRI:
        ggml_vk_tri(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_DIAG:
        ggml_vk_diag(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_CLAMP:
        ggml_vk_clamp(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_PAD:
        ggml_vk_pad(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_PAD_REFLECT_1D:
        ggml_vk_pad_reflect_1d(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_ROLL:
        ggml_vk_roll(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_CPY:
    స్థితి GGML_OP_CONT:
    స్థితి GGML_OP_DUP:
        ggml_vk_cpy(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
        ggml_vk_సమితి_అడ్డువరుసలు(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_SILU_వెనుక:
        ggml_vk_silu_వెనుక(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_NORM:
        ggml_vk_norm(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_సమూహం_NORM:
        ggml_vk_సమూహం_norm(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_RMS_NORM:
        ggml_vk_rms_norm(ctx, compute_ctx, cgraph, కణుపు_idx, (దశాంశం *)కణుపు->op_params);
        ఆపు;
    స్థితి GGML_OP_RMS_NORM_వెనుక:
        ggml_vk_rms_norm_వెనుక(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_L2_NORM:
        ggml_vk_l2_norm(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_UNARY:
        ఒకవేళ (ctx->fused_topk_moe_బహుళకం != TOPK_MOE_లెక్క) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, కణుపు_idx);
            ఆపు;
        }
        ఒకవేళ (ctx->అంకె_additional_fused_ops) {
            ggml_vk_unary_mul(ctx, compute_ctx, cgraph, కణుపు_idx);
            ఆపు;
        }

        ఎంపిక (ggml_get_unary_op(కణుపు)) {
        స్థితి GGML_UNARY_OP_ELU:
        స్థితి GGML_UNARY_OP_EXP:
        స్థితి GGML_UNARY_OP_EXPM1:
        స్థితి GGML_UNARY_OP_SILU:
        స్థితి GGML_UNARY_OP_GELU:
        స్థితి GGML_UNARY_OP_GELU_ERF:
        స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:
        స్థితి GGML_UNARY_OP_RELU:
        స్థితి GGML_UNARY_OP_NEG:
        స్థితి GGML_UNARY_OP_TANH:
        స్థితి GGML_UNARY_OP_SIGMOID:
        స్థితి GGML_UNARY_OP_HARDSIGMOID:
        స్థితి GGML_UNARY_OP_HARDSWISH:
        స్థితి GGML_UNARY_OP_ABS:
        స్థితి GGML_UNARY_OP_SOFTPLUS:
        స్థితి GGML_UNARY_OP_అడుగు:
        స్థితి GGML_UNARY_OP_విడత:
        స్థితి GGML_UNARY_OP_CEIL:
        స్థితి GGML_UNARY_OP_FLOOR:
        స్థితి GGML_UNARY_OP_TRUNC:
        స్థితి GGML_UNARY_OP_SGN:
            ggml_vk_unary(ctx, compute_ctx, src0, కణుపు);
            ఆపు;
        స్థితి GGML_UNARY_OP_XIELU:
            ggml_vk_xielu(ctx, compute_ctx, src0, కణుపు);
            ఆపు;
        అప్రమేయ:
            తిరిగి అబద్ధం;
        }
        ఆపు;
    స్థితి GGML_OP_GLU:
        ఎంపిక (ggml_get_glu_op(కణుపు)) {
        స్థితి GGML_GLU_OP_GEGLU:
        స్థితి GGML_GLU_OP_REGLU:
        స్థితి GGML_GLU_OP_SWIGLU:
        స్థితి GGML_GLU_OP_SWIGLU_OAI:
        స్థితి GGML_GLU_OP_GEGLU_ERF:
        స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
        స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
            ggml_vk_glu(ctx, compute_ctx, src0, src1, కణుపు);
            ఆపు;
        అప్రమేయ:
            తిరిగి అబద్ధం;
        }
        ఆపు;
    స్థితి GGML_OP_DIAG_ముసుగు_INF:
        ggml_vk_diag_ముసుగు_inf(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_SOFT_MAX:
        ఒకవేళ (ctx->fused_topk_moe_బహుళకం != TOPK_MOE_లెక్క) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, కణుపు_idx);
        } లేకపోతే {
            ggml_vk_soft_max(ctx, compute_ctx, src0, src1, src2, కణుపు);
        }

        ఆపు;
    స్థితి GGML_OP_SOFT_MAX_వెనుక:
        ggml_vk_soft_max_వెనుక(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_ROPE:
        ggml_vk_rope(ctx, compute_ctx, cgraph, కణుపు_idx, అబద్ధం);

        ఆపు;
    స్థితి GGML_OP_ROPE_వెనుక:
        ggml_vk_rope(ctx, compute_ctx, cgraph, కణుపు_idx, నిజం);

        ఆపు;
    స్థితి GGML_OP_ARGSORT:
        ఒకవేళ (ctx->fused_topk_moe_బహుళకం != TOPK_MOE_లెక్క) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, కణుపు_idx);
        } లేకపోతే {
            ggml_vk_argsort(ctx, compute_ctx, src0, కణుపు);
        }

        ఆపు;
    స్థితి GGML_OP_శిఖరం_K:
        ggml_vk_topk(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_కూడిక:
        ggml_vk_కూడిక(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
        ggml_vk_కూడిక_అడ్డువరుసలు(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_CUMSUM:
        ggml_vk_cumsum(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_DSV4_HC_COMB:
        ggml_vk_dsv4_hc_comb(ctx, compute_ctx, src0, src1, src2, కణుపు);

        ఆపు;
    స్థితి GGML_OP_DSV4_HC_PRE:
        ggml_vk_dsv4_hc_pre(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_DSV4_HC_POST:
        ggml_vk_dsv4_hc_post(ctx, compute_ctx, src0, src1, src2, src3, కణుపు);

        ఆపు;
    స్థితి GGML_OP_సగటు:
        ggml_vk_సగటు(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_ARGMAX:
        ggml_vk_argmax(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_CROSS_ENTROPY_LOSS:
        ggml_vk_cross_entropy_loss(ctx, compute_ctx, కణుపు);

        ఆపు;
    స్థితి GGML_OP_CROSS_ENTROPY_LOSS_వెనుక:
        ggml_vk_cross_entropy_loss_వెనుక(ctx, compute_ctx, కణుపు);

        ఆపు;
    స్థితి GGML_OP_లెక్క_EQUAL:
        ggml_vk_లెక్క_equal(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_SOLVE_TRI:
        ggml_vk_solve_tri(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_IM2_నిలువు:
        ggml_vk_im2col(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_IM2_నిలువు_3_D:
        ggml_vk_im2col_3d(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_TIMESTEP_EMBEDDING:
        ggml_vk_timestep_embedding(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_COL2IM_1D:
        ggml_vk_col2im_1d(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_CONV_TRANSPOSE_1D:
        ggml_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_POOL_1D:
        ggml_vk_pool_1d(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_POOL_2D:
        ggml_vk_pool_2d(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_CONV_2D:
    స్థితి GGML_OP_CONV_TRANSPOSE_2D:
        ggml_vk_conv_2d(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_CONV_3D:
        ggml_vk_conv_3d(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_CONV_2D_DW:
        ggml_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, కణుపు);

        ఆపు;
    స్థితి GGML_OP_LEAKY_RELU:
        ggml_vk_leaky_relu(ctx, compute_ctx, src0, కణుపు);

        ఆపు;
    స్థితి GGML_OP_MUL_MAT:
        ggml_vk_mul_mat(ctx, compute_ctx, cgraph, కణుపు_idx);

        ఆపు;
    స్థితి GGML_OP_MUL_MAT_ID:
        ggml_vk_mul_mat_id(ctx, compute_ctx, cgraph, కణుపు_idx);

        ఆపు;

    స్థితి GGML_OP_FLASH_ATTN_EXT:
        ggml_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, కణుపు->src[4], కణుపు);

        ఆపు;

    స్థితి GGML_OP_RWKV_WKV6:
        ggml_vk_rwkv_wkv6(ctx, compute_ctx, కణుపు);

        ఆపు;

    స్థితి GGML_OP_RWKV_WKV7:
        ggml_vk_rwkv_wkv7(ctx, compute_ctx, కణుపు);

        ఆపు;

    స్థితి GGML_OP_GATED_రేఖీయ_ATTN:
        ggml_vk_gated_రేఖీయ_attn(ctx, compute_ctx, కణుపు);

        ఆపు;

    స్థితి GGML_OP_LIGHTNING_INDEXER:
        ggml_vk_lightning_indexer(ctx, compute_ctx, కణుపు);

        ఆపు;

    స్థితి GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, కణుపు);

        ఆపు;

    స్థితి GGML_OP_SSM_పరిశీలన:
        ggml_vk_ssm_పరిశీలన(ctx, compute_ctx, కణుపు);

        ఆపు;

    స్థితి GGML_OP_SSM_CONV:
        ggml_vk_ssm_conv(ctx, compute_ctx, cgraph, కణుపు_idx);

        ఆపు;

    స్థితి GGML_OP_OPT_అడుగు_ADAMW:
        ggml_vk_opt_అడుగు_adamw(ctx, compute_ctx, కణుపు);

        ఆపు;

    స్థితి GGML_OP_OPT_అడుగు_SGD:
        ggml_vk_opt_అడుగు_sgd(ctx, compute_ctx, src0, src1, src2, కణుపు);

        ఆపు;
    అప్రమేయ:
        తిరిగి అబద్ధం;
    }

    // the submit path below can end the command buffer, so close the region first
    dbg.close();

    ctx->tensor_ctxs[కణుపు_idx] = compute_ctx;

#ఒకవేళ defined(GGML_VULKAN_CHECK_RESULTS)
    // Force context reset on each node so that each tensor ends up in its own context
    // and can be run and compared to its CPU equivalent separately
    చివరి_కణుపు = నిజం;
#అయితే_ముగింపు

    ఒకవేళ (submit || చివరి_కణుపు) {
        ggml_vk_ctx_చివర(compute_ctx);

        // TODO probably it'd be better to pass a exit_node flag to ggml_vk_compute_forward
        ఒకవేళ (చివరి_కణుపు) {
            compute_ctx->exit_tensor_idx = కణుపు_idx_begin;
        }
        లేకపోతే {
            compute_ctx->exit_tensor_idx = -1;
        }

        ctx->compute_ctx.reset();

        ggml_vk_compute_forward(ctx, cgraph, కణుపు_begin, కణుపు_idx_begin, almost_ready);
    }
    తిరిగి నిజం;
}

శూన్య ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ggml_tensor * tensor, పూర్ణాంకం tensor_idx, తార్కిక almost_ready) {
    GGML_వాడని(cgraph);
    GGML_వాడని(tensor);

    VK_LOG_DEBUG("ggml_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << ggml_op_name(tensor->op) << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

    vk_context subctx = ctx->tensor_ctxs[tensor_idx].తాళం();

    // Only run if ctx hasn't been submitted yet
    ఒకవేళ (!subctx->seqs.ఖాళీలు()) {
#నిర్వచితమైతే GGML_VULKAN_CHECK_RESULTS
        ggml_vk_check_results_0(ctx, cgraph, tensor_idx);
#అయితే_ముగింపు

        // Do staging buffer copies
        కొరకు (స్వయంచాలక& cpy : subctx->లోపల_memcpys) {
            జ్ఞాపకం_నకలు(cpy.dst, cpy.src, cpy.n);
        }

        కొరకు (స్వయంచాలక& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ఒకవేళ (ctx->device->serialize_submissions) {
            ggml_vk_submit(subctx, ctx->fence);
        } లేకపోతే ఒకవేళ (almost_ready && !ctx->almost_ready_fence_pending) {
            ggml_vk_submit(subctx, ctx->almost_ready_fence);
            ctx->almost_ready_fence_pending = నిజం;
        } లేకపోతే {
            ggml_vk_submit(subctx, {});
        }
        ctx->submit_pending = నిజం;

#నిర్వచితమైతే GGML_VULKAN_CHECK_RESULTS
        ggml_vk_synchronize(ctx);
        ggml_vk_check_results_1(ctx, cgraph, tensor_idx);
#అయితే_ముగింపు
    }

    ఒకవేళ (tensor_idx == subctx->exit_tensor_idx) {
        // Do staging buffer copies
        కొరకు (స్వయంచాలక& cpy : subctx->బహిర్గతం_memcpys) {
            జ్ఞాపకం_నకలు(cpy.dst, cpy.src, cpy.n);
        }
        subctx->లోపల_memcpys.clear();
        subctx->బహిర్గతం_memcpys.clear();
        subctx->memsets.clear();
    }
}

శూన్య ggml_vk_లేఖాచిత్రం_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_graph_cleanup()");
    ctx->prealloc_y_చివరి_pipeline_వాడిన = {};
    ctx->prealloc_y_చివరి_tensor_వాడిన = శూన్యసూచిక;
    ctx->prealloc_y_చివరి_k_padded = అబద్ధం;

    ctx->unsynced_nodes_written.clear();
    ctx->unsynced_nodes_read.clear();
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_విభజించు_k_need_sync = అబద్ధం;

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    ఒకవేళ (ctx->device->async_use_transfer_వరుస) {
        ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);
    }

    కొరకు (size_t i = 0; i < ctx->gc.semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.semaphores[i].s });
    }
    ctx->gc.semaphores.clear();

    కొరకు (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.tl_semaphores[i].s });
    }
    ctx->gc.tl_semaphores.clear();
    ctx->semaphore_idx = 0;

    ctx->event_idx = 0;

    కొరకు (స్వయంచాలక& event : ctx->gc.events) {
        ctx->device->device.resetEvent(event);
    }

    ctx->tensor_ctxs.clear();
    ctx->gc.contexts.clear();
    ctx->pipeline_descriptor_సమితి_requirements = 0;
    ctx->descriptor_సమితి_idx = 0;
}

శూన్య ggml_vk_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_cleanup(" << ctx->name << ")");
    // discard any unsubmitted command buffers
    ctx->compute_ctx.reset();
    // wait for any pending command buffers to finish
    ggml_vk_synchronize(ctx);

    ggml_vk_లేఖాచిత్రం_cleanup(ctx);

    ggml_vk_destroy_బఫరు(ctx->prealloc_x);
    ggml_vk_destroy_బఫరు(ctx->prealloc_y);
    ggml_vk_destroy_బఫరు(ctx->prealloc_విభజించు_k);
    ggml_vk_destroy_బఫరు(ctx->prealloc_కూడు_rms_partials);
    ggml_vk_destroy_బఫరు(ctx->sync_staging);

    ctx->prealloc_y_చివరి_pipeline_వాడిన = శూన్యసూచిక;
    ctx->prealloc_y_చివరి_tensor_వాడిన = శూన్యసూచిక;
    ctx->prealloc_y_చివరి_k_padded = అబద్ధం;

    ctx->prealloc_కొలత_x = 0;
    ctx->prealloc_కొలత_y = 0;
    ctx->prealloc_కొలత_విభజించు_k = 0;

    కొరకు (స్వయంచాలక& event : ctx->gc.events) {
        ctx->device->device.destroyEvent(event);
    }
    ctx->gc.events.clear();

    ctx->device->device.destroyFence(ctx->fence);
    ctx->device->device.destroyFence(ctx->almost_ready_fence);

    కొరకు (స్వయంచాలక& pool : ctx->descriptor_pools) {
        ctx->device->device.destroyDescriptorPool(pool);
    }
    ctx->descriptor_pools.clear();
    ctx->descriptor_sets.clear();
    ctx->descriptor_సమితి_bindings.clear();

    ctx->compute_cmd_pool.destroy(ctx->device->device);
    ఒకవేళ (ctx->device->async_use_transfer_వరుస) {
        ctx->device->device.destroySemaphore(ctx->transfer_semaphore.s);

        ctx->transfer_cmd_pool.destroy(ctx->device->device);
    }
    ఒకవేళ (vk_perf_logger_enabled) {
        ctx->perf_logger->print_timings(నిజం);
    }
}

పూర్ణాంకం ggml_vk_get_device_లెక్క() {
    ggml_vk_instance_init();

    తిరిగి vk_instance.device_indices.size();
}

శూన్య ggml_vk_get_device_description(పూర్ణాంకం device, అక్షరం * description, size_t description_కొలత) {
    ggml_vk_instance_init();

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    vk::PhysicalDeviceProperties props;
    devices[device].getProperties(&props);

    snprintf(description, description_కొలత, "%s", props.deviceName.దత్తాంశం());
}

తార్కిక ggml_backend_బఫరు_is_vk(ggml_backend_బఫరు_t బఫరు) {
    తిరిగి బఫరు->buft->iface.get_name == ggml_backend_vk_బఫరు_type_name;
}

శూన్య ggml_backend_vk_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    VK_LOG_జ్ఞాపకం("ggml_backend_vk_buffer_free_buffer()");
    ggml_backend_vk_బఫరు_context * ctx = (ggml_backend_vk_బఫరు_context *)బఫరు->context;
    ggml_vk_destroy_బఫరు(ctx->dev_బఫరు);
    తొలగించు ctx;
}

శూన్య * ggml_backend_vk_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    తిరిగి vk_ptr_పునాది;

    వాడని(బఫరు);
}

గణన ggml_స్థితి ggml_backend_vk_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_init_tensor(" << బఫరు << " (" << బఫరు->context << "), " << tensor << ")");
    ఒకవేళ (tensor->view_src != శూన్యసూచిక) {
        GGML_ASSERT(tensor->view_src->బఫరు->buft == బఫరు->buft);
    }
    తిరిగి GGML_స్థితి_SUCCESS;
}

శూన్య ggml_backend_vk_బఫరు_memset_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, uint8_t value, size_t స్థానభ్రంశం, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_memset_tensor(" << బఫరు << ", " << tensor << ", " << value << ", " << స్థానభ్రంశం << ", " << size << ")");
    ggml_backend_vk_బఫరు_context * buf_ctx = (ggml_backend_vk_బఫరు_context *)బఫరు->context;
    vk_బఫరు buf = buf_ctx->dev_బఫరు;

    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    uint32_t val32 = (uint32_t)value * 0x01010101;
    ggml_vk_బఫరు_memset(buf, vk_tensor_స్థానభ్రంశం(tensor) + tensor->view_offs + స్థానభ్రంశం, val32, size);
}

శూన్య ggml_backend_vk_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor(" << బఫరు << ", " << tensor << ", " << దత్తాంశం << ", " << స్థానభ్రంశం << ", " << size << ")");
    ggml_backend_vk_బఫరు_context * buf_ctx = (ggml_backend_vk_బఫరు_context *)బఫరు->context;
    vk_బఫరు buf = buf_ctx->dev_బఫరు;

    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    ggml_vk_బఫరు_write(buf, vk_tensor_స్థానభ్రంశం(tensor) + tensor->view_offs + స్థానభ్రంశం, దత్తాంశం, size);
}

శూన్య ggml_backend_vk_బఫరు_సమితి_tensor_2d(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor_2d(" << బఫరు << ", " << tensor << ", " << దత్తాంశం << ", " << స్థానభ్రంశం << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_దత్తాంశం << ")");
    ggml_backend_vk_బఫరు_context * buf_ctx = (ggml_backend_vk_బఫరు_context *)బఫరు->context;
    vk_బఫరు buf = buf_ctx->dev_బఫరు;

    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    ggml_vk_బఫరు_write_2d(buf, vk_tensor_స్థానభ్రంశం(tensor) + tensor->view_offs + స్థానభ్రంశం, దత్తాంశం, stride_దత్తాంశం, stride_tensor, size, n_copies);
}

శూన్య ggml_backend_vk_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor(" << బఫరు << ", " << tensor << ", " << దత్తాంశం << ", " << స్థానభ్రంశం << ", " << size << ")");
    ggml_backend_vk_బఫరు_context * buf_ctx = (ggml_backend_vk_బఫరు_context *)బఫరు->context;

    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    vk_బఫరు buf = buf_ctx->dev_బఫరు;

    ggml_vk_బఫరు_read(buf, vk_tensor_స్థానభ్రంశం(tensor) + tensor->view_offs + స్థానభ్రంశం, దత్తాంశం, size);
}

శూన్య ggml_backend_vk_బఫరు_get_tensor_2d(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor_2d(" << బఫరు << ", " << tensor << ", " << దత్తాంశం << ", " << స్థానభ్రంశం << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_దత్తాంశం << ")");
    ggml_backend_vk_బఫరు_context * buf_ctx = (ggml_backend_vk_బఫరు_context *)బఫరు->context;

    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    vk_బఫరు buf = buf_ctx->dev_బఫరు;

    ggml_vk_బఫరు_read_2d(buf, vk_tensor_స్థానభ్రంశం(tensor) + tensor->view_offs + స్థానభ్రంశం, దత్తాంశం, stride_tensor, stride_దత్తాంశం, size, n_copies);
}

తార్కిక ggml_backend_vk_బఫరు_cpy_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst) {
    ఒకవేళ (ggml_nbytes(src) == 0) {
        తిరిగి నిజం;
    }

    ఒకవేళ (ggml_backend_బఫరు_is_vk(src->బఫరు)) {
        ggml_backend_vk_బఫరు_context * src_buf_ctx = (ggml_backend_vk_బఫరు_context *)src->బఫరు->context;
        ggml_backend_vk_బఫరు_context * dst_buf_ctx = (ggml_backend_vk_బఫరు_context *)dst->బఫరు->context;

        vk_బఫరు src_buf = src_buf_ctx->dev_బఫరు;
        vk_బఫరు dst_buf = dst_buf_ctx->dev_బఫరు;

        ggml_vk_బఫరు_copy(dst_buf, vk_tensor_స్థానభ్రంశం(dst) + dst->view_offs, src_buf, vk_tensor_స్థానభ్రంశం(src) + src->view_offs, ggml_nbytes(src));

        తిరిగి నిజం;
    }
    తిరిగి అబద్ధం;

    వాడని(బఫరు);
}

శూన్య ggml_backend_vk_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    ggml_backend_vk_బఫరు_context * ctx = (ggml_backend_vk_బఫరు_context *)బఫరు->context;

    ggml_vk_బఫరు_memset(ctx->dev_బఫరు, 0, value, బఫరు->size);
}

స్థిరరాశి అక్షరం * ggml_backend_vk_బఫరు_type_name(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_vk_బఫరు_type_context * ctx = (ggml_backend_vk_బఫరు_type_context *)buft->context;

    తిరిగి ctx->name.c_str();
}

ggml_backend_బఫరు_t ggml_backend_vk_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    VK_LOG_జ్ఞాపకం("ggml_backend_vk_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_vk_బఫరు_type_context * ctx = (ggml_backend_vk_బఫరు_type_context *) buft->context;

    vk_బఫరు dev_బఫరు = శూన్యసూచిక;
    ప్రయత్నించు {
        dev_బఫరు = ggml_vk_create_బఫరు_device(ctx->device, size);
    } పట్టుకో (స్థిరరాశి vk::SystemError& e) {
        తిరిగి శూన్యసూచిక;
    }

    ggml_backend_vk_బఫరు_context * bufctx = కొత్త ggml_backend_vk_బఫరు_context(ctx->device, std::move(dev_బఫరు), ctx->name);

    తిరిగి ggml_backend_బఫరు_init(buft, ggml_backend_vk_బఫరు_interface, bufctx, size);
}

size_t ggml_backend_vk_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_vk_బఫరు_type_context * ctx = (ggml_backend_vk_బఫరు_type_context *) buft->context;
    తిరిగి ctx->device->properties.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment;
}

size_t ggml_backend_vk_బఫరు_type_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_vk_బఫరు_type_context * ctx = (ggml_backend_vk_బఫరు_type_context *) buft->context;
    తిరిగి ctx->device->suballocation_ఖండం_కొలత;
}

size_t ggml_backend_vk_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి ggml_tensor * tensor) {
    తిరిగి ggml_nbytes(tensor);

    వాడని(buft);
}

ggml_backend_బఫరు_type_t ggml_backend_vk_బఫరు_type(size_t dev_అంకె) {
    ggml_vk_instance_init();

    VK_LOG_DEBUG("ggml_backend_vk_buffer_type(" << dev_అంకె << ")");

    vk_device dev = ggml_vk_get_device(dev_అంకె);

    తిరిగి &dev->బఫరు_type;
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_vk_host_బఫరు_type_name(ggml_backend_బఫరు_type_t buft) {
    తిరిగి GGML_VK_NAME "_Host";

    వాడని(buft);
}

స్థిర శూన్య ggml_backend_vk_host_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    VK_LOG_జ్ఞాపకం("ggml_backend_vk_host_buffer_free_buffer()");
    ggml_vk_host_free(బఫరు->context);
}

స్థిర ggml_backend_బఫరు_t ggml_backend_vk_host_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    VK_LOG_జ్ఞాపకం("ggml_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");

    size += 32;  // Behave like the CPU buffer type
    శూన్య * ptr = ggml_vk_host_malloc(size);
    ఒకవేళ (ptr == శూన్యసూచిక) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory\n");
        // fallback to cpu buffer
        తిరిగి ggml_backend_buft_alloc_బఫరు(ggml_backend_cpu_బఫరు_type(), size);
    }

    ggml_backend_బఫరు_t బఫరు = ggml_backend_cpu_బఫరు_from_ptr(ptr, size);
    బఫరు->buft = buft;
    బఫరు->iface.free_బఫరు = ggml_backend_vk_host_బఫరు_free_బఫరు;

    తిరిగి బఫరు;

    వాడని(buft);
}

స్థిర size_t ggml_backend_vk_host_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    తిరిగి vk_instance.devices[0]->properties.limits.min_జ్ఞాపకం_Map_Alignment;

    వాడని(buft);
}

స్థిర size_t ggml_backend_vk_host_బఫరు_type_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {
    తిరిగి vk_instance.devices[0]->suballocation_ఖండం_కొలత;

    వాడని(buft);
}

ggml_backend_బఫరు_type_t ggml_backend_vk_host_బఫరు_type() {
    స్థిర నిర్మాణం ggml_backend_బఫరు_type ggml_backend_vk_బఫరు_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_vk_host_బఫరు_type_name,
            /* .alloc_buffer        = */ ggml_backend_vk_host_బఫరు_type_alloc_బఫరు,
            /* .alloc_buffer_n      = */ శూన్యసూచిక,
            /* .get_alignment       = */ ggml_backend_vk_host_బఫరు_type_get_alignment,
            /* .get_max_size        = */ ggml_backend_vk_host_బఫరు_type_get_max_కొలత,
            /* .get_alloc_size      = */ ggml_backend_cpu_బఫరు_type()->iface.get_alloc_కొలత,
            /* .get_alloc_size_n    = */ ఖాళీ,
            /* .is_host             = */ ggml_backend_cpu_బఫరు_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0),
        /* .context  = */ శూన్యసూచిక,
    };

    // Make sure device 0 is initialized
    ggml_vk_instance_init();
    ggml_vk_get_device(0);

    తిరిగి &ggml_backend_vk_బఫరు_type_host;
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_vk_name(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    తిరిగి ctx->name.c_str();
}

శూన్య ggml_backend_vk_free(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    VK_LOG_DEBUG("ggml_backend_vk_free(" << ctx->name << ")");

    ggml_vk_cleanup(ctx);

    తొలగించు ctx;
    తొలగించు backend;
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_vk_get_default_బఫరు_type(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    తిరిగి &ctx->device->బఫరు_type;
}

స్థిర శూన్య ggml_backend_vk_సమితి_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->బఫరు->buft == ggml_backend_vk_get_default_బఫరు_type(backend) || tensor->బఫరు->buft == ggml_backend_vk_host_బఫరు_type()) && "unsupported buffer type");

    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    ggml_backend_vk_బఫరు_context * buf_ctx = (ggml_backend_vk_బఫరు_context *)tensor->బఫరు->context;

    vk_context cpy_ctx;

    ఒకవేళ (ctx->device->async_use_transfer_వరుస) {
        cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
    } లేకపోతే {
        cpy_ctx = ggml_vk_get_compute_ctx(ctx);
    }

    vk_బఫరు buf = buf_ctx->dev_బఫరు;

    స్వయంచాలక dst_స్థానభ్రంశం = vk_tensor_స్థానభ్రంశం(tensor) + tensor->view_offs + స్థానభ్రంశం;

    తార్కిక ret = ggml_vk_బఫరు_write_2d_async(cpy_ctx, buf, dst_స్థానభ్రంశం, దత్తాంశం, stride_దత్తాంశం, stride_tensor, size, n_copies);

    ఒకవేళ (!ret) {
        స్థిరరాశి size_t staging_కొలత = size * n_copies;
        ggml_vk_ensure_sync_staging_బఫరు(ctx, staging_కొలత);
        ggml_vk_sync_buffers(శూన్యసూచిక, cpy_ctx);

        std::vector<vk::బఫరు_Copy> slices(1);
        ఒకవేళ (size == stride_tensor) {
            slices[0].src_స్థానభ్రంశం_2 = 0;
            slices[0].dst_స్థానభ్రంశం_2 = dst_స్థానభ్రంశం;
            slices[0].size = staging_కొలత;
        } లేకపోతే {
            slices.resize(n_copies);
            కొరకు (size_t i = 0; i < n_copies; i++) {
                slices[i].src_స్థానభ్రంశం_2 = i * size;
                slices[i].dst_స్థానభ్రంశం_2 = dst_స్థానభ్రంశం + i * stride_tensor;
                slices[i].size = size;
            }
        }

        cpy_ctx->s->బఫరు->buf.copy_బఫరు(ctx->sync_staging->బఫరు, buf->బఫరు, slices);

        ఒకవేళ (size == stride_దత్తాంశం) {
            deferred_memcpy(ctx->sync_staging->ptr, దత్తాంశం, staging_కొలత, &cpy_ctx->లోపల_memcpys);
        } లేకపోతే {
            కొరకు (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)ctx->sync_staging->ptr + i * size, (స్థిరరాశి uint8_t *)దత్తాంశం + i * stride_దత్తాంశం, size, &cpy_ctx->లోపల_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

స్థిర శూన్య ggml_backend_vk_సమితి_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_async(" << size << ")");
    ggml_backend_vk_సమితి_tensor_2d_async(backend, tensor, దత్తాంశం, స్థానభ్రంశం, size, 1, size, size);
}

స్థిర శూన్య ggml_backend_vk_get_tensor_2d_async(ggml_backend_t backend, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->బఫరు->buft == ggml_backend_vk_get_default_బఫరు_type(backend) || tensor->బఫరు->buft == ggml_backend_vk_host_బఫరు_type()) && "unsupported buffer type");

    ఒకవేళ (size == 0) {
        తిరిగి;
    }

    ggml_backend_vk_బఫరు_context * buf_ctx = (ggml_backend_vk_బఫరు_context *)tensor->బఫరు->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    vk_బఫరు buf = buf_ctx->dev_బఫరు;

    స్వయంచాలక src_స్థానభ్రంశం = vk_tensor_స్థానభ్రంశం(tensor) + tensor->view_offs + స్థానభ్రంశం;
    తార్కిక ret = ggml_vk_బఫరు_read_2d_async(compute_ctx, buf, src_స్థానభ్రంశం, దత్తాంశం, stride_tensor, stride_దత్తాంశం, size, n_copies);

    ఒకవేళ (!ret) {
        స్థిరరాశి size_t staging_కొలత = size * n_copies;
        ggml_vk_ensure_sync_staging_బఫరు(ctx, staging_కొలత);
        ggml_vk_sync_buffers(శూన్యసూచిక, compute_ctx);

        std::vector<vk::బఫరు_Copy> slices(1);
        ఒకవేళ (size == stride_tensor) {
            slices[0].src_స్థానభ్రంశం_2 = src_స్థానభ్రంశం;
            slices[0].dst_స్థానభ్రంశం_2 = 0;
            slices[0].size = staging_కొలత;
        } లేకపోతే {
            slices.resize(n_copies);
            కొరకు (size_t i = 0; i < n_copies; i++) {
                slices[i].src_స్థానభ్రంశం_2 = src_స్థానభ్రంశం + i * stride_tensor;
                slices[i].dst_స్థానభ్రంశం_2 = i * size;
                slices[i].size = size;
            }
        }

        compute_ctx->s->బఫరు->buf.copy_బఫరు(buf->బఫరు, ctx->sync_staging->బఫరు, slices);

        ఒకవేళ (size == stride_దత్తాంశం) {
            deferred_memcpy(దత్తాంశం, ctx->sync_staging->ptr, staging_కొలత, &compute_ctx->బహిర్గతం_memcpys);
        } లేకపోతే {
            కొరకు (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)దత్తాంశం + i * stride_దత్తాంశం, (స్థిరరాశి uint8_t *)ctx->sync_staging->ptr + i * size, size, &compute_ctx->బహిర్గతం_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

స్థిర శూన్య ggml_backend_vk_get_tensor_async(ggml_backend_t backend, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_async(" << size << ")");
    ggml_backend_vk_get_tensor_2d_async(backend, tensor, దత్తాంశం, స్థానభ్రంశం, size, 1, size, size);
}

స్థిర తార్కిక ggml_backend_vk_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_backend_vk_cpy_tensor_async(" << src << " -> " << dst << ", size=" << ggml_nbytes(src) << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend_dst->context;

    // Skip zero-size tensors
    ఒకవేళ (ggml_nbytes(src) == 0) {
        తిరిగి నిజం;
    }

    ఒకవేళ (dst->బఫరు->buft != ggml_backend_vk_get_default_బఫరు_type(backend_dst)) {
        తిరిగి అబద్ధం;
    }

    ggml_backend_vk_బఫరు_context * dst_buf_ctx = (ggml_backend_vk_బఫరు_context *)dst->బఫరు->context;
    vk_బఫరు dst_buf = dst_buf_ctx->dev_బఫరు;

    ఒకవేళ (ggml_backend_బఫరు_is_vk(src->బఫరు)) {
        ggml_backend_vk_బఫరు_context * src_buf_ctx = (ggml_backend_vk_బఫరు_context *)src->బఫరు->context;

        // Async copy only works within the same device
        ఒకవేళ (src_buf_ctx->dev_బఫరు->device != dst_buf->device) {
            తిరిగి అబద్ధం;
        }

        vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_బఫరు_copy_async(compute_ctx, dst_buf, vk_tensor_స్థానభ్రంశం(dst) + dst->view_offs,
                                   src_buf_ctx->dev_బఫరు, vk_tensor_స్థానభ్రంశం(src) + src->view_offs,
                                   ggml_nbytes(src));
        తిరిగి నిజం;
    }

    ఒకవేళ (ggml_backend_బఫరు_is_host(src->బఫరు)) {
        vk_బఫరు pinned_buf = శూన్యసూచిక;
        size_t pinned_స్థానభ్రంశం = 0;
        ggml_vk_host_get(ctx->device, src->దత్తాంశం, pinned_buf, pinned_స్థానభ్రంశం);
        ఒకవేళ (pinned_buf == శూన్యసూచిక) {
            తిరిగి అబద్ధం;
        }

        // If the backend is idle, use a CPU copy to avoid GPU synchronization overhead.
        స్థిర constexpr size_t max_cpu_copy_కొలత = 128 * 1024;
        స్థిరరాశి తార్కిక src_backend_synchronous = backend_src->iface.synchronize == శూన్యసూచిక;
        స్థిరరాశి తార్కిక transfer_idle = !ctx->device->async_use_transfer_వరుస ||
                                   ctx->transfer_semaphore_చివరి_submitted == ctx->transfer_semaphore.value;
        స్థిరరాశి తార్కిక backend_idle = ctx->compute_ctx.expired() && ctx->transfer_ctx.expired() &&
                                  !ctx->submit_pending && !ctx->almost_ready_fence_pending && transfer_idle;
        స్థిరరాశి తార్కిక dst_host_coherent =
            (dst_buf->జ్ఞాపకం_property_flags & (vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent)) ==
            (vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent);

        ఒకవేళ ((backend_src == backend_dst || src_backend_synchronous) && backend_idle && dst_host_coherent && ggml_nbytes(src) <= max_cpu_copy_కొలత) {
            ggml_vk_బఫరు_write(dst_buf, vk_tensor_స్థానభ్రంశం(dst) + dst->view_offs, src->దత్తాంశం, ggml_nbytes(src));
            తిరిగి నిజం;
        }

        vk_context cpy_ctx;
        ఒకవేళ (ctx->device->async_use_transfer_వరుస) {
            cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
        } లేకపోతే {
            cpy_ctx = ggml_vk_get_compute_ctx(ctx);
        }

        తిరిగి ggml_vk_బఫరు_write_async(cpy_ctx, dst_buf,
                                          vk_tensor_స్థానభ్రంశం(dst) + dst->view_offs,
                                          src->దత్తాంశం, ggml_nbytes(src));
    }

    తిరిగి అబద్ధం;
}

శూన్య ggml_vk_synchronize(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_synchronize()");

    తార్కిక do_transfer = !ctx->compute_ctx.expired();

    ఒకవేళ (ggml_vk_submit_transfer_ctx(ctx)) {
        ctx->submit_pending = నిజం;
    }

    vk_context compute_ctx;
    vk_command_బఫరు* cmd_buf = శూన్యసూచిక;
    ఒకవేళ (do_transfer) {
        compute_ctx = ctx->compute_ctx.తాళం();
        ఒకవేళ (compute_ctx->s) {
            cmd_buf = compute_ctx->s->బఫరు;
        }

        ggml_vk_ctx_చివర(compute_ctx);

        కొరకు (స్వయంచాలక& cpy : compute_ctx->లోపల_memcpys) {
            జ్ఞాపకం_నకలు(cpy.dst, cpy.src, cpy.n);
        }

        ఒకవేళ (ctx->device->serialize_submissions) {
            ggml_vk_submit(compute_ctx, ctx->fence);
            VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, నిజం, UINT64_MAX), "synchronize waitForFences", ctx->device);
            ctx->device->device.resetFences({ ctx->fence });
        } లేకపోతే {
            ggml_vk_submit(compute_ctx, {});
        }
        ctx->submit_pending = నిజం;
    }

    ఒకవేళ (ctx->submit_pending) {
        ఒకవేళ (ctx->device->serialize_submissions) {
            ctx->submit_pending = అబద్ధం;
        } లేకపోతే ఒకవేళ (ctx->device->async_use_transfer_వరుస && ctx->transfer_semaphore_చివరి_submitted < ctx->transfer_semaphore.value) {
            vk::TimelineSemaphoreSubmitInfo tl_info{
                1, &ctx->transfer_semaphore.value,
                0, శూన్యసూచిక,
            };
            vk::PipelineStageFlags stage = ctx->device->transfer_వరుస->stage_flags;
            vk::SubmitInfo si{
                1, &ctx->transfer_semaphore.s, &stage,
                0, శూన్యసూచిక,
                0, శూన్యసూచిక,
            };
            si.సమితి_PNext(&tl_info);
            ctx->device->compute_వరుస->handle->submit({ si }, ctx->fence);
            ctx->transfer_semaphore_చివరి_submitted = ctx->transfer_semaphore.value;
        } లేకపోతే {
            ctx->device->compute_వరుస->handle->submit({}, ctx->fence);
        }
        ఒకవేళ (!ctx->device->serialize_submissions) {
            ggml_vk_wait_for_fence(ctx);
        }
        ctx->submit_pending = అబద్ధం;
        ఒకవేళ (cmd_buf) {
            cmd_buf->లోపల_use = అబద్ధం;
            cmd_buf->buf.reset();
        }
    }

    ఒకవేళ (do_transfer) {
        కొరకు (స్వయంచాలక& cpy : compute_ctx->బహిర్గతం_memcpys) {
            జ్ఞాపకం_నకలు(cpy.dst, cpy.src, cpy.n);
        }
        ctx->compute_ctx.reset();
    }
}

స్థిర శూన్య ggml_backend_vk_synchronize(ggml_backend_t backend) {
    VK_LOG_DEBUG("ggml_backend_vk_synchronize()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ggml_vk_synchronize(ctx);

    ggml_vk_లేఖాచిత్రం_cleanup(ctx);
}

తార్కిక ggml_vk_is_ఖాళీలు(ggml_tensor * కణుపు) {
    తిరిగి ggml_is_ఖాళీలు(కణుపు) || కణుపు->op == GGML_OP_NONE || కణుపు->op == GGML_OP_RESHAPE || కణుపు->op == GGML_OP_TRANSPOSE || కణుపు->op == GGML_OP_VIEW || కణుపు->op == GGML_OP_క్రమమార్పు;
}

స్థిర తార్కిక ggml_vk_can_fuse_unary_mul(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం unary_idx, పూర్ణాంకం mul_idx) {
    స్థిరరాశి ggml_tensor * unary = cgraph->nodes[unary_idx];
    స్థిరరాశి ggml_tensor * mul = cgraph->nodes[mul_idx];

    ఒకవేళ (ggml_vk_unary_mul_op_సూచీ(ggml_get_unary_op(unary)) < 0) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (unary->type != mul->type) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (mul->src[0] != unary && mul->src[1] != unary) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
    ఒకవేళ (other == శూన్యసూచిక || other->type != unary->type) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0])) {
        తిరిగి అబద్ధం;
    }
    // fastmod needs src to tile into dst
    ఒకవేళ (mul->src[0] == unary) {
        తిరిగి ggml_can_repeat(other, unary);
    }
    తిరిగి ggml_can_repeat(unary, mul->src[0]);
}

స్థిర తార్కిక ggml_vk_can_fuse_unary_mul_pair(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    స్థిరరాశి గణన ggml_op ops[]    = { GGML_OP_UNARY, GGML_OP_MUL };
    స్థిరరాశి పూర్ణాంకం           outputs[] = { కణుపు_idx + 1 };
    తిరిగి ggml_can_fuse_subgraph(cgraph, కణుపు_idx, 2, ops, outputs, 1) &&
           ggml_vk_can_fuse_unary_mul(cgraph, కణుపు_idx, కణుపు_idx + 1);
}

స్థిర తార్కిక ggml_vk_can_fuse_hc_post_gate(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    స్థిరరాశి ggml_tensor * scale_లోపల  = cgraph->nodes[కణుపు_idx];
    స్థిరరాశి ggml_tensor * sigmoid   = cgraph->nodes[కణుపు_idx + 1];
    స్థిరరాశి ggml_tensor * scale_బహిర్గతం = cgraph->nodes[కణుపు_idx + 2];

    // the shader folds scale -> sigmoid -> scale; a bias on either scale is not handled
    తిరిగి ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
           ggml_get_op_params_f32(scale_లోపల, 1) == 0.0f &&
           ggml_get_op_params_f32(scale_బహిర్గతం, 1) == 0.0f &&
           scale_లోపల->src[0]->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(scale_లోపల->src[0], scale_బహిర్గతం);
}

తార్కిక ggml_vk_can_fuse(స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, std::initializer_list<గణన ggml_op> ops) {
    ఒకవేళ (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL) {
        తిరిగి ggml_vk_can_fuse_unary_mul_pair(cgraph, కణుపు_idx);
    }

    ఒకవేళ (!ggml_can_fuse(cgraph, కణుపు_idx, ops)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ ((ops.size() == 2 || ops.size() == 3 || ops.size() == 4) &&
        ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        స్థిరరాశి ggml_tensor *rms_norm = cgraph->nodes[కణుపు_idx];
        స్థిరరాశి ggml_tensor *mul = cgraph->nodes[కణుపు_idx + 1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        // rms_norm only supports f32
        ఒకవేళ (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }
        // if rms_norm is the B operand, then we don't handle broadcast
        ఒకవేళ (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            తిరిగి అబద్ధం;
        }
        // rms_norm shader assumes contiguous rows
        ఒకవేళ (!ggml_is_contiguous_అడ్డువరుసలు(mul->src[0]) || !ggml_is_contiguous_అడ్డువరుసలు(mul->src[1])) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (ops.size() >= 3 && ops.begin()[2] == GGML_OP_కూడు) {
            స్థిరరాశి ggml_tensor *కూడు_2 = cgraph->nodes[కణుపు_idx + 2];
            స్థిరరాశి ggml_tensor *residual = కూడు_2->src[0] == mul ? కూడు_2->src[1] : కూడు_2->src[0];
            ఒకవేళ (కూడు_2->src[0] != mul && కూడు_2->src[1] != mul) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (residual->type != GGML_TYPE_F32 || కూడు_2->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(కూడు_2, residual) || !ggml_is_contiguous(residual) ||
                !ggml_is_contiguous(కూడు_2) || get_misalign_బైట్లు(ctx, residual) != 0) {
                తిరిగి అబద్ధం;
            }

            స్థిరరాశి ggml_tensor *dst = కూడు_2;
            ఒకవేళ (ops.size() == 4) {
                ఒకవేళ (ops.begin()[3] != GGML_OP_MUL) {
                    తిరిగి అబద్ధం;
                }

                స్థిరరాశి ggml_tensor *post_mul = cgraph->nodes[కణుపు_idx + 3];
                స్థిరరాశి ggml_tensor *scale = post_mul->src[0] == కూడు_2 ? post_mul->src[1] : post_mul->src[0];
                ఒకవేళ (post_mul->src[0] != కూడు_2 && post_mul->src[1] != కూడు_2) {
                    తిరిగి అబద్ధం;
                }
                // The shader reads data_e[0], so the final multiply must use a scalar.
                ఒకవేళ (scale->type != GGML_TYPE_F32 || post_mul->type != GGML_TYPE_F32 ||
                    ggml_nelements(scale) != 1 || !ggml_is_contiguous(post_mul) ||
                    get_misalign_బైట్లు(ctx, scale) != 0) {
                    తిరిగి అబద్ధం;
                }
                dst = post_mul;
            }

            ఒకవేళ (get_misalign_బైట్లు(ctx, dst) != 0) {
                తిరిగి అబద్ధం;
            }
        }
    }

    స్వయంచాలక స్థిరరాశి &mm_కూడు_ok = [&](స్థిరరాశి ggml_tensor *mul, స్థిరరాశి ggml_tensor *కూడు_2) {
        స్థిరరాశి ggml_tensor *bias = కూడు_2->src[0] == mul ? కూడు_2->src[1] : కూడు_2->src[0];

        // mat-vec only
        ఒకవేళ (ggml_nrows(mul) != 1) {
            తిరిగి అబద్ధం;
        }
        // shaders assume the types match
        ఒకవేళ (mul->type != bias->type) {
            తిరిగి అబద్ధం;
        }
        // shaders reuse the D shape for bias
        ఒకవేళ (!ggml_are_same_shape(mul, bias) ||
            !ggml_are_same_stride(mul, bias)) {
            తిరిగి అబద్ధం;
        }
        // unaligned bias isn't handled
        ఒకవేళ (get_misalign_బైట్లు(ctx, bias) != 0) {
            తిరిగి అబద్ధం;
        }
        తిరిగి నిజం;
    };

    ఒకవేళ ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_కూడు) {
        // additional constraints specific to this fusion
        స్థిరరాశి ggml_tensor *mul = cgraph->nodes[కణుపు_idx];
        స్థిరరాశి ggml_tensor *కూడు_2 = cgraph->nodes[కణుపు_idx + 1];

        ఒకవేళ (!mm_కూడు_ok(mul, కూడు_2)) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (ops.size() == 3) {
            ఒకవేళ (ops.begin()[2] != GGML_OP_కూడు) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (!mm_కూడు_ok(కూడు_2, cgraph->nodes[కణుపు_idx + 2])) {
                తిరిగి అబద్ధం;
            }
        }
    }

    స్వయంచాలక స్థిరరాశి &mmid_mul_ok = [&](స్థిరరాశి ggml_tensor *mmid, స్థిరరాశి ggml_tensor *mul) {
        స్థిరరాశి ggml_tensor *scale = mul->src[1];

        ఒకవేళ (mmid != mul->src[0]) {
            తిరిగి అబద్ధం;
        }
        // mat-vec only
        ఒకవేళ (!ggml_vk_use_mul_mat_vec_id(cgraph, కణుపు_idx)) {
            తిరిగి అబద్ధం;
        }
        // shaders assume the types match
        ఒకవేళ (mmid->type != scale->type) {
            తిరిగి అబద్ధం;
        }
        // shaders assume the bias is contiguous
        ఒకవేళ (!ggml_is_contiguous(scale)) {
            తిరిగి అబద్ధం;
        }
        // unaligned bias isn't handled
        ఒకవేళ (get_misalign_బైట్లు(ctx, scale) != 0) {
            తిరిగి అబద్ధం;
        }
        // shader only indexes by expert index
        ఒకవేళ (scale->ne[0] != 1 ||
            scale->ne[1] != mul->ne[1] ||
            scale->ne[2] != 1 ||
            scale->ne[3] != 1) {
            తిరిగి అబద్ధం;
        }
        తిరిగి నిజం;
    };

    ఒకవేళ ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_కూడు_ID) {
        // additional constraints specific to this fusion
        స్థిరరాశి ggml_tensor *mul = cgraph->nodes[కణుపు_idx];
        స్థిరరాశి ggml_tensor *కూడు_2 = cgraph->nodes[కణుపు_idx + 1];
        స్థిరరాశి ggml_tensor *bias = కూడు_2->src[1];

        ఒకవేళ (mul != కూడు_2->src[0]) {
            తిరిగి అబద్ధం;
        }
        // mat-vec only
        ఒకవేళ (!ggml_vk_use_mul_mat_vec_id(cgraph, కణుపు_idx)) {
            తిరిగి అబద్ధం;
        }
        // shaders assume the types match
        ఒకవేళ (mul->type != bias->type) {
            తిరిగి అబద్ధం;
        }
        // shaders assume the bias is contiguous
        ఒకవేళ (!ggml_is_contiguous(bias)) {
            తిరిగి అబద్ధం;
        }
        // the ID tensor must be the same for mul_mat_id and add_id
        ఒకవేళ (mul->src[2] != కూడు_2->src[2]) {
            తిరిగి అబద్ధం;
        }
        // unaligned bias isn't handled
        ఒకవేళ (get_misalign_బైట్లు(ctx, bias) != 0) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (ops.size() == 3) {
            ఒకవేళ (ops.begin()[2] != GGML_OP_MUL) {
                తిరిగి అబద్ధం;
            }
            స్థిరరాశి ggml_tensor *mul = cgraph->nodes[కణుపు_idx + 2];
            తిరిగి mmid_mul_ok(కూడు_2, mul);
        }
    }

    ఒకవేళ (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        స్థిరరాశి ggml_tensor *mmid = cgraph->nodes[కణుపు_idx];
        స్థిరరాశి ggml_tensor *mul = cgraph->nodes[కణుపు_idx + 1];

        ఒకవేళ (!mmid_mul_ok(mmid, mul)) {
            తిరిగి అబద్ధం;
        }
    }

    తిరిగి నిజం;
}

తార్కిక ggml_vk_can_fuse_ssm_conv(స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph,
                                      పూర్ణాంకం కణుపు_idx, పూర్ణాంకం అంకె_extra) {
    స్థిరరాశి ggml_tensor * conv = cgraph->nodes[కణుపు_idx];
    ఒకవేళ (conv->op != GGML_OP_SSM_CONV) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి ggml_tensor * silu = శూన్యసూచిక;
    స్థిరరాశి ggml_tensor * bias = శూన్యసూచిక;

    ఒకవేళ (అంకె_extra == 1) {
        ఒకవేళ (!ggml_can_fuse(cgraph, కణుపు_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY })) {
            తిరిగి అబద్ధం;
        }
        silu = cgraph->nodes[కణుపు_idx + 1];
    } లేకపోతే ఒకవేళ (అంకె_extra == 2) {
        ఒకవేళ (!ggml_can_fuse(cgraph, కణుపు_idx, { GGML_OP_SSM_CONV, GGML_OP_కూడు, GGML_OP_UNARY })) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి ggml_tensor * కూడు_2 = cgraph->nodes[కణుపు_idx + 1];
        silu = cgraph->nodes[కణుపు_idx + 2];
        bias = (కూడు_2->src[0] == conv) ? కూడు_2->src[1] : కూడు_2->src[0];

        ఒకవేళ (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            తిరిగి అబద్ధం;
        }
        // bias must be channel-wise (one element per channel of the conv output)
        ఒకవేళ (ggml_nelements(bias) != conv->ne[0] || bias->ne[0] != conv->ne[0]) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (కూడు_2->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }
        // The shader doesn't apply per-tensor offsets, so reject misaligned bias.
        ఒకవేళ (get_misalign_బైట్లు(ctx, bias) != 0) {
            తిరిగి అబద్ధం;
        }
    } లేకపోతే {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }
    // The shader writes to the fused dst using its own strides, but the push constants don't
    // carry a per-tensor offset, so the binding must be naturally aligned.
    ఒకవేళ (get_misalign_బైట్లు(ctx, silu) != 0) {
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
}

తార్కిక ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph,
                                      పూర్ణాంకం కణుపు_idx, topk_moe_బహుళకం mode) {

    స్థిరరాశి ggml_tensor * softmax;
    స్థిరరాశి ggml_tensor * weights;
    స్థిరరాశి ggml_tensor * get_అడ్డువరుసలు;
    స్థిరరాశి ggml_tensor * argsort;

    ఎంపిక (mode) {
    స్థితి TOPK_MOE_EARLY_SOFTMAX_NORM:
        softmax = cgraph->nodes[కణుపు_idx + 0];
        weights = cgraph->nodes[కణుపు_idx + 9];
        get_అడ్డువరుసలు = cgraph->nodes[కణుపు_idx + 4];
        argsort = cgraph->nodes[కణుపు_idx + 2];
        ఆపు;
    స్థితి TOPK_MOE_SIGMOID_NORM_BIAS:
        softmax = cgraph->nodes[కణుపు_idx + 0]; // really sigmoid
        weights = cgraph->nodes[కణుపు_idx + 10];
        get_అడ్డువరుసలు = cgraph->nodes[కణుపు_idx + 5];
        argsort = cgraph->nodes[కణుపు_idx + 3];
        ఒకవేళ (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SIGMOID) {
            తిరిగి అబద్ధం;
        }
        // bias is expected to be 1D
        ఒకవేళ (ggml_nrows(cgraph->nodes[కణుపు_idx + 2]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[కణుపు_idx + 2]->src[1])) {
            తిరిగి అబద్ధం;
        }
        // sigmoid fusion seems to generate infinities on moltenvk
        ఒకవేళ (ctx->device->driver_id == vk::DriverId::eMoltenvk) {
            తిరిగి అబద్ధం;
        }
        ఆపు;
    స్థితి TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS:
        softmax = cgraph->nodes[కణుపు_idx + 0]; // really softplus
        weights = cgraph->nodes[కణుపు_idx + 11];
        get_అడ్డువరుసలు = cgraph->nodes[కణుపు_idx + 6];
        argsort = cgraph->nodes[కణుపు_idx + 4];
        ఒకవేళ (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SOFTPLUS) {
            తిరిగి అబద్ధం;
        }
        // bias is expected to be 1D
        ఒకవేళ (ggml_nrows(cgraph->nodes[కణుపు_idx + 3]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[కణుపు_idx + 3]->src[1])) {
            తిరిగి అబద్ధం;
        }
        ఆపు;
    స్థితి TOPK_MOE_EARLY_SOFTMAX:
        softmax = cgraph->nodes[కణుపు_idx + 0];
        weights = cgraph->nodes[కణుపు_idx + 4];
        get_అడ్డువరుసలు = cgraph->nodes[కణుపు_idx + 4];
        argsort = cgraph->nodes[కణుపు_idx + 2];
        ఆపు;
    స్థితి TOPK_MOE_LATE_SOFTMAX:
        softmax = cgraph->nodes[కణుపు_idx + 4];
        weights = cgraph->nodes[కణుపు_idx + 5];
        get_అడ్డువరుసలు = cgraph->nodes[కణుపు_idx + 2];
        argsort = cgraph->nodes[కణుపు_idx + 0];
        ఆపు;
    అప్రమేయ:
        తిరిగి అబద్ధం;
    }

    ggml_tensor * probs = get_అడ్డువరుసలు->src[0];
    ఒకవేళ (probs->op != GGML_OP_RESHAPE) {
        తిరిగి అబద్ధం;
    }
    probs = probs->src[0];
    ggml_tensor * selection_probs = argsort->src[0];

    ఒకవేళ (probs != selection_probs &&
        mode != TOPK_MOE_SIGMOID_NORM_BIAS &&
        mode != TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (softmax->op == GGML_OP_SOFT_MAX) {
        స్థిరరాశి దశాంశం * op_params = (స్థిరరాశి దశాంశం *)softmax->op_params;

        దశాంశం scale = op_params[0];
        దశాంశం max_bias = op_params[1];

        ఒకవేళ (scale != 1.0f || max_bias != 0.0f) {
            తిరిగి అబద్ధం;
        }

        // don't fuse when masks or sinks are present
        ఒకవేళ (softmax->src[1] || softmax->src[2]) {
            తిరిగి అబద్ధం;
        }
    }

    స్థిరరాశి పూర్ణాంకం n_expert = softmax->ne[0];
    ఒకవేళ (n_expert > (1 << (అంకె_topk_moe_pipelines-1))) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ctx->device->subgroup_arithmetic ||
        !ctx->device->subgroup_shuffle ||
        !ctx->device->subgroup_require_నిండిన_support ||
        ctx->device->disable_fusion) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_vk_సరిపోలిక_ops(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx,
                              స్థిరరాశి std::initializer_list<ggml_op> & ops) {
    ఒకవేళ (కణుపు_idx + (పూర్ణాంకం) ops.size() > cgraph->n_nodes) {
        తిరిగి అబద్ధం;
    }
    కొరకు (size_t j = 0; j < ops.size(); ++j) {
        స్థిరరాశి ggml_tensor * కణుపు = cgraph->nodes[కణుపు_idx + j];
        ఒకవేళ (కణుపు->op != ops.begin()[j] ||
            (కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            (కణుపు->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            తిరిగి అబద్ధం;
        }
    }
    తిరిగి నిజం;
}

తార్కిక ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    ఒకవేళ (ctx->device->disable_fusion || !ctx->device->pipeline_topk_radix_qsa) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి పూర్ణాంకం n_ops = topk_qsa_నమూనా.size();
    ఒకవేళ (!ggml_vk_సరిపోలిక_ops(cgraph, కణుపు_idx, topk_qsa_నమూనా) ||
        !ggml_check_అంచులు(cgraph, కణుపు_idx, topk_qsa_అంచులు)) {
        తిరిగి అబద్ధం;
    }

    // elided nodes must be single-use (cpy counts its own src[1] self-reference)
    కొరకు (పూర్ణాంకం j = 0; j < n_ops - 1; ++j) {
        స్థిరరాశి ggml_tensor * కణుపు = cgraph->nodes[కణుపు_idx + j];
        స్థిరరాశి int32_t want = కణుపు->op == GGML_OP_CPY ? 2 : 1;
        ఒకవేళ (ggml_కణుపు_get_use_లెక్క(cgraph, కణుపు_idx + j) != want) {
            తిరిగి అబద్ధం;
        }
    }

    స్థిరరాశి ggml_tensor * get_అడ్డువరుసలు = cgraph->nodes[కణుపు_idx + 0];
    స్థిరరాశి ggml_tensor * కూడు_2      = cgraph->nodes[కణుపు_idx + n_ops - 2];
    స్థిరరాశి ggml_tensor * శిఖరం_k    = cgraph->nodes[కణుపు_idx + n_ops - 1];

    స్థిరరాశి ggml_tensor * scores   = get_అడ్డువరుసలు->src[0]; // [n_tps, n_blocks, n_stream]
    స్థిరరాశి ggml_tensor * cell_blk = get_అడ్డువరుసలు->src[1]; // [n_kv, n_stream]
    స్థిరరాశి ggml_tensor * expanded = కూడు_2->src[0];      // [n_kv, n_tps, n_stream]

    // raw mask: follow the reshape/cpy chain back to the materialized f16 input
    స్థిరరాశి ggml_tensor * ముసుగు = కూడు_2->src[1];
    అయితే (ముసుగు && (ముసుగు->op == GGML_OP_RESHAPE || ముసుగు->op == GGML_OP_CPY)) {
        ముసుగు = ముసుగు->src[0];
    }
    ఒకవేళ (!ముసుగు || ముసుగు->type != GGML_TYPE_F16) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (scores->type != GGML_TYPE_F32 || cell_blk->type != GGML_TYPE_I32 || శిఖరం_k->type != GGML_TYPE_I32) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_is_contiguous(scores) || !ggml_is_contiguous(cell_blk) || !ggml_is_contiguous(ముసుగు) ||
        !ggml_is_contiguous(expanded) || !ggml_is_contiguous(శిఖరం_k)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int64_t n_tps    = scores->ne[0];
    స్థిరరాశి int64_t n_ఖండాలు = scores->ne[1];
    స్థిరరాశి int64_t n_ప్రవాహం = scores->ne[2];
    స్థిరరాశి int64_t n_kv     = cell_blk->ne[0];
    స్థిరరాశి int64_t వెడల్పు    = శిఖరం_k->ne[0];

    // pin the indexer layout the shader's addressing assumes
    ఒకవేళ (scores->ne[3] != 1 || cell_blk->ne[1] != n_ప్రవాహం || ggml_nrows(cell_blk) != n_ప్రవాహం ||
        ggml_nelements(ముసుగు) != n_kv * n_tps * n_ప్రవాహం ||
        expanded->ne[0] != n_kv || expanded->ne[1] != n_tps || expanded->ne[2] != n_ప్రవాహం ||
        శిఖరం_k->ne[1] != n_tps || శిఖరం_k->ne[2] != n_ప్రవాహం || శిఖరం_k->ne[3] != 1 ||
        n_ఖండాలు <= 0 || n_kv <= 0 || వెడల్పు <= 0 || వెడల్పు > n_kv) {
        తిరిగి అబద్ధం;
    }

    // only worth it in the radix regime; small k uses the faster tournament unfused
    స్థిరరాశి uint32_t k_min_pipeline = std::max((uint32_t) log2f(దశాంశం(వెడల్పు)) + 1, ctx->device->subgroup_కొలత_log2);
    ఒకవేళ (k_min_pipeline < అంకె_topk_pipelines && ctx->device->pipeline_topk_f32[k_min_pipeline]) {
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
}

తార్కిక ggml_vk_can_fuse_rope_సమితి_అడ్డువరుసలు(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph,
                                           పూర్ణాంకం కణుపు_idx) {
    స్థిరరాశి ggml_tensor *rope = cgraph->nodes[కణుపు_idx + 0];
    స్థిరరాశి ggml_tensor *view = cgraph->nodes[కణుపు_idx + 1];
    స్థిరరాశి ggml_tensor *సమితి_అడ్డువరుసలు = cgraph->nodes[కణుపు_idx + 2];

    // The set_rows epilogue uses one index per ne2 slice and does not encode ne3.
    ఒకవేళ (rope->src[0]->ne[3] != 1) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (సమితి_అడ్డువరుసలు->type != GGML_TYPE_F32 && సమితి_అడ్డువరుసలు->type != GGML_TYPE_F16) {
        తిరిగి అబద్ధం;
    }

    // The shader reads each aligned I64 index as a uvec2 and uses its low 32 bits.
    ఒకవేళ (సమితి_అడ్డువరుసలు->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(సమితి_అడ్డువరుసలు->src[1]) ||
        సమితి_అడ్డువరుసలు->nb[0] != ggml_type_కొలత(సమితి_అడ్డువరుసలు->type) || get_misalign_బైట్లు(ctx, సమితి_అడ్డువరుసలు->src[1]) != 0) {
        తిరిగి అబద్ధం;
    }

    // SET_ROWS consumes one flattened [ne0*ne1] row for each ne2 slice.
    ఒకవేళ (!ggml_is_contiguous(view) ||
        view->ne[0] != rope->ne[0] * rope->ne[1] || view->ne[1] != rope->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(సమితి_అడ్డువరుసలు->src[1]) != rope->ne[2]) {
        తిరిగి అబద్ధం;
    }

    // Only norm/neox/mrope/imrope shaders have the fusion code
    స్థిరరాశి పూర్ణాంకం mode = ((స్థిరరాశి int32_t *) rope->op_params)[2];
    ఒకవేళ (mode != GGML_ROPE_TYPE_సాధారణ && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

తార్కిక ggml_vk_can_fuse_rms_norm_సమితి_అడ్డువరుసలు(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph,
                                               పూర్ణాంకం కణుపు_idx) {
    స్థిరరాశి ggml_tensor * rms = cgraph->nodes[కణుపు_idx];
    స్థిరరాశి ggml_tensor * view = cgraph->nodes[కణుపు_idx + 1];
    స్థిరరాశి ggml_tensor * సమితి_అడ్డువరుసలు = cgraph->nodes[కణుపు_idx + 2];

    // The RMS kernel reads F32 and writes directly to the F32 or F16 SET_ROWS destination.
    ఒకవేళ (rms->src[0]->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 ||
        (సమితి_అడ్డువరుసలు->type != GGML_TYPE_F32 && సమితి_అడ్డువరుసలు->type != GGML_TYPE_F16) ||
        సమితి_అడ్డువరుసలు->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(సమితి_అడ్డువరుసలు->src[1]) ||
        సమితి_అడ్డువరుసలు->nb[0] != ggml_type_కొలత(సమితి_అడ్డువరుసలు->type) || get_misalign_బైట్లు(ctx, సమితి_అడ్డువరుసలు->src[1]) != 0) {
        తిరిగి అబద్ధం;
    }
    // As with the ROPE epilogue, each ne2 slice supplies one flattened row and ne3 is not encoded.
    ఒకవేళ (rms->ne[3] != 1 || !ggml_is_contiguous(rms->src[0]) || !ggml_is_contiguous(view)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (view->ne[0] != rms->ne[0] * rms->ne[1] || view->ne[1] != rms->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(సమితి_అడ్డువరుసలు->src[1]) != rms->ne[2]) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

తార్కిక ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    GGML_వాడని(ctx);
    ఒకవేళ (!ggml_can_fuse(cgraph, కణుపు_idx, snake_నమూనా)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి ggml_tensor * mul0     = cgraph->nodes[కణుపు_idx + 0];
    స్థిరరాశి ggml_tensor * sin_కణుపు = cgraph->nodes[కణుపు_idx + 1];
    స్థిరరాశి ggml_tensor * sqr      = cgraph->nodes[కణుపు_idx + 2];
    స్థిరరాశి ggml_tensor * mul1     = cgraph->nodes[కణుపు_idx + 3];
    స్థిరరాశి ggml_tensor * కూడు_2      = cgraph->nodes[కణుపు_idx + 4];

    స్థిరరాశి ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    స్థిరరాశి ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    స్థిరరాశి ggml_tensor * inv_b    = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];
    స్థిరరాశి ggml_tensor * x_లోపల_కూడు = (కూడు_2->src[0] == mul1) ? కూడు_2->src[1] : కూడు_2->src[0];

    ఒకవేళ (x_లోపల_కూడు != x) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_BF16) {
        తిరిగి అబద్ధం;
    }
    // Shader bindings: data_a is A_TYPE so it follows x's precision, while
    // data_b and data_c are hardcoded float, so the broadcast operands must
    // be F32 regardless of x's type.
    ఒకవేళ (a->type     != GGML_TYPE_F32) తిరిగి అబద్ధం;
    ఒకవేళ (inv_b->type != GGML_TYPE_F32) తిరిగి అబద్ధం;
    // Chain intermediates and output share x's precision (single A_TYPE / D_TYPE pipeline).
    ఒకవేళ (mul0->type     != x->type) తిరిగి అబద్ధం;
    ఒకవేళ (sin_కణుపు->type != x->type) తిరిగి అబద్ధం;
    ఒకవేళ (sqr->type      != x->type) తిరిగి అబద్ధం;
    ఒకవేళ (mul1->type     != x->type) తిరిగి అబద్ధం;
    ఒకవేళ (కూడు_2->type      != x->type) తిరిగి అబద్ధం;
    ఒకవేళ (!ggml_are_same_shape(a, inv_b)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (a->ne[0] != 1 || a->ne[1] != x->ne[1]) {
        తిరిగి అబద్ధం;
    }
    // Dispatch is 2D over (ne0, ne1), so x and add must be 2D and a / inv_b
    // must collapse to [1, C, 1, 1]. Higher dims are not handled by the shader.
    ఒకవేళ (x->ne[2]     != 1 || x->ne[3]     != 1) తిరిగి అబద్ధం;
    ఒకవేళ (కూడు_2->ne[2]   != 1 || కూడు_2->ne[3]   != 1) తిరిగి అబద్ధం;
    ఒకవేళ (a->ne[2]     != 1 || a->ne[3]     != 1) తిరిగి అబద్ధం;
    ఒకవేళ (inv_b->ne[2] != 1 || inv_b->ne[3] != 1) తిరిగి అబద్ధం;
    // Shader uses idx = i0 + i1 * ne0 and reads data_b[i1] / data_c[i1],
    // so every operand must be contiguous.
    ఒకవేళ (!ggml_is_contiguous(x) || !ggml_is_contiguous(కూడు_2) ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(inv_b)) {
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
}

తార్కిక ggml_vk_tensors_overlap(స్థిరరాశి ggml_tensor * a, స్థిరరాశి ggml_tensor * b, తార్కిక elementwise) {
    ggml_backend_vk_బఫరు_context * a_buf_ctx = (ggml_backend_vk_బఫరు_context *)a->బఫరు->context;
    vk_బఫరు a_buf = a_buf_ctx->dev_బఫరు;
    ggml_backend_vk_బఫరు_context * b_buf_ctx = (ggml_backend_vk_బఫరు_context *)b->బఫరు->context;
    vk_బఫరు b_buf = b_buf_ctx->dev_బఫరు;
    ఒకవేళ (a_buf == b_buf) {
        స్వయంచాలక a_పునాది = vk_tensor_స్థానభ్రంశం(a) + a->view_offs;
        స్వయంచాలక a_కొలత = ggml_nbytes(a);
        స్వయంచాలక b_పునాది = vk_tensor_స్థానభ్రంశం(b) + b->view_offs;
        స్వయంచాలక b_కొలత = ggml_nbytes(b);

        ఒకవేళ (elementwise && a_పునాది == b_పునాది && a_కొలత == b_కొలత) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ ((b_పునాది <= a_పునాది && a_పునాది < b_పునాది + b_కొలత) ||
            (a_పునాది <= b_పునాది && b_పునాది < a_పునాది + a_కొలత)) {
            తిరిగి నిజం;
        }
    }
    తిరిగి అబద్ధం;
}

తార్కిక ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph,
                                               పూర్ణాంకం కణుపు_idx) {
    స్థిరరాశి ggml_tensor *rms = cgraph->nodes[కణుపు_idx + 0];
    స్థిరరాశి ggml_tensor *mul = cgraph->nodes[కణుపు_idx + 1];
    స్థిరరాశి ggml_tensor *rope = cgraph->nodes[కణుపు_idx + 2];

    స్థిరరాశి పూర్ణాంకం mode = ((స్థిరరాశి int32_t *) rope->op_params)[2];

    // noncontig tensors aren't tested, and don't seem common in practice
    ఒకవేళ (!ggml_is_contiguous(rms) ||
        !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous(rope)) {
        తిరిగి అబద్ధం;
    }

    // only norm/neox are handled in the shader
    ఒకవేళ (mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_సాధారణ) {
        తిరిగి అబద్ధం;
    }

    // shared memory size for passing data from mul->rope
    ఒకవేళ (mul->ne[0] > 1024) {
        తిరిగి అబద్ధం;
    }

    // conditions for pipeline creation
    ఒకవేళ (పరిమాణం(vk_op_rms_norm_mul_rope_నెట్టు_constants) > ctx->device->properties.limits.max_నెట్టు_Constants_కొలత) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

uint32_t ggml_vk_fuse_multi_కూడు(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {

    స్థిరరాశి ggml_tensor *మొదటి_కణుపు = cgraph->nodes[కణుపు_idx];
    ఒకవేళ (మొదటి_కణుపు->op != GGML_OP_కూడు) {
        తిరిగి 0;
    }

    ఒకవేళ (!ctx->device->multi_కూడు) {
        తిరిగి 0;
    }

    int32_t అంకె_adds = 1;
    అయితే (కణుపు_idx + అంకె_adds < cgraph->n_nodes &&
           cgraph->nodes[కణుపు_idx + అంకె_adds]->op == GGML_OP_కూడు &&
           అంకె_adds < MAX_FUSED_ADDS) {
        అంకె_adds++;
    }

    // The shader currently requires same shapes (but different strides are allowed),
    // everything f32, and no misalignment
    కొరకు (int32_t i = 0; i < అంకె_adds; ++i) {
        స్థిరరాశి ggml_tensor *తదుపరి_కణుపు = cgraph->nodes[కణుపు_idx + i];
        ఒకవేళ (!ggml_are_same_shape(మొదటి_కణుపు, తదుపరి_కణుపు->src[0]) ||
            !ggml_are_same_shape(మొదటి_కణుపు, తదుపరి_కణుపు->src[1]) ||
            తదుపరి_కణుపు->type != GGML_TYPE_F32 ||
            తదుపరి_కణుపు->src[0]->type != GGML_TYPE_F32 ||
            తదుపరి_కణుపు->src[1]->type != GGML_TYPE_F32 ||
            get_misalign_బైట్లు(ctx, తదుపరి_కణుపు) ||
            get_misalign_బైట్లు(ctx, తదుపరి_కణుపు->src[0]) ||
            get_misalign_బైట్లు(ctx, తదుపరి_కణుపు->src[1])) {
            అంకె_adds = i;
        }
    }

    // Verify we can fuse these
    ggml_op adds[MAX_FUSED_ADDS];
    కొరకు (int32_t i = 0; i < అంకె_adds; ++i) {
        adds[i] = GGML_OP_కూడు;
    }

    // decrease num_adds if they can't all be fused
    అయితే (అంకె_adds > 1 && !ggml_can_fuse(cgraph, కణుపు_idx, adds, అంకె_adds)) {
        అంకె_adds--;
    }

    // a single add is not "fused", so just return zero
    ఒకవేళ (అంకె_adds == 1) {
        తిరిగి 0;
    }
    తిరిగి అంకె_adds;
}

స్థిర int32_t కనుగొను_మొదటి_సమితి(uint32_t x) {
    int32_t ret = 0;
    ఒకవేళ (!x) {
        తిరిగి -1;
    }
    అయితే (!(x & 1)) {
        x >>= 1;
        ret++;
    }
    తిరిగి ret;
}

స్థిర ggml_స్థితి ggml_backend_vk_లేఖాచిత్రం_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    VK_LOG_DEBUG("ggml_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ctx->device->diag_cgraph = శూన్యసూచిక;
    ctx->device->diag_prev_ప్రారంభం = -1;
    ctx->device->diag_prev_చివర = -1;

    ఒకవేళ (vk_instance.debug_utils_support) {
        ctx->device->debug_cmdbuf_idx = 0;
    }

    // queue scope, so it encloses every submit this evaluation makes.
    // closed when the function returns
    ggml_vk_debug_label వరుస_dbg(ctx->device->compute_వరుస->handle.get(), "ggml_backend_vk_graph_compute");

    ctx->prealloc_కొలత_కూడు_rms_partials_స్థానభ్రంశం = 0;
    ctx->do_కూడు_rms_partials = అబద్ధం;
    ctx->do_కూడు_rms_partials_స్థానభ్రంశం_calculation = అబద్ధం;

    పూర్ణాంకం చివరి_కణుపు = cgraph->n_nodes - 1;

    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    అయితే (చివరి_కణుపు > 0 && (ggml_vk_is_ఖాళీలు(cgraph->nodes[చివరి_కణుపు]) || ((cgraph->nodes[చివరి_కణుపు]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0))) {
        చివరి_కణుపు -= 1;
    }

    // Reserve tensor context space for all nodes
    ctx->tensor_ctxs.resize(cgraph->n_nodes);

    తార్కిక మొదటి_కణుపు_లోపల_batch = నిజం; // true if next node will be first node in a batch
    పూర్ణాంకం submit_కణుపు_idx = 0; // index to first node in a batch

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx;
    ఒకవేళ (vk_perf_logger_enabled) {
        // allocate/resize the query pool
        ఒకవేళ (ctx->అంకె_queries < cgraph->n_nodes + 1) {
            ఒకవేళ (ctx->query_pool) {
                ctx->device->device.destroyQueryPool(ctx->query_pool);
            }
            vk::QueryPoolCreateInfo query_create_info;
            query_create_info.queryType = vk::QueryType::eTimestamp;
            query_create_info.query_లెక్క = cgraph->n_nodes + 100;
            ctx->query_pool = ctx->device->device.createQueryPool(query_create_info);
            ctx->అంకె_queries = query_create_info.query_లెక్క;
            ctx->query_fusion_పేర్లు.resize(ctx->అంకె_queries);
            ctx->query_fusion_కణుపు_లెక్క.resize(ctx->అంకె_queries);
            ctx->query_nodes.resize(ctx->అంకె_queries);
            ctx->query_కణుపు_idx.resize(ctx->అంకె_queries);
        }

        ctx->device->device.resetQueryPool(ctx->query_pool, 0, cgraph->n_nodes+1);
        std::fill(ctx->query_fusion_పేర్లు.begin(), ctx->query_fusion_పేర్లు.end(), శూన్యసూచిక);
        std::fill(ctx->query_fusion_కణుపు_లెక్క.begin(), ctx->query_fusion_కణుపు_లెక్క.end(), 0);
        std::fill(ctx->query_nodes.begin(), ctx->query_nodes.end(), శూన్యసూచిక);
        std::fill(ctx->query_కణుపు_idx.begin(), ctx->query_కణుపు_idx.end(), 0);

        GGML_ASSERT(ctx->compute_ctx.expired());
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        ctx->query_idx = 0;
        compute_ctx->s->బఫరు->buf.writeTimestamp(vk::Pipeline_Stage_Flag_బిట్లు::eAllCommands, ctx->query_pool, ctx->query_idx++);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    ctx->prealloc_y_చివరి_pipeline_వాడిన = శూన్యసూచిక;
    ctx->prealloc_y_చివరి_tensor_వాడిన = శూన్యసూచిక;
    ctx->prealloc_y_చివరి_k_padded = అబద్ధం;

    ఒకవేళ (ctx->prealloc_కొలత_కూడు_rms_partials) {
        ggml_vk_preallocate_buffers(ctx, శూన్యసూచిక);
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        // initialize partial sums to zero.
        ggml_vk_బఫరు_memset_async(compute_ctx, ctx->prealloc_కూడు_rms_partials, 0, 0, ctx->prealloc_కొలత_కూడు_rms_partials);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    // Submit after enough work has accumulated, to overlap CPU cmdbuffer generation with GPU execution.
    // Estimate the amount of compute work using flops, and submit every 200 GFLOP
    // (and scaled down based on total graph flops, so smaller models submit earlier).
    // Also submit at least every 100 nodes, in case there are workloads without heavy compute.
    uint32_t submitted_nodes = 0;
    uint32_t submit_లెక్క = 0;
    uint64_t batch_flops = 0;
    uint64_t మొత్తం_flops = 0;
    uint64_t flops_cap = 200'000'000'000ULL;

    // On weaker AMD GPUs larger submissions can hit a driver timeout, submit more often to avoid this
    ఒకవేళ (ctx->device->vendor_id == VK_VENDOR_ID_AMD && ctx->device->shader_core_లెక్క > 0) {
        ఒకవేళ (ctx->device->architecture == AMD_GCN && ctx->device->shader_core_లెక్క < 32) {
            flops_cap = 500'000'000ULL * ctx->device->shader_core_లెక్క;
        } లేకపోతే ఒకవేళ (ctx->device->architecture != AMD_GCN && ctx->device->shader_core_లెక్క < 24) {
            flops_cap = 2'000'000'000ULL * ctx->device->shader_core_లెక్క;
        }
    }
    uint64_t flops_చొప్పున_submit = std::min(flops_cap, ctx->చివరి_మొత్తం_flops / 40u);

    స్వయంచాలక స్థిరరాశి submit_after = [&](పూర్ణాంకం start, పూర్ణాంకం end) {
        ఒకవేళ (ctx->device->serialize_submissions) {
            ప్రయత్నించు {
                స్వయంచాలక res = ctx->device->device.waitForFences({ ctx->fence }, నిజం, UINT64_MAX);
                ఒకవేళ (res != vk::ఫలితం_2::eSuccess) {
                    GGML_LOG_ERROR("ggml_vulkan: waitForFences error during serialized submission\n");
                    విసురు vk::SystemError(vk::make_error_code(res), "ggml_vulkan: waitForFences during serialized submission");
                }
            } పట్టుకో (vk::Device_కోల్పోయిన_Error &) {
                ggml_vk_print_device_fault_info(ctx->device);
                GGML_LOG_ERROR("ggml_vulkan: device lost on %s waiting for submission (nodes %d to %d):\n",
                        ctx->device->name.c_str(), start, end);
                ggml_vk_print_కణుపు_list(cgraph, start, end);
                విసురు;
            }
            ctx->device->device.resetFences({ ctx->fence });
            ctx->submit_pending = అబద్ధం;
            ctx->device->diag_cgraph = cgraph;
            ctx->device->diag_prev_ప్రారంభం = start;
            ctx->device->diag_prev_చివర = end;
        }
        మొదటి_కణుపు_లోపల_batch = నిజం;
        submitted_nodes = 0;
        batch_flops = 0;
        ఒకవేళ (submit_లెక్క < 3) {
            flops_చొప్పున_submit *= 2;
        }
        submit_లెక్క++;
    };

    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        ఒకవేళ (మొదటి_కణుపు_లోపల_batch) {
            submit_కణుపు_idx = i;
        }

        {
            స్వయంచాలక కణుపు_flops = ggml_vk_get_కణుపు_flops(cgraph->nodes[i]);
            మొత్తం_flops += కణుపు_flops;

            // Flush the current batch before recording a node that would push it over the flop threshold
            ఒకవేళ (flops_చొప్పున_submit != 0 && submitted_nodes > 0 && batch_flops + కణుపు_flops >= flops_చొప్పున_submit) {
                vk_context flush_ctx = ggml_vk_get_compute_ctx(ctx);
                ggml_vk_ctx_చివర(flush_ctx);
                flush_ctx->exit_tensor_idx = -1;
                ctx->compute_ctx.reset();
                ggml_vk_compute_forward(ctx, cgraph, cgraph->nodes[submit_కణుపు_idx], submit_కణుపు_idx, అబద్ధం);
                submit_after(submit_కణుపు_idx, i - 1);
                submit_కణుపు_idx = i;
            }

            batch_flops += కణుపు_flops;
        }

        // op_srcs_fused_elementwise indicates whether an op's srcs all contribute to
        // the fused result in an elementwise-way. This affects whether the memory for
        // the src is allowed to overlap the memory for the destination.
        // The array is sized to handle the largest fusion (asserted later).
        తార్కిక op_srcs_fused_elementwise[13];

        ctx->fused_topk_moe_బహుళకం = TOPK_MOE_లెక్క;
        ctx->fused_topk_moe_scale = అబద్ధం;
        ctx->fused_topk_qsa = అబద్ధం;
        ctx->fused_hc_post_gate = అబద్ధం;
        ctx->fused_rms_norm_బహుళకం = RMS_NORM_లెక్క;
        స్థిరరాశి అక్షరం *fusion_string {};
        ఒకవేళ (!ctx->device->disable_fusion) {
            uint32_t అంకె_adds = ggml_vk_fuse_multi_కూడు(ctx, cgraph, i);
            ఒకవేళ (అంకె_adds) {
                ctx->అంకె_additional_fused_ops = అంకె_adds - 1;
                fusion_string = "MULTI_ADD";
                std::fill_n(op_srcs_fused_elementwise, ctx->అంకె_additional_fused_ops + 1, నిజం);
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_కూడు, GGML_OP_కూడు })) {
                ctx->అంకె_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ADD_ADD";
                op_srcs_fused_elementwise[0] = అబద్ధం;
                op_srcs_fused_elementwise[1] = నిజం;
                op_srcs_fused_elementwise[2] = నిజం;
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_కూడు })) {
                ctx->అంకె_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ADD";
                op_srcs_fused_elementwise[0] = అబద్ధం;
                op_srcs_fused_elementwise[1] = నిజం;
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_కూడు_ID, GGML_OP_MUL })) {
                ctx->అంకె_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ID_ADD_ID_MUL";
                op_srcs_fused_elementwise[0] = అబద్ధం;
                op_srcs_fused_elementwise[1] = నిజం;
                op_srcs_fused_elementwise[2] = నిజం;
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_కూడు_ID })) {
                ctx->అంకె_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_ADD_ID";
                op_srcs_fused_elementwise[0] = అబద్ధం;
                op_srcs_fused_elementwise[1] = నిజం;
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL })) {
                ctx->అంకె_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_MUL";
                op_srcs_fused_elementwise[0] = అబద్ధం;
                op_srcs_fused_elementwise[1] = నిజం;
            } లేకపోతే ఒకవేళ (ggml_can_fuse_subgraph(cgraph, i, rms_norm_mul_rope_view_సమితి_అడ్డువరుసలు_నమూనా, { i + 4 }) &&
                       ggml_check_అంచులు(cgraph, i, rms_norm_mul_rope_view_సమితి_అడ్డువరుసలు_అంచులు) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i) &&
                       ggml_vk_can_fuse_rope_సమితి_అడ్డువరుసలు(ctx, cgraph, i + 2)) {
                ctx->అంకె_additional_fused_ops = 4;
                ctx->fused_rms_norm_బహుళకం = RMS_NORM_MUL_ROPE_VIEW_సమితి_అడ్డువరుసలు;
                fusion_string = "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = అబద్ధం;
                op_srcs_fused_elementwise[1] = అబద్ధం;
                op_srcs_fused_elementwise[2] = అబద్ధం;
                op_srcs_fused_elementwise[3] = అబద్ధం;
                op_srcs_fused_elementwise[4] = అబద్ధం;
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i)) {
                ctx->అంకె_additional_fused_ops = 2;
                ctx->fused_rms_norm_బహుళకం = RMS_NORM_MUL_ROPE;
                fusion_string = "RMS_NORM_MUL_ROPE";
                // rope is approximately elementwise - whole rows are done by a single workgroup and it's row-wise
                op_srcs_fused_elementwise[0] = అబద్ధం;
                op_srcs_fused_elementwise[1] = నిజం;
                op_srcs_fused_elementwise[2] = నిజం;
            } లేకపోతే ఒకవేళ (ggml_can_fuse_subgraph(cgraph, i, hc_post_gate_నమూనా, { i + 3 }) &&
                       ggml_check_అంచులు(cgraph, i, hc_post_gate_అంచులు) &&
                       ggml_vk_can_fuse_hc_post_gate(cgraph, i)) {
                ctx->అంకె_additional_fused_ops = hc_post_gate_నమూనా.size() - 1;
                ctx->fused_hc_post_gate = నిజం;
                fusion_string = "HC_POST_GATE";
                std::fill_n(op_srcs_fused_elementwise, ctx->అంకె_additional_fused_ops + 1, అబద్ధం);
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_కూడు_mul_నమూనా)) {
                ctx->అంకె_additional_fused_ops = 3;
                ctx->fused_rms_norm_బహుళకం = RMS_NORM_MUL_కూడు_MUL;
                fusion_string = "RMS_NORM_MUL_ADD_MUL";
                std::fill_n(op_srcs_fused_elementwise, 4, నిజం);
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_కూడు_నమూనా)) {
                ctx->అంకె_additional_fused_ops = 2;
                ctx->fused_rms_norm_బహుళకం = RMS_NORM_MUL_కూడు;
                fusion_string = "RMS_NORM_MUL_ADD";
                std::fill_n(op_srcs_fused_elementwise, 3, నిజం);
            } లేకపోతే ఒకవేళ (ggml_can_fuse_subgraph(cgraph, i, rms_norm_view_సమితి_అడ్డువరుసలు_నమూనా, { i + 2 }) &&
                       ggml_check_అంచులు(cgraph, i, rms_norm_view_సమితి_అడ్డువరుసలు_అంచులు) &&
                       ggml_vk_can_fuse_rms_norm_సమితి_అడ్డువరుసలు(ctx, cgraph, i)) {
                ctx->అంకె_additional_fused_ops = 2;
                ctx->fused_rms_norm_బహుళకం = RMS_NORM_VIEW_సమితి_అడ్డువరుసలు;
                fusion_string = "RMS_NORM_VIEW_SET_ROWS";
                std::fill_n(op_srcs_fused_elementwise, 3, అబద్ధం);
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                ctx->అంకె_additional_fused_ops = 1;
                ctx->fused_rms_norm_బహుళకం = RMS_NORM_MUL;
                fusion_string = "RMS_NORM_MUL";
                // rms_norm is not elementwise, but whole rows must be consumed and the scale factor computed before
                // they are overwritten, and one workgroup per row. So close enough.
                op_srcs_fused_elementwise[0] = నిజం;
                op_srcs_fused_elementwise[1] = నిజం;
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL })) {
                ctx->అంకె_additional_fused_ops = 1;
                ఎంపిక (ggml_get_unary_op(cgraph->nodes[i])) {
                    స్థితి GGML_UNARY_OP_GELU:     fusion_string = "GELU_MUL";     ఆపు;
                    స్థితి GGML_UNARY_OP_SIGMOID:  fusion_string = "SIGMOID_MUL";  ఆపు;
                    స్థితి GGML_UNARY_OP_SILU:     fusion_string = "SILU_MUL";     ఆపు;
                    అప్రమేయ:                     fusion_string = "SOFTPLUS_MUL"; ఆపు;
                }
                op_srcs_fused_elementwise[0] = నిజం;
                op_srcs_fused_elementwise[1] = నిజం;
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 2)) {
                ctx->అంకె_additional_fused_ops = 2;
                fusion_string = "SSM_CONV_BIAS_SILU";
                // ssm_conv reads multiple input tokens per output, so it's not elementwise w.r.t. its srcs.
                // The downstream add and silu are elementwise on the conv output.
                op_srcs_fused_elementwise[0] = అబద్ధం;
                op_srcs_fused_elementwise[1] = నిజం;
                op_srcs_fused_elementwise[2] = నిజం;
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 1)) {
                ctx->అంకె_additional_fused_ops = 1;
                fusion_string = "SSM_CONV_SILU";
                op_srcs_fused_elementwise[0] = అబద్ధం;
                op_srcs_fused_elementwise[1] = నిజం;
            } లేకపోతే ఒకవేళ (ggml_can_fuse_subgraph(cgraph, i, rope_view_సమితి_అడ్డువరుసలు_నమూనా, { i + 2 }) &&
                       ggml_check_అంచులు(cgraph, i, rope_view_సమితి_అడ్డువరుసలు_అంచులు) &&
                       ggml_vk_can_fuse_rope_సమితి_అడ్డువరుసలు(ctx, cgraph, i)) {
                ctx->అంకె_additional_fused_ops = 2;
                fusion_string = "ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = అబద్ధం;
                op_srcs_fused_elementwise[1] = అబద్ధం;
                op_srcs_fused_elementwise[2] = అబద్ధం;
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse_snake(ctx, cgraph, i)) {
                ctx->అంకె_additional_fused_ops = 4;
                fusion_string = "SNAKE";
                // elementwise=true: snake.comp is safe under exact aliasing because each
                // thread reads data_x[idx] into a register before writing data_d[idx]
                // with a data dependency on that register. The overlap check still
                // rejects partial overlaps (different base or size).
                std::fill_n(op_srcs_fused_elementwise, 5, నిజం);
            } లేకపోతే ఒకవేళ (ggml_vk_can_fuse_topk_qsa(ctx, cgraph, i)) {
                ctx->అంకె_additional_fused_ops = topk_qsa_నమూనా.size() - 1;
                ctx->fused_topk_qsa = నిజం;
                fusion_string = "TOPK_QSA";
                std::fill_n(op_srcs_fused_elementwise, ctx->అంకె_additional_fused_ops + 1, అబద్ధం);
            } లేకపోతే ఒకవేళ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax_norm, { i + 3, i + 9 }) &&
                       ggml_check_అంచులు(cgraph, i, topk_moe_early_softmax_norm_అంచులు) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX_NORM)) {
                ctx->అంకె_additional_fused_ops = topk_moe_early_softmax_norm.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ముసుగు |= 1 << 3;
                ctx->fused_topk_moe_బహుళకం = TOPK_MOE_EARLY_SOFTMAX_NORM;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX_NORM";
                std::fill_n(op_srcs_fused_elementwise, ctx->అంకె_additional_fused_ops + 1, అబద్ధం);
            } లేకపోతే ఒకవేళ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sigmoid_norm_bias, { i + 4, i + 10 }) &&
                       ggml_check_అంచులు(cgraph, i, topk_moe_sigmoid_norm_bias_అంచులు) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SIGMOID_NORM_BIAS)) {
                ctx->అంకె_additional_fused_ops = topk_moe_sigmoid_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ముసుగు |= 1 << 4;
                ctx->fused_topk_moe_బహుళకం = TOPK_MOE_SIGMOID_NORM_BIAS;
                fusion_string = "TOPK_MOE_SIGMOID_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->అంకె_additional_fused_ops + 1, అబద్ధం);
            } లేకపోతే ఒకవేళ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sqrt_softplus_norm_bias, { i + 5, i + 11 }) &&
                       ggml_check_అంచులు(cgraph, i, topk_moe_sqrt_softplus_norm_bias_అంచులు) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS)) {
                ctx->అంకె_additional_fused_ops = topk_moe_sqrt_softplus_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ముసుగు |= 1 << 5;
                ctx->fused_topk_moe_బహుళకం = TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
                fusion_string = "TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->అంకె_additional_fused_ops + 1, అబద్ధం);
            } లేకపోతే ఒకవేళ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax, { i + 3, i + 4 }) &&
                       ggml_check_అంచులు(cgraph, i, topk_moe_early_softmax_అంచులు) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX)) {
                ctx->అంకె_additional_fused_ops = topk_moe_early_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ముసుగు |= 1 << 3;
                ctx->fused_topk_moe_బహుళకం = TOPK_MOE_EARLY_SOFTMAX;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->అంకె_additional_fused_ops + 1, అబద్ధం);
            } లేకపోతే ఒకవేళ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_late_softmax, { i + 1, i + 5 }) &&
                       ggml_check_అంచులు(cgraph, i, topk_moe_late_softmax_అంచులు) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_LATE_SOFTMAX)) {
                ctx->అంకె_additional_fused_ops = topk_moe_late_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ముసుగు |= 1 << 1;
                ctx->fused_topk_moe_బహుళకం = TOPK_MOE_LATE_SOFTMAX;
                fusion_string = "TOPK_MOE_LATE_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->అంకె_additional_fused_ops + 1, అబద్ధం);
            }
            ఒకవేళ (ctx->fused_topk_moe_బహుళకం != TOPK_MOE_లెక్క) {
                // Look for an additional scale op to fuse - occurs in deepseek2 and nemotron3 nano.
                ఒకవేళ (ggml_can_fuse_subgraph(cgraph, i + ctx->అంకె_additional_fused_ops - 1, { GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE }, { i + ctx->అంకె_additional_fused_ops + 1 }) ||
                    ggml_can_fuse_subgraph(cgraph, i + ctx->అంకె_additional_fused_ops, { GGML_OP_GET_అడ్డువరుసలు, GGML_OP_SCALE }, { i + ctx->అంకె_additional_fused_ops + 1 })) {
                    ctx->fused_topk_moe_scale = నిజం;
                    ctx->అంకె_additional_fused_ops++;
                    op_srcs_fused_elementwise[ctx->అంకె_additional_fused_ops] = అబద్ధం;
                }
            }
        }
        GGML_ASSERT(ctx->అంకె_additional_fused_ops < (పూర్ణాంకం)(పరిమాణం(op_srcs_fused_elementwise) / పరిమాణం(op_srcs_fused_elementwise[0])));
        ctx->fused_ops_write_ముసుగు |= 1 << ctx->అంకె_additional_fused_ops;

        // Check whether fusion would overwrite src operands while they're still in use.
        // If so, disable fusion.
        ఒకవేళ (ctx->అంకె_additional_fused_ops) {
            // There are up to two output nodes - topk_moe has two.
            uint32_t బిట్లు = ctx->fused_ops_write_ముసుగు & ~(1 << ctx->అంకె_additional_fused_ops);
            ggml_tensor *output_nodes[2] {};
            output_nodes[0] = cgraph->nodes[i + ctx->అంకె_additional_fused_ops];
            ఒకవేళ (బిట్లు) {
                పూర్ణాంకం output_idx = కనుగొను_మొదటి_సమితి(బిట్లు);
                GGML_ASSERT(బిట్లు == (1u << output_idx));
                output_nodes[1] = cgraph->nodes[i + output_idx];
            }

            తార్కిక need_disable = అబద్ధం;

            కొరకు (పూర్ణాంకం j = 0; j < 2; ++j) {
                ggml_tensor *dst = output_nodes[j];
                ఒకవేళ (!dst) {
                    కొనసాగు;
                }
                // Loop over all srcs of all nodes in the fusion. If the src overlaps
                // the destination and the src is not an intermediate node that's being
                // elided, then disable fusion.
                కొరకు (పూర్ణాంకం k = 0; k <= ctx->అంకె_additional_fused_ops; ++k) {
                    కొరకు (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                        ggml_tensor *src = cgraph->nodes[i + k]->src[s];
                        ఒకవేళ (!src || src->op == GGML_OP_NONE) {
                            కొనసాగు;
                        }
                        ఒకవేళ (ggml_vk_tensors_overlap(src, dst, op_srcs_fused_elementwise[k])) {
                            తార్కిక దొరికింది = అబద్ధం;
                            కొరకు (పూర్ణాంకం n = 0; n < k; ++n) {
                                ఒకవేళ (cgraph->nodes[i + n] == src) {
                                    దొరికింది = నిజం;
                                    ఆపు;
                                }
                            }
                            ఒకవేళ (!దొరికింది) {
                                need_disable = నిజం;
                            }
                        }
                    }
                }
            }
            ఒకవేళ (need_disable) {
                ctx->అంకె_additional_fused_ops = 0;
                ctx->fused_ops_write_ముసుగు = 1;
                ctx->fused_topk_moe_బహుళకం = TOPK_MOE_లెక్క;
                ctx->fused_topk_moe_scale = అబద్ధం;
                ctx->fused_topk_qsa = అబద్ధం;
                ctx->fused_hc_post_gate = అబద్ధం;
                ctx->fused_rms_norm_బహుళకం = RMS_NORM_లెక్క;
                fusion_string = శూన్యసూచిక;
            }
        }

        // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
        తార్కిక almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
        తార్కిక submit = (submitted_nodes >= ctx->device->max_nodes_చొప్పున_submit) ||
                      (flops_చొప్పున_submit != 0 && batch_flops >= flops_చొప్పున_submit) ||
                      (i + ctx->అంకె_additional_fused_ops >= చివరి_కణుపు) ||
                      (almost_ready && !ctx->almost_ready_fence_pending);

        తార్కిక enqueued = ggml_vk_build_లేఖాచిత్రం(ctx, cgraph, i, cgraph->nodes[submit_కణుపు_idx], submit_కణుపు_idx, i + ctx->అంకె_additional_fused_ops >= చివరి_కణుపు, almost_ready, submit);

        ఒకవేళ (vk_perf_logger_enabled && enqueued) {
            compute_ctx = ggml_vk_get_compute_ctx(ctx);
            ఒకవేళ (!vk_perf_logger_concurrent) {
                // track a single node/fusion for the current query
                ctx->query_nodes[ctx->query_idx] = cgraph->nodes[i];
                ctx->query_fusion_పేర్లు[ctx->query_idx] = fusion_string;
                compute_ctx->s->బఫరు->buf.writeTimestamp(vk::Pipeline_Stage_Flag_బిట్లు::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            } లేకపోతే {
                // track a fusion string and number of fused ops for the current node_idx
                ctx->query_fusion_పేర్లు[i] = fusion_string;
                ctx->query_fusion_కణుపు_లెక్క[i] = ctx->అంకె_additional_fused_ops;
            }
        }

        ఒకవేళ (enqueued) {
            ++submitted_nodes;

#నిర్వచితం_కాకపోతే GGML_VULKAN_CHECK_RESULTS
            ఒకవేళ (మొదటి_కణుపు_లోపల_batch) {
                మొదటి_కణుపు_లోపల_batch = అబద్ధం;
            }
#అయితే_ముగింపు
        }

        ఒకవేళ (submit && enqueued) {
            submit_after(submit_కణుపు_idx, i + (పూర్ణాంకం)ctx->అంకె_additional_fused_ops);
        }
        i += ctx->అంకె_additional_fused_ops;
        ctx->అంకె_additional_fused_ops = 0;
        ctx->fused_ops_write_ముసుగు = 0;
    }

    ctx->చివరి_మొత్తం_flops = మొత్తం_flops;

    ఒకవేళ (vk_perf_logger_enabled) {
        // End the command buffer and submit/wait
        GGML_ASSERT(!ctx->compute_ctx.expired());
        compute_ctx = ctx->compute_ctx.తాళం();
        ggml_vk_ctx_చివర(compute_ctx);

        ggml_vk_submit(compute_ctx, ctx->device->fence);
        VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, నిజం, UINT64_MAX), "GGML_VULKAN_PERF waitForFences", ctx->device);
        ctx->device->device.resetFences({ ctx->device->fence });
        ctx->compute_ctx.reset();

        // Get the results and pass them to the logger
        std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
        VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->query_pool, 0, ctx->query_idx, (cgraph->n_nodes + 1)*పరిమాణం(uint64_t), timestamps.దత్తాంశం(), పరిమాణం(uint64_t), vk::Query_ఫలితం_Flag_బిట్లు::e64 | vk::Query_ఫలితం_Flag_బిట్లు::eWait), "get timestamp results", ctx->device);
        ఒకవేళ (!vk_perf_logger_concurrent) {
            // Log each op separately
            కొరకు (పూర్ణాంకం i = 1; i < ctx->query_idx; i++) {
                స్వయంచాలక కణుపు = ctx->query_nodes[i];
                స్వయంచాలక name = ctx->query_fusion_పేర్లు[i];
                ctx->perf_logger->log_timing(కణుపు, name, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        } లేకపోతే {
            // Log each group of nodes
            పూర్ణాంకం prev_కణుపు_idx = 0;
            కొరకు (పూర్ణాంకం i = 1; i < ctx->query_idx; i++) {
                స్వయంచాలక cur_కణుపు_idx = ctx->query_కణుపు_idx[i];
                std::vector<ggml_tensor *> nodes;
                std::vector<స్థిరరాశి అక్షరం *> పేర్లు;
                కొరకు (పూర్ణాంకం కణుపు_idx = prev_కణుపు_idx; కణుపు_idx < cur_కణుపు_idx; ++కణుపు_idx) {
                    ఒకవేళ (ggml_op_is_ఖాళీలు(cgraph->nodes[కణుపు_idx]->op)) {
                        కొనసాగు;
                    }
                    nodes.push_back(cgraph->nodes[కణుపు_idx]);
                    పేర్లు.push_back(ctx->query_fusion_పేర్లు[కణుపు_idx]);
                    కణుపు_idx += ctx->query_fusion_కణుపు_లెక్క[కణుపు_idx];
                }
                prev_కణుపు_idx = cur_కణుపు_idx;
                ctx->perf_logger->log_timing(nodes, పేర్లు, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        }
        ctx->perf_logger->print_timings();
    }

    ఒకవేళ (!ctx->device->support_async) {
        ggml_vk_synchronize(ctx);
    }

    తిరిగి GGML_స్థితి_SUCCESS;

    వాడని(backend);
}

శూన్య ggml_vk_లేఖాచిత్రం_optimize(ggml_backend_t backend, నిర్మాణం ggml_cgraph * graph, నిర్మాణం ggml_backend_లేఖాచిత్రం_optimize_params * params)
{
    VK_LOG_DEBUG("ggml_vk_graph_optimize(" << graph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ఒకవేళ (ctx->device->disable_లేఖాచిత్రం_optimize) {
        తిరిగి;
    }

    స్వయంచాలక స్థిరరాశి &is_ఖాళీలు = [](స్థిరరాశి ggml_tensor * కణుపు) -> తార్కిక {
        తిరిగి కణుపు->op == GGML_OP_NONE || కణుపు->op == GGML_OP_RESHAPE || కణుపు->op == GGML_OP_TRANSPOSE || కణుపు->op == GGML_OP_VIEW || కణుపు->op == GGML_OP_క్రమమార్పు;
    };

    స్వయంచాలక స్థిరరాశి &is_src_యొక్క = [&is_ఖాళీలు](స్థిరరాశి ggml_tensor *dst, స్థిరరాశి ggml_tensor *src) -> తార్కిక {
        స్వయంచాలక స్థిరరాశి &పునాది = [](స్థిరరాశి ggml_tensor * tensor) {
            తిరిగి tensor->view_src ? tensor->view_src : tensor;
        };
        కొరకు (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            ఒకవేళ (dst->src[s] == src) {
                తిరిగి నిజం;
            }
            ఒకవేళ (is_ఖాళీలు(dst) || is_ఖాళీలు(src)) {
                కొనసాగు;
            }
            // A source view of dst may read storage written through a different view by src.
            ఒకవేళ (dst->src[s] && పునాది(dst->src[s]) == పునాది(src)) {
                తిరిగి నిజం;
            }
            // Moving dst forward may overwrite storage still read through a view by src.
            ఒకవేళ (src->src[s] && పునాది(dst) == పునాది(src->src[s])) {
                తిరిగి నిజం;
            }
        }
        // implicit dependency if they view the same tensor
        ఒకవేళ (పునాది(dst) == పునాది(src)) {
            తిరిగి నిజం;
        }
        తిరిగి అబద్ధం;
    };

    std::vector<ggml_tensor *> new_క్రమం;
    std::vector<తార్కిక> వాడిన(graph->n_nodes, అబద్ధం);
    std::set<ggml_tensor *> వాడిన_కణుపు_సమితి;

    పూర్ణాంకం మొదటి_వాడని = 0;

    // scheduled or zero-compute nodes in [lo, hi)
    స్వయంచాలక స్థిరరాశి &ఖాళీలు_or_scheduled_between = [&](పూర్ణాంకం lo, పూర్ణాంకం hi) -> తార్కిక {
        కొరకు (పూర్ణాంకం v = lo; v < hi; ++v) {
            ఒకవేళ (!వాడిన[v] && !is_ఖాళీలు(graph->nodes[v])) {
                తిరిగి అబద్ధం;
            }
        }
        తిరిగి నిజం;
    };
    అయితే (మొదటి_వాడని < graph->n_nodes) {
        std::vector<పూర్ణాంకం> current_సమితి;

        // Check for fusion patterns and avoid reordering them
        స్వయంచాలక స్థిరరాశి &సరిపోలిక_నమూనా = [&](స్థిరరాశి std::initializer_list<ggml_op> &pattern, పూర్ణాంకం start) -> తార్కిక {
            ఒకవేళ (start + (పూర్ణాంకం)pattern.size() <= graph->n_nodes) {
                తార్కిక is_నమూనా = నిజం;
                కొరకు (size_t j = 0; j < pattern.size(); ++j) {
                    ఒకవేళ (graph->nodes[start + j]->op != pattern.begin()[j] || వాడిన[start + j]) {
                        is_నమూనా = అబద్ధం;
                    }
                }
                తిరిగి is_నమూనా;
            }
            తిరిగి అబద్ధం;
        };

        స్వయంచాలక స్థిరరాశి &keep_నమూనా = [&](స్థిరరాశి std::initializer_list<ggml_op> &pattern) -> తార్కిక {
            ఒకవేళ (సరిపోలిక_నమూనా(pattern, మొదటి_వాడని)) {
                కొరకు (size_t j = 0; j < pattern.size(); ++j) {
                    new_క్రమం.push_back(graph->nodes[మొదటి_వాడని + j]);
                    వాడిన_కణుపు_సమితి.insert(graph->nodes[మొదటి_వాడని + j]);
                    వాడిన[మొదటి_వాడని + j] = నిజం;
                }
                అయితే (మొదటి_వాడని < graph->n_nodes && వాడిన[మొదటి_వాడని]) {
                    మొదటి_వాడని++;
                }
                తిరిగి నిజం;
            }
            తిరిగి అబద్ధం;
        };

        స్వయంచాలక స్థిరరాశి &కూడు_నమూనా_alloc_deps = [&](స్థిరరాశి std::initializer_list<ggml_op> &pattern, పూర్ణాంకం చివరి_కణుపు) {
            // Keep external inputs alive through the fused output.
            std::set<ggml_tensor *> చూసిన;
            కొరకు (size_t j = 0; j < pattern.size(); ++j) {
                ggml_tensor * కణుపు = graph->nodes[మొదటి_వాడని + j];
                కొరకు (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                    ggml_tensor * src = కణుపు->src[s];
                    ఒకవేళ (src && చూసిన.insert(src).second) {
                        params->కూడు_alloc_dep(params->user_దత్తాంశం, src, graph->nodes[చివరి_కణుపు]);
                    }
                }
                చూసిన.insert(కణుపు);
            }
        };

        స్వయంచాలక స్థిరరాశి &keep_topk_moe_నమూనా = [&](స్థిరరాశి std::initializer_list<ggml_op> &pattern) -> తార్కిక {
            ఒకవేళ (!సరిపోలిక_నమూనా(pattern, మొదటి_వాడని)) {
                తిరిగి అబద్ధం;
            }

            పూర్ణాంకం చివరి_కణుపు = మొదటి_వాడని + (పూర్ణాంకం) pattern.size() - 1;
            // Some TOPK_MOE variants fuse a trailing scale.
            ఒకవేళ (చివరి_కణుపు + 1 < graph->n_nodes && graph->nodes[చివరి_కణుపు + 1]->op == GGML_OP_SCALE) {
                చివరి_కణుపు++;
            }

            కూడు_నమూనా_alloc_deps(pattern, చివరి_కణుపు);

            తిరిగి keep_నమూనా(pattern);
        };

        ఒకవేళ (keep_topk_moe_నమూనా(topk_moe_early_softmax_norm)) {
            కొనసాగు;
        }
        ఒకవేళ (keep_topk_moe_నమూనా(topk_moe_sigmoid_norm_bias)) {
            కొనసాగు;
        }
        ఒకవేళ (keep_topk_moe_నమూనా(topk_moe_sqrt_softplus_norm_bias)) {
            కొనసాగు;
        }
        ఒకవేళ (keep_topk_moe_నమూనా(topk_moe_early_softmax)) {
            కొనసాగు;
        }
        ఒకవేళ (keep_topk_moe_నమూనా(topk_moe_late_softmax)) {
            కొనసాగు;
        }
        ఒకవేళ (keep_నమూనా(snake_నమూనా)) {
            కొనసాగు;
        }
        ఒకవేళ (keep_నమూనా(topk_qsa_నమూనా)) {
            కొనసాగు;
        }

        ఒకవేళ (keep_నమూనా(rms_norm_mul_కూడు_mul_నమూనా)) {
            కొనసాగు;
        }
        ఒకవేళ (keep_నమూనా(rms_norm_mul_కూడు_నమూనా)) {
            కొనసాగు;
        }
        ఒకవేళ (keep_నమూనా(rms_norm_mul_rope_view_సమితి_అడ్డువరుసలు_నమూనా)) {
            కొనసాగు;
        }
        ఒకవేళ (keep_నమూనా(rms_norm_view_సమితి_అడ్డువరుసలు_నమూనా)) {
            కొనసాగు;
        }
        ఒకవేళ (keep_నమూనా(rope_view_సమితి_అడ్డువరుసలు_నమూనా)) {
            కొనసాగు;
        }
        ఒకవేళ (సరిపోలిక_నమూనా(hc_post_gate_నమూనా, మొదటి_వాడని)) {
            కూడు_నమూనా_alloc_deps(hc_post_gate_నమూనా, మొదటి_వాడని + (పూర్ణాంకం) hc_post_gate_నమూనా.size() - 1);
            keep_నమూనా(hc_post_gate_నమూనా);
            కొనసాగు;
        }

        // First, grab the next unused node.
        current_సమితి.push_back(మొదటి_వాడని);

        // Loop through the next N nodes. Grab any that don't depend on other nodes that
        // haven't already been run. Nodes that have already been run have used[i] set
        // to true. Allow nodes that depend on the previous node if it's a fusion pattern
        // that we support (e.g. RMS_NORM + MUL).
        // This first pass only grabs "real" (non-view nodes). Second pass grabs view nodes.
        // The goal is to not interleave real and view nodes in a way that breaks fusion.
        స్థిరరాశి పూర్ణాంకం అంకె_గమ్యం_CHECK = 20;
        కొరకు (పూర్ణాంకం j = మొదటి_వాడని+1; j < std::min(మొదటి_వాడని + అంకె_గమ్యం_CHECK, graph->n_nodes); ++j) {
            ఒకవేళ (వాడిన[j]) {
                కొనసాగు;
            }
            ఒకవేళ (is_ఖాళీలు(graph->nodes[j])) {
                కొనసాగు;
            }
            // Protect every interior QSA node (not just the start): the mask branch is
            // independent, so it gets pulled out and breaks keep_pattern otherwise.
            స్వయంచాలక స్థిరరాశి &లోపల_qsa_నమూనా = [&](పూర్ణాంకం n) -> తార్కిక {
                కొరకు (పూర్ణాంకం o = 0; o < (పూర్ణాంకం) topk_qsa_నమూనా.size(); ++o) {
                    ఒకవేళ (n - o >= 0 && సరిపోలిక_నమూనా(topk_qsa_నమూనా, n - o)) {
                        తిరిగి నిజం;
                    }
                }
                తిరిగి అబద్ధం;
            };
            ఒకవేళ (సరిపోలిక_నమూనా(topk_moe_early_softmax_norm, j) ||
                సరిపోలిక_నమూనా(topk_moe_sigmoid_norm_bias, j) ||
                సరిపోలిక_నమూనా(topk_moe_sqrt_softplus_norm_bias, j) ||
                సరిపోలిక_నమూనా(topk_moe_early_softmax, j) ||
                సరిపోలిక_నమూనా(topk_moe_late_softmax, j) ||
                సరిపోలిక_నమూనా(snake_నమూనా, j) ||
                లోపల_qsa_నమూనా(j) ||
                సరిపోలిక_నమూనా(rms_norm_mul_కూడు_mul_నమూనా, j) ||
                సరిపోలిక_నమూనా(rms_norm_mul_కూడు_నమూనా, j) ||
                సరిపోలిక_నమూనా(rms_norm_mul_rope_view_సమితి_అడ్డువరుసలు_నమూనా, j) ||
                సరిపోలిక_నమూనా(rms_norm_view_సమితి_అడ్డువరుసలు_నమూనా, j) ||
                సరిపోలిక_నమూనా(rope_view_సమితి_అడ్డువరుసలు_నమూనా, j) ||
                సరిపోలిక_నమూనా(hc_post_gate_నమూనా, j)) {
                కొనసాగు;
            }
            తార్కిక ok = నిజం;
            కొరకు (పూర్ణాంకం c = మొదటి_వాడని; c < j; ++c) {
                ఒకవేళ (!వాడిన[c] &&
                    is_src_యొక్క(graph->nodes[j], graph->nodes[c]) &&
                    !(c == current_సమితి.back() && graph->nodes[c]->op == GGML_OP_RMS_NORM && graph->nodes[j]->op == GGML_OP_MUL && ఖాళీలు_or_scheduled_between(c+1, j)) &&
                    !(c == current_సమితి.back() && graph->nodes[c]->op == GGML_OP_UNARY && graph->nodes[j]->op == GGML_OP_MUL && ఖాళీలు_or_scheduled_between(c+1, j)) &&
                    !(j == c+1 && c == current_సమితి.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT && graph->nodes[j]->op == GGML_OP_కూడు) &&
                    !(j == c+1 && c == current_సమితి.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_కూడు_ID) &&
                    !(j == c+1 && c == current_సమితి.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_సమితి.back() && graph->nodes[c]->op == GGML_OP_కూడు && graph->nodes[j]->op == GGML_OP_కూడు) &&
                    !(j == c+1 && c == current_సమితి.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_కూడు) &&
                    !(j == c+1 && c == current_సమితి.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_UNARY)) {
                    ok = అబద్ధం;
                    ఆపు;
                }
            }
            ఒకవేళ (ok) {
                current_సమితి.push_back(j);

                పూర్ణాంకం rope_idx = j;

                // When we've found RMS_NORM + MUL, try to find a ROPE that uses it
                ఒకవేళ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_MUL &&
                    graph->nodes[j-1]->op == GGML_OP_RMS_NORM) {
                    కొరకు (పూర్ణాంకం k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ఒకవేళ (graph->nodes[k]->op == GGML_OP_ROPE &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // Check that other srcs are already valid
                            graph->nodes[k]->src[1]->op == GGML_OP_NONE &&
                            (graph->nodes[k]->src[2] == శూన్యసూచిక || graph->nodes[k]->src[2]->op == GGML_OP_NONE)) {
                            rope_idx = k;
                            current_సమితి.push_back(rope_idx);
                            వాడిన[rope_idx] = నిజం;
                            ఆపు;
                        }
                    }
                }
                // Look for ROPE/RMS_NORM + VIEW + SET_ROWS and make them consecutive
                ఒకవేళ (graph->nodes[rope_idx]->op == GGML_OP_ROPE || graph->nodes[rope_idx]->op == GGML_OP_RMS_NORM) {
                    పూర్ణాంకం view_idx = -1;
                    పూర్ణాంకం సమితి_అడ్డువరుసలు_idx = -1;
                    కొరకు (పూర్ణాంకం k = rope_idx + 1; k < std::min(rope_idx + 15, graph->n_nodes); ++k) {
                        ఒకవేళ (వాడిన[k]) {
                            కొనసాగు;
                        }
                        ఒకవేళ (view_idx == -1 && graph->nodes[k]->op == GGML_OP_VIEW && graph->nodes[k]->src[0] == graph->nodes[rope_idx]) {
                            view_idx = k;
                            కొనసాగు;
                        }
                        ఒకవేళ (view_idx != -1 && graph->nodes[k]->op == GGML_OP_సమితి_అడ్డువరుసలు && graph->nodes[k]->src[0] == graph->nodes[view_idx]) {
                            సమితి_అడ్డువరుసలు_idx = k;
                            ఆపు;
                        }
                    }
                    ఒకవేళ (సమితి_అడ్డువరుసలు_idx != -1) {
                        స్థిరరాశి పూర్ణాంకం కణుపు_idxs[] = { rope_idx, view_idx, సమితి_అడ్డువరుసలు_idx };
                        స్థిరరాశి ggml_op ops[] = { graph->nodes[rope_idx]->op, GGML_OP_VIEW, GGML_OP_సమితి_అడ్డువరుసలు };
                        తార్కిక can_pull = ggml_can_fuse_subgraph_ext(graph, కణుపు_idxs, 3, ops, &సమితి_అడ్డువరుసలు_idx, 1);

                        కొరకు (పూర్ణాంకం c = rope_idx + 1; can_pull && c < సమితి_అడ్డువరుసలు_idx; ++c) {
                            ఒకవేళ (!వాడిన[c] && c != view_idx && !is_ఖాళీలు(graph->nodes[c]) &&
                                is_src_యొక్క(graph->nodes[సమితి_అడ్డువరుసలు_idx], graph->nodes[c])) {
                                can_pull = అబద్ధం;
                            }
                        }

                        ఒకవేళ (can_pull) {
                            current_సమితి.push_back(view_idx);
                            current_సమితి.push_back(సమితి_అడ్డువరుసలు_idx);
                            వాడిన[view_idx] = నిజం;
                            వాడిన[సమితి_అడ్డువరుసలు_idx] = నిజం;
                        }
                    }
                }
                // Look for MUL_MAT_ID + ADD_ID + MUL
                ఒకవేళ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_కూడు_ID &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT_ID) {
                    కొరకు (పూర్ణాంకం k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ఒకవేళ (graph->nodes[k]->op == GGML_OP_MUL &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || వాడిన_కణుపు_సమితి.find(graph->nodes[k]->src[1]) != వాడిన_కణుపు_సమితి.end())) {
                            current_సమితి.push_back(k);
                            వాడిన[k] = నిజం;
                            ఆపు;
                        }
                    }
                }
                // Look for MUL_MAT + ADD + ADD
                ఒకవేళ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_కూడు &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT) {
                    కొరకు (పూర్ణాంకం k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ఒకవేళ (graph->nodes[k]->op == GGML_OP_కూడు &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || వాడిన_కణుపు_సమితి.find(graph->nodes[k]->src[1]) != వాడిన_కణుపు_సమితి.end())) {
                            current_సమితి.push_back(k);
                            వాడిన[k] = నిజం;
                            ఆపు;
                        }
                    }
                }
                // SSM_CONV + ADD + UNARY: pull the consuming UNARY forward
                ఒకవేళ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_కూడు &&
                    graph->nodes[j-1]->op == GGML_OP_SSM_CONV) {
                    కొరకు (పూర్ణాంకం k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ఒకవేళ (graph->nodes[k]->op == GGML_OP_UNARY &&
                            graph->nodes[k]->src[0] == graph->nodes[j]) {
                            current_సమితి.push_back(k);
                            వాడిన[k] = నిజం;
                            ఆపు;
                        }
                    }
                }
                // UNARY + MUL: pull the consuming MUL forward
                ఒకవేళ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_UNARY) {
                    కొరకు (పూర్ణాంకం k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ggml_tensor * mul = graph->nodes[k];
                        ఒకవేళ (mul->op != GGML_OP_MUL || (mul->src[0] != graph->nodes[j] && mul->src[1] != graph->nodes[j])) {
                            కొనసాగు;
                        }
                        ggml_tensor * other = (mul->src[0] == graph->nodes[j]) ? mul->src[1] : mul->src[0];
                        // the other src must either be weights or already processed
                        ఒకవేళ (!(other->op == GGML_OP_NONE || వాడిన_కణుపు_సమితి.find(other) != వాడిన_కణుపు_సమితి.end())) {
                            కొనసాగు;
                        }
                        ఒకవేళ (!ggml_vk_can_fuse_unary_mul(graph, j, k)) {
                            కొనసాగు;
                        }
                        current_సమితి.push_back(k);
                        వాడిన[k] = నిజం;
                        ఆపు;
                    }
                }
            }
        }
        // Second pass grabs view nodes.
        // Skip this if it would break a fusion optimization (don't split up add->rms_norm or add->add).
        ఒకవేళ (graph->nodes[current_సమితి.back()]->op != GGML_OP_కూడు) {
            కొరకు (పూర్ణాంకం j = మొదటి_వాడని+1; j < std::min(మొదటి_వాడని + అంకె_గమ్యం_CHECK, graph->n_nodes); ++j) {
                ఒకవేళ (వాడిన[j]) {
                    కొనసాగు;
                }
                ఒకవేళ (!is_ఖాళీలు(graph->nodes[j])) {
                    కొనసాగు;
                }
                తార్కిక ok = నిజం;
                కొరకు (పూర్ణాంకం c = మొదటి_వాడని; c < j; ++c) {
                    తార్కిక c_లోపల_current_సమితి = std::find(current_సమితి.begin(), current_సమితి.end(), c) != current_సమితి.end();
                    // skip views whose srcs haven't been processed.
                    ఒకవేళ (!వాడిన[c] &&
                        is_src_యొక్క(graph->nodes[j], graph->nodes[c]) &&
                        !c_లోపల_current_సమితి) {
                        ok = అబద్ధం;
                        ఆపు;
                    }
                }
                ఒకవేళ (ok) {
                    current_సమితి.push_back(j);
                }
            }
        }

        // Push the current set into new_order
        కొరకు (స్వయంచాలక c : current_సమితి) {
            new_క్రమం.push_back(graph->nodes[c]);
            వాడిన_కణుపు_సమితి.insert(graph->nodes[c]);
            వాడిన[c] = నిజం;
        }
        అయితే (మొదటి_వాడని < graph->n_nodes && వాడిన[మొదటి_వాడని]) {
            మొదటి_వాడని++;
        }
    }
    // Replace the graph with the new order.
    కొరకు (పూర్ణాంకం i = 0; i < graph->n_nodes; ++i) {
        graph->nodes[i] = new_క్రమం[i];
    }
}

స్థిర శూన్య ggml_backend_vk_event_రికార్డు(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_record(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
    స్వయంచాలక* cmd_buf = compute_ctx->s->బఫరు; // retrieve pointer before it gets reset

    ఒకవేళ (vkev->has_event) {
        // Move existing event into submitted
        vkev->events_submitted.push_back(vkev->event);
    }

    // Grab the next event and record it, create one if necessary
    ఒకవేళ (vkev->events_free.ఖాళీలు()) {
        vkev->event = ctx->device->device.createEvent({});
    } లేకపోతే {
        vkev->event = vkev->events_free.back();
        vkev->events_free.pop_back();
    }

    vkev->has_event = నిజం;

    ggml_vk_సమితి_event(compute_ctx, vkev->event);

    vkev->tl_semaphore.value++;
    compute_ctx->s->signal_semaphores.push_back(vkev->tl_semaphore);
    ggml_vk_ctx_చివర(compute_ctx);

    ggml_vk_submit(compute_ctx, {});
    ctx->submit_pending = నిజం;
    vkev->cmd_బఫరు = cmd_buf;
    vkev->cmd_బఫరు_use_counter = cmd_buf->use_counter;
    ctx->compute_ctx.reset();
}

స్థిర శూన్య ggml_backend_vk_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_wait(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    ఒకవేళ (vkev->has_event) {
        // Wait for latest event
        ggml_vk_wait_events(compute_ctx, { vkev->event });

        ఒకవేళ (ctx->device->async_use_transfer_వరుస) {
            vk_context transfer_ctx = ggml_vk_get_transfer_ctx(ctx);
            transfer_ctx->s->wait_semaphores.push_back(vkev->tl_semaphore);
        }
    }
}

స్థిర ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .set_tensor_async        = */ ggml_backend_vk_సమితి_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_vk_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_vk_సమితి_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_vk_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ ఖాళీ,
    /* .graph_plan_free         = */ ఖాళీ,
    /* .graph_plan_update       = */ ఖాళీ,
    /* .graph_plan_compute      = */ ఖాళీ,
    /* .graph_compute           = */ ggml_backend_vk_లేఖాచిత్రం_compute,
    /* .event_record            = */ ggml_backend_vk_event_రికార్డు,
    /* .event_wait              = */ ggml_backend_vk_event_wait,
    /* .graph_optimize          = */ ggml_vk_లేఖాచిత్రం_optimize,
};

స్థిర ggml_guid_t ggml_backend_vk_guid() {
    స్థిర ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    తిరిగి &guid;
}

ggml_backend_t ggml_backend_vk_init(size_t dev_అంకె) {
    VK_LOG_DEBUG("ggml_backend_vk_init(" << dev_అంకె << ")");

    ggml_backend_vk_context * ctx = కొత్త ggml_backend_vk_context;
    ggml_vk_init(ctx, dev_అంకె);

    ggml_backend_t vk_backend = కొత్త ggml_backend {
        /* .guid    = */ ggml_backend_vk_guid(),
        /* .iface   = */ ggml_backend_vk_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), dev_అంకె),
        /* .context = */ ctx,
    };

    ఒకవేళ (!ctx->device->support_async) {
        vk_backend->iface.get_tensor_async = శూన్యసూచిక;
    }

    తిరిగి vk_backend;
}

తార్కిక ggml_backend_is_vk(ggml_backend_t backend) {
    తిరిగి backend != ఖాళీ && ggml_guid_matches(backend->guid, ggml_backend_vk_guid());
}

పూర్ణాంకం ggml_backend_vk_get_device_లెక్క() {
    తిరిగి ggml_vk_get_device_లెక్క();
}

శూన్య ggml_backend_vk_get_device_description(పూర్ణాంకం device, అక్షరం * description, size_t description_కొలత) {
    GGML_ASSERT(device < (పూర్ణాంకం) vk_instance.device_indices.size());
    పూర్ణాంకం dev_idx = vk_instance.device_indices[device];
    ggml_vk_get_device_description(dev_idx, description, description_కొలత);
}

శూన్య ggml_backend_vk_get_device_జ్ఞాపకం(పూర్ణాంకం device, size_t * విడుదల, size_t * మొత్తం) {
    GGML_ASSERT(device < (పూర్ణాంకం) vk_instance.device_indices.size());
    GGML_ASSERT(device < (పూర్ణాంకం) vk_instance.device_supports_membudget.size());

    vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];
    vk::Physical_Device_జ్ఞాపకం_Budget_Properties_EXT budgetprops;
    vk::Physical_Device_జ్ఞాపకం_Properties2 memprops = {};
    స్థిరరాశి తార్కిక membudget_supported = vk_instance.device_supports_membudget[device];
    స్థిరరాశి తార్కిక is_integrated_gpu = vkdev.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    ఒకవేళ (membudget_supported) {
        memprops.p_తదుపరి = &budgetprops;
    }
    vkdev.get_జ్ఞాపకం_Properties2(&memprops);

    *మొత్తం = 0;
    *విడుదల = 0;

    కొరకు (uint32_t i = 0; i < memprops.జ్ఞాపకం_Properties.జ్ఞాపకం_కుప్ప_లెక్క; ++i) {
        స్థిరరాశి vk::జ్ఞాపకం_కుప్ప & కుప్ప = memprops.జ్ఞాపకం_Properties.జ్ఞాపకం_Heaps[i];

        ఒకవేళ (is_integrated_gpu || (కుప్ప.flags & vk::జ్ఞాపకం_కుప్ప_Flag_బిట్లు::eDeviceLocal)) {
            *మొత్తం += కుప్ప.size;

            ఒకవేళ (membudget_supported && i < budgetprops.కుప్ప_Usage.size()) {
                // Driver may report heapBudget > heap.size so we add a safeguard here
                // See https://github.com/ggml-org/llama.cpp/issues/29277
                స్థిరరాశి vk::Device_కొలత budget = std::min(budgetprops.కుప్ప_Budget[i], కుప్ప.size);
                *విడుదల += budget > budgetprops.కుప్ప_Usage[i] ? budget - budgetprops.కుప్ప_Usage[i] : 0;
            } లేకపోతే {
                *విడుదల += కుప్ప.size;
            }
        }
    }
}

స్థిర vk::PhysicalDeviceType ggml_backend_vk_get_device_type(పూర్ణాంకం device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (పూర్ణాంకం) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    vk::PhysicalDeviceProperties2 props = {};
    device.getProperties2(&props);

    తిరిగి props.properties.deviceType;
}

స్థిర std::string ggml_backend_vk_get_device_pci_id(పూర్ణాంకం device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (పూర్ణాంకం) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    స్థిరరాశి std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

    తార్కిక ext_support = అబద్ధం;

    కొరకు (స్థిరరాశి స్వయంచాలక& properties : ext_props) {
        ఒకవేళ (strcmp("VK_EXT_pci_bus_info", properties.extensionName) == 0) {
            ext_support = నిజం;
            ఆపు;
        }
    }

    ఒకవేళ (!ext_support) {
        తిరిగి "";
    }

    vk::PhysicalDeviceProperties2 props = {};
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pci_bus_info = {};

    props.p_తదుపరి = &pci_bus_info;

    device.getProperties2(&props);

    స్థిరరాశి uint32_t pci_domain = pci_bus_info.pciDomain;
    స్థిరరాశి uint32_t pci_bus = pci_bus_info.pciBus;
    స్థిరరాశి uint32_t pci_device = pci_bus_info.pciDevice;
    స్థిరరాశి uint8_t pci_function = (uint8_t) pci_bus_info.pciFunction; // pci function is between 0 and 7, prevent printf overflow warning

    అక్షరం pci_bus_id[16] = {};
    snprintf(pci_bus_id, పరిమాణం(pci_bus_id), "%04x:%02x:%02x.%x", pci_domain, pci_bus, pci_device, pci_function);

    తిరిగి std::string(pci_bus_id);
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_vk_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    తిరిగి ctx->name.c_str();
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_vk_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    తిరిగి ctx->description.c_str();
}

స్థిర శూన్య ggml_backend_vk_device_get_జ్ఞాపకం(ggml_backend_dev_t device, size_t * విడుదల, size_t * మొత్తం) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)device->context;
    ggml_backend_vk_get_device_జ్ఞాపకం(ctx->device, విడుదల, మొత్తం);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_vk_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    తిరిగి ggml_backend_vk_బఫరు_type(ctx->device);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_vk_device_get_host_బఫరు_type(ggml_backend_dev_t dev) {
    వాడని(dev);
    తిరిగి ggml_backend_vk_host_బఫరు_type();
}

స్థిర గణన ggml_backend_dev_type ggml_backend_vk_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    తిరిగి ctx->is_integrated_gpu ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

స్థిర శూన్య ggml_backend_vk_device_get_props(ggml_backend_dev_t dev, నిర్మాణం ggml_backend_dev_props * props) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    props->name        = ggml_backend_vk_device_get_name(dev);
    props->description = ggml_backend_vk_device_get_description(dev);
    props->type        = ggml_backend_vk_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.ఖాళీలు() ? శూన్యసూచిక : ctx->pci_bus_id.c_str();
    ggml_backend_vk_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);
    props->caps = {
        /* .async                 = */ నిజం,
        /* .host_buffer           = */ నిజం,
        /* .buffer_from_host_ptr  = */ అబద్ధం,
        /* .events                = */ నిజం,
        /* .mmap_support          = */ !ctx->is_integrated_gpu,
    };
}

స్థిర ggml_backend_t ggml_backend_vk_device_init(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    వాడని(params);
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    తిరిగి ggml_backend_vk_init(ctx->device);
}

స్థిర తార్కిక ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    స్థిరరాశి vk_device& device = ggml_vk_get_device(ctx->device);

    స్థిరరాశి తార్కిక ఉపయోగిస్తుంది_bda = (op->op == GGML_OP_IM2_నిలువు || op->op == GGML_OP_IM2_నిలువు_3_D) &&
                          device->shader_int64 && device->బఫరు_device_చిరునామా;

    స్వయంచాలక స్థిరరాశి & tensor_కొలత_supported = [&](size_t tensor_కొలత) {
        ఒకవేళ (tensor_కొలత > device->max_బఫరు_కొలత) {
            తిరిగి అబద్ధం;
        }
        // For im2col shaders using BDA, maxStorageBufferRange limit doesn't apply.
        // If shader64BitIndexing is enabled, maxStorageBufferRange limit doesn't apply.
        ఒకవేళ (!ఉపయోగిస్తుంది_bda && !device->shader_64b_indexing) {
            ఒకవేళ (tensor_కొలత > device->properties.limits.max_Storage_బఫరు_Range) {
                తిరిగి అబద్ధం;
            }
        }
        తిరిగి నిజం;
    };
    // reject any tensors larger than the max buffer size
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        ఒకవేళ (op->src[i] && !tensor_కొలత_supported(ggml_nbytes(op->src[i]))) {
            తిరిగి అబద్ధం;
        }
    }
    ఒకవేళ (!tensor_కొలత_supported(ggml_nbytes(op))) {
        తిరిగి అబద్ధం;
    }

    ఎంపిక (op->op) {
        స్థితి GGML_OP_UNARY:
            ఎంపిక (ggml_get_unary_op(op)) {
                స్థితి GGML_UNARY_OP_EXP:
                స్థితి GGML_UNARY_OP_EXPM1:
                స్థితి GGML_UNARY_OP_ELU:
                స్థితి GGML_UNARY_OP_GELU:
                స్థితి GGML_UNARY_OP_GELU_ERF:
                స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:
                స్థితి GGML_UNARY_OP_SILU:
                స్థితి GGML_UNARY_OP_RELU:
                స్థితి GGML_UNARY_OP_XIELU:
                స్థితి GGML_UNARY_OP_NEG:
                స్థితి GGML_UNARY_OP_TANH:
                స్థితి GGML_UNARY_OP_SIGMOID:
                స్థితి GGML_UNARY_OP_HARDSIGMOID:
                స్థితి GGML_UNARY_OP_HARDSWISH:
                స్థితి GGML_UNARY_OP_ABS:
                స్థితి GGML_UNARY_OP_SOFTPLUS:
                స్థితి GGML_UNARY_OP_అడుగు:
                స్థితి GGML_UNARY_OP_విడత:
                స్థితి GGML_UNARY_OP_CEIL:
                స్థితి GGML_UNARY_OP_FLOOR:
                స్థితి GGML_UNARY_OP_TRUNC:
                స్థితి GGML_UNARY_OP_SGN:
                    తిరిగి (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                అప్రమేయ:
                    తిరిగి అబద్ధం;
            }
        స్థితి GGML_OP_GLU:
            ఎంపిక (ggml_get_glu_op(op)) {
                స్థితి GGML_GLU_OP_GEGLU:
                స్థితి GGML_GLU_OP_REGLU:
                స్థితి GGML_GLU_OP_SWIGLU:
                స్థితి GGML_GLU_OP_SWIGLU_OAI:
                స్థితి GGML_GLU_OP_GEGLU_ERF:
                స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
                స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
                    తిరిగి (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type) &&
                           (!op->src[1] || op->src[1]->type == op->src[0]->type);
                అప్రమేయ:
                    తిరిగి అబద్ధం;
            }
        స్థితి GGML_OP_MUL_MAT:
        స్థితి GGML_OP_MUL_MAT_ID:
            {
                ggml_type src0_type = op->src[0]->type;
                ఒకవేళ (op->op == GGML_OP_MUL_MAT_ID) {
                    ఒకవేళ (!ggml_vk_use_mul_mat_vec_id(op)) {
                        స్థిరరాశి ggml_tensor * ids = op->src[2];
                        // Shaders store row IDs as uint16_t slot and token indices.
                        ఒకవేళ (ids->ne[0] > int64_t(UINT16_MAX) + 1 || ids->ne[1] > int64_t(UINT16_MAX) + 1) {
                            తిరిగి అబద్ధం;
                        }
                    }
                    ఒకవేళ (!device->mul_mat_id_s[src0_type] && !device->mul_mat_id_m[src0_type] && !device->mul_mat_id_l[src0_type]) {
                        // If there's not enough shared memory for row_ids and the result tile, fallback to CPU
                        తిరిగి అబద్ధం;
                    }
                    ఒకవేళ (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                        తిరిగి అబద్ధం;
                    }
                }
                ఎంపిక (src0_type) {
                    స్థితి GGML_TYPE_F32:
                    స్థితి GGML_TYPE_F16:
                    స్థితి GGML_TYPE_BF16:
                    స్థితి GGML_TYPE_Q1_0:
                    స్థితి GGML_TYPE_Q2_0:
                    స్థితి GGML_TYPE_Q4_0:
                    స్థితి GGML_TYPE_Q4_1:
                    స్థితి GGML_TYPE_Q5_0:
                    స్థితి GGML_TYPE_Q5_1:
                    స్థితి GGML_TYPE_Q8_0:
                    స్థితి GGML_TYPE_Q2_K:
                    స్థితి GGML_TYPE_Q3_K:
                    స్థితి GGML_TYPE_Q4_K:
                    స్థితి GGML_TYPE_Q5_K:
                    స్థితి GGML_TYPE_Q6_K:
                    స్థితి GGML_TYPE_IQ1_S:
                    స్థితి GGML_TYPE_IQ1_M:
                    స్థితి GGML_TYPE_IQ2_XXS:
                    స్థితి GGML_TYPE_IQ2_XS:
                    స్థితి GGML_TYPE_IQ2_S:
                    స్థితి GGML_TYPE_IQ3_XXS:
                    స్థితి GGML_TYPE_IQ3_S:
                    స్థితి GGML_TYPE_IQ4_XS:
                    స్థితి GGML_TYPE_IQ4_NL:
                    స్థితి GGML_TYPE_MXFP4:
                    స్థితి GGML_TYPE_NVFP4:
                    స్థితి GGML_TYPE_TQ1_0:
                    స్థితి GGML_TYPE_TQ2_0:
                        ఆపు;
                    అప్రమేయ:
                        తిరిగి అబద్ధం;
                }
                నిర్మాణం ggml_tensor * a;
                నిర్మాణం ggml_tensor * b;
                ఒకవేళ (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } లేకపోతే {
                    a = op->src[2];
                    b = op->src[1];
                }
                ఒకవేళ (a->ne[3] != b->ne[3]) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (!(ggml_vk_dim01_contiguous(op->src[0]) || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) ||
                    !(ggml_vk_dim01_contiguous(op->src[1]) || op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16)) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F16) {
                    // We currently don't have a bf16 x f16 shader, or an fp16->bf16 copy shader.
                    // So don't support this combination for now.
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->src[1]->type == GGML_TYPE_BF16 && op->src[0]->type != GGML_TYPE_BF16 && op->op != GGML_OP_MUL_MAT) {
                    // BF16 in src1 (widened to F32/F16) is only wired up for MUL_MAT
                    తిరిగి అబద్ధం;
                }

                తిరిగి నిజం;
            }
        స్థితి GGML_OP_FLASH_ATTN_EXT:
            {
                తార్కిక coopmat2 = device->coopmat2;
                uint32_t HSK = op->src[1]->ne[0];
                uint32_t HSV = op->src[2]->ne[0];
                ఒకవేళ ((HSK % 8) != 0 || (HSV % 8) != 0) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->src[4] && op->src[4]->type != GGML_TYPE_F32) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->src[0]->type != GGML_TYPE_F32) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->type != GGML_TYPE_F32) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->src[3] && op->src[3]->type != GGML_TYPE_F16) {
                    తిరిగి అబద్ధం;
                }
                స్వయంచాలక fa_kv_ok = [](ggml_type t) {
                    ఎంపిక (t) {
                    స్థితి GGML_TYPE_F32:
                    స్థితి GGML_TYPE_F16:
                    స్థితి GGML_TYPE_BF16:
                    స్థితి GGML_TYPE_Q8_0:
                    స్థితి GGML_TYPE_Q5_1:
                    స్థితి GGML_TYPE_Q5_0:
                    స్థితి GGML_TYPE_Q4_1:
                    స్థితి GGML_TYPE_Q4_0:
                    స్థితి GGML_TYPE_IQ4_NL:
                        తిరిగి నిజం;
                    అప్రమేయ:
                        తిరిగి అబద్ధం;
                    }
                };
                ఒకవేళ (!fa_kv_ok(op->src[1]->type) || !fa_kv_ok(op->src[2]->type)) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ ((op->src[1]->type == GGML_TYPE_BF16) != (op->src[2]->type == GGML_TYPE_BF16)) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (!coopmat2 && !(device->subgroup_shuffle && device->subgroup_vote)) {
                    // scalar/coopmat1 FA uses subgroupShuffle/subgroupAll
                    తిరిగి అబద్ధం;
                }
                తిరిగి నిజం;
            }
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            {
                ఎంపిక (op->src[0]->type) {
                    స్థితి GGML_TYPE_F32:
                    స్థితి GGML_TYPE_F16:
                    స్థితి GGML_TYPE_BF16:
                    స్థితి GGML_TYPE_Q1_0:
                    స్థితి GGML_TYPE_Q2_0:
                    స్థితి GGML_TYPE_Q4_0:
                    స్థితి GGML_TYPE_Q4_1:
                    స్థితి GGML_TYPE_Q5_0:
                    స్థితి GGML_TYPE_Q5_1:
                    స్థితి GGML_TYPE_Q8_0:
                    స్థితి GGML_TYPE_Q2_K:
                    స్థితి GGML_TYPE_Q3_K:
                    స్థితి GGML_TYPE_Q4_K:
                    స్థితి GGML_TYPE_Q5_K:
                    స్థితి GGML_TYPE_Q6_K:
                    స్థితి GGML_TYPE_IQ1_S:
                    స్థితి GGML_TYPE_IQ1_M:
                    స్థితి GGML_TYPE_IQ2_XXS:
                    స్థితి GGML_TYPE_IQ2_XS:
                    స్థితి GGML_TYPE_IQ2_S:
                    స్థితి GGML_TYPE_IQ3_XXS:
                    స్థితి GGML_TYPE_IQ3_S:
                    స్థితి GGML_TYPE_IQ4_XS:
                    స్థితి GGML_TYPE_IQ4_NL:
                    స్థితి GGML_TYPE_MXFP4:
                    స్థితి GGML_TYPE_NVFP4:
                    స్థితి GGML_TYPE_TQ1_0:
                    స్థితి GGML_TYPE_TQ2_0:
                    స్థితి GGML_TYPE_I32:
                        తిరిగి నిజం;
                    అప్రమేయ:
                        తిరిగి అబద్ధం;
                }
            }
        స్థితి GGML_OP_GET_అడ్డువరుసలు_వెనుక:
            తిరిగి op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            {
                ఒకవేళ ((op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_F16) ||
                    (op->src[1]->type != GGML_TYPE_I32 && op->src[1]->type != GGML_TYPE_I64)) {
                    తిరిగి అబద్ధం;
                }
                ఎంపిక (op->type) {
                    స్థితి GGML_TYPE_F32:
                    స్థితి GGML_TYPE_F16:
                    స్థితి GGML_TYPE_BF16:
                    స్థితి GGML_TYPE_Q1_0:
                    స్థితి GGML_TYPE_Q2_0:
                    స్థితి GGML_TYPE_Q4_0:
                    స్థితి GGML_TYPE_Q4_1:
                    స్థితి GGML_TYPE_Q5_0:
                    స్థితి GGML_TYPE_Q5_1:
                    స్థితి GGML_TYPE_Q8_0:
                    స్థితి GGML_TYPE_IQ4_NL:
                        తిరిగి నిజం;
                    అప్రమేయ:
                        తిరిగి అబద్ధం;
                }
            }
        స్థితి GGML_OP_CONT:
        స్థితి GGML_OP_CPY:
        స్థితి GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1] != శూన్యసూచిక ? op->src[1]->type : src0_type;

                ఒకవేళ (src0_type == GGML_TYPE_F32) {
                    ఎంపిక (src1_type) {
                    స్థితి GGML_TYPE_F32:
                    స్థితి GGML_TYPE_F16:
                    స్థితి GGML_TYPE_BF16:
                    స్థితి GGML_TYPE_Q1_0:
                    స్థితి GGML_TYPE_Q2_0:
                    స్థితి GGML_TYPE_Q4_0:
                    స్థితి GGML_TYPE_Q4_1:
                    స్థితి GGML_TYPE_Q5_0:
                    స్థితి GGML_TYPE_Q5_1:
                    స్థితి GGML_TYPE_Q8_0:
                    స్థితి GGML_TYPE_IQ4_NL:
                        తిరిగి నిజం;
                    అప్రమేయ:
                        ఆపు;
                    }
                }
                ఒకవేళ (src1_type == GGML_TYPE_F32) {
                    ఎంపిక (src0_type) {
                    స్థితి GGML_TYPE_F16:
                    స్థితి GGML_TYPE_BF16:
                    స్థితి GGML_TYPE_Q1_0:
                    స్థితి GGML_TYPE_Q2_0:
                    స్థితి GGML_TYPE_Q4_0:
                    స్థితి GGML_TYPE_Q4_1:
                    స్థితి GGML_TYPE_Q5_0:
                    స్థితి GGML_TYPE_Q5_1:
                    స్థితి GGML_TYPE_Q8_0:
                    స్థితి GGML_TYPE_IQ4_NL:
                        తిరిగి నిజం;
                    అప్రమేయ:
                        ఆపు;
                    }
                }

                ఒకవేళ (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    తిరిగి నిజం;
                }

                ఒకవేళ (
                    (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) ||
                    (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32)
                ) {
                    తిరిగి నిజం;
                }

                // We can handle copying from a type to the same type if it's
                // either not quantized or is quantized and contiguous.
                // We use f16 or f32 shaders to do the copy,
                // so the type/block size must be a multiple of 4.
                ఒకవేళ (src0_type == src1_type &&
                    (!ggml_is_quantized(src0_type) || (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op))) &&
                    (ggml_type_కొలత(src0_type) % 2) == 0) {
                    తిరిగి నిజం;
                }
                తిరిగి అబద్ధం;
            }
        స్థితి GGML_OP_REPEAT:
            తిరిగి ggml_type_కొలత(op->type) == ggml_type_కొలత(op->src[0]->type) &&
                  (ggml_type_కొలత(op->type) == పరిమాణం(దశాంశం) || ggml_type_కొలత(op->type) == 2);
        స్థితి GGML_OP_REPEAT_వెనుక:
            తిరిగి op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_ROPE:
            తిరిగి ggml_is_contiguous_అడ్డువరుసలు(op) && ggml_is_contiguous_అడ్డువరుసలు(op->src[0]);
        స్థితి GGML_OP_ROPE_వెనుక:
        స్థితి GGML_OP_NONE:
        స్థితి GGML_OP_RESHAPE:
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_క్రమమార్పు:
        స్థితి GGML_OP_TRANSPOSE:
        స్థితి GGML_OP_RMS_NORM:
            తిరిగి నిజం;
        స్థితి GGML_OP_సమూహం_NORM:
            తిరిగి ggml_is_contiguous(op->src[0]);
        స్థితి GGML_OP_NORM:
        స్థితి GGML_OP_L2_NORM:
            తిరిగి op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        స్థితి GGML_OP_కూడు:
        స్థితి GGML_OP_SUB:
        స్థితి GGML_OP_MUL:
        స్థితి GGML_OP_DIV:
            తిరిగి (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        స్థితి GGML_OP_కూడు_ID:
            తిరిగి op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32;
        స్థితి GGML_OP_SILU_వెనుక:
        స్థితి GGML_OP_RMS_NORM_వెనుక:
            తిరిగి ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_SQR:
        స్థితి GGML_OP_SQRT:
        స్థితి GGML_OP_SIN:
        స్థితి GGML_OP_COS:
        స్థితి GGML_OP_CLAMP:
        స్థితి GGML_OP_LEAKY_RELU:
            తిరిగి (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        స్థితి GGML_OP_OPT_అడుగు_ADAMW:
        స్థితి GGML_OP_OPT_అడుగు_SGD:
            తిరిగి ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_బహిర్గతం_PROD:
            తిరిగి ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && op->type == GGML_TYPE_F32;
        స్థితి GGML_OP_LOG:
        స్థితి GGML_OP_TRI:
        స్థితి GGML_OP_DIAG:
            తిరిగి (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        స్థితి GGML_OP_ARGSORT:
            {
                ఒకవేళ (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    తిరిగి అబద్ధం;
                }
                // pipeline_argsort_large_f32 requires vulkan memory model.
                ఒకవేళ (device->vulkan_జ్ఞాపకం_model) {
                    తిరిగి నిజం;
                } లేకపోతే {
                    తిరిగి op->ne[0] <= (1 << std::min(device->max_workgroup_కొలత_log2, అంకె_argsort_pipelines - 1));
                }
            }
        స్థితి GGML_OP_శిఖరం_K:
            {
                ఒకవేళ (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    తిరిగి అబద్ధం;
                }
                // large k falls back to radix-select
                స్థిరరాశి uint32_t min_pipeline =
                    std::max((uint32_t) log2f(దశాంశం(op->ne[0])) + 1, device->subgroup_కొలత_log2);
                ఒకవేళ (min_pipeline < అంకె_topk_pipelines && device->pipeline_topk_f32[min_pipeline]) {
                    తిరిగి నిజం;
                }
                తిరిగి device->pipeline_topk_radix_f32 != శూన్యసూచిక;
            }
        స్థితి GGML_OP_UPSCALE:
            ఒకవేళ (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                ఒకవేళ ((op->op_params[0] & 0xFF) != GGML_SCALE_బహుళకం_BILINEAR) {
                    తిరిగి అబద్ధం;
                }
            }
            తిరిగి op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_ACC:
            తిరిగి op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        స్థితి GGML_OP_సమితి:
            తిరిగి op->src[0]->type == op->src[1]->type && op->src[0]->type == op->type &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_I32);
        స్థితి GGML_OP_CONCAT: {
            తిరిగి ggml_vk_concat_supported(op->src[0], op->src[1], op);
        }
        స్థితి GGML_OP_ADD1:
            తిరిగి (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16);
        స్థితి GGML_OP_ARANGE:
            తిరిగి op->type == GGML_TYPE_F32;
        స్థితి GGML_OP_FILL:
            తిరిగి op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
        స్థితి GGML_OP_SCALE:
            తిరిగి ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_PAD:
        స్థితి GGML_OP_PAD_REFLECT_1D:
        స్థితి GGML_OP_ROLL:
            తిరిగి op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_DIAG_ముసుగు_INF:
            తిరిగి ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_SOFT_MAX:
            తిరిగి ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && (!op->src[1] || (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16));
        స్థితి GGML_OP_SOFT_MAX_వెనుక:
            తిరిగి ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_కూడిక:
            తిరిగి op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguously_allocated(op->src[0]);
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
        స్థితి GGML_OP_సగటు:
            తిరిగి op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_అడ్డువరుసలు(op->src[0]);
        స్థితి GGML_OP_CUMSUM:
            {
                ఒకవేళ (device->subgroup_arithmetic && device->subgroup_require_నిండిన_support) {
                    తిరిగి op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_అడ్డువరుసలు(op->src[0]);
                }
                తిరిగి అబద్ధం;
            }
        స్థితి GGML_OP_DSV4_HC_COMB:
        స్థితి GGML_OP_DSV4_HC_PRE:
        స్థితి GGML_OP_DSV4_HC_POST:
            {
                ఒకవేళ (op->type != GGML_TYPE_F32) {
                    తిరిగి అబద్ధం;
                }
                కొరకు (uint32_t i = 0; i < GGML_MAX_SRC; ++i) {
                    ఒకవేళ (op->src[i] && op->src[i]->type != GGML_TYPE_F32) {
                        తిరిగి అబద్ధం;
                    }
                }
                // hc is hardcoded to 4 in the shaders. ggml only constrains it
                // to 4 for COMB, so PRE/POST have to be checked here.
                ఒకవేళ (op->op == GGML_OP_DSV4_HC_PRE && op->src[0]->ne[1] != 4) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->op == GGML_OP_DSV4_HC_POST && op->src[1]->ne[1] != 4) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->op == GGML_OP_DSV4_HC_COMB) {
                    తిరిగి device->pipeline_dsv4_hc_comb_f32 != శూన్యసూచిక;
                }
                తిరిగి నిజం;
            }
        స్థితి GGML_OP_SOLVE_TRI:
            {
                ఒకవేళ (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
                    తిరిగి అబద్ధం;
                }
                స్థిరరాశి uint32_t N = op->src[0]->ne[0];
                స్థిరరాశి uint32_t K = op->src[1]->ne[0];
                // K dimension limited to workgroup size
                ఒకవేళ (K > 1u << device->max_workgroup_కొలత_log2) {
                    తిరిగి అబద్ధం;
                }
                స్థిరరాశి uint32_t batch_N = device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత / ((N + K) * పరిమాణం(దశాంశం));

                ఒకవేళ (batch_N == 0) {
                    తిరిగి అబద్ధం;
                }
                తిరిగి నిజం;
            }
        స్థితి GGML_OP_ARGMAX:
            తిరిగి ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_CROSS_ENTROPY_LOSS:
            తిరిగి ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[0], op->src[1])
                && ggml_is_contiguous(op) && ggml_is_scalar(op) && op->type == GGML_TYPE_F32;
        స్థితి GGML_OP_CROSS_ENTROPY_LOSS_వెనుక:
            తిరిగి ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32 && ggml_is_scalar(op->src[0])
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[2]) && op->src[2]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[1], op->src[2])
                && ggml_are_same_shape(op->src[1], op)
                && ggml_is_contiguous(op) && op->type == GGML_TYPE_F32;
        స్థితి GGML_OP_లెక్క_EQUAL:
            తిరిగి ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_I32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_I32;
        స్థితి GGML_OP_IM2_నిలువు:
            తిరిగి ggml_is_contiguous(op->src[1])
                && op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        స్థితి GGML_OP_IM2_నిలువు_3_D:
            తిరిగి op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        స్థితి GGML_OP_TIMESTEP_EMBEDDING:
            తిరిగి op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_CONV_2D_DW:
            తిరిగి (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)
                && op->src[1]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_POOL_1D:
            తిరిగి ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_POOL_2D:
            తిరిగి ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_RWKV_WKV6:
        స్థితి GGML_OP_RWKV_WKV7:
            తిరిగి నిజం; // all inputs are contiguous, see ggml.c
        స్థితి GGML_OP_GATED_రేఖీయ_ATTN:
            // the shader block size is hardcoded to head_size 64
            తిరిగి op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->src[0]->ne[0] == 64;
        స్థితి GGML_OP_LIGHTNING_INDEXER:
            {
                స్థిరరాశి ggml_tensor * q = op->src[0];
                స్థిరరాశి ggml_tensor * k = op->src[1];
                స్థిరరాశి ggml_tensor * w = op->src[2];
                స్థిరరాశి ggml_tensor * m = op->src[3];

                // the q/w/m types and the shape relationships between q, k, w, m and dst
                // are already asserted in ggml_lightning_indexer()
                ఒకవేళ (!ggml_vk_lightning_indexer_k_type_supported(k->type) || !device->fp16) {
                    తిరిగి అబద్ధం;
                }

                // the shader block size is hardcoded to head size 128
                ఒకవేళ (q->ne[0] != 128) {
                    తిరిగి అబద్ధం;
                }

                // the shader indexes the buffers by element stride, and is dispatched
                // without allow_misalign
                కొరకు (స్థిరరాశి ggml_tensor * t : {q, k, w, m, op}) {
                    ఒకవేళ (t->nb[0] != ggml_type_కొలత(t->type) ||
                        (vk_tensor_స్థానభ్రంశం(t) + t->view_offs) % device->properties.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment != 0) {
                        తిరిగి అబద్ధం;
                    }
                    // the strides get scaled down from bytes, so the division must be exact
                    కొరకు (పూర్ణాంకం i = 1; i < GGML_MAX_DIMS; ++i) {
                        ఒకవేళ (t->nb[i] % ggml_type_కొలత(t->type) != 0) {
                            తిరిగి అబద్ధం;
                        }
                    }
                }
                తిరిగి నిజం;
            }
        స్థితి GGML_OP_GATED_DELTA_NET:
            {
                స్థిరరాశి uint32_t S_v = op->src[2]->ne[0];
                ఒకవేళ (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128) {
                    తిరిగి అబద్ధం;
                }
                కొరకు (పూర్ణాంకం i = 0; i < 6; i++) {
                    ఒకవేళ (op->src[i] == శూన్యసూచిక || op->src[i]->type != GGML_TYPE_F32) {
                        తిరిగి అబద్ధం;
                    }
                }
                తిరిగి op->type == GGML_TYPE_F32;
            }
        స్థితి GGML_OP_SSM_పరిశీలన:
            {
                కొరకు (పూర్ణాంకం i = 0; i < 6; i++) {
                    ఒకవేళ (op->src[i] && ggml_is_quantized(op->src[i]->type)) {
                        తిరిగి అబద్ధం;
                    }
                }
                ఒకవేళ (op->src[6] && op->src[6]->type != GGML_TYPE_I32) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    తిరిగి అబద్ధం;
                }

                స్థిరరాశి uint32_t d_state = op->src[0]->ne[0];
                స్థిరరాశి uint32_t తల_dim = op->src[0]->ne[1];

                తార్కిక is_mamba2 = (op->src[3] && op->src[3]->nb[1] == పరిమాణం(దశాంశం));
                ఒకవేళ (!is_mamba2) {
                    తిరిగి అబద్ధం;
                }

                ఒకవేళ ((d_state != 128 && d_state != 256) || తల_dim % 16 != 0) {
                    తిరిగి అబద్ధం;
                }

                size_t shmem_కొలత = d_state * పరిమాణం(దశాంశం);

                ఒకవేళ (shmem_కొలత > device->properties.limits.max_Compute_Shared_జ్ఞాపకం_కొలత) {
                    తిరిగి అబద్ధం;
                }

                ఒకవేళ (!device->subgroup_basic) {
                    తిరిగి అబద్ధం;
                }

                తిరిగి నిజం;
            }
        స్థితి GGML_OP_SSM_CONV:
            తిరిగి op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_CONV_TRANSPOSE_1D:
            తిరిగి op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_COL2IM_1D:
            తిరిగి (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16 ||
                    op->src[0]->type == GGML_TYPE_BF16) &&
                   op->type == op->src[0]->type &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op);
        స్థితి GGML_OP_CONV_2D:
        స్థితి GGML_OP_CONV_TRANSPOSE_2D:
            {
                స్థిరరాశి తార్కిక transpose = op->op == GGML_OP_CONV_TRANSPOSE_2D;
                స్థిరరాశి int64_t బయటకు = !transpose ? op->src[0]->ne[3] : op->src[0]->ne[2];
                స్థిరరాశి int64_t లోపలికి  = !transpose ? op->src[0]->ne[2] : op->src[0]->ne[3];

                // Channel-contiguous format is not supported yet.
                తిరిగి ((op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    (op->src[0]->nb[0] == పరిమాణం(దశాంశం) || op->src[0]->nb[0] == పరిమాణం(ggml_fp16_t) ) &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->type == GGML_TYPE_F32 &&
                    బయటకు == op->ne[2] &&
                    లోపలికి == op->src[1]->ne[2] &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op));
            }
        స్థితి GGML_OP_CONV_3D:
            తిరిగి (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op);
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }

    వాడని(dev);
}

స్థిర తార్కిక ggml_backend_vk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    ఒకవేళ (buft->iface.get_name != ggml_backend_vk_బఫరు_type_name) {
        తిరిగి అబద్ధం;
    }

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ggml_backend_vk_బఫరు_type_context * buft_ctx = (ggml_backend_vk_బఫరు_type_context *)buft->context;

    తిరిగి buft_ctx->device->idx == ctx->device;
}

స్థిర తార్కిక ggml_backend_vk_device_offload_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    ggml_backend_vk_device_context * dev_ctx = (ggml_backend_vk_device_context *)dev->context;

    తిరిగి ggml_vk_get_op_batch_కొలత(op) >= dev_ctx->op_offload_min_batch_కొలత;
}

స్థిర ggml_backend_event_t ggml_backend_vk_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    స్వయంచాలక device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = కొత్త vk_event;
    ఒకవేళ (!vkev) {
        తిరిగి శూన్యసూచిక;
    }

    // No events initially, they get created on demand
    vkev->has_event = అబద్ధం;

    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.సమితి_PNext(&tci);
    vkev->tl_semaphore = { device->device.createSemaphore(ci), 0 };

    తిరిగి కొత్త ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ vkev,
    };
}

స్థిర శూన్య ggml_backend_vk_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    స్వయంచాలక device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = (vk_event *)event->context;

    device->device.destroySemaphore(vkev->tl_semaphore.s);
    కొరకు (స్వయంచాలక& event : vkev->events_free) {
        device->device.destroyEvent(event);
    }
    కొరకు (స్వయంచాలక& event : vkev->events_submitted) {
        device->device.destroyEvent(event);
    }
    ఒకవేళ (vkev->has_event) {
        device->device.destroyEvent(vkev->event);
    }
    తొలగించు vkev;
    తొలగించు event;
}

స్థిర శూన్య ggml_backend_vk_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_device_event_synchronize(backend=" << dev << ", event=" << event << ")");
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    స్వయంచాలక device = ggml_vk_get_device(ctx->device);
    vk_event *vkev = (vk_event *)event->context;

    // Only do something if the event has actually been used
    ఒకవేళ (vkev->has_event) {
        vk::Semaphore sem = vkev->tl_semaphore.s;
        uint64_t val = vkev->tl_semaphore.value;
        vk::SemaphoreWaitInfo swi{vk::SemaphoreWaitFlags{}, sem, val};
        VK_CHECK(device->device.waitSemaphores(swi, UINT64_MAX), "event_synchronize", device);

        // Reset and move submitted events
        కొరకు (స్వయంచాలక& event : vkev->events_submitted) {
            device->device.resetEvent(event);
        }
        vkev->events_free.insert(vkev->events_free.end(), vkev->events_submitted.begin(), vkev->events_submitted.end());
        vkev->events_submitted.clear();

        // Finished using current command buffer so we flag for reuse
        ఒకవేళ (vkev->cmd_బఫరు) {
            // Only flag for reuse if it hasn't been reused already
            ఒకవేళ (vkev->cmd_బఫరు_use_counter == vkev->cmd_బఫరు->use_counter) {
                vkev->cmd_బఫరు->లోపల_use = అబద్ధం;
                vkev->cmd_బఫరు->buf.reset();
            }
            vkev->cmd_బఫరు = శూన్యసూచిక;
        }
    }
}

స్థిర ggml_backend_బఫరు_t ggml_backend_vk_device_బఫరు_from_host_ptr(ggml_backend_dev_t dev, శూన్య * ptr, size_t size, size_t max_tensor_కొలత) {
    VK_LOG_DEBUG("ggml_backend_vk_device_buffer_from_host_ptr(backend=" << dev << ", ptr=" << ptr << ", size=" << size << ")");
    GGML_వాడని(max_tensor_కొలత);

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    స్వయంచాలక device = ggml_vk_get_device(ctx->device);

    vk_బఫరు buf = ggml_vk_బఫరు_from_host_ptr(device, ptr, size);

    ఒకవేళ (!buf) {
        తిరిగి {};
    }

    ggml_backend_vk_బఫరు_context * bufctx = కొత్త ggml_backend_vk_బఫరు_context(device, std::move(buf), device->name);

    ggml_backend_బఫరు_t ret = ggml_backend_బఫరు_init(ggml_backend_vk_device_get_బఫరు_type(dev), ggml_backend_vk_బఫరు_interface, bufctx, size);

    తిరిగి ret;
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_device_i ggml_backend_vk_device_i = {
    /* .get_name             = */ ggml_backend_vk_device_get_name,
    /* .get_description      = */ ggml_backend_vk_device_get_description,
    /* .get_memory           = */ ggml_backend_vk_device_get_జ్ఞాపకం,
    /* .get_type             = */ ggml_backend_vk_device_get_type,
    /* .get_props            = */ ggml_backend_vk_device_get_props,
    /* .init_backend         = */ ggml_backend_vk_device_init,
    /* .get_buffer_type      = */ ggml_backend_vk_device_get_బఫరు_type,
    /* .get_host_buffer_type = */ ggml_backend_vk_device_get_host_బఫరు_type,
    /* .buffer_from_host_ptr = */ ggml_backend_vk_device_బఫరు_from_host_ptr,
    /* .supports_op          = */ ggml_backend_vk_device_supports_op,
    /* .supports_buft        = */ ggml_backend_vk_device_supports_buft,
    /* .offload_op           = */ ggml_backend_vk_device_offload_op,
    /* .event_new            = */ ggml_backend_vk_device_event_new,
    /* .event_free           = */ ggml_backend_vk_device_event_free,
    /* .event_synchronize    = */ ggml_backend_vk_device_event_synchronize,
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_vk_reg_get_name(ggml_backend_reg_t reg) {
    వాడని(reg);
    తిరిగి GGML_VK_NAME;
}

స్థిర size_t ggml_backend_vk_reg_get_device_లెక్క(ggml_backend_reg_t reg) {
    వాడని(reg);
    తిరిగి ggml_backend_vk_get_device_లెక్క();
}

స్థిర ggml_backend_dev_t ggml_backend_vk_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    స్థిర std::vector<ggml_backend_dev_t> devices;

    స్థిర తార్కిక initialized = అబద్ధం;

    {
        స్థిర std::mutex mutex;
        std::lock_guard<std::mutex> తాళం(mutex);
        ఒకవేళ (!initialized) {
            స్థిరరాశి పూర్ణాంకం min_batch_కొలత = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
            కొరకు (పూర్ణాంకం i = 0; i < ggml_backend_vk_get_device_లెక్క(); i++) {
                ggml_backend_vk_device_context * ctx = కొత్త ggml_backend_vk_device_context;
                అక్షరం desc[256];
                ggml_backend_vk_get_device_description(i, desc, పరిమాణం(desc));
                ctx->device = i;
                ctx->name = GGML_VK_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->is_integrated_gpu = ggml_backend_vk_get_device_type(i) == vk::PhysicalDeviceType::eIntegratedGpu;
                ctx->pci_bus_id = ggml_backend_vk_get_device_pci_id(i);
                ctx->op_offload_min_batch_కొలత = min_batch_కొలత;
                devices.push_back(కొత్త ggml_backend_device {
                    /* .iface   = */ ggml_backend_vk_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                });
            }
            initialized = నిజం;
        }
    }

    GGML_ASSERT(device < devices.size());
    తిరిగి devices[device];
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_reg_i ggml_backend_vk_reg_i = {
    /* .get_name         = */ ggml_backend_vk_reg_get_name,
    /* .get_device_count = */ ggml_backend_vk_reg_get_device_లెక్క,
    /* .get_device       = */ ggml_backend_vk_reg_get_device,
    /* .get_proc_address = */ ఖాళీ,
};

ggml_backend_reg_t ggml_backend_vk_reg() {
    స్థిర ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_సంస్కరణ,
        /* .iface       = */ ggml_backend_vk_reg_i,
        /* .context     = */ శూన్యసూచిక,
    };
    ప్రయత్నించు {
        ggml_vk_instance_init();
        తిరిగి &reg;
    } పట్టుకో (స్థిరరాశి vk::SystemError& e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: System error: " << e.what());
        తిరిగి శూన్యసూచిక;
    } పట్టుకో (స్థిరరాశి std::exception &e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: " << e.what());
        తిరిగి శూన్యసూచిక;
    } పట్టుకో (...) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: unknown exception during Vulkan init");
        తిరిగి శూన్యసూచిక;
    }
}

తార్కిక ggml_vk_instance_layer_settings_అందుబాటు() {
#నిర్వచితమైతే GGML_VULKAN_VALIDATE
    // Check if validation layer provides the extension
    స్థిరరాశి std::string layer_name = "VK_LAYER_KHRONOS_validation";
    కొరకు (స్థిరరాశి స్వయంచాలక& layer : vk::enumerateInstanceLayerProperties()) {
        ఒకవేళ (layer_name == layer.layerName.దత్తాంశం()) {
            కొరకు (స్థిరరాశి స్వయంచాలక& ext : vk::enumerateInstanceExtensionProperties(layer_name)) {
                ఒకవేళ (strcmp("VK_EXT_layer_settings", ext.extensionName.దత్తాంశం()) == 0) {
                    తిరిగి నిజం;
                }
            }
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Validation layer or layer extension VK_EXT_layer_settings not found." << std::పంక్తి_ముగింపు;
#అయితే_ముగింపు
    తిరిగి అబద్ధం;
}

తార్కిక ggml_vk_instance_portability_enumeration_ext_అందుబాటు(స్థిరరాశి std::vector<vk::ExtensionProperties>& instance_extensions) {
#నిర్వచితమైతే __APPLE__
    // Check for portability enumeration extension for MoltenVK support
    కొరకు (స్థిరరాశి స్వయంచాలక& properties : instance_extensions) {
        ఒకవేళ (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            తిరిగి నిజం;
        }
    }
    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::పంక్తి_ముగింపు;
#అయితే_ముగింపు
    తిరిగి అబద్ధం;

    వాడని(instance_extensions);
}

తార్కిక ggml_vk_instance_debug_utils_ext_అందుబాటు(
    స్థిరరాశి std::vector<vk::ExtensionProperties> & instance_extensions) {
    // Check for portability enumeration extension for MoltenVK support
    కొరకు (స్థిరరాశి స్వయంచాలక & properties : instance_extensions) {
        ఒకవేళ (strcmp("VK_EXT_debug_utils", properties.extensionName) == 0) {
            తిరిగి నిజం;
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found." << std::పంక్తి_ముగింపు;
    తిరిగి అబద్ధం;

    వాడని(instance_extensions);
}

తార్కిక ggml_vk_device_is_supported(స్థిరరాశి vk::PhysicalDevice & vkdev) {
    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_తదుపరి = శూన్యసూచిక;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_తదుపరి = &vk11_features;

    vkGetPhysicalDeviceFeatures2(vkdev, &device_features2);

    తిరిగి vk11_features.storage_Buffer16_బిట్టు_Access;
}

తార్కిక ggml_vk_khr_cooperative_మాతృక_support(స్థిరరాశి vk::PhysicalDeviceProperties& props, స్థిరరాశి vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch) {
    ఎంపిక (props.vendorID) {
    స్థితి VK_VENDOR_ID_INTEL:
        // Only allowing Xe2/Xe3 GPU and integrated Xe GPUs at the moment since older hardware (ex. Arc A770) has performance regressions.
        తిరిగి (arch == vk_device_architecture::INTEL_XE2) ||
            (arch == vk_device_architecture::INTEL_XE1 && props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu && driver_props.driverID == vk::DriverId::eIntelProprietaryWindows);
    స్థితి VK_VENDOR_ID_AMD:
        ఒకవేళ (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::e_Amd_Open_ఆధారం) {
            // Workaround for AMD proprietary driver reporting support on all GPUs
            తిరిగి arch == vk_device_architecture::AMD_RDNA3 || arch == vk_device_architecture::AMD_RDNA4;
        }
        తిరిగి నిజం;
    స్థితి VK_VENDOR_ID_QUALCOMM:
        // Only allow Adreno GPUs with hardware matrix cores (Gen 6+).
        తిరిగి arch == vk_device_architecture::QUALCOMM_ADRENO;
    అప్రమేయ:
        తిరిగి నిజం;
    }
}

uint32_t ggml_vk_intel_shader_core_లెక్క(స్థిరరాశి vk::PhysicalDevice& vkdev) {
    VkPhysicalDeviceProperties2 props = vkdev.getProperties2();

    ఒకవేళ (props.properties.vendorID != VK_VENDOR_ID_INTEL) {
        తిరిగి 0;
    }

    స్థిరరాశి uint32_t device_id = props.properties.deviceID;

    ఎంపిక (device_id) {
    స్థితి 0x56A6:  // A310
        తిరిగి 6;
    స్థితి 0x5693:  // A370M
    స్థితి 0x56A5:  // A380
    స్థితి 0x56B1:  // Pro A40/A50
        తిరిగి 8;
    స్థితి 0x5697:  // A530M
        తిరిగి 12;
    స్థితి 0x5692:  // A550M
    స్థితి 0x56B3:  // Pro A60
        తిరిగి 16;
    స్థితి 0x56A2:  // A580
        తిరిగి 24;
    స్థితి 0x5691:  // A730M
    స్థితి 0x56A1:  // A750
        తిరిగి 28;
    స్థితి 0x56A0:  // A770
    స్థితి 0x5690:  // A770M
        తిరిగి 32;
    స్థితి 0xE212:  // Pro B50
        తిరిగి 16;
    స్థితి 0xE20C:  // B570
        తిరిగి 18;
    స్థితి 0xE20B:  // B580
    స్థితి 0xE211:  // Pro B60
    స్థితి 0xE222:  // Pro B65
        తిరిగి 20;
    స్థితి 0xE223:  // Pro B70
        తిరిగి 32;
    స్థితి 0xB080:  // PTL Xe3 LPG 2x6 (12 subslices)
        తిరిగి 12;
    అప్రమేయ:
        తిరిగి 0;
    }
}

తార్కిక ggml_vk_intel_windows_driver_లోపల_range(uint32_t driver_సంస్కరణ, uint32_t క్రింది_major, uint32_t క్రింది_minor, uint32_t పై_major, uint32_t పై_minor) {
#ఒకవేళ defined(_WIN32)
    // Intel Windows encodes xxx.yyyy as [31:14].[13:0].
    స్థిరరాశి uint32_t major = driver_సంస్కరణ >> 14;
    స్థిరరాశి uint32_t minor = driver_సంస్కరణ & 0x3fff;

    స్థిరరాశి తార్కిక ge_క్రింది = major > క్రింది_major || (major == క్రింది_major && minor >= క్రింది_minor);
    స్థిరరాశి తార్కిక lt_పై = major < పై_major || (major == పై_major && minor < పై_minor);

    తిరిగి ge_క్రింది && lt_పై;
#లేకపోతే
    GGML_వాడని(driver_సంస్కరణ);
    GGML_వాడని(క్రింది_major);
    GGML_వాడని(క్రింది_minor);
    GGML_వాడని(పై_major);
    GGML_వాడని(పై_minor);
    తిరిగి నిజం;
#అయితే_ముగింపు
}

GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)


// out-of-lined header method definitions

శూన్య vk_వరుస_handle_synchronized::submit(vk::ArrayProxy<స్థిరరాశి vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    ఒకవేళ (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    std::lock_guard<std::mutex> guard(mutex);
    ప్రయత్నించు {
        queue.submit(submits, fence);
    } పట్టుకో (vk::Device_కోల్పోయిన_Error &) {
        ఒకవేళ (స్వయంచాలక dev = device.తాళం()) {
            ggml_vk_print_device_కోల్పోయిన_info(dev);
        }
        విసురు;
    }
}

శూన్య vk_వరుస_handle_unsynchronized::submit(vk::ArrayProxy<స్థిరరాశి vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    ఒకవేళ (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    ప్రయత్నించు {
        queue.submit(submits, fence);
    } పట్టుకో (vk::Device_కోల్పోయిన_Error &) {
        ఒకవేళ (స్వయంచాలక dev = device.తాళం()) {
            ggml_vk_print_device_కోల్పోయిన_info(dev);
        }
        విసురు;
    }
}

vk_device_struct::~vk_device_struct() {
    // the build failed before the device was created, nothing to destroy
    ఒకవేళ (device == VK_NULL_HANDLE) {
        తిరిగి;
    }

    VK_LOG_DEBUG("destroy device " << name);

    device.destroyFence(fence);

    ggml_vk_destroy_బఫరు(sync_staging);

    ఒకవేళ (compute_వరుస) compute_వరుస->cmd_pool.destroy(device);
    ఒకవేళ (transfer_వరుస) transfer_వరుస->cmd_pool.destroy(device);

    // Explicitly clear to ensure queues drop their shared_ptrs to handles
    // before the Vulkan logical device instance is destroyed
    compute_వరుస.reset();
    transfer_వరుస.reset();

    కొరకు (స్వయంచాలక& pipeline : all_pipelines) {
        ఒకవేళ (pipeline.expired()) {
            కొనసాగు;
        }

        vk_pipeline pl = pipeline.తాళం();
        ggml_vk_destroy_pipeline(device, pl);
    }
    all_pipelines.clear();

    device.destroy_Descriptor_సమితి_Layout(dsl);

    device.destroy();
}

శూన్య vk_perf_logger::print_timings(తార్కిక force) {
    ఒకవేళ (timings.ఖాళీలు()) {
        తిరిగి;
    }
    print_లెక్క++;
    ఒకవేళ ((print_లెక్క % vk_perf_logger_frequency) != 0 && !force) {
        తిరిగి;
    }
    print_లెక్క = 0;
    uint64_t మొత్తం_all_op_times = 0;
    std::cerr << "----------------\nVulkan Timings:" << std::పంక్తి_ముగింపు;
    కొరకు (స్థిరరాశి స్వయంచాలక & t : timings) {
        uint64_t మొత్తం_op_times = 0;
        కొరకు (స్థిరరాశి స్వయంచాలక & సమయం : t.second) {
            మొత్తం_op_times += సమయం;
        }
        std::cerr << t.మొదటి << ": " << t.second.size() << " x " << (మొత్తం_op_times / t.second.size() / 1000.0)
                  << " us = " << (మొత్తం_op_times / 1000.0) << " us";

        // If we have as many flops entries as timing entries for the op, then compute and log the flops/S.
        స్వయంచాలక it = flops.find(t.మొదటి);
        ఒకవేళ (it != flops.end() && (it->second).size() == t.second.size()) {
            uint64_t మొత్తం_op_flops = 0;
            కొరకు (స్థిరరాశి స్వయంచాలక & elem : it->second) {
                మొత్తం_op_flops += elem;
            }
            std::cerr << " ("
                      << (ద్విగుణ(మొత్తం_op_flops) / (1000.0 * 1000.0 * 1000.0)) /
                             (ద్విగుణ(మొత్తం_op_times) / (1000.0 * 1000.0 * 1000.0))
                      << " GFLOPS/s)";
        }

        మొత్తం_all_op_times += మొత్తం_op_times;

        std::cerr << std::పంక్తి_ముగింపు;
    }

    ఒకవేళ (timings.size() > 0) {
        std::cerr << "Total time: " << మొత్తం_all_op_times / 1000.0 << " us." << std::పంక్తి_ముగింపు;
    }

    timings.clear();
    flops.clear();
}

std::string vk_perf_logger::get_కణుపు_fusion_name(స్థిరరాశి ggml_tensor * కణుపు, స్థిరరాశి అక్షరం *fusion_name, uint64_t *n_flops) {
    *n_flops = ggml_vk_get_కణుపు_flops(కణుపు);
    std::string fusion_str;
    ఒకవేళ (fusion_name) {
        fusion_str = fusion_name + std::string(" ");
    }
    ఒకవేళ (కణుపు->op == GGML_OP_UNARY) {
        తిరిగి fusion_str + ggml_unary_op_name(ggml_get_unary_op(కణుపు));
    }
    ఒకవేళ (కణుపు->op == GGML_OP_MUL_MAT || కణుపు->op == GGML_OP_MUL_MAT_ID) {
        స్థిరరాశి uint64_t m     = కణుపు->ne[0];
        స్థిరరాశి uint64_t n     = కణుపు->ne[1];
        స్థిరరాశి uint64_t k     = కణుపు->src[1]->ne[0];
        స్థిరరాశి uint64_t batch = కణుపు->ne[2] * కణుపు->ne[3];
        std::string    name  = ggml_op_name(కణుపు->op);
        ఒకవేళ ((కణుపు->op == GGML_OP_MUL_MAT && n <= mul_mat_vec_max_నిలువుల_సంఖ్య) ||
            (కణుపు->op == GGML_OP_MUL_MAT_ID && కణుపు->src[2]->ne[1] == 1)) {
            name += "_VEC";
        }
        name += " ";
        name += ggml_type_name(కణుపు->src[0]->type);
        name += " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
        ఒకవేళ (కణుపు->op == GGML_OP_MUL_MAT_ID) {
            name += " n_expert=" + std::to_string(కణుపు->src[0]->ne[2]);
        }
        ఒకవేళ (batch > 1) {
            name += " batch=" + std::to_string(batch);
        }
        తిరిగి fusion_str + name;
    }
    ఒకవేళ (కణుపు->op == GGML_OP_CONV_2D || కణుపు->op == GGML_OP_CONV_TRANSPOSE_2D) {
        std::string   name    = ggml_op_name(కణుపు->op);
        స్థిరరాశి ggml_tensor * knl = కణుపు->src[0];
        uint64_t      Cout    = కణుపు->ne[2];
        uint64_t      కొలత_K  = కణుపు->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        uint64_t      కొలత_N_2  = కణుపు->ne[3] * కణుపు->ne[0] * కణుపు->ne[1];
        name += " M=Cout=" + std::to_string(Cout) + ", K=Cin*KW*KH=" + std::to_string(కొలత_K) +
                ", N=N*OW*OH=" + std::to_string(కొలత_N_2);
        తిరిగి fusion_str + name;
    }
    ఒకవేళ (కణుపు->op == GGML_OP_RMS_NORM) {
        std::string   name    = ggml_op_name(కణుపు->op);
        name += "(" + std::to_string(కణుపు->ne[0]) + "," + std::to_string(కణుపు->ne[1]) + "," + std::to_string(కణుపు->ne[2]) + "," + std::to_string(కణుపు->ne[3]) + ")";
        తిరిగి fusion_str + name;
    }
    ఒకవేళ (కణుపు->op == GGML_OP_FLASH_ATTN_EXT) {
        స్థిరరాశి ggml_tensor * dst = కణుపు;
        స్థిరరాశి ggml_tensor * q = కణుపు->src[0];
        స్థిరరాశి ggml_tensor * k = కణుపు->src[1];
        స్థిరరాశి ggml_tensor * v = కణుపు->src[2];
        స్థిరరాశి ggml_tensor * m = కణుపు->src[3];
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(కణుపు->op) <<
            " dst(" << dst->ne[0] << "," << dst->ne[1] << "," << dst->ne[2] << "," << dst->ne[3] << "), " <<
            " q(" << q->ne[0] << "," << q->ne[1] << "," << q->ne[2] << "," << q->ne[3] << "), " <<
            " k(" << k->ne[0] << "," << k->ne[1] << "," << k->ne[2] << "," << k->ne[3] << "), " <<
            " v(" << v->ne[0] << "," << v->ne[1] << "," << v->ne[2] << "," << v->ne[3] << "), " <<
            " m(" << (m?m->ne[0]:0) << "," << (m?m->ne[1]:0) << "," << (m?m->ne[2]:0) << "," << (m?m->ne[3]:0) << ")";
        తిరిగి name.str();
    }
    ఒకవేళ (కణుపు->op == GGML_OP_శిఖరం_K) {
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(కణుపు->op) <<
            " K=" << కణుపు->ne[0] <<
            " (" << కణుపు->src[0]->ne[0] << "," << కణుపు->src[0]->ne[1] << "," << కణుపు->src[0]->ne[2] << "," << కణుపు->src[0]->ne[3] << ")";
        తిరిగి name.str();
    }
    తిరిగి fusion_str + ggml_op_name(కణుపు->op);
}

ggml_backend_vk_బఫరు_context::~ggml_backend_vk_బఫరు_context() {
    ggml_vk_destroy_బఫరు(dev_బఫరు);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, స్థిరరాశి std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2) {
    ఒకవేళ (!vk_instance.debug_utils_support || ctx->s == శూన్యసూచిక) {
        తిరిగి;
    }
    begin(ctx, pipeline_name + " (" + std::to_string(wg0) + "," + std::to_string(wg1) + "," + std::to_string(wg2) + ")");
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, స్థిరరాశి ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, పూర్ణాంకం n_fused) {
    ఒకవేళ (!vk_instance.debug_utils_support || ctx->s == శూన్యసూచిక) {
        తిరిగి;
    }
    std::string name = ggml_op_name(cgraph->nodes[కణుపు_idx]->op);
    కొరకు (పూర్ణాంకం i = 1; i <= n_fused; i++) {
        name += "+";
        name += ggml_op_name(cgraph->nodes[కణుపు_idx + i]->op);
    }
    name += " ";
    name += cgraph->nodes[కణుపు_idx]->name;
    begin(ctx, name);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_వరుస_handle * handle, స్థిరరాశి అక్షరం * name) {
    ఒకవేళ (!vk_instance.debug_utils_support || handle == శూన్యసూచిక) {
        తిరిగి;
    }
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<దశాంశం, 4>{1.0f, 1.0f, 1.0f, 1.0f};

    qhandle = handle;
    std::lock_guard<vk_వరుస_handle> guard(*qhandle);
    vk_instance.pfn_vk_వరుస_Begin_Debug_Utils_Label_EXT(qhandle->queue, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

శూన్య ggml_vk_debug_label::close() {
    ఒకవేళ (subctx != శూన్యసూచిక) {
        // close on the current command buffer, which may differ from the one begin used
        ఒకవేళ (subctx->s != శూన్యసూచిక) {
            vk_instance.pfn_vk_Cmd_చివర_Debug_Utils_Label_EXT(subctx->s->బఫరు->buf);
        }
        subctx->debug_labels.pop_back();
        subctx = శూన్యసూచిక;
    }
    ఒకవేళ (qhandle != శూన్యసూచిక) {
        std::lock_guard<vk_వరుస_handle> guard(*qhandle);
        vk_instance.pfn_vk_వరుస_చివర_Debug_Utils_Label_EXT(qhandle->queue);
        qhandle = శూన్యసూచిక;
    }
}

శూన్య ggml_vk_debug_label::begin(vk_context & ctx, స్థిరరాశి std::string & name) {
    ఒకవేళ (!vk_instance.debug_utils_support || ctx->s == శూన్యసూచిక) {
        తిరిగి;
    }
    subctx = ctx.get();
    subctx->debug_labels.push_back(name);
    ggml_vk_cmd_label_begin(subctx->s->బఫరు->buf, subctx->debug_labels.back().c_str());
}

