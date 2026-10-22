#अन्तर्भाव <iostream>
#अन्तर्भाव <fstream>
#अन्तर्भाव <sstream>
#अन्तर्भाव <string>
#अन्तर्भाव <stdexcept>
#अन्तर्भाव <array>
#अन्तर्भाव <vector>
#अन्तर्भाव <map>
#अन्तर्भाव <thread>
#अन्तर्भाव <mutex>
#अन्तर्भाव <future>
#अन्तर्भाव <queue>
#अन्तर्भाव <condition_variable>
#अन्तर्भाव <atomic>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <cstring>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cassert>
#अन्तर्भाव <algorithm>
#अन्तर्भाव <sys/stat.h>
#अन्तर्भाव <sys/types.h>
#अन्तर्भाव <filesystem>

#ifdef _WIN32
    #define NOMINMAX
    #अन्तर्भाव <windows.h>
    #अन्तर्भाव <direct.h> // For _mkdir on Windows
#अन्यथा
    #अन्तर्भाव <unistd.h>
    #अन्तर्भाव <sys/wait.h>
    #अन्तर्भाव <fcntl.h>
#endif

#define ASYNCIO_CONCURRENCY 64

std::mutex lock;
std::vector<std::pair<std::string, std::string>> shader_fnames;
// Set when any shader subprocess fails (non-zero exit / stderr / launch failure) so the
// build is stopped instead of silently producing a broken libggml-vulkan. (issue #24393)
स्थिर std::atomic<तार्किक> compile_failed{असत्य};
std::locale c_locale("C");

std::string GLSLC = "glslc";
std::string निवेश_filepath = "";
std::string output_dir = "/tmp";
std::string target_hpp = "";
std::string target_cpp = "";

ध्रुव std::vector<std::string> type_names = {
    "f32",
    "f16",
    "q1_0",
    "q2_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq1_s",
    "iq1_m",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq3_xxs",
    "iq3_s",
    "iq4_xs",
    "iq4_nl",
    "mxfp4",
    "nvfp4",
    "tq1_0",
    "tq2_0",
    "bf16",
};

गणना MatMulIdType {
    NONE,
    DEFAULT,
    SUBGROUP,
};

नामाकाश {

पूर्णाङ्क execute_command(std::vector<std::string>& command, std::string& stdout_str, std::string& stderr_str) {
#ifdef _WIN32
    HANDLE stdout_read, stdout_write;
    HANDLE stderr_read, stderr_write;
    SECURITY_ATTRIBUTES sa = { परिमाण(SECURITY_ATTRIBUTES), रिक्त, TRUE };

    यदि (!CreatePipe(&stdout_read, &stdout_write, &sa, 0) ||
        !SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0)) {
        क्षिप std::runtime_error("Failed to create stdout pipe");
    }

    यदि (!CreatePipe(&stderr_read, &stderr_write, &sa, 0) ||
        !SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0)) {
        क्षिप std::runtime_error("Failed to create stderr pipe");
    }

    PROCESS_INFORMATION pi;
    STARTUPINFOA si = {};
    si.cb = परिमाण(STARTUPINFOA);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = stdout_write;
    si.hStdError = stderr_write;

    std::string cmd;
    प्रति (ध्रुव स्वतः& part : command) {
        cmd += part + " ";
    }

    यदि (!CreateProcessA(रिक्त, cmd.data(), रिक्त, रिक्त, TRUE, 0, रिक्त, रिक्त, &si, &pi)) {
        क्षिप std::runtime_error("Failed to create process");
    }

    CloseHandle(stdout_write);
    CloseHandle(stderr_write);

    std::array<वर्ण, 128> buffer;
    DWORD bytes_read;

    यावत् (ReadFile(stdout_read, buffer.data(), (DWORD)buffer.size(), &bytes_read, रिक्त) && bytes_read > 0) {
        stdout_str.append(buffer.data(), bytes_read);
    }

    यावत् (ReadFile(stderr_read, buffer.data(), (DWORD)buffer.size(), &bytes_read, रिक्त) && bytes_read > 0) {
        stderr_str.append(buffer.data(), bytes_read);
    }

    CloseHandle(stdout_read);
    CloseHandle(stderr_read);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    प्रत्यागम (पूर्णाङ्क)exit_code;
#अन्यथा
    पूर्णाङ्क stdout_pipe[2];
    पूर्णाङ्क stderr_pipe[2];

    यदि (प्रणाल(stdout_pipe) != 0 || प्रणाल(stderr_pipe) != 0) {
        क्षिप std::runtime_error("Failed to create pipes");
    }

    pid_t pid = विभाजन();
    यदि (pid < 0) {
        std::cerr << म_त्रुटि(त्रुटि_सं) << "\n";
        क्षिप std::runtime_error("Failed to fork process");
    }

    std::vector<वर्ण*> argv;
    प्रति (std::string& part : command) {
        argv.push_back(part.data());
    }
    argv.push_back(nullptr);

    यदि (pid == 0) {
        बंद(stdout_pipe[0]);
        बंद(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        बंद(stdout_pipe[1]);
        बंद(stderr_pipe[1]);
        execvp(argv[0], argv.data());
        _exit(निकास_त्रुटि);
    } अन्यथा {
        बंद(stdout_pipe[1]);
        बंद(stderr_pipe[1]);

        std::array<वर्ण, 128> buffer;
        ssize_t bytes_read;

        यावत् ((bytes_read = पठ(stdout_pipe[0], buffer.data(), buffer.size())) > 0) {
            stdout_str.append(buffer.data(), bytes_read);
        }

        यावत् ((bytes_read = पठ(stderr_pipe[0], buffer.data(), buffer.size())) > 0) {
            stderr_str.append(buffer.data(), bytes_read);
        }

        बंद(stdout_pipe[0]);
        बंद(stderr_pipe[0]);
        पूर्णाङ्क status = 0;
        waitpid(pid, &status, 0);
        प्रत्यागम WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
#endif
}

तार्किक directory_exists(ध्रुव std::string& path) {
    संरचना stat info;
    यदि (stat(path.c_str(), &info) != 0) {
        प्रत्यागम असत्य; // Path doesn't exist or can't be accessed
    }
    प्रत्यागम (info.st_mode & S_IFDIR) != 0; // Check if it is a directory
}

तार्किक create_directory(ध्रुव std::string& path) {
#ifdef _WIN32
    प्रत्यागम _mkdir(path.c_str()) == 0 || त्रुटि_सं == EEXIST; // EEXIST means the directory already exists
#अन्यथा
    प्रत्यागम सूची_गढ़ो(path.c_str(), 0755) == 0 || त्रुटि_सं == EEXIST; // 0755 is the directory permissions
#endif
}

std::string to_uppercase(ध्रुव std::string& निवेश) {
    std::string result = निवेश;
    प्रति (वर्ण& c : result) {
        c = std::बड़े(c);
    }
    प्रत्यागम result;
}

तार्किक string_starts_with(ध्रुव std::string& str, ध्रुव std::string& prefix) {
    यदि (prefix.size() > str.size()) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम std::equal(prefix.begin(), prefix.end(), str.begin());
}

तार्किक string_ends_with(ध्रुव std::string& str, ध्रुव std::string& suffix) {
    यदि (suffix.size() > str.size()) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

तार्किक is_quantized_type(ध्रुव std::string& type_name) {
    प्रत्यागम type_name != "f32" && type_name != "f16" && type_name != "bf16";
}

तार्किक is_legacy_quant(ध्रुव std::string& type_name) {
    प्रत्यागम type_name == "q2_0" || type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";
}

तार्किक is_k_quant(ध्रुव std::string& type_name) {
    प्रत्यागम string_ends_with(type_name, "_k");
}

तार्किक is_iq_quant(ध्रुव std::string& type_name) {
    प्रत्यागम string_starts_with(type_name, "iq");
}

तार्किक is_lut_quant(ध्रुव std::string& type_name) {
    प्रत्यागम is_iq_quant(type_name) || type_name == "mxfp4" || type_name == "nvfp4";
}

std::string lut_load_vec_a(ध्रुव std::string& type_name) {
    यदि (type_name == "iq1_s" || type_name == "iq1_m" || type_name == "iq2_xxs" || type_name == "iq2_xs" || type_name == "iq2_s" || type_name == "iq4_xs") {
        प्रत्यागम "8";
    }
    प्रत्यागम "4";
}

स्थिर ध्रुव वर्ण path_separator = '/';

std::string join_paths(ध्रुव std::string& path1, ध्रुव std::string& path2) {
    प्रत्यागम path1 + path_separator + path2;
}

std::string basename(ध्रुव std::string &path) {
    प्रत्यागम path.substr(path.find_last_of("/\\") + 1);
}

std::stringstream make_generic_stringstream() {
    std::stringstream ss;
    ss.imbue(c_locale);
    प्रत्यागम ss;
}

std::string read_द्विआधारी_file(ध्रुव std::string& path, तार्किक may_not_exist = असत्य) {
    खाता* f = ख_खोलो(path.c_str(), "rb");
    यदि (!f) {
        यदि (!may_not_exist) {
            std::cerr << "Error opening file: " << path << " (" << म_त्रुटि(त्रुटि_सं) << ")\n";
        }
        प्रत्यागम {};
    }

    ख_जाओ(f, 0, अंत_से);
    माप_प्रकार size = ख_बताओ(f);
    ख_जाओ(f, 0, शुरू_से);

    std::string data(size, '\0');
    माप_प्रकार read_size = ख_पढ़ो(data.data(), 1, size, f);
    ख_बंद(f);
    यदि (read_size != size) {
        std::cerr << "Error reading file: " << path << " (" << म_त्रुटि(त्रुटि_सं) << ")\n";
        प्रत्यागम {};
    }

    प्रत्यागम data;
}

शून्य write_द्विआधारी_file(ध्रुव std::string& path, ध्रुव std::string& content) {
    खाता* f = ख_खोलो(path.c_str(), "wb");
    यदि (!f) {
        std::cerr << "Error opening file for writing: " << path << " (" << म_त्रुटि(त्रुटि_सं) << ")\n";
        प्रत्यागम;
    }

    माप_प्रकार write_size = ख_डालो(content.data(), 1, content.size(), f);
    ख_बंद(f);
    यदि (write_size != content.size()) {
        std::cerr << "Error writing file: " << path << " (" << म_त्रुटि(त्रुटि_सं) << ")\n";
        प्रत्यागम;
    }
}

शून्य write_file_if_changed(ध्रुव std::string& path, ध्रुव std::string& content) {
    std::string existing = read_द्विआधारी_file(path, सत्य);
    यदि (existing != content) {
        write_द्विआधारी_file(path, content);
    }
}


// variables to track number of compiles in progress
स्थिर uint32_t compile_गिनती = 0;
स्थिर std::mutex compile_गिनती_mutex;
स्थिर std::condition_variable compile_गिनती_cond;
स्थिर तार्किक generate_dep_file = सत्य;

शून्य decrement_compile_गिनती(uint32_t * count) {
    यदि (count) {
        std::lock_guard<std::mutex> guard(compile_गिनती_mutex);
        प्रतिज्ञा(compile_गिनती > 0);
        compile_गिनती--;
        compile_गिनती_cond.notify_all();
    }
}

using compile_गिनती_guard = std::unique_ptr<uint32_t, decltype(&decrement_compile_गिनती)>;

compile_गिनती_guard acquire_compile_slot() {
    // wait until fewer than N compiles are in progress.
    // 16 is an arbitrary limit, the goal is to avoid "failed to create pipe" errors.
    uint32_t N = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::unique_lock<std::mutex> guard(compile_गिनती_mutex);
    compile_गिनती_cond.रुको(guard, [N] { प्रत्यागम compile_गिनती < N; });
    compile_गिनती++;
    प्रत्यागम compile_गिनती_guard(&compile_गिनती, &decrement_compile_गिनती);
}

शून्य string_to_spv_func(std::string name, std::string in_path, std::string out_path, std::map<std::string, std::string> defines, तार्किक coopmat, तार्किक dep_file, compile_गिनती_guard slot) {
    std::string target_env = (name.find("_cm2") != std::string::npos) ? "--target-env=vulkan1.3" : "--target-env=vulkan1.2";

    #ifdef _WIN32
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", target_env, "\"" + in_path + "\"", "-o", "\"" + out_path + "\""};
    #अन्यथा
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", target_env, in_path, "-o", out_path};
    #endif

    // disable spirv-opt for coopmat shaders for https://github.com/ggml-org/llama.cpp/issues/10734
    // disable spirv-opt for bf16 shaders for https://github.com/ggml-org/llama.cpp/issues/15344
    // disable spirv-opt for rope shaders for https://github.com/ggml-org/llama.cpp/issues/16860
    // disable spirv-opt for dot2 shaders (spirv-opt doesn't recognize SPV_VALVE_mixed_float_dot_product capability)
    यदि (!coopmat && name.find("bf16") == std::string::npos && name.find("rope") == std::string::npos && name.find("_dot2") == std::string::npos) {
        cmd.push_back("-O");
    }

    यदि (dep_file) {
        cmd.push_back("-MD");
        cmd.push_back("-MF");
#ifdef _WIN32
        cmd.push_back("\"" + target_cpp + ".d\"");
#अन्यथा
        cmd.push_back(target_cpp + ".d");
#endif
    }

    #ifdef GGML_VULKAN_SHADER_DEBUG_INFO
        cmd.push_back("-g");
    #endif

    प्रति (ध्रुव स्वतः& define : defines) {
        cmd.push_back("-D" + define.first + "=" + define.second);
    }

    std::string command;
    प्रति (ध्रुव स्वतः& part : cmd) {
        command += part + " ";
    }

    std::string stdout_str, stderr_str;
    यत्न {
        // std::cout << "Executing command: ";
        // for (const auto& part : cmd) {
        //     std::cout << part << " ";
        // }
        // std::cout << std::endl;

        पूर्णाङ्क exit_code = execute_command(cmd, stdout_str, stderr_str);
        यदि (exit_code != 0 || !stderr_str.empty()) {
            std::cerr << "cannot compile " << name << " (exit code " << exit_code << ")\n\n";
            प्रति (ध्रुव स्वतः& part : cmd) {
                std::cerr << part << " ";
            }
            std::cerr << "\n\n" << stderr_str << std::endl;
            compile_failed = सत्य;
            प्रत्यागम;
        }

        यदि (dep_file) {
            // replace .spv output path with the embed .cpp path which is used as output in CMakeLists.txt
            std::string dep = read_द्विआधारी_file(target_cpp + ".d", सत्य);
            यदि (!dep.empty()) {
                माप_प्रकार pos = dep.find(out_path);
                यदि (pos != std::string::npos) {
                    dep.replace(pos, out_path.length(), target_cpp);
                }
                write_द्विआधारी_file(target_cpp + ".d", dep);
            }
        }

        std::lock_guard<std::mutex> guard(lock);
        shader_fnames.push_back(std::make_pair(name, out_path));
    } गृहाण (ध्रुव std::exception& e) {
        std::cerr << "Error executing command for " << name << ": " << e.what() << std::endl;
        compile_failed = सत्य;
    }
}

std::map<std::string, std::string> merge_maps(ध्रुव std::map<std::string, std::string>& a, ध्रुव std::map<std::string, std::string>& b) {
    std::map<std::string, std::string> result = a;
    result.insert(b.begin(), b.end());
    प्रत्यागम result;
}

स्थिर std::deque<std::future<शून्य>> compiles;
शून्य string_to_spv(std::string name, ध्रुव std::string& source, ध्रुव std::map<std::string, std::string>& defines, तार्किक fp16 = सत्य, तार्किक coopmat = असत्य, तार्किक coopmat2 = असत्य, तार्किक f16acc = असत्य, ध्रुव std::string& suffix = "") {
    name = name + (f16acc ? "_f16acc" : "") + (coopmat ? "_cm1" : "") + (coopmat2 ? "_cm2" : (fp16 ? "" : "_fp32")) + suffix;
    std::string out_path = join_paths(output_dir, name + ".spv");

    यदि (निवेश_filepath == "") {
        // No input source to compile, only generate header for all shaders
        shader_fnames.push_back(std::pair(name, out_path));
        प्रत्यागम;
    } अन्यथा यदि (basename(निवेश_filepath) != source) {
        // Only compile shader variants matching the input filename
        प्रत्यागम;
    }

    compile_गिनती_guard slot = acquire_compile_slot();
    compiles.push_back(std::async(
        string_to_spv_func, name, निवेश_filepath, out_path, defines, coopmat, generate_dep_file, std::move(slot)));
    // Don't write the same dep file from multiple processes
    generate_dep_file = असत्य;

    // Clean up completed futures - don't accumulate virtual memory for completed threads' stacks.
    यावत् (!compiles.empty() && compiles.front().wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        compiles.pop_front();
    }
}

शून्य matmul_shaders(तार्किक fp16, MatMulIdType matmul_id_type, तार्किक coopmat, तार्किक coopmat2, तार्किक f16acc, तार्किक dot2 = असत्य) {
    std::string load_vec = coopmat2 ? "1" : fp16 ? "8" : "4";
    std::string aligned_b_type_f32 = coopmat2 ? "float" : fp16 ? "mat2x4" : "vec4";
    std::string aligned_b_type_f16 = coopmat2 ? "float16_t" : fp16 ? "f16mat2x4" : "f16vec4";
    std::string dot2_sfx = dot2 ? "_dot2" : "";

    std::map<std::string, std::string> base_dict;
    std::string shader_name = "matmul";

    यदि (matmul_id_type == MatMulIdType::DEFAULT) {
        base_dict["MUL_MAT_ID"] = "1";
        shader_name = "matmul_id";
    } अन्यथा यदि (matmul_id_type == MatMulIdType::SUBGROUP) {
        base_dict["MUL_MAT_ID"] = "1";
        base_dict["MUL_MAT_ID_USE_SUBGROUPS"] = "1";
        shader_name = "matmul_id_subgroup";
    }

    यदि (fp16) {
        base_dict["FLOAT16"] = "1";
    }

    base_dict["ACC_TYPE"     ] = f16acc ? "float16_t" : "float";
    base_dict["ACC_TYPEV2"   ] = f16acc ? "f16vec2"   : "vec2";
    base_dict["ACC_TYPE_VEC4"] = f16acc ? "f16vec4"   : "vec4";
    यदि (f16acc) {
        base_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
        base_dict["ACC_F16"] = "1";
    }

    यदि (coopmat) {
        base_dict["COOPMAT"] = "1";
    }
#यदि defined(GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT)
    यदि (coopmat2) {
        base_dict["GGML_VULKAN_COOPMAT2_DECODE_VECTOR"] = "1";
    }
#endif

    यदि (dot2) {
        base_dict["DOT2_F16"] = "1";
    }

    ध्रुव std::string source_name = coopmat2 ? "mul_mm_cm2.comp" : "mul_mm.comp";

    स्वतः ध्रुव &FLOAT_TYPE = [&](पूर्णाङ्क vec, ध्रुव std::string &t) -> std::string {
        विकल्प (vec) {
        पक्ष 1:
            यदि (t == "bf16") {
                // scalar path promotes to float
                यदि (!coopmat && !coopmat2) {
                    प्रत्यागम "float";
                }
                प्रत्यागम "bfloat16_t";
            }
            यदि (coopmat2 || fp16) {
                प्रत्यागम "float16_t";
            }
            प्रत्यागम "float";
        पक्ष 2:
            यदि (t == "bf16") {
                // scalar path promotes to float
                यदि (!coopmat && !coopmat2) {
                    प्रत्यागम "vec2";
                }
                प्रत्यागम "bf16vec2";
            }
            यदि (coopmat2 || fp16) {
                प्रत्यागम "f16vec2";
            }
            प्रत्यागम "vec2";
        पक्ष 4:
            यदि (t == "bf16") {
                // scalar path promotes to float
                यदि (!coopmat && !coopmat2) {
                    प्रत्यागम "vec4";
                }
                प्रत्यागम "bf16vec4";
            }
            यदि (coopmat2 || fp16) {
                प्रत्यागम "f16vec4";
            }
            प्रत्यागम "vec4";
        पक्ष 8:
            यदि (t == "bf16") {
                // scalar path promotes to float
                यदि (!coopmat && !coopmat2) {
                    प्रत्यागम "mat2x4";
                }
                क्षिप std::runtime_error("bf16 vec8 not supported");
            }
            यदि (coopmat2 || fp16) {
                प्रत्यागम "f16mat2x4";
            }
            प्रत्यागम "mat2x4";
        स्वभाव:
            क्षिप std::runtime_error("invalid vector size");
        }
    };

    ध्रुव std::map<std::string, std::string> float_type_dict_f16 = {
        {"FLOAT_TYPE",   FLOAT_TYPE(1, "f16")},
        {"FLOAT_TYPEV2", FLOAT_TYPE(2, "f16")},
        {"FLOAT_TYPEV4", FLOAT_TYPE(4, "f16")},
        {"FLOAT_TYPEV8", FLOAT_TYPE(8, "f16")},
    };

    // Shaders with f16 B_TYPE
    string_to_spv(shader_name + "_f32_f16" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict_f16), {{"DATA_A_F32", "1"}, {"LOAD_VEC_A", load_vec}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

    string_to_spv(shader_name + "_f16" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict_f16), {{"DATA_A_F16", "1"}, {"LOAD_VEC_A", load_vec}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

    // bf16
    {
        // For aligned matmul loads
        std::string load_vec_a = coopmat2 ? "1" : "4";

        // scalar path promotes to float
        std::string to_float_type = (coopmat || coopmat2) ? "uintBitsToBFloat16EXT" : "bf16_to_fp32";

        ध्रुव std::map<std::string, std::string> float_type_dict_bf16 = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "bf16")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "bf16")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "bf16")},
        };

        // If bfloat16 is not supported, then only compile the scalar (promote to fp32) shader
#यदि !defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        यदि (!(coopmat || coopmat2))
#endif
        {
            यदि (!dot2) {
                string_to_spv(shader_name + "_bf16", source_name, merge_maps(merge_maps(base_dict, float_type_dict_bf16), {{"TO_FLOAT_TYPE", to_float_type}, {"DATA_A_BF16", "1"}, {"LOAD_VEC_A", load_vec_a}, {"LOAD_VEC_B", "4"}, {"B_TYPE", coopmat2 ? "bfloat16_t" : "u16vec4"}, {"B_TYPE_SCALAR", coopmat2 ? "bfloat16_t" : "uint16_t"}, {"B_TYPEV4", "bf16vec4"}, {"D_TYPE", "float"}, {"B_IS_FLOAT", "1"}, {"DATA_B_BF16", "1"}}), fp16, coopmat, coopmat2, f16acc);
            }
        }
    }

    प्रति (ध्रुव स्वतः& tname : type_names) {
        यदि (tname == "bf16") {
            अनुवर्त;
        }

        // Float types keep per-type compilation (different accumulation loop structure)
        यदि (tname == "f32" || tname == "f16") {
            std::string data_a_कुंजी = "DATA_A_" + to_uppercase(tname);

            ध्रुव std::map<std::string, std::string> float_type_dict = {
                {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
                {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
                {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
                {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
            };

            यदि (!coopmat2) {
                string_to_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_कुंजी, "1"}, {"LOAD_VEC_A", load_vec}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }
            अनुवर्त;
        }

        std::string data_a_कुंजी = "DATA_A_" + to_uppercase(tname);
        ध्रुव std::map<std::string, std::string> float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
        };

#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        यदि (!f16acc && !coopmat && !coopmat2 && !dot2 && (is_legacy_quant(tname) || is_k_quant(tname) || tname == "mxfp4" || tname == "iq3_s" || tname == "iq4_xs")) {
            string_to_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq.comp", merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_कुंजी, "1"}, {"D_TYPE", "float"},}), fp16, coopmat, coopmat2, f16acc);
        }
#endif

        यदि (!f16acc && coopmat && (tname == "q4_0" || tname == "q4_1" || tname == "q5_0" || tname == "q5_1" || tname == "q8_0" || tname == "iq4_nl" || tname == "iq4_xs" || tname == "mxfp4"
                     || tname == "q3_k" || tname == "q4_k" || tname == "q5_k" || tname == "q6_k" || tname == "nvfp4")) {
            string_to_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq_cm1.comp", merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_कुंजी, "1"}, {"D_TYPE", "float"}, {"D_TYPE_VEC4", "vec4"}}), fp16, coopmat, coopmat2, f16acc);
        }

        यदि (is_lut_quant(tname)) {
            std::string lva = lut_load_vec_a(tname);

            string_to_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_कुंजी, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

            यदि (!coopmat2) {
                string_to_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_कुंजी, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }

#यदि defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            यदि ((tname == "mxfp4" || tname == "nvfp4") && (coopmat || coopmat2)) {
                string_to_spv(shader_name + "_" + tname + "_f16_ocp" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_कुंजी, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                यदि (!coopmat2) {
                    string_to_spv(shader_name + "_" + tname + "_f32_ocp" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_कुंजी, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                }
            }
#endif
            अनुवर्त;
        }

        // dedicated shader needed due to regression on Ampere
        यदि (coopmat2 && (tname == "q4_k" || tname == "q5_k")) {
            string_to_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_कुंजी, "1"}, {"LOAD_VEC_A", load_vec}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }

    // Quant shader: one SPIR-V for all quant types, selected via MmTypeA spec constant
    {
        ध्रुव std::map<std::string, std::string> quant_float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "q4_0")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "q4_0")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "q4_0")},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, "q4_0")},
        };

        string_to_spv(shader_name + "_quant_f16" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

        यदि (!coopmat2) {
            string_to_spv(shader_name + "_quant_f32" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }
}

शून्य process_shaders() {
    // matmul
    प्रति (ध्रुव MatMulIdType& matmul_id_type : {MatMulIdType::NONE, MatMulIdType::DEFAULT, MatMulIdType::SUBGROUP}) {
        // No coopmats
        // fp32
        matmul_shaders(असत्य, matmul_id_type, असत्य, असत्य, असत्य);

        // fp16, fp32acc and fp16acc
        matmul_shaders(सत्य, matmul_id_type, असत्य, असत्य, असत्य);
        matmul_shaders(सत्य, matmul_id_type, असत्य, असत्य, सत्य);

        // dot2 variants (scalar fp16 only)
        matmul_shaders(सत्य, matmul_id_type, असत्य, असत्य, असत्य, सत्य);
        matmul_shaders(सत्य, matmul_id_type, असत्य, असत्य, सत्य,  सत्य);

        यदि (matmul_id_type != MatMulIdType::DEFAULT) {
#यदि defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            // Coopmat, fp32acc and fp16acc
            matmul_shaders(सत्य, matmul_id_type, सत्य, असत्य, असत्य);
            matmul_shaders(सत्य, matmul_id_type, सत्य, असत्य, सत्य);
#endif

#यदि defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            // Coopmat2, fp32acc and fp16acc
            matmul_shaders(सत्य, matmul_id_type, असत्य, सत्य, असत्य);
            matmul_shaders(सत्य, matmul_id_type, असत्य, सत्य, सत्य);
#endif
        }
    }

    प्रति (ध्रुव तार्किक& fp16 : {असत्य, सत्य}) {
        std::map<std::string, std::string> base_dict;
        यदि (fp16) {
            base_dict = {{"FLOAT_TYPE", "float16_t"}, {"FLOAT_TYPEV2", "f16vec2"}, {"FLOAT_TYPEV4", "f16vec4"}, {"FLOAT16", "1"}, {"FLOAT_TYPE_MAX", "float16_t(65504.0)"}};
        } अन्यथा {
            base_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"FLOAT_TYPEV4", "vec4"}};
        }

        // flash attention
        प्रति (ध्रुव तार्किक& f16acc : {असत्य, सत्य}) {
            std::map<std::string, std::string> fa_base_dict = base_dict;
            fa_base_dict["ACC_TYPE"] = fp16 && f16acc ? "float16_t" : "float";
            fa_base_dict["ACC_TYPEV2"] = fp16 && f16acc ? "f16vec2" : "vec2";
            fa_base_dict["ACC_TYPEV4"] = fp16 && f16acc ? "f16vec4" : "vec4";
            // Compile IQ4_NL support into all FA variants so its shared LUT is available when K or V uses it.
            fa_base_dict["DATA_A_IQ4_NL"] = "1";
            यदि (fp16 && f16acc) {
                fa_base_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
            }

            यदि (fp16) {
#यदि defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                string_to_spv("flash_attn_f32_f16", "flash_attn_cm2.comp",
                    merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, असत्य, सत्य, f16acc);
#endif

#यदि defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                string_to_spv("flash_attn_f32_f16", "flash_attn_cm1.comp",
                    merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}), fp16, सत्य, असत्य, f16acc);
#endif
            }

            string_to_spv("flash_attn_f32_f16", "flash_attn.comp",
                merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, असत्य, असत्य, f16acc);

            यदि (fp16) {
                string_to_spv("flash_attn_f32_f16_dot2", "flash_attn.comp",
                    merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"DOT2_F16", "1"}}), fp16, असत्य, असत्य, f16acc);
            }

#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            string_to_spv("flash_attn_f32_f16", "flash_attn.comp",
                merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"MMQ", "1"}, {"FA_MMQ_MIXED", "1"}}), fp16, असत्य, असत्य, f16acc, "_int8");
#endif
        }
    }

    ध्रुव std::map<std::string, std::string> fa_bf16_dict = {
        {"FLOAT_TYPE",   "bfloat16_t"},
        {"FLOAT_TYPEV2", "bf16vec2"},
        {"FLOAT_TYPEV4", "bf16vec4"},
        {"ACC_TYPE",     "float"},
        {"ACC_TYPEV2",   "vec2"},
        {"ACC_TYPEV4",   "vec4"},
        {"BFLOAT16",     "1"},
    };

#यदि defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_to_spv("flash_attn_f32_f16_bf16", "flash_attn_cm1.comp",
        merge_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}),
        सत्य, सत्य, असत्य, असत्य);
#endif

#यदि defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    string_to_spv("flash_attn_f32_f16_bf16", "flash_attn_cm2.comp",
        merge_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}),
        सत्य, असत्य, सत्य, असत्य);
#endif

    std::map<std::string, std::string> base_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}};

    प्रति (ध्रुव स्वतः& tname : type_names) {
        // mul mat vec
        std::string data_a_कुंजी = "DATA_A_" + to_uppercase(tname);
        std::string shader = (string_ends_with(tname, "_k") || string_starts_with(tname, "iq1_") || string_starts_with(tname, "iq2_") || string_starts_with(tname, "iq3_") || tname == "iq4_xs" || tname == "tq2_0" || tname == "tq1_0") ? "mul_mat_vec_" + tname + ".comp" : "mul_mat_vec.comp";

        string_to_spv("mul_mat_vec_" + tname + "_f32_f32", shader, merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_to_spv("mul_mat_vec_" + tname + "_f16_f32", shader, merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));

        string_to_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup", shader, merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_to_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup", shader, merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));

        string_to_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup_no_shmem", shader, merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        string_to_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup_no_shmem", shader, merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

#यदि defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        यदि (tname == "mxfp4" || tname == "nvfp4") {
            string_to_spv("mul_mat_vec_" + tname + "_f32_f32_ocp", shader, merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_to_spv("mul_mat_vec_" + tname + "_f16_f32_ocp", shader, merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));
            string_to_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup", shader, merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_to_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup", shader, merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_to_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_to_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup_no_shmem", shader, merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_कुंजी, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_कुंजी, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_कुंजी, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#endif

        string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_कुंजी, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_कुंजी, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup_no_shmem", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_कुंजी, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

        // mul mat vec with integer dot product
#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        यदि (is_legacy_quant(tname) || tname == "mxfp4" || is_k_quant(tname) || tname == "iq1_s" || tname == "iq1_m" || tname == "iq4_xs") {
            string_to_spv("mul_mat_vec_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_to_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_to_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

            string_to_spv("mul_mat_vec_id_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_कुंजी, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_to_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_कुंजी, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_to_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_कुंजी, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#endif

        // Dequant shaders
        यदि (tname != "f16" && tname != "bf16") {
            string_to_spv("dequant_" + tname, "dequant_" + tname + ".comp", merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"D_TYPE", "float16_t"}}));
        }
        // Fused dequant+transpose variant for FA quant-KV (per-head-contiguous f16 scratch).
        यदि (tname == "q8_0") {
            string_to_spv("dequant_" + tname + "_transpose", "dequant_" + tname + ".comp", merge_maps(base_dict, {{data_a_कुंजी, "1"}, {"D_TYPE", "float16_t"}, {"DEQUANT_TRANSPOSE", "1"}}));
        }

        shader = (tname == "f32" || tname == "f16" || tname == "bf16") ? "get_rows.comp" : "get_rows_quant.comp";

        यदि (tname == "f16") {
            string_to_spv("get_rows_" + tname, shader, merge_maps(base_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {data_a_कुंजी, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}}));
        } अन्यथा {
            string_to_spv("get_rows_" + tname, shader, merge_maps(base_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {data_a_कुंजी, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}}));
        }
        string_to_spv("get_rows_" + tname + "_f32", shader, merge_maps(base_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {data_a_कुंजी, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}}));
    }

    string_to_spv("get_rows_i32", "get_rows.comp", {{"TEMP_TYPE", "uint"}, {"A_TYPE", "uint"}, {"B_TYPE", "int"}, {"D_TYPE", "uint"}});

    string_to_spv("mul_mat_vec_p021_f16_f32_subgroup_add", "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});
    string_to_spv("mul_mat_vec_p021_f16_f32",              "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}});
    string_to_spv("mul_mat_vec_nc_f16_f32", "mul_mat_vec_nc.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}});

    // Norms
    string_to_spv("norm_f32", "norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("group_norm_f32", "group_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("rms_norm_f32", "rms_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("rms_norm_mul_add_f32", "rms_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_ADD_FUSION", "1"}}));
    string_to_spv("rms_norm_mul_add_partials_f32", "rms_norm_partials.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_ADD_FUSION", "1"}}));
    string_to_spv("rms_norm_set_rows_f32_f32", "rms_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_SET_ROWS_FUSION", "1"}}));
    string_to_spv("rms_norm_set_rows_f32_f16", "rms_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"RMS_NORM_SET_ROWS_FUSION", "1"}}));
    string_to_spv("rms_norm_partials_f32", "rms_norm_partials.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("rms_norm_mul_rope_f32_f32", "rms_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
    string_to_spv("rms_norm_mul_rope_f32_f16", "rms_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
    string_to_spv("rms_norm_back_f32", "rms_norm_back.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("l2_norm_f32", "l2_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("cpy_32", "copy.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_to_spv("cpy_f32_f16", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_to_spv("cpy_16", "copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_to_spv("cpy_f16_f32", "copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_to_spv("cpy_f32_bf16","copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_to_spv("cpy_bf16_f32","copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
    string_to_spv("cpy_bf16_f16","copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float16_t"}, {"DATA_A_BF16", "1"}});
    string_to_spv("contig_cpy_32", "contig_copy.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_to_spv("contig_cpy_f32_i32", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_to_spv("contig_cpy_i32_f32", "contig_copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});
    string_to_spv("contig_cpy_f32_f16", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_to_spv("contig_cpy_16", "contig_copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_to_spv("contig_cpy_f16_f32", "contig_copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_to_spv("contig_cpy_f32_bf16","contig_copy.comp",{{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_to_spv("contig_cpy_bf16_f32","contig_copy.comp",{{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
    string_to_spv("contig_cpy_bf16_f16","contig_copy.comp",{{"A_TYPE", "uint16_t"}, {"D_TYPE", "float16_t"}, {"DATA_A_BF16", "1"}});
    string_to_spv("cpy_f32_i32", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_to_spv("cpy_i32_f32", "copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});

    string_to_spv("cpy_transpose_16", "copy_transpose.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_to_spv("cpy_transpose_32", "copy_transpose.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_to_spv("cpy_transpose_02_16", "copy_transpose_02.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_to_spv("cpy_transpose_02_32", "copy_transpose_02.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});

    प्रति (std::string t : {"q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
        string_to_spv("cpy_f32_" + t, "copy_to_quant.comp", {{"DATA_A_" + to_uppercase(t), "1"}, {"S_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        string_to_spv("cpy_" + t + "_f32", "copy_from_quant.comp", {{"DATA_A_" + to_uppercase(t), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }

    प्रति (स्वतः src : {std::pair{"f32", "float"}, std::pair{"f16", "float16_t"}}) {
        प्रति (std::string dst : {"f32", "f16", "bf16", "q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
            string_to_spv("set_rows_" + std::string(src.first) + "_" + dst + "_i32", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + to_uppercase(dst), "1"}, {"B_TYPE", "uint"}, {"B_SIZE", "32"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
            string_to_spv("set_rows_" + std::string(src.first) + "_" + dst + "_i64", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + to_uppercase(dst), "1"}, {"B_TYPE", "uvec2"}, {"B_SIZE", "64"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        }
    }

    स्वतः get_type_str = [](तार्किक f16) {
        प्रत्यागम f16 ? "float16_t" : "float";
    };
    स्वतः get_suffix = [](तार्किक src0_f16, तार्किक src1_f16, तार्किक dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        प्रत्यागम s;
    };
    प्रति (std::string op : {"add", "sub", "mul", "div", "add_rms", }) {
    प्रति (स्वतः src0_f16 : {असत्य, सत्य}) {
    प्रति (स्वतः src1_f16 : {असत्य, सत्य}) {
    प्रति (स्वतः dst_f16  : {असत्य, सत्य}) {
        स्वतः source = op == "add_rms" ? std::string("add") : op;
        स्वतः name = op + get_suffix(src0_f16, src1_f16, dst_f16);
        स्वतः add_rms = op == "add_rms" ? "1" : "0";
        string_to_spv(name.c_str(), source + ".comp", {{"A_TYPE", get_type_str(src0_f16)}, {"B_TYPE", get_type_str(src1_f16)}, {"D_TYPE", get_type_str(dst_f16)}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , add_rms}});
    }
    }
    }
    }

    string_to_spv("sub_f32", "sub.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("acc_f32", "acc.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("split_k_reduce", "mul_mat_split_k_reduce.comp", {});
    string_to_spv("fa_split_k_reduce", "flash_attn_split_k_reduce.comp", {});

    string_to_spv("fa_mask_opt", "flash_attn_mask_opt.comp", {});

#यदि defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_to_spv("fa_decode_ph1", "flash_attn_decode_phase_1.comp", {}, सत्य, सत्य, असत्य, असत्य);
    string_to_spv("fa_decode_ph2", "flash_attn_decode_phase_2.comp", {}, सत्य, सत्य, असत्य, असत्य);
#endif

    string_to_spv("fa_sparse_compact", "flash_attn_sparse_compact.comp", {});
    string_to_spv("fa_sparse_compact_subgroup", "flash_attn_sparse_compact.comp", {{"USE_SUBGROUPS", "1"}});

    string_to_spv("quantize_q8_1", "quantize_q8_1.comp", {});
    string_to_spv("quantize_q8_1_subgroup", "quantize_q8_1.comp", {{"USE_SUBGROUPS", "1"}});

    string_to_spv("quantize_q8_1_x4", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}});
    string_to_spv("quantize_q8_1_x4_subgroup", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}, {"USE_SUBGROUPS", "1"}});

    string_to_spv("mul_f32", "mul.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("div_f32", "div.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("repeat_i32", "repeat.comp", {{"A_TYPE", "int32_t"}, {"D_TYPE", "int32_t"}});
    string_to_spv("repeat_back_f32", "repeat_back.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("get_rows_back_f32", "get_rows_back.comp", {{"A_TYPE", "float"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}});

    string_to_spv("repeat_i16", "repeat.comp", {{"A_TYPE", "int16_t"}, {"D_TYPE", "int16_t"}});

    string_to_spv("scale_f32", "scale.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("pad_f32", "pad.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("pad_reflect_1d_f32", "pad_reflect_1d.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("concat_i8", "concat.comp", {{"A_TYPE", "uint8_t"}, {"B_TYPE", "uint8_t"}, {"D_TYPE", "uint8_t"}});
    string_to_spv("concat_i16", "concat.comp", {{"A_TYPE", "uint16_t"}, {"B_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_to_spv("concat_i32", "concat.comp", {{"A_TYPE", "uint"}, {"B_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_to_spv("concat_i64", "concat.comp", {{"A_TYPE", "uvec2"}, {"B_TYPE", "uvec2"}, {"D_TYPE", "uvec2"}});

    string_to_spv("upscale_f32", "upscale.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("exp_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_exp"}});
    string_to_spv("exp_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_exp"}});
    string_to_spv("expm1_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_expm1"}});
    string_to_spv("expm1_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_expm1"}});

    string_to_spv("log_f16",        "log.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("log_f32",        "log.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("gelu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu"}});
    string_to_spv("gelu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu"}});
    string_to_spv("gelu_erf_f16",   "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu_erf"}});
    string_to_spv("gelu_erf_f32",   "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu_erf"}});
    string_to_spv("gelu_quick_f16", "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu_quick"}});
    string_to_spv("gelu_quick_f32", "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu_quick"}});
    string_to_spv("silu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_silu"}});
    string_to_spv("silu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_silu"}});
    string_to_spv("relu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_relu"}});
    string_to_spv("relu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_relu"}});
    string_to_spv("sqr_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sqr"}});
    string_to_spv("sqr_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sqr"}});
    string_to_spv("sqrt_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sqrt"}});
    string_to_spv("sqrt_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sqrt"}});
    string_to_spv("sin_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sin"}});
    string_to_spv("sin_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sin"}});
    string_to_spv("cos_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_cos"}});
    string_to_spv("cos_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_cos"}});
    string_to_spv("clamp_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_clamp"}});
    string_to_spv("clamp_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_clamp"}});
    string_to_spv("leaky_relu_f16", "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_leaky_relu"}});
    string_to_spv("leaky_relu_f32", "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_leaky_relu"}});
    string_to_spv("neg_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_neg"}});
    string_to_spv("neg_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_neg"}});
    string_to_spv("tanh_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_tanh"}});
    string_to_spv("tanh_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_tanh"}});
    string_to_spv("sigmoid_f16",    "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sigmoid"}});
    string_to_spv("sigmoid_f32",    "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sigmoid"}});
    string_to_spv("hardsigmoid_f16","unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_hardsigmoid"}});
    string_to_spv("hardsigmoid_f32","unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_hardsigmoid"}});
    string_to_spv("hardswish_f16",  "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_hardswish"}});
    string_to_spv("hardswish_f32",  "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_hardswish"}});
    string_to_spv("abs_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_abs"}});
    string_to_spv("abs_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_abs"}});
    string_to_spv("elu_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_elu"}});
    string_to_spv("elu_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_elu"}});
    string_to_spv("xielu_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_xielu"}});
    string_to_spv("xielu_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_xielu"}});
    string_to_spv("sgn_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sgn"}});
    string_to_spv("sgn_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sgn"}});

    string_to_spv("tri_f16",        "tri.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("tri_f32",        "tri.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("diag_f16",       "diag.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("diag_f32",       "diag.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_to_spv("softplus_f16",   "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_softplus"}});
    string_to_spv("softplus_f32",   "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_softplus"}});

    string_to_spv("gelu_mul_f32",    "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_gelu"},     {"UNARY_MUL_FUSION", "1"}});
    string_to_spv("gelu_mul_f16",    "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_gelu"},     {"UNARY_MUL_FUSION", "1"}});
    string_to_spv("sigmoid_mul_f32", "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_sigmoid"},  {"UNARY_MUL_FUSION", "1"}});
    string_to_spv("sigmoid_mul_f16", "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_sigmoid"},  {"UNARY_MUL_FUSION", "1"}});
    string_to_spv("silu_mul_f32",    "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_silu"},     {"UNARY_MUL_FUSION", "1"}});
    string_to_spv("silu_mul_f16",    "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_silu"},     {"UNARY_MUL_FUSION", "1"}});
    string_to_spv("softplus_mul_f32","unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_softplus"}, {"UNARY_MUL_FUSION", "1"}});
    string_to_spv("softplus_mul_f16","unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_softplus"}, {"UNARY_MUL_FUSION", "1"}});

    string_to_spv("add1_f16_f16",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_to_spv("add1_f16_f32",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_to_spv("add1_f32_f32",   "add1.comp",        {{"A_TYPE", "float"},       {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_to_spv("arange_f32",     "arange.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_to_spv("fill_f32",       "fill.comp",        {{"D_TYPE", "float"},       {"FLOAT_TYPE", "float"}});
    string_to_spv("fill_f16",       "fill.comp",        {{"D_TYPE", "float16_t"},   {"FLOAT_TYPE", "float"}});
    string_to_spv("step_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_step"}});
    string_to_spv("step_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_step"}});
    string_to_spv("round_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_round"}});
    string_to_spv("round_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_round"}});
    string_to_spv("ceil_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_ceil"}});
    string_to_spv("ceil_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_ceil"}});
    string_to_spv("floor_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_floor"}});
    string_to_spv("floor_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_floor"}});
    string_to_spv("trunc_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_trunc"}});
    string_to_spv("trunc_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_trunc"}});

    string_to_spv("geglu_f16",      "geglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("geglu_f32",      "geglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("reglu_f16",      "reglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("reglu_f32",      "reglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("swiglu_f16",     "swiglu.comp",      {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("swiglu_f32",     "swiglu.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("swiglu_oai_f16", "swiglu_oai.comp",  {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("swiglu_oai_f32", "swiglu_oai.comp",  {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("swiglu_clamp_f16", "swiglu_clamp.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_to_spv("swiglu_clamp_f32", "swiglu_clamp.comp", {{"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_to_spv("geglu_erf_f16",  "geglu_erf.comp",   {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("geglu_erf_f32",  "geglu_erf.comp",   {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("geglu_quick_f16","geglu_quick.comp", {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("geglu_quick_f32","geglu_quick.comp", {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_to_spv("silu_back_f32",  "silu_back.comp",   {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("diag_mask_inf_f32", "diag_mask_inf.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("soft_max_f32", "soft_max.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_f32_f16", "soft_max.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_back_f32", "soft_max_back.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("soft_max_large1_f32", "soft_max_large1.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_large2_f32", "soft_max_large2.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_large3_f32", "soft_max_large3.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_large1_f32_f16", "soft_max_large1.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_large2_f32_f16", "soft_max_large2.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_large3_f32_f16", "soft_max_large3.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));

    string_to_spv("rope_norm_f32", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_to_spv("rope_norm_f16", "rope_norm.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_to_spv("rope_norm_f32_f16", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_to_spv("rope_neox_f32", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_to_spv("rope_neox_f16", "rope_neox.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_to_spv("rope_neox_f32_f16", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_to_spv("rope_multi_f32", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_to_spv("rope_multi_f16", "rope_multi.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_to_spv("rope_multi_f32_f16", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_to_spv("rope_vision_f32", "rope_vision.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_to_spv("rope_vision_f16", "rope_vision.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});

    string_to_spv("argsort_f32", "argsort.comp", {{"A_TYPE", "float"}});
    string_to_spv("argsort_large_f32", "argsort_large.comp", {{"A_TYPE", "float"}});

    string_to_spv("topk_argsort_f32", "topk_argsort.comp", {{"A_TYPE", "float"}});
    string_to_spv("topk_nary_search_f32", "topk_nary_search.comp", {{"A_TYPE", "float"}});
    string_to_spv("topk_radix_select_f32", "topk_radix_select.comp", {{"A_TYPE", "float"}});

    string_to_spv("argmax_f32", "argmax.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "int"}}));
    string_to_spv("sum_rows_f32", "sum_rows.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("cross_entropy_loss_f32", "cross_entropy_loss.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("cross_entropy_loss_back_f32", "cross_entropy_loss_back.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("fwht_f32", "fwht.comp", {});
    string_to_spv("fwht_shmem_f32", "fwht.comp", {{"FWHT_SHMEM", "1"}});
    string_to_spv("count_equal_i32", "count_equal.comp", merge_maps(base_dict, {{"A_TYPE", "int"}, {"B_TYPE", "int"}, {"D_TYPE", "int"}}));
    string_to_spv("dsv4_hc_comb_f32", "dsv4_hc_comb.comp", {});
    string_to_spv("dsv4_hc_pre_f32",  "dsv4_hc_pre.comp",  {});
    string_to_spv("dsv4_hc_post_f32", "dsv4_hc_post.comp", {});
    string_to_spv("cumsum_f32", "cumsum.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("cumsum_multipass1_f32", "cumsum_multipass1.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("cumsum_multipass2_f32", "cumsum_multipass2.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("count_experts", "count_experts.comp", merge_maps(base_dict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}}));
    string_to_spv("count_experts_subgroup", "count_experts.comp", merge_maps(base_dict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}, {"USE_SUBGROUPS", "1"}}));

    प्रति (std::string dim_str : {"", "_3d"}) {
        प्रति (तार्किक bda : {असत्य, सत्य}) {
            std::string bda_str = bda ? "_bda" : "";
            std::string bda_def = bda ? "1" : "0";
            string_to_spv("im2col" + dim_str + "_f32" + bda_str, "im2col" + dim_str + ".comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"D_SIZE", "4"}, {"BDA", bda_def}}));
            string_to_spv("im2col" + dim_str + "_f32_f16" + bda_str, "im2col" + dim_str + ".comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"D_SIZE", "2"}, {"BDA", bda_def}}));
        }
    }

    string_to_spv("out_prod_f32", "out_prod.comp", {});

    string_to_spv("timestep_embedding_f32", "timestep_embedding.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("conv_transpose_1d_f32", "conv_transpose_1d.comp", {{"A_TYPE", "float"},  {"B_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("col2im_1d_f32",  "col2im_1d.comp", {{"DATA_A_F32", "1"},  {"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_to_spv("col2im_1d_f16",  "col2im_1d.comp", {{"DATA_A_F16", "1"},  {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_to_spv("col2im_1d_bf16", "col2im_1d.comp", {{"DATA_A_BF16", "1"}, {"A_TYPE", "uint16_t"},  {"D_TYPE", "uint16_t"}});

    string_to_spv("snake_f32",  "snake.comp", {{"DATA_A_F32", "1"},  {"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_to_spv("snake_f16",  "snake.comp", {{"DATA_A_F16", "1"},  {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_to_spv("snake_bf16", "snake.comp", {{"DATA_A_BF16", "1"}, {"DATA_D_BF16", "1"}, {"A_TYPE", "uint16_t"},  {"D_TYPE", "uint16_t"}});

    string_to_spv("pool1d_f32", "pool1d.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("pool2d_f32", "pool2d.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("pool2d_subgroup_f32", "pool2d.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"USE_SUBGROUP", "1"}}));

    string_to_spv("rwkv_wkv6_f32", "wkv6.comp", merge_maps(base_dict, {{"A_TYPE", "float"}}));

    string_to_spv("gated_linear_attn_f32", "gla.comp", merge_maps(base_dict, {{"A_TYPE", "float"}}));

    // Compile IQ4_NL support in so its shared LUT is available when K uses it.
    // K quant type is selected at runtime via the FaTypeK spec constant.
    std::map<std::string, std::string> li_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV4", "vec4"}, {"DATA_A_IQ4_NL", "1"}};
    string_to_spv("lightning_indexer_f32", "lightning_indexer.comp", li_dict);

    string_to_spv("rwkv_wkv7_f32", "wkv7.comp", merge_maps(base_dict, {{"A_TYPE", "float"}}));

    string_to_spv("gated_delta_net_f32", "gated_delta_net.comp", merge_maps(base_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "1"}}));
    string_to_spv("gated_delta_net_f32_nocluster", "gated_delta_net.comp", merge_maps(base_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));
    string_to_spv("gated_delta_net_f32_shmem", "gated_delta_net.comp", merge_maps(base_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "0"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));

    string_to_spv("opt_step_adamw_f32", "opt_step_adamw.comp", merge_maps(base_dict, {{"A_TYPE", "float"}}));
    string_to_spv("opt_step_sgd_f32", "opt_step_sgd.comp", merge_maps(base_dict, {{"A_TYPE", "float"}}));

    string_to_spv("solve_tri_f32", "solve_tri.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    प्रति (स्वतः transpose : {असत्य, सत्य}) {
        प्रति (स्वतः unroll : {असत्य, सत्य}) {
            प्रति (स्वतः a_f16 : {असत्य, सत्य}) {
                std::map<std::string, std::string> defines = {
                    {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                    {"USE_COLLECTIVES", "1"}, {"UNROLL", unroll ? "[[unroll]]" : ""},
                };
                यदि (transpose) defines["TRANSPOSE"] = "1";
                std::string name = std::string(transpose ? "conv_transpose_2d": "conv2d")
                    + (a_f16 ? "_f16" : "") + "_f32";
                string_to_spv(name + (unroll ? "_unroll" : ""), "conv2d_mm.comp", defines);
#यदि defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                यदि (unroll) {
                    स्वतः cm2_defines = defines;
                    cm2_defines["COOPMAT2"] = "1";
                    string_to_spv(name, "conv2d_mm.comp", cm2_defines, सत्य, असत्य, सत्य);
                }
#endif
#यदि defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                यदि (unroll) {
                    स्वतः cm1_defines = defines;
                    cm1_defines["COOPMAT"] = "1";
                    string_to_spv(name, "conv2d_mm.comp", cm1_defines, सत्य, सत्य, असत्य);
                }
#endif
            }
        }
    }

    प्रति (स्वतः unroll : {असत्य, सत्य}) {
        प्रति (स्वतः a_f16 : {असत्य, सत्य}) {
            std::map<std::string, std::string> defines = {
                {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                {"UNROLL", unroll ? "[[unroll]]" : ""},
            };
            std::string name = std::string("conv3d") + (a_f16 ? "_f16" : "") + "_f32";
            string_to_spv(name + (unroll ? "_unroll" : ""), "conv3d_mm.comp", defines);
#यदि defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            यदि (unroll) {
                स्वतः cm2_defines = defines;
                cm2_defines["COOPMAT2"] = "1";
                string_to_spv(name, "conv3d_mm.comp", cm2_defines, सत्य, असत्य, सत्य);
            }
#endif
#यदि defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            यदि (unroll) {
                स्वतः cm1_defines = defines;
                cm1_defines["COOPMAT"] = "1";
                string_to_spv(name, "conv3d_mm.comp", cm1_defines, सत्य, सत्य, असत्य);
            }
#endif
        }
    }

    string_to_spv("conv2d_dw_whcn_f32", "conv2d_dw.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_to_spv("conv2d_dw_cwhn_f32", "conv2d_dw.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));
    string_to_spv("conv2d_dw_whcn_f16_f32", "conv2d_dw.comp", merge_maps(base_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_to_spv("conv2d_dw_cwhn_f16_f32", "conv2d_dw.comp", merge_maps(base_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));

    string_to_spv("roll_f32", "roll.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("add_id_f32", "add_id.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("multi_add_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , "0"}});
    string_to_spv("multi_add_rms_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , "1"}});

    string_to_spv("ssm_scan_f32",          "ssm_scan.comp", {{"A_TYPE", "float"}});
    string_to_spv("ssm_scan_subgroup_f32", "ssm_scan.comp", {{"A_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});

    string_to_spv("ssm_conv_f32", "ssm_conv.comp", {{"A_TYPE", "float"}});

    string_to_spv("topk_moe_f32", "topk_moe.comp", {});

    प्रति (स्वतः &c : compiles) {
        c.रुको();
    }
}

शून्य write_output_files() {
    std::stringstream hdr = make_generic_stringstream();
    std::stringstream src = make_generic_stringstream();

    hdr << "#include <cstdint>\n\n";
    src << "#include \"" << basename(target_hpp) << "\"\n\n";

    std::sort(shader_fnames.begin(), shader_fnames.end());
    प्रति (ध्रुव स्वतः& pair : shader_fnames) {
        ध्रुव std::string& name = pair.first;
        #ifdef _WIN32
            std::string path = pair.second;
            std::replace(path.begin(), path.end(), '/', '\\' );
        #अन्यथा
            ध्रुव std::string& path = pair.second;
        #endif

        hdr << "extern const uint64_t " << name << "_len;\n";
        hdr << "extern const unsigned char " << name << "_data[];\n\n";

        यदि (निवेश_filepath != "") {
            std::string data = read_द्विआधारी_file(path);
            यदि (data.empty()) {
                अनुवर्त;
            }

            src << "const uint64_t " << name << "_len = " << data.size() << ";\n";
            src << "const unsigned char " << name << "_data[" << data.size() << "] = {\n" << std::hex;
            स्वतः bytes = reinterpret_cast<ध्रुव uint8_t*>(data.data());
            प्रति (माप_प्रकार i = 0; i < data.size(); ++i) {
                src << "0x" << static_cast<पूर्णाङ्क>(bytes[i]) << ",";
                यदि ((i + 1) % 12 == 0) src << "\n";
            }
            src << std::dec << "\n};\n\n";
        }
    }

    std::string suffixes[2] = {"_f32", "_f16"};
    प्रति (std::string op : {"add", "sub", "mul", "div", "add_rms"}) {
        hdr << "extern const void * " << op << "_data[2][2][2];\n";
        hdr << "extern const uint64_t " << op << "_len[2][2][2];\n";

        std::string op_file = op == "add_rms" ? "add.comp" : std::string(op) + ".comp";
        यदि (basename(निवेश_filepath) != op_file) {
            अनुवर्त;
        }
        std::stringstream data = make_generic_stringstream();
        std::stringstream len  = make_generic_stringstream();
        data << "const void * " << op << "_data[2][2][2] = ";
        len  << "const uint64_t " << op << "_len[2][2][2] = ";
        प्रति (uint32_t t0 = 0; t0 < 2; ++t0) {
            यदि (t0 == 0) {
                data << "{";
                len  << "{";
            }
            प्रति (uint32_t t1 = 0; t1 < 2; ++t1) {
                यदि (t1 == 0) {
                    data << "{";
                    len  << "{";
                }
                प्रति (uint32_t t2 = 0; t2 < 2; ++t2) {
                    यदि (t2 == 0) {
                        data << "{";
                        len  << "{";
                    }
                    data << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    len  << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    data << "_data,";
                    len  << "_len,";
                    यदि (t2 == 1) {
                        data << "}, ";
                        len  << "}, ";
                    }
                }
                यदि (t1 == 1) {
                    data << "}, ";
                    len  << "}, ";
                }
            }
            यदि (t0 == 1) {
                data << "};\n";
                len  << "};\n";
            }
        }
        src << data.str();
        src << len.str();
    }

    std::vector<std::string> btypes = {"f16", "f32"};

#यदि defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    btypes.push_back("q8_1");
#endif

    प्रति (ध्रुव std::string& btype : btypes) {
    प्रति (ध्रुव स्वतः& tname : type_names) {
        यदि (btype == "q8_1" && !is_legacy_quant(tname) && tname != "mxfp4" && !is_k_quant(tname) && tname != "iq1_s" && tname != "iq1_m" && tname != "iq4_xs") {
            अनुवर्त;
        }
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3];\n";
        यदि (basename(निवेश_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }

        यदि (btype == "f16") {
            अनुवर्त;
        }
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3];\n";
        यदि (basename(निवेश_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_id_" << tname << "_" << btype << "_f32_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_id_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_id_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }
    }
    }

#यदि defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    प्रति (ध्रुव std::string& btype : {"f16", "f32"}) {
    प्रति (ध्रुव std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3];\n";
        यदि (basename(निवेश_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
    }
    प्रति (ध्रुव std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3];\n";
        यदि (basename(निवेश_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3] = {mul_mat_vec_id_" << tname << "_f32_f32_ocp_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3] =  {mul_mat_vec_id_" << tname << "_f32_f32_ocp_len,  mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_len, mul_mat_vec_id_"  << tname << "_f32_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
#endif

    यदि (निवेश_filepath == "") {
        write_file_if_changed(target_hpp, hdr.str());
    }
    यदि (target_cpp != "") {
        write_द्विआधारी_file(target_cpp, src.str());
    }
}

} // namespace

पूर्णाङ्क main(पूर्णाङ्क argc, वर्ण** argv) {
    std::map<std::string, std::string> args;
    प्रति (पूर्णाङ्क i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        यदि (arg.rfind("--", 0) == 0) {
            यदि (i + 1 < argc && argv[i + 1][0] != '-') {
                args[arg] = argv[i + 1];
                ++i;
            } अन्यथा {
                args[arg] = "";
            }
        }
    }

    यदि (args.find("--glslc") != args.end()) {
        GLSLC = args["--glslc"]; // Path to glslc
    }
    यदि (args.find("--source") != args.end()) {
        निवेश_filepath = args["--source"]; // The shader source file to compile
    }
    यदि (args.find("--output-dir") != args.end()) {
        output_dir = args["--output-dir"]; // Directory for containing SPIR-V output
    }
    यदि (args.find("--target-hpp") != args.end()) {
        target_hpp = args["--target-hpp"]; // Path to generated header file
    }
    यदि (args.find("--target-cpp") != args.end()) {
        target_cpp = args["--target-cpp"]; // Path to generated cpp file
    }

    यदि (!directory_exists(output_dir)) {
        यदि (!create_directory(output_dir)) {
            std::cerr << "Error creating output directory: " << output_dir << "\n";
            प्रत्यागम निकास_त्रुटि;
        }
    }

    process_shaders();

    यदि (compile_failed) {
        std::cerr << "vulkan-shaders-gen: one or more shaders failed to compile" << std::endl;
        प्रत्यागम निकास_त्रुटि;
    }

    write_output_files();

    प्रत्यागम निकास_सफल;
}
