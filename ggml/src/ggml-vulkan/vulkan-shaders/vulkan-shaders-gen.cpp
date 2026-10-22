#અન્તર્ભાવ <iostream>
#અન્તર્ભાવ <fstream>
#અન્તર્ભાવ <sstream>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <stdexcept>
#અન્તર્ભાવ <array>
#અન્તર્ભાવ <vector>
#અન્તર્ભાવ <map>
#અન્તર્ભાવ <thread>
#અન્તર્ભાવ <mutex>
#અન્તર્ભાવ <future>
#અન્તર્ભાવ <queue>
#અન્તર્ભાવ <condition_variable>
#અન્તર્ભાવ <atomic>
#અન્તર્ભાવ <cstdio>
#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <cassert>
#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <sys/stat.h>
#અન્તર્ભાવ <sys/types.h>
#અન્તર્ભાવ <filesystem>

#ifdef _WIN32
    #define NOMINMAX
    #અન્તર્ભાવ <windows.h>
    #અન્તર્ભાવ <direct.h> // For _mkdir on Windows
#નહીંતર
    #અન્તર્ભાવ <unistd.h>
    #અન્તર્ભાવ <sys/wait.h>
    #અન્તર્ભાવ <fcntl.h>
#endif

#define ASYNCIO_CONCURRENCY 64

std::mutex lock;
std::vector<std::pair<std::string, std::string>> shader_fnames;
// Set when any shader subprocess fails (non-zero exit / stderr / launch failure) so the
// build is stopped instead of silently producing a broken libggml-vulkan. (issue #24393)
સ્થિર std::atomic<તાર્કિક> compile_failed{ખોટું};
std::locale c_locale("C");

std::string GLSLC = "glslc";
std::string આગત_filepath = "";
std::string output_dir = "/tmp";
std::string target_hpp = "";
std::string target_cpp = "";

અચલ std::vector<std::string> type_names = {
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

ગણના MatMulIdType {
    NONE,
    DEFAULT,
    SUBGROUP,
};

નામાકાશ {

પૂર્ણાંક execute_command(std::vector<std::string>& command, std::string& stdout_str, std::string& stderr_str) {
#ifdef _WIN32
    HANDLE stdout_read, stdout_write;
    HANDLE stderr_read, stderr_write;
    SECURITY_ATTRIBUTES sa = { કદ(SECURITY_ATTRIBUTES), ખાલી, TRUE };

    જો (!CreatePipe(&stdout_read, &stdout_write, &sa, 0) ||
        !SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0)) {
        ફેંકો std::runtime_error("Failed to create stdout pipe");
    }

    જો (!CreatePipe(&stderr_read, &stderr_write, &sa, 0) ||
        !SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0)) {
        ફેંકો std::runtime_error("Failed to create stderr pipe");
    }

    PROCESS_INFORMATION pi;
    STARTUPINFOA si = {};
    si.cb = કદ(STARTUPINFOA);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = stdout_write;
    si.hStdError = stderr_write;

    std::string cmd;
    માટે (અચલ auto& part : command) {
        cmd += part + " ";
    }

    જો (!CreateProcessA(ખાલી, cmd.data(), ખાલી, ખાલી, TRUE, 0, ખાલી, ખાલી, &si, &pi)) {
        ફેંકો std::runtime_error("Failed to create process");
    }

    CloseHandle(stdout_write);
    CloseHandle(stderr_write);

    std::array<અક્ષર, 128> buffer;
    DWORD bytes_read;

    જ્યારે (ReadFile(stdout_read, buffer.data(), (DWORD)buffer.size(), &bytes_read, ખાલી) && bytes_read > 0) {
        stdout_str.append(buffer.data(), bytes_read);
    }

    જ્યારે (ReadFile(stderr_read, buffer.data(), (DWORD)buffer.size(), &bytes_read, ખાલી) && bytes_read > 0) {
        stderr_str.append(buffer.data(), bytes_read);
    }

    CloseHandle(stdout_read);
    CloseHandle(stderr_read);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    પાછા (પૂર્ણાંક)exit_code;
#નહીંતર
    પૂર્ણાંક stdout_pipe[2];
    પૂર્ણાંક stderr_pipe[2];

    જો (પ્રણાલ(stdout_pipe) != 0 || પ્રણાલ(stderr_pipe) != 0) {
        ફેંકો std::runtime_error("Failed to create pipes");
    }

    pid_t pid = fork();
    જો (pid < 0) {
        std::cerr << strerror(errno) << "\n";
        ફેંકો std::runtime_error("Failed to fork process");
    }

    std::vector<અક્ષર*> argv;
    માટે (std::string& part : command) {
        argv.push_back(part.data());
    }
    argv.push_back(nullptr);

    જો (pid == 0) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        execvp(argv[0], argv.data());
        _exit(EXIT_FAILURE);
    } નહીંતર {
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        std::array<અક્ષર, 128> buffer;
        ssize_t bytes_read;

        જ્યારે ((bytes_read = પઠ(stdout_pipe[0], buffer.data(), buffer.size())) > 0) {
            stdout_str.append(buffer.data(), bytes_read);
        }

        જ્યારે ((bytes_read = પઠ(stderr_pipe[0], buffer.data(), buffer.size())) > 0) {
            stderr_str.append(buffer.data(), bytes_read);
        }

        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        પૂર્ણાંક status = 0;
        waitpid(pid, &status, 0);
        પાછા WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
#endif
}

તાર્કિક directory_exists(અચલ std::string& path) {
    સંરચના stat info;
    જો (stat(path.c_str(), &info) != 0) {
        પાછા ખોટું; // Path doesn't exist or can't be accessed
    }
    પાછા (info.st_mode & S_IFDIR) != 0; // Check if it is a directory
}

તાર્કિક create_directory(અચલ std::string& path) {
#ifdef _WIN32
    પાછા _mkdir(path.c_str()) == 0 || errno == EEXIST; // EEXIST means the directory already exists
#નહીંતર
    પાછા mkdir(path.c_str(), 0755) == 0 || errno == EEXIST; // 0755 is the directory permissions
#endif
}

std::string to_uppercase(અચલ std::string& આગત) {
    std::string result = આગત;
    માટે (અક્ષર& c : result) {
        c = std::toupper(c);
    }
    પાછા result;
}

તાર્કિક string_starts_with(અચલ std::string& str, અચલ std::string& prefix) {
    જો (prefix.size() > str.size()) {
        પાછા ખોટું;
    }
    પાછા std::equal(prefix.begin(), prefix.end(), str.begin());
}

તાર્કિક string_ends_with(અચલ std::string& str, અચલ std::string& suffix) {
    જો (suffix.size() > str.size()) {
        પાછા ખોટું;
    }
    પાછા std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

તાર્કિક is_quantized_type(અચલ std::string& type_name) {
    પાછા type_name != "f32" && type_name != "f16" && type_name != "bf16";
}

તાર્કિક is_legacy_quant(અચલ std::string& type_name) {
    પાછા type_name == "q2_0" || type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";
}

તાર્કિક is_k_quant(અચલ std::string& type_name) {
    પાછા string_ends_with(type_name, "_k");
}

તાર્કિક is_iq_quant(અચલ std::string& type_name) {
    પાછા string_starts_with(type_name, "iq");
}

તાર્કિક is_lut_quant(અચલ std::string& type_name) {
    પાછા is_iq_quant(type_name) || type_name == "mxfp4" || type_name == "nvfp4";
}

std::string lut_load_vec_a(અચલ std::string& type_name) {
    જો (type_name == "iq1_s" || type_name == "iq1_m" || type_name == "iq2_xxs" || type_name == "iq2_xs" || type_name == "iq2_s" || type_name == "iq4_xs") {
        પાછા "8";
    }
    પાછા "4";
}

સ્થિર અચલ અક્ષર path_separator = '/';

std::string join_paths(અચલ std::string& path1, અચલ std::string& path2) {
    પાછા path1 + path_separator + path2;
}

std::string basename(અચલ std::string &path) {
    પાછા path.substr(path.find_last_of("/\\") + 1);
}

std::stringstream make_generic_stringstream() {
    std::stringstream ss;
    ss.imbue(c_locale);
    પાછા ss;
}

std::string read_દ્વિઅંકી_file(અચલ std::string& path, તાર્કિક may_not_exist = ખોટું) {
    FILE* f = fopen(path.c_str(), "rb");
    જો (!f) {
        જો (!may_not_exist) {
            std::cerr << "Error opening file: " << path << " (" << strerror(errno) << ")\n";
        }
        પાછા {};
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::string data(size, '\0');
    size_t read_size = fread(data.data(), 1, size, f);
    fclose(f);
    જો (read_size != size) {
        std::cerr << "Error reading file: " << path << " (" << strerror(errno) << ")\n";
        પાછા {};
    }

    પાછા data;
}

શૂન્ય write_દ્વિઅંકી_file(અચલ std::string& path, અચલ std::string& content) {
    FILE* f = fopen(path.c_str(), "wb");
    જો (!f) {
        std::cerr << "Error opening file for writing: " << path << " (" << strerror(errno) << ")\n";
        પાછા;
    }

    size_t write_size = fwrite(content.data(), 1, content.size(), f);
    fclose(f);
    જો (write_size != content.size()) {
        std::cerr << "Error writing file: " << path << " (" << strerror(errno) << ")\n";
        પાછા;
    }
}

શૂન્ય write_file_if_changed(અચલ std::string& path, અચલ std::string& content) {
    std::string existing = read_દ્વિઅંકી_file(path, સાચું);
    જો (existing != content) {
        write_દ્વિઅંકી_file(path, content);
    }
}


// variables to track number of compiles in progress
સ્થિર uint32_t compile_ગણતરી = 0;
સ્થિર std::mutex compile_ગણતરી_mutex;
સ્થિર std::condition_variable compile_ગણતરી_cond;
સ્થિર તાર્કિક generate_dep_file = સાચું;

શૂન્ય decrement_compile_ગણતરી(uint32_t * count) {
    જો (count) {
        std::lock_guard<std::mutex> guard(compile_ગણતરી_mutex);
        પ્રતિજ્ઞા(compile_ગણતરી > 0);
        compile_ગણતરી--;
        compile_ગણતરી_cond.notify_all();
    }
}

using compile_ગણતરી_guard = std::unique_ptr<uint32_t, decltype(&decrement_compile_ગણતરી)>;

compile_ગણતરી_guard acquire_compile_slot() {
    // wait until fewer than N compiles are in progress.
    // 16 is an arbitrary limit, the goal is to avoid "failed to create pipe" errors.
    uint32_t N = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::unique_lock<std::mutex> guard(compile_ગણતરી_mutex);
    compile_ગણતરી_cond.wait(guard, [N] { પાછા compile_ગણતરી < N; });
    compile_ગણતરી++;
    પાછા compile_ગણતરી_guard(&compile_ગણતરી, &decrement_compile_ગણતરી);
}

શૂન્ય string_to_spv_func(std::string name, std::string in_path, std::string out_path, std::map<std::string, std::string> defines, તાર્કિક coopmat, તાર્કિક dep_file, compile_ગણતરી_guard slot) {
    std::string target_env = (name.find("_cm2") != std::string::npos) ? "--target-env=vulkan1.3" : "--target-env=vulkan1.2";

    #ifdef _WIN32
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", target_env, "\"" + in_path + "\"", "-o", "\"" + out_path + "\""};
    #નહીંતર
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", target_env, in_path, "-o", out_path};
    #endif

    // disable spirv-opt for coopmat shaders for https://github.com/ggml-org/llama.cpp/issues/10734
    // disable spirv-opt for bf16 shaders for https://github.com/ggml-org/llama.cpp/issues/15344
    // disable spirv-opt for rope shaders for https://github.com/ggml-org/llama.cpp/issues/16860
    // disable spirv-opt for dot2 shaders (spirv-opt doesn't recognize SPV_VALVE_mixed_float_dot_product capability)
    જો (!coopmat && name.find("bf16") == std::string::npos && name.find("rope") == std::string::npos && name.find("_dot2") == std::string::npos) {
        cmd.push_back("-O");
    }

    જો (dep_file) {
        cmd.push_back("-MD");
        cmd.push_back("-MF");
#ifdef _WIN32
        cmd.push_back("\"" + target_cpp + ".d\"");
#નહીંતર
        cmd.push_back(target_cpp + ".d");
#endif
    }

    #ifdef GGML_VULKAN_SHADER_DEBUG_INFO
        cmd.push_back("-g");
    #endif

    માટે (અચલ auto& define : defines) {
        cmd.push_back("-D" + define.first + "=" + define.second);
    }

    std::string command;
    માટે (અચલ auto& part : cmd) {
        command += part + " ";
    }

    std::string stdout_str, stderr_str;
    પ્રયાસ {
        // std::cout << "Executing command: ";
        // for (const auto& part : cmd) {
        //     std::cout << part << " ";
        // }
        // std::cout << std::endl;

        પૂર્ણાંક exit_code = execute_command(cmd, stdout_str, stderr_str);
        જો (exit_code != 0 || !stderr_str.empty()) {
            std::cerr << "cannot compile " << name << " (exit code " << exit_code << ")\n\n";
            માટે (અચલ auto& part : cmd) {
                std::cerr << part << " ";
            }
            std::cerr << "\n\n" << stderr_str << std::endl;
            compile_failed = સાચું;
            પાછા;
        }

        જો (dep_file) {
            // replace .spv output path with the embed .cpp path which is used as output in CMakeLists.txt
            std::string dep = read_દ્વિઅંકી_file(target_cpp + ".d", સાચું);
            જો (!dep.empty()) {
                size_t pos = dep.find(out_path);
                જો (pos != std::string::npos) {
                    dep.replace(pos, out_path.length(), target_cpp);
                }
                write_દ્વિઅંકી_file(target_cpp + ".d", dep);
            }
        }

        std::lock_guard<std::mutex> guard(lock);
        shader_fnames.push_back(std::make_pair(name, out_path));
    } પકડો (અચલ std::exception& e) {
        std::cerr << "Error executing command for " << name << ": " << e.what() << std::endl;
        compile_failed = સાચું;
    }
}

std::map<std::string, std::string> merge_maps(અચલ std::map<std::string, std::string>& a, અચલ std::map<std::string, std::string>& b) {
    std::map<std::string, std::string> result = a;
    result.insert(b.begin(), b.end());
    પાછા result;
}

સ્થિર std::deque<std::future<શૂન્ય>> compiles;
શૂન્ય string_to_spv(std::string name, અચલ std::string& source, અચલ std::map<std::string, std::string>& defines, તાર્કિક fp16 = સાચું, તાર્કિક coopmat = ખોટું, તાર્કિક coopmat2 = ખોટું, તાર્કિક f16acc = ખોટું, અચલ std::string& suffix = "") {
    name = name + (f16acc ? "_f16acc" : "") + (coopmat ? "_cm1" : "") + (coopmat2 ? "_cm2" : (fp16 ? "" : "_fp32")) + suffix;
    std::string out_path = join_paths(output_dir, name + ".spv");

    જો (આગત_filepath == "") {
        // No input source to compile, only generate header for all shaders
        shader_fnames.push_back(std::pair(name, out_path));
        પાછા;
    } નહીંતર જો (basename(આગત_filepath) != source) {
        // Only compile shader variants matching the input filename
        પાછા;
    }

    compile_ગણતરી_guard slot = acquire_compile_slot();
    compiles.push_back(std::async(
        string_to_spv_func, name, આગત_filepath, out_path, defines, coopmat, generate_dep_file, std::move(slot)));
    // Don't write the same dep file from multiple processes
    generate_dep_file = ખોટું;

    // Clean up completed futures - don't accumulate virtual memory for completed threads' stacks.
    જ્યારે (!compiles.empty() && compiles.front().wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        compiles.pop_front();
    }
}

શૂન્ય matmul_shaders(તાર્કિક fp16, MatMulIdType matmul_id_type, તાર્કિક coopmat, તાર્કિક coopmat2, તાર્કિક f16acc, તાર્કિક dot2 = ખોટું) {
    std::string load_vec = coopmat2 ? "1" : fp16 ? "8" : "4";
    std::string aligned_b_type_f32 = coopmat2 ? "float" : fp16 ? "mat2x4" : "vec4";
    std::string aligned_b_type_f16 = coopmat2 ? "float16_t" : fp16 ? "f16mat2x4" : "f16vec4";
    std::string dot2_sfx = dot2 ? "_dot2" : "";

    std::map<std::string, std::string> base_dict;
    std::string shader_name = "matmul";

    જો (matmul_id_type == MatMulIdType::DEFAULT) {
        base_dict["MUL_MAT_ID"] = "1";
        shader_name = "matmul_id";
    } નહીંતર જો (matmul_id_type == MatMulIdType::SUBGROUP) {
        base_dict["MUL_MAT_ID"] = "1";
        base_dict["MUL_MAT_ID_USE_SUBGROUPS"] = "1";
        shader_name = "matmul_id_subgroup";
    }

    જો (fp16) {
        base_dict["FLOAT16"] = "1";
    }

    base_dict["ACC_TYPE"     ] = f16acc ? "float16_t" : "float";
    base_dict["ACC_TYPEV2"   ] = f16acc ? "f16vec2"   : "vec2";
    base_dict["ACC_TYPE_VEC4"] = f16acc ? "f16vec4"   : "vec4";
    જો (f16acc) {
        base_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
        base_dict["ACC_F16"] = "1";
    }

    જો (coopmat) {
        base_dict["COOPMAT"] = "1";
    }
#જો defined(GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT)
    જો (coopmat2) {
        base_dict["GGML_VULKAN_COOPMAT2_DECODE_VECTOR"] = "1";
    }
#endif

    જો (dot2) {
        base_dict["DOT2_F16"] = "1";
    }

    અચલ std::string source_name = coopmat2 ? "mul_mm_cm2.comp" : "mul_mm.comp";

    auto અચલ &FLOAT_TYPE = [&](પૂર્ણાંક vec, અચલ std::string &t) -> std::string {
        પસંદગી (vec) {
        સ્થિતિ 1:
            જો (t == "bf16") {
                // scalar path promotes to float
                જો (!coopmat && !coopmat2) {
                    પાછા "float";
                }
                પાછા "bfloat16_t";
            }
            જો (coopmat2 || fp16) {
                પાછા "float16_t";
            }
            પાછા "float";
        સ્થિતિ 2:
            જો (t == "bf16") {
                // scalar path promotes to float
                જો (!coopmat && !coopmat2) {
                    પાછા "vec2";
                }
                પાછા "bf16vec2";
            }
            જો (coopmat2 || fp16) {
                પાછા "f16vec2";
            }
            પાછા "vec2";
        સ્થિતિ 4:
            જો (t == "bf16") {
                // scalar path promotes to float
                જો (!coopmat && !coopmat2) {
                    પાછા "vec4";
                }
                પાછા "bf16vec4";
            }
            જો (coopmat2 || fp16) {
                પાછા "f16vec4";
            }
            પાછા "vec4";
        સ્થિતિ 8:
            જો (t == "bf16") {
                // scalar path promotes to float
                જો (!coopmat && !coopmat2) {
                    પાછા "mat2x4";
                }
                ફેંકો std::runtime_error("bf16 vec8 not supported");
            }
            જો (coopmat2 || fp16) {
                પાછા "f16mat2x4";
            }
            પાછા "mat2x4";
        મૂળભૂત:
            ફેંકો std::runtime_error("invalid vector size");
        }
    };

    અચલ std::map<std::string, std::string> float_type_dict_f16 = {
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

        અચલ std::map<std::string, std::string> float_type_dict_bf16 = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "bf16")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "bf16")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "bf16")},
        };

        // If bfloat16 is not supported, then only compile the scalar (promote to fp32) shader
#જો !defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        જો (!(coopmat || coopmat2))
#endif
        {
            જો (!dot2) {
                string_to_spv(shader_name + "_bf16", source_name, merge_maps(merge_maps(base_dict, float_type_dict_bf16), {{"TO_FLOAT_TYPE", to_float_type}, {"DATA_A_BF16", "1"}, {"LOAD_VEC_A", load_vec_a}, {"LOAD_VEC_B", "4"}, {"B_TYPE", coopmat2 ? "bfloat16_t" : "u16vec4"}, {"B_TYPE_SCALAR", coopmat2 ? "bfloat16_t" : "uint16_t"}, {"B_TYPEV4", "bf16vec4"}, {"D_TYPE", "float"}, {"B_IS_FLOAT", "1"}, {"DATA_B_BF16", "1"}}), fp16, coopmat, coopmat2, f16acc);
            }
        }
    }

    માટે (અચલ auto& tname : type_names) {
        જો (tname == "bf16") {
            ચાલુ;
        }

        // Float types keep per-type compilation (different accumulation loop structure)
        જો (tname == "f32" || tname == "f16") {
            std::string data_a_ચાવી = "DATA_A_" + to_uppercase(tname);

            અચલ std::map<std::string, std::string> float_type_dict = {
                {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
                {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
                {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
                {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
            };

            જો (!coopmat2) {
                string_to_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_ચાવી, "1"}, {"LOAD_VEC_A", load_vec}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }
            ચાલુ;
        }

        std::string data_a_ચાવી = "DATA_A_" + to_uppercase(tname);
        અચલ std::map<std::string, std::string> float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
        };

#જો defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        જો (!f16acc && !coopmat && !coopmat2 && !dot2 && (is_legacy_quant(tname) || is_k_quant(tname) || tname == "mxfp4" || tname == "iq3_s" || tname == "iq4_xs")) {
            string_to_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq.comp", merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_ચાવી, "1"}, {"D_TYPE", "float"},}), fp16, coopmat, coopmat2, f16acc);
        }
#endif

        જો (!f16acc && coopmat && (tname == "q4_0" || tname == "q4_1" || tname == "q5_0" || tname == "q5_1" || tname == "q8_0" || tname == "iq4_nl" || tname == "iq4_xs" || tname == "mxfp4"
                     || tname == "q3_k" || tname == "q4_k" || tname == "q5_k" || tname == "q6_k" || tname == "nvfp4")) {
            string_to_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq_cm1.comp", merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_ચાવી, "1"}, {"D_TYPE", "float"}, {"D_TYPE_VEC4", "vec4"}}), fp16, coopmat, coopmat2, f16acc);
        }

        જો (is_lut_quant(tname)) {
            std::string lva = lut_load_vec_a(tname);

            string_to_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_ચાવી, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

            જો (!coopmat2) {
                string_to_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_ચાવી, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }

#જો defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            જો ((tname == "mxfp4" || tname == "nvfp4") && (coopmat || coopmat2)) {
                string_to_spv(shader_name + "_" + tname + "_f16_ocp" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_ચાવી, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                જો (!coopmat2) {
                    string_to_spv(shader_name + "_" + tname + "_f32_ocp" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_ચાવી, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                }
            }
#endif
            ચાલુ;
        }

        // dedicated shader needed due to regression on Ampere
        જો (coopmat2 && (tname == "q4_k" || tname == "q5_k")) {
            string_to_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_ચાવી, "1"}, {"LOAD_VEC_A", load_vec}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }

    // Quant shader: one SPIR-V for all quant types, selected via MmTypeA spec constant
    {
        અચલ std::map<std::string, std::string> quant_float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "q4_0")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "q4_0")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "q4_0")},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, "q4_0")},
        };

        string_to_spv(shader_name + "_quant_f16" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

        જો (!coopmat2) {
            string_to_spv(shader_name + "_quant_f32" + dot2_sfx, source_name, merge_maps(merge_maps(base_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }
}

શૂન્ય process_shaders() {
    // matmul
    માટે (અચલ MatMulIdType& matmul_id_type : {MatMulIdType::NONE, MatMulIdType::DEFAULT, MatMulIdType::SUBGROUP}) {
        // No coopmats
        // fp32
        matmul_shaders(ખોટું, matmul_id_type, ખોટું, ખોટું, ખોટું);

        // fp16, fp32acc and fp16acc
        matmul_shaders(સાચું, matmul_id_type, ખોટું, ખોટું, ખોટું);
        matmul_shaders(સાચું, matmul_id_type, ખોટું, ખોટું, સાચું);

        // dot2 variants (scalar fp16 only)
        matmul_shaders(સાચું, matmul_id_type, ખોટું, ખોટું, ખોટું, સાચું);
        matmul_shaders(સાચું, matmul_id_type, ખોટું, ખોટું, સાચું,  સાચું);

        જો (matmul_id_type != MatMulIdType::DEFAULT) {
#જો defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            // Coopmat, fp32acc and fp16acc
            matmul_shaders(સાચું, matmul_id_type, સાચું, ખોટું, ખોટું);
            matmul_shaders(સાચું, matmul_id_type, સાચું, ખોટું, સાચું);
#endif

#જો defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            // Coopmat2, fp32acc and fp16acc
            matmul_shaders(સાચું, matmul_id_type, ખોટું, સાચું, ખોટું);
            matmul_shaders(સાચું, matmul_id_type, ખોટું, સાચું, સાચું);
#endif
        }
    }

    માટે (અચલ તાર્કિક& fp16 : {ખોટું, સાચું}) {
        std::map<std::string, std::string> base_dict;
        જો (fp16) {
            base_dict = {{"FLOAT_TYPE", "float16_t"}, {"FLOAT_TYPEV2", "f16vec2"}, {"FLOAT_TYPEV4", "f16vec4"}, {"FLOAT16", "1"}, {"FLOAT_TYPE_MAX", "float16_t(65504.0)"}};
        } નહીંતર {
            base_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"FLOAT_TYPEV4", "vec4"}};
        }

        // flash attention
        માટે (અચલ તાર્કિક& f16acc : {ખોટું, સાચું}) {
            std::map<std::string, std::string> fa_base_dict = base_dict;
            fa_base_dict["ACC_TYPE"] = fp16 && f16acc ? "float16_t" : "float";
            fa_base_dict["ACC_TYPEV2"] = fp16 && f16acc ? "f16vec2" : "vec2";
            fa_base_dict["ACC_TYPEV4"] = fp16 && f16acc ? "f16vec4" : "vec4";
            // Compile IQ4_NL support into all FA variants so its shared LUT is available when K or V uses it.
            fa_base_dict["DATA_A_IQ4_NL"] = "1";
            જો (fp16 && f16acc) {
                fa_base_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
            }

            જો (fp16) {
#જો defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                string_to_spv("flash_attn_f32_f16", "flash_attn_cm2.comp",
                    merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, ખોટું, સાચું, f16acc);
#endif

#જો defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                string_to_spv("flash_attn_f32_f16", "flash_attn_cm1.comp",
                    merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}), fp16, સાચું, ખોટું, f16acc);
#endif
            }

            string_to_spv("flash_attn_f32_f16", "flash_attn.comp",
                merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, ખોટું, ખોટું, f16acc);

            જો (fp16) {
                string_to_spv("flash_attn_f32_f16_dot2", "flash_attn.comp",
                    merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"DOT2_F16", "1"}}), fp16, ખોટું, ખોટું, f16acc);
            }

#જો defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            string_to_spv("flash_attn_f32_f16", "flash_attn.comp",
                merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"MMQ", "1"}, {"FA_MMQ_MIXED", "1"}}), fp16, ખોટું, ખોટું, f16acc, "_int8");
#endif
        }
    }

    અચલ std::map<std::string, std::string> fa_bf16_dict = {
        {"FLOAT_TYPE",   "bfloat16_t"},
        {"FLOAT_TYPEV2", "bf16vec2"},
        {"FLOAT_TYPEV4", "bf16vec4"},
        {"ACC_TYPE",     "float"},
        {"ACC_TYPEV2",   "vec2"},
        {"ACC_TYPEV4",   "vec4"},
        {"BFLOAT16",     "1"},
    };

#જો defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_to_spv("flash_attn_f32_f16_bf16", "flash_attn_cm1.comp",
        merge_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}),
        સાચું, સાચું, ખોટું, ખોટું);
#endif

#જો defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    string_to_spv("flash_attn_f32_f16_bf16", "flash_attn_cm2.comp",
        merge_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}),
        સાચું, ખોટું, સાચું, ખોટું);
#endif

    std::map<std::string, std::string> base_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}};

    માટે (અચલ auto& tname : type_names) {
        // mul mat vec
        std::string data_a_ચાવી = "DATA_A_" + to_uppercase(tname);
        std::string shader = (string_ends_with(tname, "_k") || string_starts_with(tname, "iq1_") || string_starts_with(tname, "iq2_") || string_starts_with(tname, "iq3_") || tname == "iq4_xs" || tname == "tq2_0" || tname == "tq1_0") ? "mul_mat_vec_" + tname + ".comp" : "mul_mat_vec.comp";

        string_to_spv("mul_mat_vec_" + tname + "_f32_f32", shader, merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_to_spv("mul_mat_vec_" + tname + "_f16_f32", shader, merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));

        string_to_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup", shader, merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_to_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup", shader, merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));

        string_to_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup_no_shmem", shader, merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        string_to_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup_no_shmem", shader, merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

#જો defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        જો (tname == "mxfp4" || tname == "nvfp4") {
            string_to_spv("mul_mat_vec_" + tname + "_f32_f32_ocp", shader, merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_to_spv("mul_mat_vec_" + tname + "_f16_f32_ocp", shader, merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));
            string_to_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup", shader, merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_to_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup", shader, merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_to_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_to_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup_no_shmem", shader, merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_ચાવી, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_ચાવી, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_ચાવી, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#endif

        string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_ચાવી, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_ચાવી, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup_no_shmem", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_ચાવી, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

        // mul mat vec with integer dot product
#જો defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        જો (is_legacy_quant(tname) || tname == "mxfp4" || is_k_quant(tname) || tname == "iq1_s" || tname == "iq1_m" || tname == "iq4_xs") {
            string_to_spv("mul_mat_vec_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_to_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_to_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

            string_to_spv("mul_mat_vec_id_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_ચાવી, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_to_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_ચાવી, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_to_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_ચાવી, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#endif

        // Dequant shaders
        જો (tname != "f16" && tname != "bf16") {
            string_to_spv("dequant_" + tname, "dequant_" + tname + ".comp", merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"D_TYPE", "float16_t"}}));
        }
        // Fused dequant+transpose variant for FA quant-KV (per-head-contiguous f16 scratch).
        જો (tname == "q8_0") {
            string_to_spv("dequant_" + tname + "_transpose", "dequant_" + tname + ".comp", merge_maps(base_dict, {{data_a_ચાવી, "1"}, {"D_TYPE", "float16_t"}, {"DEQUANT_TRANSPOSE", "1"}}));
        }

        shader = (tname == "f32" || tname == "f16" || tname == "bf16") ? "get_rows.comp" : "get_rows_quant.comp";

        જો (tname == "f16") {
            string_to_spv("get_rows_" + tname, shader, merge_maps(base_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {data_a_ચાવી, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}}));
        } નહીંતર {
            string_to_spv("get_rows_" + tname, shader, merge_maps(base_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {data_a_ચાવી, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}}));
        }
        string_to_spv("get_rows_" + tname + "_f32", shader, merge_maps(base_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {data_a_ચાવી, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}}));
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

    માટે (std::string t : {"q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
        string_to_spv("cpy_f32_" + t, "copy_to_quant.comp", {{"DATA_A_" + to_uppercase(t), "1"}, {"S_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        string_to_spv("cpy_" + t + "_f32", "copy_from_quant.comp", {{"DATA_A_" + to_uppercase(t), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }

    માટે (auto src : {std::pair{"f32", "float"}, std::pair{"f16", "float16_t"}}) {
        માટે (std::string dst : {"f32", "f16", "bf16", "q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
            string_to_spv("set_rows_" + std::string(src.first) + "_" + dst + "_i32", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + to_uppercase(dst), "1"}, {"B_TYPE", "uint"}, {"B_SIZE", "32"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
            string_to_spv("set_rows_" + std::string(src.first) + "_" + dst + "_i64", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + to_uppercase(dst), "1"}, {"B_TYPE", "uvec2"}, {"B_SIZE", "64"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        }
    }

    auto get_type_str = [](તાર્કિક f16) {
        પાછા f16 ? "float16_t" : "float";
    };
    auto get_suffix = [](તાર્કિક src0_f16, તાર્કિક src1_f16, તાર્કિક dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        પાછા s;
    };
    માટે (std::string op : {"add", "sub", "mul", "div", "add_rms", }) {
    માટે (auto src0_f16 : {ખોટું, સાચું}) {
    માટે (auto src1_f16 : {ખોટું, સાચું}) {
    માટે (auto dst_f16  : {ખોટું, સાચું}) {
        auto source = op == "add_rms" ? std::string("add") : op;
        auto name = op + get_suffix(src0_f16, src1_f16, dst_f16);
        auto add_rms = op == "add_rms" ? "1" : "0";
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

#જો defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_to_spv("fa_decode_ph1", "flash_attn_decode_phase_1.comp", {}, સાચું, સાચું, ખોટું, ખોટું);
    string_to_spv("fa_decode_ph2", "flash_attn_decode_phase_2.comp", {}, સાચું, સાચું, ખોટું, ખોટું);
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

    માટે (std::string dim_str : {"", "_3d"}) {
        માટે (તાર્કિક bda : {ખોટું, સાચું}) {
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

    માટે (auto transpose : {ખોટું, સાચું}) {
        માટે (auto unroll : {ખોટું, સાચું}) {
            માટે (auto a_f16 : {ખોટું, સાચું}) {
                std::map<std::string, std::string> defines = {
                    {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                    {"USE_COLLECTIVES", "1"}, {"UNROLL", unroll ? "[[unroll]]" : ""},
                };
                જો (transpose) defines["TRANSPOSE"] = "1";
                std::string name = std::string(transpose ? "conv_transpose_2d": "conv2d")
                    + (a_f16 ? "_f16" : "") + "_f32";
                string_to_spv(name + (unroll ? "_unroll" : ""), "conv2d_mm.comp", defines);
#જો defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                જો (unroll) {
                    auto cm2_defines = defines;
                    cm2_defines["COOPMAT2"] = "1";
                    string_to_spv(name, "conv2d_mm.comp", cm2_defines, સાચું, ખોટું, સાચું);
                }
#endif
#જો defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                જો (unroll) {
                    auto cm1_defines = defines;
                    cm1_defines["COOPMAT"] = "1";
                    string_to_spv(name, "conv2d_mm.comp", cm1_defines, સાચું, સાચું, ખોટું);
                }
#endif
            }
        }
    }

    માટે (auto unroll : {ખોટું, સાચું}) {
        માટે (auto a_f16 : {ખોટું, સાચું}) {
            std::map<std::string, std::string> defines = {
                {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                {"UNROLL", unroll ? "[[unroll]]" : ""},
            };
            std::string name = std::string("conv3d") + (a_f16 ? "_f16" : "") + "_f32";
            string_to_spv(name + (unroll ? "_unroll" : ""), "conv3d_mm.comp", defines);
#જો defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            જો (unroll) {
                auto cm2_defines = defines;
                cm2_defines["COOPMAT2"] = "1";
                string_to_spv(name, "conv3d_mm.comp", cm2_defines, સાચું, ખોટું, સાચું);
            }
#endif
#જો defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            જો (unroll) {
                auto cm1_defines = defines;
                cm1_defines["COOPMAT"] = "1";
                string_to_spv(name, "conv3d_mm.comp", cm1_defines, સાચું, સાચું, ખોટું);
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

    માટે (auto &c : compiles) {
        c.wait();
    }
}

શૂન્ય write_output_files() {
    std::stringstream hdr = make_generic_stringstream();
    std::stringstream src = make_generic_stringstream();

    hdr << "#include <cstdint>\n\n";
    src << "#include \"" << basename(target_hpp) << "\"\n\n";

    std::sort(shader_fnames.begin(), shader_fnames.end());
    માટે (અચલ auto& pair : shader_fnames) {
        અચલ std::string& name = pair.first;
        #ifdef _WIN32
            std::string path = pair.second;
            std::replace(path.begin(), path.end(), '/', '\\' );
        #નહીંતર
            અચલ std::string& path = pair.second;
        #endif

        hdr << "extern const uint64_t " << name << "_len;\n";
        hdr << "extern const unsigned char " << name << "_data[];\n\n";

        જો (આગત_filepath != "") {
            std::string data = read_દ્વિઅંકી_file(path);
            જો (data.empty()) {
                ચાલુ;
            }

            src << "const uint64_t " << name << "_len = " << data.size() << ";\n";
            src << "const unsigned char " << name << "_data[" << data.size() << "] = {\n" << std::hex;
            auto bytes = reinterpret_cast<અચલ uint8_t*>(data.data());
            માટે (size_t i = 0; i < data.size(); ++i) {
                src << "0x" << static_cast<પૂર્ણાંક>(bytes[i]) << ",";
                જો ((i + 1) % 12 == 0) src << "\n";
            }
            src << std::dec << "\n};\n\n";
        }
    }

    std::string suffixes[2] = {"_f32", "_f16"};
    માટે (std::string op : {"add", "sub", "mul", "div", "add_rms"}) {
        hdr << "extern const void * " << op << "_data[2][2][2];\n";
        hdr << "extern const uint64_t " << op << "_len[2][2][2];\n";

        std::string op_file = op == "add_rms" ? "add.comp" : std::string(op) + ".comp";
        જો (basename(આગત_filepath) != op_file) {
            ચાલુ;
        }
        std::stringstream data = make_generic_stringstream();
        std::stringstream len  = make_generic_stringstream();
        data << "const void * " << op << "_data[2][2][2] = ";
        len  << "const uint64_t " << op << "_len[2][2][2] = ";
        માટે (uint32_t t0 = 0; t0 < 2; ++t0) {
            જો (t0 == 0) {
                data << "{";
                len  << "{";
            }
            માટે (uint32_t t1 = 0; t1 < 2; ++t1) {
                જો (t1 == 0) {
                    data << "{";
                    len  << "{";
                }
                માટે (uint32_t t2 = 0; t2 < 2; ++t2) {
                    જો (t2 == 0) {
                        data << "{";
                        len  << "{";
                    }
                    data << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    len  << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    data << "_data,";
                    len  << "_len,";
                    જો (t2 == 1) {
                        data << "}, ";
                        len  << "}, ";
                    }
                }
                જો (t1 == 1) {
                    data << "}, ";
                    len  << "}, ";
                }
            }
            જો (t0 == 1) {
                data << "};\n";
                len  << "};\n";
            }
        }
        src << data.str();
        src << len.str();
    }

    std::vector<std::string> btypes = {"f16", "f32"};

#જો defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    btypes.push_back("q8_1");
#endif

    માટે (અચલ std::string& btype : btypes) {
    માટે (અચલ auto& tname : type_names) {
        જો (btype == "q8_1" && !is_legacy_quant(tname) && tname != "mxfp4" && !is_k_quant(tname) && tname != "iq1_s" && tname != "iq1_m" && tname != "iq4_xs") {
            ચાલુ;
        }
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3];\n";
        જો (basename(આગત_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }

        જો (btype == "f16") {
            ચાલુ;
        }
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3];\n";
        જો (basename(આગત_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_id_" << tname << "_" << btype << "_f32_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_id_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_id_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }
    }
    }

#જો defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    માટે (અચલ std::string& btype : {"f16", "f32"}) {
    માટે (અચલ std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3];\n";
        જો (basename(આગત_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
    }
    માટે (અચલ std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3];\n";
        જો (basename(આગત_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3] = {mul_mat_vec_id_" << tname << "_f32_f32_ocp_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3] =  {mul_mat_vec_id_" << tname << "_f32_f32_ocp_len,  mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_len, mul_mat_vec_id_"  << tname << "_f32_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
#endif

    જો (આગત_filepath == "") {
        write_file_if_changed(target_hpp, hdr.str());
    }
    જો (target_cpp != "") {
        write_દ્વિઅંકી_file(target_cpp, src.str());
    }
}

} // namespace

પૂર્ણાંક મુખ્ય(પૂર્ણાંક argc, અક્ષર** argv) {
    std::map<std::string, std::string> args;
    માટે (પૂર્ણાંક i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        જો (arg.rfind("--", 0) == 0) {
            જો (i + 1 < argc && argv[i + 1][0] != '-') {
                args[arg] = argv[i + 1];
                ++i;
            } નહીંતર {
                args[arg] = "";
            }
        }
    }

    જો (args.find("--glslc") != args.end()) {
        GLSLC = args["--glslc"]; // Path to glslc
    }
    જો (args.find("--source") != args.end()) {
        આગત_filepath = args["--source"]; // The shader source file to compile
    }
    જો (args.find("--output-dir") != args.end()) {
        output_dir = args["--output-dir"]; // Directory for containing SPIR-V output
    }
    જો (args.find("--target-hpp") != args.end()) {
        target_hpp = args["--target-hpp"]; // Path to generated header file
    }
    જો (args.find("--target-cpp") != args.end()) {
        target_cpp = args["--target-cpp"]; // Path to generated cpp file
    }

    જો (!directory_exists(output_dir)) {
        જો (!create_directory(output_dir)) {
            std::cerr << "Error creating output directory: " << output_dir << "\n";
            પાછા EXIT_FAILURE;
        }
    }

    process_shaders();

    જો (compile_failed) {
        std::cerr << "vulkan-shaders-gen: one or more shaders failed to compile" << std::endl;
        પાછા EXIT_FAILURE;
    }

    write_output_files();

    પાછા EXIT_SUCCESS;
}
