#అన్తర్భావ <iostream>
#అన్తర్భావ <fstream>
#అన్తర్భావ <sstream>
#అన్తర్భావ <string>
#అన్తర్భావ <stdexcept>
#అన్తర్భావ <array>
#అన్తర్భావ <vector>
#అన్తర్భావ <map>
#అన్తర్భావ <thread>
#అన్తర్భావ <mutex>
#అన్తర్భావ <future>
#అన్తర్భావ <queue>
#అన్తర్భావ <condition_variable>
#అన్తర్భావ <atomic>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <cstring>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <cassert>
#అన్తర్భావ <algorithm>
#అన్తర్భావ <sys/stat.h>
#అన్తర్భావ <sys/types.h>
#అన్తర్భావ <filesystem>

#నిర్వచితమైతే _WIN32
    #నిర్వచించు NOMINMAX
    #అన్తర్భావ <windows.h>
    #అన్తర్భావ <direct.h> // For _mkdir on Windows
#లేకపోతే
    #అన్తర్భావ <unistd.h>
    #అన్తర్భావ <sys/wait.h>
    #అన్తర్భావ <fcntl.h>
#అయితే_ముగింపు

#నిర్వచించు ASYNCIO_CONCURRENCY 64

std::mutex తాళం;
std::vector<std::pair<std::string, std::string>> shader_fnames;
// Set when any shader subprocess fails (non-zero exit / stderr / launch failure) so the
// build is stopped instead of silently producing a broken libggml-vulkan. (issue #24393)
స్థిర std::atomic<తార్కిక> compile_failed{అబద్ధం};
std::locale c_locale("C");

std::string GLSLC = "glslc";
std::string ఆదానం_filepath = "";
std::string output_dir = "/tmp";
std::string లక్ష్యం_hpp = "";
std::string లక్ష్యం_cpp = "";

స్థిరరాశి std::vector<std::string> type_పేర్లు = {
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

గణన MatMulIdType {
    NONE,
    DEFAULT,
    SUBGROUP,
};

నామాకాశ {

పూర్ణాంకం execute_command(std::vector<std::string>& command, std::string& stdout_str, std::string& stderr_str) {
#నిర్వచితమైతే _WIN32
    HANDLE stdout_read, stdout_write;
    HANDLE stderr_read, stderr_write;
    SECURITY_లక్షణాలు sa = { పరిమాణం(SECURITY_లక్షణాలు), ఖాళీ, TRUE };

    ఒకవేళ (!CreatePipe(&stdout_read, &stdout_write, &sa, 0) ||
        !సమితి_Handle_Information(stdout_read, HANDLE_FLAG_INHERIT, 0)) {
        విసురు std::runtime_error("Failed to create stdout pipe");
    }

    ఒకవేళ (!CreatePipe(&stderr_read, &stderr_write, &sa, 0) ||
        !సమితి_Handle_Information(stderr_read, HANDLE_FLAG_INHERIT, 0)) {
        విసురు std::runtime_error("Failed to create stderr pipe");
    }

    PROCESS_INFORMATION pi;
    STARTUPINFOA si = {};
    si.cb = పరిమాణం(STARTUPINFOA);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = stdout_write;
    si.hStdError = stderr_write;

    std::string cmd;
    కొరకు (స్థిరరాశి స్వయంచాలక& part : command) {
        cmd += part + " ";
    }

    ఒకవేళ (!CreateProcessA(ఖాళీ, cmd.దత్తాంశం(), ఖాళీ, ఖాళీ, TRUE, 0, ఖాళీ, ఖాళీ, &si, &pi)) {
        విసురు std::runtime_error("Failed to create process");
    }

    CloseHandle(stdout_write);
    CloseHandle(stderr_write);

    std::array<అక్షరం, 128> బఫరు;
    DWORD బైట్లు_read;

    అయితే (ReadFile(stdout_read, బఫరు.దత్తాంశం(), (DWORD)బఫరు.size(), &బైట్లు_read, ఖాళీ) && బైట్లు_read > 0) {
        stdout_str.append(బఫరు.దత్తాంశం(), బైట్లు_read);
    }

    అయితే (ReadFile(stderr_read, బఫరు.దత్తాంశం(), (DWORD)బఫరు.size(), &బైట్లు_read, ఖాళీ) && బైట్లు_read > 0) {
        stderr_str.append(బఫరు.దత్తాంశం(), బైట్లు_read);
    }

    CloseHandle(stdout_read);
    CloseHandle(stderr_read);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_సంకేతం = 1;
    Get_Exit_సంకేతం_Process(pi.hProcess, &exit_సంకేతం);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    తిరిగి (పూర్ణాంకం)exit_సంకేతం;
#లేకపోతే
    పూర్ణాంకం stdout_pipe[2];
    పూర్ణాంకం stderr_pipe[2];

    ఒకవేళ (ప్రణాల(stdout_pipe) != 0 || ప్రణాల(stderr_pipe) != 0) {
        విసురు std::runtime_error("Failed to create pipes");
    }

    pid_t pid = fork();
    ఒకవేళ (pid < 0) {
        std::cerr << strerror(errno) << "\n";
        విసురు std::runtime_error("Failed to fork process");
    }

    std::vector<అక్షరం*> argv;
    కొరకు (std::string& part : command) {
        argv.push_back(part.దత్తాంశం());
    }
    argv.push_back(శూన్యసూచిక);

    ఒకవేళ (pid == 0) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        execvp(argv[0], argv.దత్తాంశం());
        _exit(EXIT_FAILURE);
    } లేకపోతే {
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        std::array<అక్షరం, 128> బఫరు;
        ssize_t బైట్లు_read;

        అయితే ((బైట్లు_read = పఠ(stdout_pipe[0], బఫరు.దత్తాంశం(), బఫరు.size())) > 0) {
            stdout_str.append(బఫరు.దత్తాంశం(), బైట్లు_read);
        }

        అయితే ((బైట్లు_read = పఠ(stderr_pipe[0], బఫరు.దత్తాంశం(), బఫరు.size())) > 0) {
            stderr_str.append(బఫరు.దత్తాంశం(), బైట్లు_read);
        }

        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        పూర్ణాంకం స్థితి_2 = 0;
        waitpid(pid, &స్థితి_2, 0);
        తిరిగి WIFEXITED(స్థితి_2) ? WEXITSTATUS(స్థితి_2) : -1;
    }
#అయితే_ముగింపు
}

తార్కిక directory_exists(స్థిరరాశి std::string& path) {
    నిర్మాణం stat info;
    ఒకవేళ (stat(path.c_str(), &info) != 0) {
        తిరిగి అబద్ధం; // Path doesn't exist or can't be accessed
    }
    తిరిగి (info.st_mode & S_IFDIR) != 0; // Check if it is a directory
}

తార్కిక create_directory(స్థిరరాశి std::string& path) {
#నిర్వచితమైతే _WIN32
    తిరిగి _mkdir(path.c_str()) == 0 || errno == EEXIST; // EEXIST means the directory already exists
#లేకపోతే
    తిరిగి mkdir(path.c_str(), 0755) == 0 || errno == EEXIST; // 0755 is the directory permissions
#అయితే_ముగింపు
}

std::string గమ్యం_uppercase(స్థిరరాశి std::string& ఆదానం) {
    std::string ఫలితం = ఆదానం;
    కొరకు (అక్షరం& c : ఫలితం) {
        c = std::పెద్ద_అక్షరం(c);
    }
    తిరిగి ఫలితం;
}

తార్కిక string_starts_with(స్థిరరాశి std::string& str, స్థిరరాశి std::string& prefix) {
    ఒకవేళ (prefix.size() > str.size()) {
        తిరిగి అబద్ధం;
    }
    తిరిగి std::equal(prefix.begin(), prefix.end(), str.begin());
}

తార్కిక string_చివర్లు_with(స్థిరరాశి std::string& str, స్థిరరాశి std::string& suffix) {
    ఒకవేళ (suffix.size() > str.size()) {
        తిరిగి అబద్ధం;
    }
    తిరిగి std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

తార్కిక is_quantized_type(స్థిరరాశి std::string& type_name) {
    తిరిగి type_name != "f32" && type_name != "f16" && type_name != "bf16";
}

తార్కిక is_legacy_quant(స్థిరరాశి std::string& type_name) {
    తిరిగి type_name == "q2_0" || type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";
}

తార్కిక is_k_quant(స్థిరరాశి std::string& type_name) {
    తిరిగి string_చివర్లు_with(type_name, "_k");
}

తార్కిక is_iq_quant(స్థిరరాశి std::string& type_name) {
    తిరిగి string_starts_with(type_name, "iq");
}

తార్కిక is_lut_quant(స్థిరరాశి std::string& type_name) {
    తిరిగి is_iq_quant(type_name) || type_name == "mxfp4" || type_name == "nvfp4";
}

std::string lut_ఎక్కించు_vec_a(స్థిరరాశి std::string& type_name) {
    ఒకవేళ (type_name == "iq1_s" || type_name == "iq1_m" || type_name == "iq2_xxs" || type_name == "iq2_xs" || type_name == "iq2_s" || type_name == "iq4_xs") {
        తిరిగి "8";
    }
    తిరిగి "4";
}

స్థిర స్థిరరాశి అక్షరం మార్గం_separator = '/';

std::string join_paths(స్థిరరాశి std::string& path1, స్థిరరాశి std::string& path2) {
    తిరిగి path1 + మార్గం_separator + path2;
}

std::string basename(స్థిరరాశి std::string &path) {
    తిరిగి path.substr(path.find_last_of("/\\") + 1);
}

std::stringstream make_generic_stringstream() {
    std::stringstream ss;
    ss.imbue(c_locale);
    తిరిగి ss;
}

std::string read_ద్విమాన_file(స్థిరరాశి std::string& path, తార్కిక may_not_exist = అబద్ధం) {
    FILE* f = fopen(path.c_str(), "rb");
    ఒకవేళ (!f) {
        ఒకవేళ (!may_not_exist) {
            std::cerr << "Error opening file: " << path << " (" << strerror(errno) << ")\n";
        }
        తిరిగి {};
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::string దత్తాంశం(size, '\0');
    size_t read_కొలత = fread(దత్తాంశం.దత్తాంశం(), 1, size, f);
    fclose(f);
    ఒకవేళ (read_కొలత != size) {
        std::cerr << "Error reading file: " << path << " (" << strerror(errno) << ")\n";
        తిరిగి {};
    }

    తిరిగి దత్తాంశం;
}

శూన్య write_ద్విమాన_file(స్థిరరాశి std::string& path, స్థిరరాశి std::string& content) {
    FILE* f = fopen(path.c_str(), "wb");
    ఒకవేళ (!f) {
        std::cerr << "Error opening file for writing: " << path << " (" << strerror(errno) << ")\n";
        తిరిగి;
    }

    size_t write_కొలత = fwrite(content.దత్తాంశం(), 1, content.size(), f);
    fclose(f);
    ఒకవేళ (write_కొలత != content.size()) {
        std::cerr << "Error writing file: " << path << " (" << strerror(errno) << ")\n";
        తిరిగి;
    }
}

శూన్య write_file_if_మారింది(స్థిరరాశి std::string& path, స్థిరరాశి std::string& content) {
    std::string existing = read_ద్విమాన_file(path, నిజం);
    ఒకవేళ (existing != content) {
        write_ద్విమాన_file(path, content);
    }
}


// variables to track number of compiles in progress
స్థిర uint32_t compile_లెక్క = 0;
స్థిర std::mutex compile_లెక్క_mutex;
స్థిర std::condition_variable compile_లెక్క_cond;
స్థిర తార్కిక generate_dep_file = నిజం;

శూన్య decrement_compile_లెక్క(uint32_t * count) {
    ఒకవేళ (count) {
        std::lock_guard<std::mutex> guard(compile_లెక్క_mutex);
        ప్రతిజ్ఞా(compile_లెక్క > 0);
        compile_లెక్క--;
        compile_లెక్క_cond.notify_all();
    }
}

ఉపయోగించి compile_లెక్క_guard = std::unique_ptr<uint32_t, decltype(&decrement_compile_లెక్క)>;

compile_లెక్క_guard acquire_compile_స్థానం() {
    // wait until fewer than N compiles are in progress.
    // 16 is an arbitrary limit, the goal is to avoid "failed to create pipe" errors.
    uint32_t N = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::unique_lock<std::mutex> guard(compile_లెక్క_mutex);
    compile_లెక్క_cond.wait(guard, [N] { తిరిగి compile_లెక్క < N; });
    compile_లెక్క++;
    తిరిగి compile_లెక్క_guard(&compile_లెక్క, &decrement_compile_లెక్క);
}

శూన్య string_గమ్యం_spv_func(std::string name, std::string లోపల_మార్గం, std::string బహిర్గతం_మార్గం, std::map<std::string, std::string> నిర్వచిస్తుంది, తార్కిక coopmat, తార్కిక dep_file, compile_లెక్క_guard స్థానం) {
    std::string లక్ష్యం_env = (name.find("_cm2") != std::string::npos) ? "--target-env=vulkan1.3" : "--target-env=vulkan1.2";

    #నిర్వచితమైతే _WIN32
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", లక్ష్యం_env, "\"" + లోపల_మార్గం + "\"", "-o", "\"" + బహిర్గతం_మార్గం + "\""};
    #లేకపోతే
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", లక్ష్యం_env, లోపల_మార్గం, "-o", బహిర్గతం_మార్గం};
    #అయితే_ముగింపు

    // disable spirv-opt for coopmat shaders for https://github.com/ggml-org/llama.cpp/issues/10734
    // disable spirv-opt for bf16 shaders for https://github.com/ggml-org/llama.cpp/issues/15344
    // disable spirv-opt for rope shaders for https://github.com/ggml-org/llama.cpp/issues/16860
    // disable spirv-opt for dot2 shaders (spirv-opt doesn't recognize SPV_VALVE_mixed_float_dot_product capability)
    ఒకవేళ (!coopmat && name.find("bf16") == std::string::npos && name.find("rope") == std::string::npos && name.find("_dot2") == std::string::npos) {
        cmd.push_back("-O");
    }

    ఒకవేళ (dep_file) {
        cmd.push_back("-MD");
        cmd.push_back("-MF");
#నిర్వచితమైతే _WIN32
        cmd.push_back("\"" + లక్ష్యం_cpp + ".d\"");
#లేకపోతే
        cmd.push_back(లక్ష్యం_cpp + ".d");
#అయితే_ముగింపు
    }

    #నిర్వచితమైతే GGML_VULKAN_SHADER_DEBUG_INFO
        cmd.push_back("-g");
    #అయితే_ముగింపు

    కొరకు (స్థిరరాశి స్వయంచాలక& define : నిర్వచిస్తుంది) {
        cmd.push_back("-D" + define.మొదటి + "=" + define.second);
    }

    std::string command;
    కొరకు (స్థిరరాశి స్వయంచాలక& part : cmd) {
        command += part + " ";
    }

    std::string stdout_str, stderr_str;
    ప్రయత్నించు {
        // std::cout << "Executing command: ";
        // for (const auto& part : cmd) {
        //     std::cout << part << " ";
        // }
        // std::cout << std::endl;

        పూర్ణాంకం exit_సంకేతం = execute_command(cmd, stdout_str, stderr_str);
        ఒకవేళ (exit_సంకేతం != 0 || !stderr_str.ఖాళీలు()) {
            std::cerr << "cannot compile " << name << " (exit code " << exit_సంకేతం << ")\n\n";
            కొరకు (స్థిరరాశి స్వయంచాలక& part : cmd) {
                std::cerr << part << " ";
            }
            std::cerr << "\n\n" << stderr_str << std::పంక్తి_ముగింపు;
            compile_failed = నిజం;
            తిరిగి;
        }

        ఒకవేళ (dep_file) {
            // replace .spv output path with the embed .cpp path which is used as output in CMakeLists.txt
            std::string dep = read_ద్విమాన_file(లక్ష్యం_cpp + ".d", నిజం);
            ఒకవేళ (!dep.ఖాళీలు()) {
                size_t pos = dep.find(బహిర్గతం_మార్గం);
                ఒకవేళ (pos != std::string::npos) {
                    dep.replace(pos, బహిర్గతం_మార్గం.నిడివి(), లక్ష్యం_cpp);
                }
                write_ద్విమాన_file(లక్ష్యం_cpp + ".d", dep);
            }
        }

        std::lock_guard<std::mutex> guard(తాళం);
        shader_fnames.push_back(std::make_pair(name, బహిర్గతం_మార్గం));
    } పట్టుకో (స్థిరరాశి std::exception& e) {
        std::cerr << "Error executing command for " << name << ": " << e.what() << std::పంక్తి_ముగింపు;
        compile_failed = నిజం;
    }
}

std::map<std::string, std::string> విలీనం_maps(స్థిరరాశి std::map<std::string, std::string>& a, స్థిరరాశి std::map<std::string, std::string>& b) {
    std::map<std::string, std::string> ఫలితం = a;
    ఫలితం.insert(b.begin(), b.end());
    తిరిగి ఫలితం;
}

స్థిర std::deque<std::future<శూన్య>> compiles;
శూన్య string_గమ్యం_spv(std::string name, స్థిరరాశి std::string& ఆధారం, స్థిరరాశి std::map<std::string, std::string>& నిర్వచిస్తుంది, తార్కిక fp16 = నిజం, తార్కిక coopmat = అబద్ధం, తార్కిక coopmat2 = అబద్ధం, తార్కిక f16acc = అబద్ధం, స్థిరరాశి std::string& suffix = "") {
    name = name + (f16acc ? "_f16acc" : "") + (coopmat ? "_cm1" : "") + (coopmat2 ? "_cm2" : (fp16 ? "" : "_fp32")) + suffix;
    std::string బహిర్గతం_మార్గం = join_paths(output_dir, name + ".spv");

    ఒకవేళ (ఆదానం_filepath == "") {
        // No input source to compile, only generate header for all shaders
        shader_fnames.push_back(std::pair(name, బహిర్గతం_మార్గం));
        తిరిగి;
    } లేకపోతే ఒకవేళ (basename(ఆదానం_filepath) != ఆధారం) {
        // Only compile shader variants matching the input filename
        తిరిగి;
    }

    compile_లెక్క_guard స్థానం = acquire_compile_స్థానం();
    compiles.push_back(std::async(
        string_గమ్యం_spv_func, name, ఆదానం_filepath, బహిర్గతం_మార్గం, నిర్వచిస్తుంది, coopmat, generate_dep_file, std::move(స్థానం)));
    // Don't write the same dep file from multiple processes
    generate_dep_file = అబద్ధం;

    // Clean up completed futures - don't accumulate virtual memory for completed threads' stacks.
    అయితే (!compiles.ఖాళీలు() && compiles.front().wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        compiles.pop_front();
    }
}

శూన్య matmul_shaders(తార్కిక fp16, MatMulIdType matmul_id_type, తార్కిక coopmat, తార్కిక coopmat2, తార్కిక f16acc, తార్కిక dot2 = అబద్ధం) {
    std::string ఎక్కించు_vec = coopmat2 ? "1" : fp16 ? "8" : "4";
    std::string aligned_b_type_f32 = coopmat2 ? "float" : fp16 ? "mat2x4" : "vec4";
    std::string aligned_b_type_f16 = coopmat2 ? "float16_t" : fp16 ? "f16mat2x4" : "f16vec4";
    std::string dot2_sfx = dot2 ? "_dot2" : "";

    std::map<std::string, std::string> పునాది_dict;
    std::string shader_name = "matmul";

    ఒకవేళ (matmul_id_type == MatMulIdType::DEFAULT) {
        పునాది_dict["MUL_MAT_ID"] = "1";
        shader_name = "matmul_id";
    } లేకపోతే ఒకవేళ (matmul_id_type == MatMulIdType::SUBGROUP) {
        పునాది_dict["MUL_MAT_ID"] = "1";
        పునాది_dict["MUL_MAT_ID_USE_SUBGROUPS"] = "1";
        shader_name = "matmul_id_subgroup";
    }

    ఒకవేళ (fp16) {
        పునాది_dict["FLOAT16"] = "1";
    }

    పునాది_dict["ACC_TYPE"     ] = f16acc ? "float16_t" : "float";
    పునాది_dict["ACC_TYPEV2"   ] = f16acc ? "f16vec2"   : "vec2";
    పునాది_dict["ACC_TYPE_VEC4"] = f16acc ? "f16vec4"   : "vec4";
    ఒకవేళ (f16acc) {
        పునాది_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
        పునాది_dict["ACC_F16"] = "1";
    }

    ఒకవేళ (coopmat) {
        పునాది_dict["COOPMAT"] = "1";
    }
#ఒకవేళ defined(GGML_VULKAN_COOPMAT2_DECODE_సదిశ_GLSLC_SUPPORT)
    ఒకవేళ (coopmat2) {
        పునాది_dict["GGML_VULKAN_COOPMAT2_DECODE_VECTOR"] = "1";
    }
#అయితే_ముగింపు

    ఒకవేళ (dot2) {
        పునాది_dict["DOT2_F16"] = "1";
    }

    స్థిరరాశి std::string ఆధారం_name = coopmat2 ? "mul_mm_cm2.comp" : "mul_mm.comp";

    స్వయంచాలక స్థిరరాశి &FLOAT_TYPE = [&](పూర్ణాంకం vec, స్థిరరాశి std::string &t) -> std::string {
        ఎంపిక (vec) {
        స్థితి 1:
            ఒకవేళ (t == "bf16") {
                // scalar path promotes to float
                ఒకవేళ (!coopmat && !coopmat2) {
                    తిరిగి "float";
                }
                తిరిగి "bfloat16_t";
            }
            ఒకవేళ (coopmat2 || fp16) {
                తిరిగి "float16_t";
            }
            తిరిగి "float";
        స్థితి 2:
            ఒకవేళ (t == "bf16") {
                // scalar path promotes to float
                ఒకవేళ (!coopmat && !coopmat2) {
                    తిరిగి "vec2";
                }
                తిరిగి "bf16vec2";
            }
            ఒకవేళ (coopmat2 || fp16) {
                తిరిగి "f16vec2";
            }
            తిరిగి "vec2";
        స్థితి 4:
            ఒకవేళ (t == "bf16") {
                // scalar path promotes to float
                ఒకవేళ (!coopmat && !coopmat2) {
                    తిరిగి "vec4";
                }
                తిరిగి "bf16vec4";
            }
            ఒకవేళ (coopmat2 || fp16) {
                తిరిగి "f16vec4";
            }
            తిరిగి "vec4";
        స్థితి 8:
            ఒకవేళ (t == "bf16") {
                // scalar path promotes to float
                ఒకవేళ (!coopmat && !coopmat2) {
                    తిరిగి "mat2x4";
                }
                విసురు std::runtime_error("bf16 vec8 not supported");
            }
            ఒకవేళ (coopmat2 || fp16) {
                తిరిగి "f16mat2x4";
            }
            తిరిగి "mat2x4";
        అప్రమేయ:
            విసురు std::runtime_error("invalid vector size");
        }
    };

    స్థిరరాశి std::map<std::string, std::string> float_type_dict_f16 = {
        {"FLOAT_TYPE",   FLOAT_TYPE(1, "f16")},
        {"FLOAT_TYPEV2", FLOAT_TYPE(2, "f16")},
        {"FLOAT_TYPEV4", FLOAT_TYPE(4, "f16")},
        {"FLOAT_TYPEV8", FLOAT_TYPE(8, "f16")},
    };

    // Shaders with f16 B_TYPE
    string_గమ్యం_spv(shader_name + "_f32_f16" + dot2_sfx, ఆధారం_name, విలీనం_maps(విలీనం_maps(పునాది_dict, float_type_dict_f16), {{"DATA_A_F32", "1"}, {"LOAD_VEC_A", ఎక్కించు_vec}, {"LOAD_VEC_B", ఎక్కించు_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

    string_గమ్యం_spv(shader_name + "_f16" + dot2_sfx, ఆధారం_name, విలీనం_maps(విలీనం_maps(పునాది_dict, float_type_dict_f16), {{"DATA_A_F16", "1"}, {"LOAD_VEC_A", ఎక్కించు_vec}, {"LOAD_VEC_B", ఎక్కించు_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

    // bf16
    {
        // For aligned matmul loads
        std::string ఎక్కించు_vec_a = coopmat2 ? "1" : "4";

        // scalar path promotes to float
        std::string గమ్యం_float_type = (coopmat || coopmat2) ? "uintBitsToBFloat16EXT" : "bf16_to_fp32";

        స్థిరరాశి std::map<std::string, std::string> float_type_dict_bf16 = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "bf16")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "bf16")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "bf16")},
        };

        // If bfloat16 is not supported, then only compile the scalar (promote to fp32) shader
#ఒకవేళ !defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ఒకవేళ (!(coopmat || coopmat2))
#అయితే_ముగింపు
        {
            ఒకవేళ (!dot2) {
                string_గమ్యం_spv(shader_name + "_bf16", ఆధారం_name, విలీనం_maps(విలీనం_maps(పునాది_dict, float_type_dict_bf16), {{"TO_FLOAT_TYPE", గమ్యం_float_type}, {"DATA_A_BF16", "1"}, {"LOAD_VEC_A", ఎక్కించు_vec_a}, {"LOAD_VEC_B", "4"}, {"B_TYPE", coopmat2 ? "bfloat16_t" : "u16vec4"}, {"B_TYPE_SCALAR", coopmat2 ? "bfloat16_t" : "uint16_t"}, {"B_TYPEV4", "bf16vec4"}, {"D_TYPE", "float"}, {"B_IS_FLOAT", "1"}, {"DATA_B_BF16", "1"}}), fp16, coopmat, coopmat2, f16acc);
            }
        }
    }

    కొరకు (స్థిరరాశి స్వయంచాలక& tname : type_పేర్లు) {
        ఒకవేళ (tname == "bf16") {
            కొనసాగు;
        }

        // Float types keep per-type compilation (different accumulation loop structure)
        ఒకవేళ (tname == "f32" || tname == "f16") {
            std::string దత్తాంశం_a_కీలకం = "DATA_A_" + గమ్యం_uppercase(tname);

            స్థిరరాశి std::map<std::string, std::string> float_type_dict = {
                {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
                {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
                {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
                {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
            };

            ఒకవేళ (!coopmat2) {
                string_గమ్యం_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, ఆధారం_name, విలీనం_maps(విలీనం_maps(పునాది_dict, float_type_dict), {{దత్తాంశం_a_కీలకం, "1"}, {"LOAD_VEC_A", ఎక్కించు_vec}, {"LOAD_VEC_B", ఎక్కించు_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }
            కొనసాగు;
        }

        std::string దత్తాంశం_a_కీలకం = "DATA_A_" + గమ్యం_uppercase(tname);
        స్థిరరాశి std::map<std::string, std::string> float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
        };

#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        ఒకవేళ (!f16acc && !coopmat && !coopmat2 && !dot2 && (is_legacy_quant(tname) || is_k_quant(tname) || tname == "mxfp4" || tname == "iq3_s" || tname == "iq4_xs")) {
            string_గమ్యం_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq.comp", విలీనం_maps(విలీనం_maps(పునాది_dict, float_type_dict), {{దత్తాంశం_a_కీలకం, "1"}, {"D_TYPE", "float"},}), fp16, coopmat, coopmat2, f16acc);
        }
#అయితే_ముగింపు

        ఒకవేళ (!f16acc && coopmat && (tname == "q4_0" || tname == "q4_1" || tname == "q5_0" || tname == "q5_1" || tname == "q8_0" || tname == "iq4_nl" || tname == "iq4_xs" || tname == "mxfp4"
                     || tname == "q3_k" || tname == "q4_k" || tname == "q5_k" || tname == "q6_k" || tname == "nvfp4")) {
            string_గమ్యం_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq_cm1.comp", విలీనం_maps(విలీనం_maps(పునాది_dict, float_type_dict), {{దత్తాంశం_a_కీలకం, "1"}, {"D_TYPE", "float"}, {"D_TYPE_VEC4", "vec4"}}), fp16, coopmat, coopmat2, f16acc);
        }

        ఒకవేళ (is_lut_quant(tname)) {
            std::string lva = lut_ఎక్కించు_vec_a(tname);

            string_గమ్యం_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, ఆధారం_name, విలీనం_maps(విలీనం_maps(పునాది_dict, float_type_dict), {{దత్తాంశం_a_కీలకం, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", ఎక్కించు_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

            ఒకవేళ (!coopmat2) {
                string_గమ్యం_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, ఆధారం_name, విలీనం_maps(విలీనం_maps(పునాది_dict, float_type_dict), {{దత్తాంశం_a_కీలకం, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", ఎక్కించు_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }

#ఒకవేళ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            ఒకవేళ ((tname == "mxfp4" || tname == "nvfp4") && (coopmat || coopmat2)) {
                string_గమ్యం_spv(shader_name + "_" + tname + "_f16_ocp" + dot2_sfx, ఆధారం_name, విలీనం_maps(విలీనం_maps(పునాది_dict, float_type_dict), {{దత్తాంశం_a_కీలకం, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", ఎక్కించు_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                ఒకవేళ (!coopmat2) {
                    string_గమ్యం_spv(shader_name + "_" + tname + "_f32_ocp" + dot2_sfx, ఆధారం_name, విలీనం_maps(విలీనం_maps(పునాది_dict, float_type_dict), {{దత్తాంశం_a_కీలకం, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", ఎక్కించు_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                }
            }
#అయితే_ముగింపు
            కొనసాగు;
        }

        // dedicated shader needed due to regression on Ampere
        ఒకవేళ (coopmat2 && (tname == "q4_k" || tname == "q5_k")) {
            string_గమ్యం_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, ఆధారం_name, విలీనం_maps(విలీనం_maps(పునాది_dict, float_type_dict), {{దత్తాంశం_a_కీలకం, "1"}, {"LOAD_VEC_A", ఎక్కించు_vec}, {"LOAD_VEC_B", ఎక్కించు_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }

    // Quant shader: one SPIR-V for all quant types, selected via MmTypeA spec constant
    {
        స్థిరరాశి std::map<std::string, std::string> quant_float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "q4_0")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "q4_0")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "q4_0")},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, "q4_0")},
        };

        string_గమ్యం_spv(shader_name + "_quant_f16" + dot2_sfx, ఆధారం_name, విలీనం_maps(విలీనం_maps(పునాది_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", ఎక్కించు_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

        ఒకవేళ (!coopmat2) {
            string_గమ్యం_spv(shader_name + "_quant_f32" + dot2_sfx, ఆధారం_name, విలీనం_maps(విలీనం_maps(పునాది_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", ఎక్కించు_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }
}

శూన్య process_shaders() {
    // matmul
    కొరకు (స్థిరరాశి MatMulIdType& matmul_id_type : {MatMulIdType::NONE, MatMulIdType::DEFAULT, MatMulIdType::SUBGROUP}) {
        // No coopmats
        // fp32
        matmul_shaders(అబద్ధం, matmul_id_type, అబద్ధం, అబద్ధం, అబద్ధం);

        // fp16, fp32acc and fp16acc
        matmul_shaders(నిజం, matmul_id_type, అబద్ధం, అబద్ధం, అబద్ధం);
        matmul_shaders(నిజం, matmul_id_type, అబద్ధం, అబద్ధం, నిజం);

        // dot2 variants (scalar fp16 only)
        matmul_shaders(నిజం, matmul_id_type, అబద్ధం, అబద్ధం, అబద్ధం, నిజం);
        matmul_shaders(నిజం, matmul_id_type, అబద్ధం, అబద్ధం, నిజం,  నిజం);

        ఒకవేళ (matmul_id_type != MatMulIdType::DEFAULT) {
#ఒకవేళ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            // Coopmat, fp32acc and fp16acc
            matmul_shaders(నిజం, matmul_id_type, నిజం, అబద్ధం, అబద్ధం);
            matmul_shaders(నిజం, matmul_id_type, నిజం, అబద్ధం, నిజం);
#అయితే_ముగింపు

#ఒకవేళ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            // Coopmat2, fp32acc and fp16acc
            matmul_shaders(నిజం, matmul_id_type, అబద్ధం, నిజం, అబద్ధం);
            matmul_shaders(నిజం, matmul_id_type, అబద్ధం, నిజం, నిజం);
#అయితే_ముగింపు
        }
    }

    కొరకు (స్థిరరాశి తార్కిక& fp16 : {అబద్ధం, నిజం}) {
        std::map<std::string, std::string> పునాది_dict;
        ఒకవేళ (fp16) {
            పునాది_dict = {{"FLOAT_TYPE", "float16_t"}, {"FLOAT_TYPEV2", "f16vec2"}, {"FLOAT_TYPEV4", "f16vec4"}, {"FLOAT16", "1"}, {"FLOAT_TYPE_MAX", "float16_t(65504.0)"}};
        } లేకపోతే {
            పునాది_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"FLOAT_TYPEV4", "vec4"}};
        }

        // flash attention
        కొరకు (స్థిరరాశి తార్కిక& f16acc : {అబద్ధం, నిజం}) {
            std::map<std::string, std::string> fa_పునాది_dict = పునాది_dict;
            fa_పునాది_dict["ACC_TYPE"] = fp16 && f16acc ? "float16_t" : "float";
            fa_పునాది_dict["ACC_TYPEV2"] = fp16 && f16acc ? "f16vec2" : "vec2";
            fa_పునాది_dict["ACC_TYPEV4"] = fp16 && f16acc ? "f16vec4" : "vec4";
            // Compile IQ4_NL support into all FA variants so its shared LUT is available when K or V uses it.
            fa_పునాది_dict["DATA_A_IQ4_NL"] = "1";
            ఒకవేళ (fp16 && f16acc) {
                fa_పునాది_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
            }

            ఒకవేళ (fp16) {
#ఒకవేళ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                string_గమ్యం_spv("flash_attn_f32_f16", "flash_attn_cm2.comp",
                    విలీనం_maps(fa_పునాది_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, అబద్ధం, నిజం, f16acc);
#అయితే_ముగింపు

#ఒకవేళ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                string_గమ్యం_spv("flash_attn_f32_f16", "flash_attn_cm1.comp",
                    విలీనం_maps(fa_పునాది_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}), fp16, నిజం, అబద్ధం, f16acc);
#అయితే_ముగింపు
            }

            string_గమ్యం_spv("flash_attn_f32_f16", "flash_attn.comp",
                విలీనం_maps(fa_పునాది_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, అబద్ధం, అబద్ధం, f16acc);

            ఒకవేళ (fp16) {
                string_గమ్యం_spv("flash_attn_f32_f16_dot2", "flash_attn.comp",
                    విలీనం_maps(fa_పునాది_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"DOT2_F16", "1"}}), fp16, అబద్ధం, అబద్ధం, f16acc);
            }

#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            string_గమ్యం_spv("flash_attn_f32_f16", "flash_attn.comp",
                విలీనం_maps(fa_పునాది_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"MMQ", "1"}, {"FA_MMQ_MIXED", "1"}}), fp16, అబద్ధం, అబద్ధం, f16acc, "_int8");
#అయితే_ముగింపు
        }
    }

    స్థిరరాశి std::map<std::string, std::string> fa_bf16_dict = {
        {"FLOAT_TYPE",   "bfloat16_t"},
        {"FLOAT_TYPEV2", "bf16vec2"},
        {"FLOAT_TYPEV4", "bf16vec4"},
        {"ACC_TYPE",     "float"},
        {"ACC_TYPEV2",   "vec2"},
        {"ACC_TYPEV4",   "vec4"},
        {"BFLOAT16",     "1"},
    };

#ఒకవేళ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_గమ్యం_spv("flash_attn_f32_f16_bf16", "flash_attn_cm1.comp",
        విలీనం_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}),
        నిజం, నిజం, అబద్ధం, అబద్ధం);
#అయితే_ముగింపు

#ఒకవేళ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    string_గమ్యం_spv("flash_attn_f32_f16_bf16", "flash_attn_cm2.comp",
        విలీనం_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}),
        నిజం, అబద్ధం, నిజం, అబద్ధం);
#అయితే_ముగింపు

    std::map<std::string, std::string> పునాది_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}};

    కొరకు (స్థిరరాశి స్వయంచాలక& tname : type_పేర్లు) {
        // mul mat vec
        std::string దత్తాంశం_a_కీలకం = "DATA_A_" + గమ్యం_uppercase(tname);
        std::string shader = (string_చివర్లు_with(tname, "_k") || string_starts_with(tname, "iq1_") || string_starts_with(tname, "iq2_") || string_starts_with(tname, "iq3_") || tname == "iq4_xs" || tname == "tq2_0" || tname == "tq1_0") ? "mul_mat_vec_" + tname + ".comp" : "mul_mat_vec.comp";

        string_గమ్యం_spv("mul_mat_vec_" + tname + "_f32_f32", shader, విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_గమ్యం_spv("mul_mat_vec_" + tname + "_f16_f32", shader, విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));

        string_గమ్యం_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup", shader, విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_గమ్యం_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup", shader, విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));

        string_గమ్యం_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup_no_shmem", shader, విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        string_గమ్యం_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup_no_shmem", shader, విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

#ఒకవేళ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ఒకవేళ (tname == "mxfp4" || tname == "nvfp4") {
            string_గమ్యం_spv("mul_mat_vec_" + tname + "_f32_f32_ocp", shader, విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_గమ్యం_spv("mul_mat_vec_" + tname + "_f16_f32_ocp", shader, విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));
            string_గమ్యం_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup", shader, విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_గమ్యం_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup", shader, విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_గమ్యం_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_గమ్యం_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup_no_shmem", shader, విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_గమ్యం_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp", shader, విలీనం_maps(పునాది_dict, {{"MUL_MAT_ID", "1"}, {దత్తాంశం_a_కీలకం, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_గమ్యం_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup", shader, విలీనం_maps(పునాది_dict, {{"MUL_MAT_ID", "1"}, {దత్తాంశం_a_కీలకం, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_గమ్యం_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, విలీనం_maps(పునాది_dict, {{"MUL_MAT_ID", "1"}, {దత్తాంశం_a_కీలకం, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#అయితే_ముగింపు

        string_గమ్యం_spv("mul_mat_vec_id_" + tname + "_f32_f32", shader, విలీనం_maps(పునాది_dict, {{"MUL_MAT_ID", "1"}, {దత్తాంశం_a_కీలకం, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_గమ్యం_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup", shader, విలీనం_maps(పునాది_dict, {{"MUL_MAT_ID", "1"}, {దత్తాంశం_a_కీలకం, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_గమ్యం_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup_no_shmem", shader, విలీనం_maps(పునాది_dict, {{"MUL_MAT_ID", "1"}, {దత్తాంశం_a_కీలకం, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

        // mul mat vec with integer dot product
#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        ఒకవేళ (is_legacy_quant(tname) || tname == "mxfp4" || is_k_quant(tname) || tname == "iq1_s" || tname == "iq1_m" || tname == "iq4_xs") {
            string_గమ్యం_spv("mul_mat_vec_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_గమ్యం_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_గమ్యం_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

            string_గమ్యం_spv("mul_mat_vec_id_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", విలీనం_maps(పునాది_dict, {{"MUL_MAT_ID", "1"}, {దత్తాంశం_a_కీలకం, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_గమ్యం_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", విలీనం_maps(పునాది_dict, {{"MUL_MAT_ID", "1"}, {దత్తాంశం_a_కీలకం, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_గమ్యం_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", విలీనం_maps(పునాది_dict, {{"MUL_MAT_ID", "1"}, {దత్తాంశం_a_కీలకం, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#అయితే_ముగింపు

        // Dequant shaders
        ఒకవేళ (tname != "f16" && tname != "bf16") {
            string_గమ్యం_spv("dequant_" + tname, "dequant_" + tname + ".comp", విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"D_TYPE", "float16_t"}}));
        }
        // Fused dequant+transpose variant for FA quant-KV (per-head-contiguous f16 scratch).
        ఒకవేళ (tname == "q8_0") {
            string_గమ్యం_spv("dequant_" + tname + "_transpose", "dequant_" + tname + ".comp", విలీనం_maps(పునాది_dict, {{దత్తాంశం_a_కీలకం, "1"}, {"D_TYPE", "float16_t"}, {"DEQUANT_TRANSPOSE", "1"}}));
        }

        shader = (tname == "f32" || tname == "f16" || tname == "bf16") ? "get_rows.comp" : "get_rows_quant.comp";

        ఒకవేళ (tname == "f16") {
            string_గమ్యం_spv("get_rows_" + tname, shader, విలీనం_maps(పునాది_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {దత్తాంశం_a_కీలకం, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}}));
        } లేకపోతే {
            string_గమ్యం_spv("get_rows_" + tname, shader, విలీనం_maps(పునాది_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {దత్తాంశం_a_కీలకం, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}}));
        }
        string_గమ్యం_spv("get_rows_" + tname + "_f32", shader, విలీనం_maps(పునాది_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {దత్తాంశం_a_కీలకం, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}}));
    }

    string_గమ్యం_spv("get_rows_i32", "get_rows.comp", {{"TEMP_TYPE", "uint"}, {"A_TYPE", "uint"}, {"B_TYPE", "int"}, {"D_TYPE", "uint"}});

    string_గమ్యం_spv("mul_mat_vec_p021_f16_f32_subgroup_add", "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});
    string_గమ్యం_spv("mul_mat_vec_p021_f16_f32",              "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}});
    string_గమ్యం_spv("mul_mat_vec_nc_f16_f32", "mul_mat_vec_nc.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}});

    // Norms
    string_గమ్యం_spv("norm_f32", "norm.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("group_norm_f32", "group_norm.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("rms_norm_f32", "rms_norm.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("rms_norm_mul_add_f32", "rms_norm.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_ADD_FUSION", "1"}}));
    string_గమ్యం_spv("rms_norm_mul_add_partials_f32", "rms_norm_partials.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_ADD_FUSION", "1"}}));
    string_గమ్యం_spv("rms_norm_set_rows_f32_f32", "rms_norm.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_SET_ROWS_FUSION", "1"}}));
    string_గమ్యం_spv("rms_norm_set_rows_f32_f16", "rms_norm.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"RMS_NORM_SET_ROWS_FUSION", "1"}}));
    string_గమ్యం_spv("rms_norm_partials_f32", "rms_norm_partials.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("rms_norm_mul_rope_f32_f32", "rms_norm.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
    string_గమ్యం_spv("rms_norm_mul_rope_f32_f16", "rms_norm.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
    string_గమ్యం_spv("rms_norm_back_f32", "rms_norm_back.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("l2_norm_f32", "l2_norm.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_గమ్యం_spv("cpy_32", "copy.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_గమ్యం_spv("cpy_f32_f16", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("cpy_16", "copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_గమ్యం_spv("cpy_f16_f32", "copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_గమ్యం_spv("cpy_f32_bf16","copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_గమ్యం_spv("cpy_bf16_f32","copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
    string_గమ్యం_spv("cpy_bf16_f16","copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float16_t"}, {"DATA_A_BF16", "1"}});
    string_గమ్యం_spv("contig_cpy_32", "contig_copy.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_గమ్యం_spv("contig_cpy_f32_i32", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_గమ్యం_spv("contig_cpy_i32_f32", "contig_copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});
    string_గమ్యం_spv("contig_cpy_f32_f16", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("contig_cpy_16", "contig_copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_గమ్యం_spv("contig_cpy_f16_f32", "contig_copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_గమ్యం_spv("contig_cpy_f32_bf16","contig_copy.comp",{{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_గమ్యం_spv("contig_cpy_bf16_f32","contig_copy.comp",{{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
    string_గమ్యం_spv("contig_cpy_bf16_f16","contig_copy.comp",{{"A_TYPE", "uint16_t"}, {"D_TYPE", "float16_t"}, {"DATA_A_BF16", "1"}});
    string_గమ్యం_spv("cpy_f32_i32", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_గమ్యం_spv("cpy_i32_f32", "copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});

    string_గమ్యం_spv("cpy_transpose_16", "copy_transpose.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_గమ్యం_spv("cpy_transpose_32", "copy_transpose.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_గమ్యం_spv("cpy_transpose_02_16", "copy_transpose_02.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_గమ్యం_spv("cpy_transpose_02_32", "copy_transpose_02.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});

    కొరకు (std::string t : {"q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
        string_గమ్యం_spv("cpy_f32_" + t, "copy_to_quant.comp", {{"DATA_A_" + గమ్యం_uppercase(t), "1"}, {"S_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        string_గమ్యం_spv("cpy_" + t + "_f32", "copy_from_quant.comp", {{"DATA_A_" + గమ్యం_uppercase(t), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }

    కొరకు (స్వయంచాలక src : {std::pair{"f32", "float"}, std::pair{"f16", "float16_t"}}) {
        కొరకు (std::string dst : {"f32", "f16", "bf16", "q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
            string_గమ్యం_spv("set_rows_" + std::string(src.మొదటి) + "_" + dst + "_i32", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + గమ్యం_uppercase(dst), "1"}, {"B_TYPE", "uint"}, {"B_SIZE", "32"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
            string_గమ్యం_spv("set_rows_" + std::string(src.మొదటి) + "_" + dst + "_i64", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + గమ్యం_uppercase(dst), "1"}, {"B_TYPE", "uvec2"}, {"B_SIZE", "64"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        }
    }

    స్వయంచాలక get_type_str = [](తార్కిక f16) {
        తిరిగి f16 ? "float16_t" : "float";
    };
    స్వయంచాలక get_suffix = [](తార్కిక src0_f16, తార్కిక src1_f16, తార్కిక dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        తిరిగి s;
    };
    కొరకు (std::string op : {"add", "sub", "mul", "div", "add_rms", }) {
    కొరకు (స్వయంచాలక src0_f16 : {అబద్ధం, నిజం}) {
    కొరకు (స్వయంచాలక src1_f16 : {అబద్ధం, నిజం}) {
    కొరకు (స్వయంచాలక dst_f16  : {అబద్ధం, నిజం}) {
        స్వయంచాలక ఆధారం = op == "add_rms" ? std::string("add") : op;
        స్వయంచాలక name = op + get_suffix(src0_f16, src1_f16, dst_f16);
        స్వయంచాలక కూడు_rms = op == "add_rms" ? "1" : "0";
        string_గమ్యం_spv(name.c_str(), ఆధారం + ".comp", {{"A_TYPE", get_type_str(src0_f16)}, {"B_TYPE", get_type_str(src1_f16)}, {"D_TYPE", get_type_str(dst_f16)}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , కూడు_rms}});
    }
    }
    }
    }

    string_గమ్యం_spv("sub_f32", "sub.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_గమ్యం_spv("acc_f32", "acc.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_గమ్యం_spv("split_k_reduce", "mul_mat_split_k_reduce.comp", {});
    string_గమ్యం_spv("fa_split_k_reduce", "flash_attn_split_k_reduce.comp", {});

    string_గమ్యం_spv("fa_mask_opt", "flash_attn_mask_opt.comp", {});

#ఒకవేళ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_గమ్యం_spv("fa_decode_ph1", "flash_attn_decode_phase_1.comp", {}, నిజం, నిజం, అబద్ధం, అబద్ధం);
    string_గమ్యం_spv("fa_decode_ph2", "flash_attn_decode_phase_2.comp", {}, నిజం, నిజం, అబద్ధం, అబద్ధం);
#అయితే_ముగింపు

    string_గమ్యం_spv("fa_sparse_compact", "flash_attn_sparse_compact.comp", {});
    string_గమ్యం_spv("fa_sparse_compact_subgroup", "flash_attn_sparse_compact.comp", {{"USE_SUBGROUPS", "1"}});

    string_గమ్యం_spv("quantize_q8_1", "quantize_q8_1.comp", {});
    string_గమ్యం_spv("quantize_q8_1_subgroup", "quantize_q8_1.comp", {{"USE_SUBGROUPS", "1"}});

    string_గమ్యం_spv("quantize_q8_1_x4", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}});
    string_గమ్యం_spv("quantize_q8_1_x4_subgroup", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}, {"USE_SUBGROUPS", "1"}});

    string_గమ్యం_spv("mul_f32", "mul.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_గమ్యం_spv("div_f32", "div.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_గమ్యం_spv("repeat_i32", "repeat.comp", {{"A_TYPE", "int32_t"}, {"D_TYPE", "int32_t"}});
    string_గమ్యం_spv("repeat_back_f32", "repeat_back.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_గమ్యం_spv("get_rows_back_f32", "get_rows_back.comp", {{"A_TYPE", "float"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}});

    string_గమ్యం_spv("repeat_i16", "repeat.comp", {{"A_TYPE", "int16_t"}, {"D_TYPE", "int16_t"}});

    string_గమ్యం_spv("scale_f32", "scale.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_గమ్యం_spv("pad_f32", "pad.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_గమ్యం_spv("pad_reflect_1d_f32", "pad_reflect_1d.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_గమ్యం_spv("concat_i8", "concat.comp", {{"A_TYPE", "uint8_t"}, {"B_TYPE", "uint8_t"}, {"D_TYPE", "uint8_t"}});
    string_గమ్యం_spv("concat_i16", "concat.comp", {{"A_TYPE", "uint16_t"}, {"B_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_గమ్యం_spv("concat_i32", "concat.comp", {{"A_TYPE", "uint"}, {"B_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_గమ్యం_spv("concat_i64", "concat.comp", {{"A_TYPE", "uvec2"}, {"B_TYPE", "uvec2"}, {"D_TYPE", "uvec2"}});

    string_గమ్యం_spv("upscale_f32", "upscale.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_గమ్యం_spv("exp_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_exp"}});
    string_గమ్యం_spv("exp_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_exp"}});
    string_గమ్యం_spv("expm1_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_expm1"}});
    string_గమ్యం_spv("expm1_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_expm1"}});

    string_గమ్యం_spv("log_f16",        "log.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("log_f32",        "log.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_గమ్యం_spv("gelu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu"}});
    string_గమ్యం_spv("gelu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu"}});
    string_గమ్యం_spv("gelu_erf_f16",   "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu_erf"}});
    string_గమ్యం_spv("gelu_erf_f32",   "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu_erf"}});
    string_గమ్యం_spv("gelu_quick_f16", "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu_quick"}});
    string_గమ్యం_spv("gelu_quick_f32", "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu_quick"}});
    string_గమ్యం_spv("silu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_silu"}});
    string_గమ్యం_spv("silu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_silu"}});
    string_గమ్యం_spv("relu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_relu"}});
    string_గమ్యం_spv("relu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_relu"}});
    string_గమ్యం_spv("sqr_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sqr"}});
    string_గమ్యం_spv("sqr_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sqr"}});
    string_గమ్యం_spv("sqrt_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sqrt"}});
    string_గమ్యం_spv("sqrt_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sqrt"}});
    string_గమ్యం_spv("sin_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sin"}});
    string_గమ్యం_spv("sin_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sin"}});
    string_గమ్యం_spv("cos_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_cos"}});
    string_గమ్యం_spv("cos_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_cos"}});
    string_గమ్యం_spv("clamp_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_clamp"}});
    string_గమ్యం_spv("clamp_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_clamp"}});
    string_గమ్యం_spv("leaky_relu_f16", "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_leaky_relu"}});
    string_గమ్యం_spv("leaky_relu_f32", "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_leaky_relu"}});
    string_గమ్యం_spv("neg_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_neg"}});
    string_గమ్యం_spv("neg_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_neg"}});
    string_గమ్యం_spv("tanh_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_tanh"}});
    string_గమ్యం_spv("tanh_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_tanh"}});
    string_గమ్యం_spv("sigmoid_f16",    "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sigmoid"}});
    string_గమ్యం_spv("sigmoid_f32",    "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sigmoid"}});
    string_గమ్యం_spv("hardsigmoid_f16","unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_hardsigmoid"}});
    string_గమ్యం_spv("hardsigmoid_f32","unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_hardsigmoid"}});
    string_గమ్యం_spv("hardswish_f16",  "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_hardswish"}});
    string_గమ్యం_spv("hardswish_f32",  "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_hardswish"}});
    string_గమ్యం_spv("abs_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_abs"}});
    string_గమ్యం_spv("abs_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_abs"}});
    string_గమ్యం_spv("elu_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_elu"}});
    string_గమ్యం_spv("elu_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_elu"}});
    string_గమ్యం_spv("xielu_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_xielu"}});
    string_గమ్యం_spv("xielu_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_xielu"}});
    string_గమ్యం_spv("sgn_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sgn"}});
    string_గమ్యం_spv("sgn_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sgn"}});

    string_గమ్యం_spv("tri_f16",        "tri.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("tri_f32",        "tri.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_గమ్యం_spv("diag_f16",       "diag.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("diag_f32",       "diag.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_గమ్యం_spv("softplus_f16",   "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_softplus"}});
    string_గమ్యం_spv("softplus_f32",   "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_softplus"}});

    string_గమ్యం_spv("gelu_mul_f32",    "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_gelu"},     {"UNARY_MUL_FUSION", "1"}});
    string_గమ్యం_spv("gelu_mul_f16",    "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_gelu"},     {"UNARY_MUL_FUSION", "1"}});
    string_గమ్యం_spv("sigmoid_mul_f32", "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_sigmoid"},  {"UNARY_MUL_FUSION", "1"}});
    string_గమ్యం_spv("sigmoid_mul_f16", "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_sigmoid"},  {"UNARY_MUL_FUSION", "1"}});
    string_గమ్యం_spv("silu_mul_f32",    "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_silu"},     {"UNARY_MUL_FUSION", "1"}});
    string_గమ్యం_spv("silu_mul_f16",    "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_silu"},     {"UNARY_MUL_FUSION", "1"}});
    string_గమ్యం_spv("softplus_mul_f32","unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_softplus"}, {"UNARY_MUL_FUSION", "1"}});
    string_గమ్యం_spv("softplus_mul_f16","unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_softplus"}, {"UNARY_MUL_FUSION", "1"}});

    string_గమ్యం_spv("add1_f16_f16",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_గమ్యం_spv("add1_f16_f32",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_గమ్యం_spv("add1_f32_f32",   "add1.comp",        {{"A_TYPE", "float"},       {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_గమ్యం_spv("arange_f32",     "arange.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_గమ్యం_spv("fill_f32",       "fill.comp",        {{"D_TYPE", "float"},       {"FLOAT_TYPE", "float"}});
    string_గమ్యం_spv("fill_f16",       "fill.comp",        {{"D_TYPE", "float16_t"},   {"FLOAT_TYPE", "float"}});
    string_గమ్యం_spv("step_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_step"}});
    string_గమ్యం_spv("step_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_step"}});
    string_గమ్యం_spv("round_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_round"}});
    string_గమ్యం_spv("round_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_round"}});
    string_గమ్యం_spv("ceil_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_ceil"}});
    string_గమ్యం_spv("ceil_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_ceil"}});
    string_గమ్యం_spv("floor_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_floor"}});
    string_గమ్యం_spv("floor_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_floor"}});
    string_గమ్యం_spv("trunc_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_trunc"}});
    string_గమ్యం_spv("trunc_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_trunc"}});

    string_గమ్యం_spv("geglu_f16",      "geglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("geglu_f32",      "geglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_గమ్యం_spv("reglu_f16",      "reglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("reglu_f32",      "reglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_గమ్యం_spv("swiglu_f16",     "swiglu.comp",      {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("swiglu_f32",     "swiglu.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_గమ్యం_spv("swiglu_oai_f16", "swiglu_oai.comp",  {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("swiglu_oai_f32", "swiglu_oai.comp",  {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_గమ్యం_spv("swiglu_clamp_f16", "swiglu_clamp.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("swiglu_clamp_f32", "swiglu_clamp.comp", {{"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_గమ్యం_spv("geglu_erf_f16",  "geglu_erf.comp",   {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("geglu_erf_f32",  "geglu_erf.comp",   {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_గమ్యం_spv("geglu_quick_f16","geglu_quick.comp", {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("geglu_quick_f32","geglu_quick.comp", {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_గమ్యం_spv("silu_back_f32",  "silu_back.comp",   {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_గమ్యం_spv("diag_mask_inf_f32", "diag_mask_inf.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_గమ్యం_spv("soft_max_f32", "soft_max.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("soft_max_f32_f16", "soft_max.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("soft_max_back_f32", "soft_max_back.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_గమ్యం_spv("soft_max_large1_f32", "soft_max_large1.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("soft_max_large2_f32", "soft_max_large2.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("soft_max_large3_f32", "soft_max_large3.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("soft_max_large1_f32_f16", "soft_max_large1.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("soft_max_large2_f32_f16", "soft_max_large2.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("soft_max_large3_f32_f16", "soft_max_large3.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));

    string_గమ్యం_spv("rope_norm_f32", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_గమ్యం_spv("rope_norm_f16", "rope_norm.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_గమ్యం_spv("rope_norm_f32_f16", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_గమ్యం_spv("rope_neox_f32", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_గమ్యం_spv("rope_neox_f16", "rope_neox.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_గమ్యం_spv("rope_neox_f32_f16", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_గమ్యం_spv("rope_multi_f32", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_గమ్యం_spv("rope_multi_f16", "rope_multi.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_గమ్యం_spv("rope_multi_f32_f16", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_గమ్యం_spv("rope_vision_f32", "rope_vision.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_గమ్యం_spv("rope_vision_f16", "rope_vision.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});

    string_గమ్యం_spv("argsort_f32", "argsort.comp", {{"A_TYPE", "float"}});
    string_గమ్యం_spv("argsort_large_f32", "argsort_large.comp", {{"A_TYPE", "float"}});

    string_గమ్యం_spv("topk_argsort_f32", "topk_argsort.comp", {{"A_TYPE", "float"}});
    string_గమ్యం_spv("topk_nary_search_f32", "topk_nary_search.comp", {{"A_TYPE", "float"}});
    string_గమ్యం_spv("topk_radix_select_f32", "topk_radix_select.comp", {{"A_TYPE", "float"}});

    string_గమ్యం_spv("argmax_f32", "argmax.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "int"}}));
    string_గమ్యం_spv("sum_rows_f32", "sum_rows.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("cross_entropy_loss_f32", "cross_entropy_loss.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("cross_entropy_loss_back_f32", "cross_entropy_loss_back.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("fwht_f32", "fwht.comp", {});
    string_గమ్యం_spv("fwht_shmem_f32", "fwht.comp", {{"FWHT_SHMEM", "1"}});
    string_గమ్యం_spv("count_equal_i32", "count_equal.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "int"}, {"B_TYPE", "int"}, {"D_TYPE", "int"}}));
    string_గమ్యం_spv("dsv4_hc_comb_f32", "dsv4_hc_comb.comp", {});
    string_గమ్యం_spv("dsv4_hc_pre_f32",  "dsv4_hc_pre.comp",  {});
    string_గమ్యం_spv("dsv4_hc_post_f32", "dsv4_hc_post.comp", {});
    string_గమ్యం_spv("cumsum_f32", "cumsum.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("cumsum_multipass1_f32", "cumsum_multipass1.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("cumsum_multipass2_f32", "cumsum_multipass2.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_గమ్యం_spv("count_experts", "count_experts.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}}));
    string_గమ్యం_spv("count_experts_subgroup", "count_experts.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}, {"USE_SUBGROUPS", "1"}}));

    కొరకు (std::string dim_str : {"", "_3d"}) {
        కొరకు (తార్కిక bda : {అబద్ధం, నిజం}) {
            std::string bda_str = bda ? "_bda" : "";
            std::string bda_def = bda ? "1" : "0";
            string_గమ్యం_spv("im2col" + dim_str + "_f32" + bda_str, "im2col" + dim_str + ".comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"D_SIZE", "4"}, {"BDA", bda_def}}));
            string_గమ్యం_spv("im2col" + dim_str + "_f32_f16" + bda_str, "im2col" + dim_str + ".comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"D_SIZE", "2"}, {"BDA", bda_def}}));
        }
    }

    string_గమ్యం_spv("out_prod_f32", "out_prod.comp", {});

    string_గమ్యం_spv("timestep_embedding_f32", "timestep_embedding.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_గమ్యం_spv("conv_transpose_1d_f32", "conv_transpose_1d.comp", {{"A_TYPE", "float"},  {"B_TYPE", "float"}, {"D_TYPE", "float"}});
    string_గమ్యం_spv("col2im_1d_f32",  "col2im_1d.comp", {{"DATA_A_F32", "1"},  {"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_గమ్యం_spv("col2im_1d_f16",  "col2im_1d.comp", {{"DATA_A_F16", "1"},  {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("col2im_1d_bf16", "col2im_1d.comp", {{"DATA_A_BF16", "1"}, {"A_TYPE", "uint16_t"},  {"D_TYPE", "uint16_t"}});

    string_గమ్యం_spv("snake_f32",  "snake.comp", {{"DATA_A_F32", "1"},  {"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_గమ్యం_spv("snake_f16",  "snake.comp", {{"DATA_A_F16", "1"},  {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_గమ్యం_spv("snake_bf16", "snake.comp", {{"DATA_A_BF16", "1"}, {"DATA_D_BF16", "1"}, {"A_TYPE", "uint16_t"},  {"D_TYPE", "uint16_t"}});

    string_గమ్యం_spv("pool1d_f32", "pool1d.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("pool2d_f32", "pool2d.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_గమ్యం_spv("pool2d_subgroup_f32", "pool2d.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"USE_SUBGROUP", "1"}}));

    string_గమ్యం_spv("rwkv_wkv6_f32", "wkv6.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}}));

    string_గమ్యం_spv("gated_linear_attn_f32", "gla.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}}));

    // Compile IQ4_NL support in so its shared LUT is available when K uses it.
    // K quant type is selected at runtime via the FaTypeK spec constant.
    std::map<std::string, std::string> li_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV4", "vec4"}, {"DATA_A_IQ4_NL", "1"}};
    string_గమ్యం_spv("lightning_indexer_f32", "lightning_indexer.comp", li_dict);

    string_గమ్యం_spv("rwkv_wkv7_f32", "wkv7.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}}));

    string_గమ్యం_spv("gated_delta_net_f32", "gated_delta_net.comp", విలీనం_maps(పునాది_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "1"}}));
    string_గమ్యం_spv("gated_delta_net_f32_nocluster", "gated_delta_net.comp", విలీనం_maps(పునాది_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));
    string_గమ్యం_spv("gated_delta_net_f32_shmem", "gated_delta_net.comp", విలీనం_maps(పునాది_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "0"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));

    string_గమ్యం_spv("opt_step_adamw_f32", "opt_step_adamw.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}}));
    string_గమ్యం_spv("opt_step_sgd_f32", "opt_step_sgd.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}}));

    string_గమ్యం_spv("solve_tri_f32", "solve_tri.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    కొరకు (స్వయంచాలక transpose : {అబద్ధం, నిజం}) {
        కొరకు (స్వయంచాలక unroll : {అబద్ధం, నిజం}) {
            కొరకు (స్వయంచాలక a_f16 : {అబద్ధం, నిజం}) {
                std::map<std::string, std::string> నిర్వచిస్తుంది = {
                    {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                    {"USE_COLLECTIVES", "1"}, {"UNROLL", unroll ? "[[unroll]]" : ""},
                };
                ఒకవేళ (transpose) నిర్వచిస్తుంది["TRANSPOSE"] = "1";
                std::string name = std::string(transpose ? "conv_transpose_2d": "conv2d")
                    + (a_f16 ? "_f16" : "") + "_f32";
                string_గమ్యం_spv(name + (unroll ? "_unroll" : ""), "conv2d_mm.comp", నిర్వచిస్తుంది);
#ఒకవేళ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                ఒకవేళ (unroll) {
                    స్వయంచాలక cm2_నిర్వచిస్తుంది = నిర్వచిస్తుంది;
                    cm2_నిర్వచిస్తుంది["COOPMAT2"] = "1";
                    string_గమ్యం_spv(name, "conv2d_mm.comp", cm2_నిర్వచిస్తుంది, నిజం, అబద్ధం, నిజం);
                }
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                ఒకవేళ (unroll) {
                    స్వయంచాలక cm1_నిర్వచిస్తుంది = నిర్వచిస్తుంది;
                    cm1_నిర్వచిస్తుంది["COOPMAT"] = "1";
                    string_గమ్యం_spv(name, "conv2d_mm.comp", cm1_నిర్వచిస్తుంది, నిజం, నిజం, అబద్ధం);
                }
#అయితే_ముగింపు
            }
        }
    }

    కొరకు (స్వయంచాలక unroll : {అబద్ధం, నిజం}) {
        కొరకు (స్వయంచాలక a_f16 : {అబద్ధం, నిజం}) {
            std::map<std::string, std::string> నిర్వచిస్తుంది = {
                {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                {"UNROLL", unroll ? "[[unroll]]" : ""},
            };
            std::string name = std::string("conv3d") + (a_f16 ? "_f16" : "") + "_f32";
            string_గమ్యం_spv(name + (unroll ? "_unroll" : ""), "conv3d_mm.comp", నిర్వచిస్తుంది);
#ఒకవేళ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            ఒకవేళ (unroll) {
                స్వయంచాలక cm2_నిర్వచిస్తుంది = నిర్వచిస్తుంది;
                cm2_నిర్వచిస్తుంది["COOPMAT2"] = "1";
                string_గమ్యం_spv(name, "conv3d_mm.comp", cm2_నిర్వచిస్తుంది, నిజం, అబద్ధం, నిజం);
            }
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            ఒకవేళ (unroll) {
                స్వయంచాలక cm1_నిర్వచిస్తుంది = నిర్వచిస్తుంది;
                cm1_నిర్వచిస్తుంది["COOPMAT"] = "1";
                string_గమ్యం_spv(name, "conv3d_mm.comp", cm1_నిర్వచిస్తుంది, నిజం, నిజం, అబద్ధం);
            }
#అయితే_ముగింపు
        }
    }

    string_గమ్యం_spv("conv2d_dw_whcn_f32", "conv2d_dw.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_గమ్యం_spv("conv2d_dw_cwhn_f32", "conv2d_dw.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));
    string_గమ్యం_spv("conv2d_dw_whcn_f16_f32", "conv2d_dw.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_గమ్యం_spv("conv2d_dw_cwhn_f16_f32", "conv2d_dw.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));

    string_గమ్యం_spv("roll_f32", "roll.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_గమ్యం_spv("add_id_f32", "add_id.comp", విలీనం_maps(పునాది_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_గమ్యం_spv("multi_add_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , "0"}});
    string_గమ్యం_spv("multi_add_rms_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , "1"}});

    string_గమ్యం_spv("ssm_scan_f32",          "ssm_scan.comp", {{"A_TYPE", "float"}});
    string_గమ్యం_spv("ssm_scan_subgroup_f32", "ssm_scan.comp", {{"A_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});

    string_గమ్యం_spv("ssm_conv_f32", "ssm_conv.comp", {{"A_TYPE", "float"}});

    string_గమ్యం_spv("topk_moe_f32", "topk_moe.comp", {});

    కొరకు (స్వయంచాలక &c : compiles) {
        c.wait();
    }
}

శూన్య write_output_files() {
    std::stringstream hdr = make_generic_stringstream();
    std::stringstream src = make_generic_stringstream();

    hdr << "#include <cstdint>\n\n";
    src << "#include \"" << basename(లక్ష్యం_hpp) << "\"\n\n";

    std::sort(shader_fnames.begin(), shader_fnames.end());
    కొరకు (స్థిరరాశి స్వయంచాలక& pair : shader_fnames) {
        స్థిరరాశి std::string& name = pair.మొదటి;
        #నిర్వచితమైతే _WIN32
            std::string path = pair.second;
            std::replace(path.begin(), path.end(), '/', '\\' );
        #లేకపోతే
            స్థిరరాశి std::string& path = pair.second;
        #అయితే_ముగింపు

        hdr << "extern const uint64_t " << name << "_len;\n";
        hdr << "extern const unsigned char " << name << "_data[];\n\n";

        ఒకవేళ (ఆదానం_filepath != "") {
            std::string దత్తాంశం = read_ద్విమాన_file(path);
            ఒకవేళ (దత్తాంశం.ఖాళీలు()) {
                కొనసాగు;
            }

            src << "const uint64_t " << name << "_len = " << దత్తాంశం.size() << ";\n";
            src << "const unsigned char " << name << "_data[" << దత్తాంశం.size() << "] = {\n" << std::hex;
            స్వయంచాలక బైట్లు = reinterpret_cast<స్థిరరాశి uint8_t*>(దత్తాంశం.దత్తాంశం());
            కొరకు (size_t i = 0; i < దత్తాంశం.size(); ++i) {
                src << "0x" << static_cast<పూర్ణాంకం>(బైట్లు[i]) << ",";
                ఒకవేళ ((i + 1) % 12 == 0) src << "\n";
            }
            src << std::dec << "\n};\n\n";
        }
    }

    std::string suffixes[2] = {"_f32", "_f16"};
    కొరకు (std::string op : {"add", "sub", "mul", "div", "add_rms"}) {
        hdr << "extern const void * " << op << "_data[2][2][2];\n";
        hdr << "extern const uint64_t " << op << "_len[2][2][2];\n";

        std::string op_file = op == "add_rms" ? "add.comp" : std::string(op) + ".comp";
        ఒకవేళ (basename(ఆదానం_filepath) != op_file) {
            కొనసాగు;
        }
        std::stringstream దత్తాంశం = make_generic_stringstream();
        std::stringstream len  = make_generic_stringstream();
        దత్తాంశం << "const void * " << op << "_data[2][2][2] = ";
        len  << "const uint64_t " << op << "_len[2][2][2] = ";
        కొరకు (uint32_t t0 = 0; t0 < 2; ++t0) {
            ఒకవేళ (t0 == 0) {
                దత్తాంశం << "{";
                len  << "{";
            }
            కొరకు (uint32_t t1 = 0; t1 < 2; ++t1) {
                ఒకవేళ (t1 == 0) {
                    దత్తాంశం << "{";
                    len  << "{";
                }
                కొరకు (uint32_t t2 = 0; t2 < 2; ++t2) {
                    ఒకవేళ (t2 == 0) {
                        దత్తాంశం << "{";
                        len  << "{";
                    }
                    దత్తాంశం << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    len  << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    దత్తాంశం << "_data,";
                    len  << "_len,";
                    ఒకవేళ (t2 == 1) {
                        దత్తాంశం << "}, ";
                        len  << "}, ";
                    }
                }
                ఒకవేళ (t1 == 1) {
                    దత్తాంశం << "}, ";
                    len  << "}, ";
                }
            }
            ఒకవేళ (t0 == 1) {
                దత్తాంశం << "};\n";
                len  << "};\n";
            }
        }
        src << దత్తాంశం.str();
        src << len.str();
    }

    std::vector<std::string> btypes = {"f16", "f32"};

#ఒకవేళ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    btypes.push_back("q8_1");
#అయితే_ముగింపు

    కొరకు (స్థిరరాశి std::string& btype : btypes) {
    కొరకు (స్థిరరాశి స్వయంచాలక& tname : type_పేర్లు) {
        ఒకవేళ (btype == "q8_1" && !is_legacy_quant(tname) && tname != "mxfp4" && !is_k_quant(tname) && tname != "iq1_s" && tname != "iq1_m" && tname != "iq4_xs") {
            కొనసాగు;
        }
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3];\n";
        ఒకవేళ (basename(ఆదానం_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }

        ఒకవేళ (btype == "f16") {
            కొనసాగు;
        }
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3];\n";
        ఒకవేళ (basename(ఆదానం_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_id_" << tname << "_" << btype << "_f32_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_id_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_id_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }
    }
    }

#ఒకవేళ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    కొరకు (స్థిరరాశి std::string& btype : {"f16", "f32"}) {
    కొరకు (స్థిరరాశి std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3];\n";
        ఒకవేళ (basename(ఆదానం_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
    }
    కొరకు (స్థిరరాశి std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3];\n";
        ఒకవేళ (basename(ఆదానం_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3] = {mul_mat_vec_id_" << tname << "_f32_f32_ocp_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3] =  {mul_mat_vec_id_" << tname << "_f32_f32_ocp_len,  mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_len, mul_mat_vec_id_"  << tname << "_f32_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
#అయితే_ముగింపు

    ఒకవేళ (ఆదానం_filepath == "") {
        write_file_if_మారింది(లక్ష్యం_hpp, hdr.str());
    }
    ఒకవేళ (లక్ష్యం_cpp != "") {
        write_ద్విమాన_file(లక్ష్యం_cpp, src.str());
    }
}

} // namespace

పూర్ణాంకం ప్రధాన(పూర్ణాంకం argc, అక్షరం** argv) {
    std::map<std::string, std::string> వాదనలు_2;
    కొరకు (పూర్ణాంకం i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        ఒకవేళ (arg.rfind("--", 0) == 0) {
            ఒకవేళ (i + 1 < argc && argv[i + 1][0] != '-') {
                వాదనలు_2[arg] = argv[i + 1];
                ++i;
            } లేకపోతే {
                వాదనలు_2[arg] = "";
            }
        }
    }

    ఒకవేళ (వాదనలు_2.find("--glslc") != వాదనలు_2.end()) {
        GLSLC = వాదనలు_2["--glslc"]; // Path to glslc
    }
    ఒకవేళ (వాదనలు_2.find("--source") != వాదనలు_2.end()) {
        ఆదానం_filepath = వాదనలు_2["--source"]; // The shader source file to compile
    }
    ఒకవేళ (వాదనలు_2.find("--output-dir") != వాదనలు_2.end()) {
        output_dir = వాదనలు_2["--output-dir"]; // Directory for containing SPIR-V output
    }
    ఒకవేళ (వాదనలు_2.find("--target-hpp") != వాదనలు_2.end()) {
        లక్ష్యం_hpp = వాదనలు_2["--target-hpp"]; // Path to generated header file
    }
    ఒకవేళ (వాదనలు_2.find("--target-cpp") != వాదనలు_2.end()) {
        లక్ష్యం_cpp = వాదనలు_2["--target-cpp"]; // Path to generated cpp file
    }

    ఒకవేళ (!directory_exists(output_dir)) {
        ఒకవేళ (!create_directory(output_dir)) {
            std::cerr << "Error creating output directory: " << output_dir << "\n";
            తిరిగి EXIT_FAILURE;
        }
    }

    process_shaders();

    ఒకవేళ (compile_failed) {
        std::cerr << "vulkan-shaders-gen: one or more shaders failed to compile" << std::పంక్తి_ముగింపు;
        తిరిగి EXIT_FAILURE;
    }

    write_output_files();

    తిరిగి EXIT_SUCCESS;
}
