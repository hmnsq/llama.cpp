దిగుమతి os
దిగుమతి re
దిగుమతి argparse


నిర్వచనం expand_includes(shader, ఆదానం_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_నమూనా = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    నిర్వచనం replacer(సరిపోలిక):
        fname = సరిపోలిక.సమూహం(1)
        file_మార్గం = os.మార్గం.join(ఆదానం_dir, fname)
        ఒకవేళ కాదు os.మార్గం.exists(file_మార్గం):
            లేపు File_Not_దొరికింది_Error(f"Included file not found: {file_path}")
        తోడుగా open(file_మార్గం, "r", encoding="utf-8") ఇవ f:
            included_సంకేతం = f.పఠ()
        # Recursively expand includes inside the included file
        తిరిగి expand_includes(included_సంకేతం, ఆదానం_dir)

    తిరిగి include_నమూనా.sub(replacer, shader)


నిర్వచనం chunk_shader(shader_సంకేతం, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    తిరిగి [shader_సంకేతం[i : i + max_chunk_len] కొరకు i లో శ్రేణి(0, పొడవు(shader_సంకేతం), max_chunk_len)]


నిర్వచనం raw_delim(shader_సంకేతం):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    అయితే f"){delim}\"" లో shader_సంకేతం:
        delim += "_x"
    తిరిగి delim


నిర్వచనం write_shader(shader_name, shader_సంకేతం, output_dir, outfile, ఆదానం_dir):
    shader_సంకేతం = expand_includes(shader_సంకేతం, ఆదానం_dir)

    ఒకవేళ output_dir:
        wgsl_filename = os.మార్గం.join(output_dir, f"{shader_name}.wgsl")
        తోడుగా open(wgsl_filename, "w", encoding="utf-8") ఇవ f_బహిర్గతం:
            f_బహిర్గతం.లిఖ(shader_సంకేతం)

    delim = raw_delim(shader_సంకేతం)
    chunks = chunk_shader(shader_సంకేతం)

    ఒకవేళ పొడవు(chunks) == 1:
        outfile.లిఖ(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    లేకపోతే:
        కొరకు idx, chunk లో enumerate(chunks):
            outfile.లిఖ(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.లిఖ(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.లిఖ('    static const std::string s = []{\n')
        outfile.లిఖ('        std::string tmp;\n')
        outfile.లిఖ(f'        tmp.reserve({len(shader_code)});\n')
        కొరకు idx లో శ్రేణి(పొడవు(chunks)):
            outfile.లిఖ(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.లిఖ('        return tmp;\n')
        outfile.లిఖ('    }();\n')
        outfile.లిఖ('    return s;\n')
        outfile.లిఖ('}\n')
        outfile.లిఖ(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


నిర్వచనం ప్రధాన():
    parser = argparse.వాదం_Parser()
    parser.కూడు_వాదం("--input_dir", required=నిజం)
    parser.కూడు_వాదం("--output_file", required=నిజం)
    వాదనలు_2 = parser.parse_వాదనలు()

    తోడుగా open(వాదనలు_2.output_file, "w", encoding="utf-8") ఇవ బహిర్గతం:
        బహిర్గతం.లిఖ("// Auto-generated shader embedding\n")
        బహిర్గతం.లిఖ("#include <string>\n\n")
        కొరకు fname లో sorted(os.listdir(వాదనలు_2.ఆదానం_dir)):
            ఒకవేళ fname.endswith(".wgsl"):
                shader_మార్గం = os.మార్గం.join(వాదనలు_2.ఆదానం_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                తోడుగా open(shader_మార్గం, "r", encoding="utf-8") ఇవ f:
                    shader_సంకేతం = f.పఠ()

                write_shader(shader_name, shader_సంకేతం, ఏమీలేదు, బహిర్గతం, వాదనలు_2.ఆదానం_dir)


ఒకవేళ __name__ == "__main__":
    ప్రధాన()
