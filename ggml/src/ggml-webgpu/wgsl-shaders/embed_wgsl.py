आनयन os
आनयन re
आनयन argparse


परिभाषा expand_includes(shader, निवेशः_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_आदर्शः = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    परिभाषा replacer(मेलनम्):
        fname = मेलनम्.गणः(1)
        file_मार्गः = os.मार्गः.join(निवेशः_dir, fname)
        यदि न os.मार्गः.exists(file_मार्गः):
            उत्थापय File_Not_प्राप्तम्_Error(f"Included file not found: {file_path}")
        सह open(file_मार्गः, "r", encoding="utf-8") इव f:
            included_संहिता = f.पठ()
        # Recursively expand includes inside the included file
        प्रत्यागम expand_includes(included_संहिता, निवेशः_dir)

    प्रत्यागम include_आदर्शः.sub(replacer, shader)


परिभाषा chunk_shader(shader_संहिता, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    प्रत्यागम [shader_संहिता[i : i + max_chunk_len] प्रति i अन्तः परिसरः(0, दीर्घता(shader_संहिता), max_chunk_len)]


परिभाषा raw_delim(shader_संहिता):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    यावत् f"){delim}\"" अन्तः shader_संहिता:
        delim += "_x"
    प्रत्यागम delim


परिभाषा write_shader(shader_name, shader_संहिता, output_dir, outfile, निवेशः_dir):
    shader_संहिता = expand_includes(shader_संहिता, निवेशः_dir)

    यदि output_dir:
        wgsl_filename = os.मार्गः.join(output_dir, f"{shader_name}.wgsl")
        सह open(wgsl_filename, "w", encoding="utf-8") इव f_बहिर्गमनम्:
            f_बहिर्गमनम्.लिख(shader_संहिता)

    delim = raw_delim(shader_संहिता)
    chunks = chunk_shader(shader_संहिता)

    यदि दीर्घता(chunks) == 1:
        outfile.लिख(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    अन्यथा:
        प्रति idx, chunk अन्तः enumerate(chunks):
            outfile.लिख(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.लिख(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.लिख('    static const std::string s = []{\n')
        outfile.लिख('        std::string tmp;\n')
        outfile.लिख(f'        tmp.reserve({len(shader_code)});\n')
        प्रति idx अन्तः परिसरः(दीर्घता(chunks)):
            outfile.लिख(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.लिख('        return tmp;\n')
        outfile.लिख('    }();\n')
        outfile.लिख('    return s;\n')
        outfile.लिख('}\n')
        outfile.लिख(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


परिभाषा मुख्यम्():
    parser = argparse.तर्कः_Parser()
    parser.सङ्कलय_तर्कः("--input_dir", required=सत्य)
    parser.सङ्कलय_तर्कः("--output_file", required=सत्य)
    तर्काः_2 = parser.parse_तर्काः()

    सह open(तर्काः_2.output_file, "w", encoding="utf-8") इव बहिः:
        बहिः.लिख("// Auto-generated shader embedding\n")
        बहिः.लिख("#include <string>\n\n")
        प्रति fname अन्तः sorted(os.listdir(तर्काः_2.निवेशः_dir)):
            यदि fname.endswith(".wgsl"):
                shader_मार्गः = os.मार्गः.join(तर्काः_2.निवेशः_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                सह open(shader_मार्गः, "r", encoding="utf-8") इव f:
                    shader_संहिता = f.पठ()

                write_shader(shader_name, shader_संहिता, किमपि_न, बहिः, तर्काः_2.निवेशः_dir)


यदि __name__ == "__main__":
    मुख्यम्()
