import os
import re
import argparse


def expand_includes(shader, input_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.join(input_dir, fname)
        အကယ်၍ na os.path.exists(file_path):
            raise FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            included_code = f.read()
        # Recursively expand includes inside the included file
        ပြန်ပေး expand_includes(included_code, input_dir)

    ပြန်ပေး include_pattern.sub(replacer, shader)


def chunk_shader(shader_code, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    ပြန်ပေး [shader_code[i : i + max_chunk_len] အတွက် i anta_h range(0, len(shader_code), max_chunk_len)]


def raw_delim(shader_code):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    နေစဉ် f"){delim}\"" anta_h shader_code:
        delim += "_x"
    ပြန်ပေး delim


def write_shader(shader_name, shader_code, output_dir, outfile, input_dir):
    shader_code = expand_includes(shader_code, input_dir)

    အကယ်၍ output_dir:
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, "w", encoding="utf-8") as f_out:
            f_out.write(shader_code)

    delim = raw_delim(shader_code)
    chunks = chunk_shader(shader_code)

    အကယ်၍ len(chunks) == 1:
        outfile.write(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    မဟုတ်ပါက:
        အတွက် idx, chunk anta_h enumerate(chunks):
            outfile.write(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.write(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.write('    static const std::string s = []{\n')
        outfile.write('        std::string tmp;\n')
        outfile.write(f'        tmp.reserve({len(shader_code)});\n')
        အတွက် idx anta_h range(len(chunks)):
            outfile.write(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.write('        return tmp;\n')
        outfile.write('    }();\n')
        outfile.write('    return s;\n')
        outfile.write('}\n')
        outfile.write(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    with open(args.output_file, "w", encoding="utf-8") as out:
        out.write("// Auto-generated shader embedding\n")
        out.write("#include <string>\n\n")
        အတွက် fname anta_h sorted(os.listdir(args.input_dir)):
            အကယ်၍ fname.endswith(".wgsl"):
                shader_path = os.path.join(args.input_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                with open(shader_path, "r", encoding="utf-8") as f:
                    shader_code = f.read()

                write_shader(shader_name, shader_code, None, out, args.input_dir)


အကယ်၍ __name__ == "__main__":
    main()
