#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#ضمن <cctype>
#ضمن <fstream>
#ضمن <sstream>
#ضمن <stdexcept>
#ضمن <string>
#ضمن <string_view>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <vector>

نطاق pre_wgsl {

//==============================================================
// Options
//==============================================================
بنية Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
ساكن std::string trim(ثابت std::string & s) {
    size_t a = 0;
    ܟܡܐ (a < s.size() && std::isspace((unsigned حرف) s[a])) {
        a++;
    }
    size_t b = s.size();
    ܟܡܐ (b > a && std::isspace((unsigned حرف) s[b - 1])) {
        b--;
    }
    ܦܢܐ s.substr(a, b - a);
}

ساكن std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ܦܢܐ trim(ss.str());
}

ساكن منطقي isIdentChar(حرف c) {
    ܦܢܐ std::isalnum(static_cast<unsigned حرف>(c)) || c == '_';
}

ساكن منطقي endsWithContinuation(ثابت std::string & line) {
    size_t i = line.size();
    ܟܡܐ (i > 0 && std::isspace((unsigned حرف) line[i - 1])) {
        i--;
    }
    ܦܢܐ i > 0 && line[i - 1] == '\\';
}

ساكن فراغ stripContinuation(std::string & line) {
    size_t i = line.size();
    ܟܡܐ (i > 0 && std::isspace((unsigned حرف) line[i - 1])) {
        i--;
    }
    ܐܢ (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

ساكن std::string expandMacrosRecursiveInternal(ثابت std::string &                                  line,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

ساكن std::string expandMacroValue(ثابت std::string &                                  name,
                                    ثابت std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ܐܢ (visiting.count(name)) {
        ارم std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    ܐܢ (it == macros.end()) {
        visiting.erase(name);
        ܦܢܐ name;
    }

    ثابت std::string & value = it->second;
    ܐܢ (value.empty()) {
        visiting.erase(name);
        ܦܢܐ "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ܦܢܐ expanded;
}

ساكن std::string expandMacrosRecursiveInternal(ثابت std::string &                                  line,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    ܟܡܐ (i < line.size()) {
        ܐܢ (isIdentChar(line[i])) {
            size_t start = i;
            ܟܡܐ (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            ܐܢ (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } ܐܚܪܢܐ {
                result += token;
            }
        } ܐܚܪܢܐ {
            result += line[i];
            i++;
        }
    }

    ܦܢܐ result;
}

ساكن std::string expandMacrosRecursive(ثابت std::string &                                  line,
                                         ثابت std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ܦܢܐ expandMacrosRecursiveInternal(line, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
صنف ExprLexer {
  عام:
    تعداد Kind { END, IDENT, NUMBER, OP, LPAREN, RPAREN };

    بنية Tok {
        Kind        kind;
        std::string text;
    };

    explicit ExprLexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        ܐܢ (pos >= src.size()) {
            ܦܢܐ { END, "" };
        }

        حرف c = src[pos];

        // number
        ܐܢ (std::isdigit((unsigned حرف) c)) {
            size_t start = pos;
            ܟܡܐ (pos < src.size() && std::isdigit((unsigned حرف) src[pos])) {
                pos++;
            }
            ܦܢܐ { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ܐܢ (std::isalpha((unsigned حرف) c) || c == '_') {
            size_t start = pos;
            ܟܡܐ (pos < src.size() && (std::isalnum((unsigned حرف) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ܦܢܐ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ܐܢ (c == '(') {
            pos++;
            ܦܢܐ { LPAREN, "(" };
        }
        ܐܢ (c == ')') {
            pos++;
            ܦܢܐ { RPAREN, ")" };
        }

        // multi-char operators
        ساكن ثابت حرف * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        لكل (auto op : two_ops) {
            ܐܢ (src.substr(pos, 2) == op) {
                pos += 2;
                ܦܢܐ { OP, std::string(op) };
            }
        }

        // single-char operators
        ܐܢ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ܦܢܐ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ܦܢܐ { END, "" };
    }

  خاص:
    std::string_view src;
    size_t           pos;

    فراغ skipWS() {
        ܟܡܐ (pos < src.size() && std::isspace((unsigned حرف) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
صنف ExprParser {
  عام:
    ExprParser(std::string_view                                     expr,
               ثابت std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(expr),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    صحيح parse() { ܦܢܐ parseLogicalOr(); }

  خاص:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    ثابت std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    فراغ advance() { tok = lex.next(); }

    منطقي acceptOp(ثابت std::string & s) {
        ܐܢ (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            ܦܢܐ ܫܪܪܐ;
        }
        ܦܢܐ ܕܓܠܐ;
    }

    منطقي acceptKind(ExprLexer::Kind k) {
        ܐܢ (tok.kind == k) {
            advance();
            ܦܢܐ ܫܪܪܐ;
        }
        ܦܢܐ ܕܓܠܐ;
    }

    صحيح parseLogicalOr() {
        صحيح v = parseLogicalAnd();
        ܟܡܐ (acceptOp("||")) {
            صحيح rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ܦܢܐ v;
    }

    صحيح parseLogicalAnd() {
        صحيح v = parseEquality();
        ܟܡܐ (acceptOp("&&")) {
            صحيح rhs = parseEquality();
            v       = (v && rhs);
        }
        ܦܢܐ v;
    }

    صحيح parseEquality() {
        صحيح v = parseRelational();
        لكل (;;) {
            ܐܢ (acceptOp("==")) {
                صحيح rhs = parseRelational();
                v       = (v == rhs);
            } ܐܚܪܢܐ ܐܢ (acceptOp("!=")) {
                صحيح rhs = parseRelational();
                v       = (v != rhs);
            } ܐܚܪܢܐ {
                اكسر;
            }
        }
        ܦܢܐ v;
    }

    صحيح parseRelational() {
        صحيح v = parseShift();
        لكل (;;) {
            ܐܢ (acceptOp("<")) {
                صحيح rhs = parseShift();
                v       = (v < rhs);
            } ܐܚܪܢܐ ܐܢ (acceptOp(">")) {
                صحيح rhs = parseShift();
                v       = (v > rhs);
            } ܐܚܪܢܐ ܐܢ (acceptOp("<=")) {
                صحيح rhs = parseShift();
                v       = (v <= rhs);
            } ܐܚܪܢܐ ܐܢ (acceptOp(">=")) {
                صحيح rhs = parseShift();
                v       = (v >= rhs);
            } ܐܚܪܢܐ {
                اكسر;
            }
        }
        ܦܢܐ v;
    }

    صحيح parseShift() {
        صحيح v = parseAdd();
        لكل (;;) {
            ܐܢ (acceptOp("<<")) {
                صحيح rhs = parseAdd();
                v       = (v << rhs);
            } ܐܚܪܢܐ ܐܢ (acceptOp(">>")) {
                صحيح rhs = parseAdd();
                v       = (v >> rhs);
            } ܐܚܪܢܐ {
                اكسر;
            }
        }
        ܦܢܐ v;
    }

    صحيح parseAdd() {
        صحيح v = parseMult();
        لكل (;;) {
            ܐܢ (acceptOp("+")) {
                صحيح rhs = parseMult();
                v       = (v + rhs);
            } ܐܚܪܢܐ ܐܢ (acceptOp("-")) {
                صحيح rhs = parseMult();
                v       = (v - rhs);
            } ܐܚܪܢܐ {
                اكسر;
            }
        }
        ܦܢܐ v;
    }

    صحيح parseMult() {
        صحيح v = parseUnary();
        لكل (;;) {
            ܐܢ (acceptOp("*")) {
                صحيح rhs = parseUnary();
                v       = (v * rhs);
            } ܐܚܪܢܐ ܐܢ (acceptOp("/")) {
                صحيح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } ܐܚܪܢܐ ܐܢ (acceptOp("%")) {
                صحيح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } ܐܚܪܢܐ {
                اكسر;
            }
        }
        ܦܢܐ v;
    }

    صحيح parseUnary() {
        ܐܢ (acceptOp("!")) {
            ܦܢܐ !parseUnary();
        }
        ܐܢ (acceptOp("-")) {
            ܦܢܐ -parseUnary();
        }
        ܐܢ (acceptOp("+")) {
            ܦܢܐ +parseUnary();
        }
        ܦܢܐ parsePrimary();
    }

    صحيح parsePrimary() {
        // '(' expr ')'
        ܐܢ (acceptKind(ExprLexer::LPAREN)) {
            صحيح v = parse();
            ܐܢ (!acceptKind(ExprLexer::RPAREN)) {
                ارم std::runtime_error("missing ')'");
            }
            ܦܢܐ v;
        }

        // number
        ܐܢ (tok.kind == ExprLexer::NUMBER) {
            صحيح v = std::stoi(tok.text);
            advance();
            ܦܢܐ v;
        }

        // defined(identifier)
        ܐܢ (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            ܐܢ (acceptKind(ExprLexer::LPAREN)) {
                ܐܢ (tok.kind != ExprLexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                ܐܢ (!acceptKind(ExprLexer::RPAREN)) {
                    ارم std::runtime_error("missing ) in defined()");
                }
                ܦܢܐ macros.count(name) ? 1 : 0;
            } ܐܚܪܢܐ {
                // defined NAME
                ܐܢ (tok.kind != ExprLexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ܦܢܐ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ܐܢ (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            ܐܢ (it == macros.end()) {
                ܦܢܐ 0;
            }
            ܐܢ (it->second.empty()) {
                ܦܢܐ 1;
            }
            ܦܢܐ evalMacroExpression(name, it->second);
        }

        // unexpected
        ܦܢܐ 0;
    }

    صحيح evalMacroExpression(ثابت std::string & name, ثابت std::string & value) {
        ܐܢ (visiting.count(name)) {
            ارم std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        صحيح        v = ep.parse();
        visiting.erase(name);
        ܦܢܐ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
صنف Preprocessor {
  عام:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ܐܢ (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ثابت std::string & filename, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::All);
        ܦܢܐ result;
    }

    std::string preprocess(ثابت std::string & contents, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        ܦܢܐ result;
    }

    std::string preprocess_includes_file(ثابت std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ܦܢܐ result;
    }

    std::string preprocess_includes(ثابت std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ܦܢܐ result;
    }

  خاص:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    تعداد صنف DirectiveMode { All, IncludesOnly };

    بنية Cond {
        منطقي parent_active;
        منطقي active;
        منطقي taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    فراغ parseMacroDefinitions(ثابت std::vector<std::string> & macro_defs) {
        لكل (ثابت auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            ܐܢ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } ܐܚܪܢܐ {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    فراغ buildMacros(ثابت std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        لكل (ثابت auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        لكل (ثابت auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            ܐܢ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } ܐܚܪܢܐ {
                name  = trim(def);
                value = "";
            }

            // Add to macros map (will override global if same name)
            macros[name] = value;
            predefined.insert(name);
        }
    }

    //----------------------------------------------------------
    // Helpers
    //----------------------------------------------------------
    std::string loadFile(ثابت std::string & fname) {
        std::ifstream f(fname);
        ܐܢ (!f.is_open()) {
            ارم std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ܦܢܐ ss.str();
    }

    منطقي condActive(ثابت std::vector<Cond> & cond) ثابت {
        ܐܢ (cond.empty()) {
            ܦܢܐ ܫܪܪܐ;
        }
        ܦܢܐ cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ثابت std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ثابت std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        ܐܢ (include_stack.count(name)) {
            ارم std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        ܦܢܐ out;
    }

    std::string processIncludeFile(ثابت std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ثابت std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ܦܢܐ processFile(full_path, macros, predefined_macros, include_stack, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ثابت std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              ثابت std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_stack,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        line;

        ܟܡܐ (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            ܐܢ (!t.empty() && t[0] == '#') {
                ܟܡܐ (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    ܐܢ (!std::getline(in, line)) {
                        اكسر;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            ܐܢ (!t.empty() && t[0] == '#') {
                منطقي handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                ܐܢ (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } ܐܚܪܢܐ {
                ܐܢ (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } ܐܚܪܢܐ ܐܢ (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        ܐܢ (mode == DirectiveMode::All && !cond.empty()) {
            ارم std::runtime_error("Unclosed #if directive");
        }

        ܦܢܐ out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    منطقي handleDirective(ثابت std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         ثابت std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_stack,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        ܐܢ (cmd == "include") {
            ܐܢ (mode == DirectiveMode::All && !condActive(cond)) {
                ܦܢܐ ܫܪܪܐ;
            }
            std::string file;
            iss >> file;
            ܐܢ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            ܦܢܐ ܫܪܪܐ;
        }

        ܐܢ (mode == DirectiveMode::IncludesOnly) {
            ܦܢܐ ܕܓܠܐ;
        }

        ܐܢ (cmd == "define") {
            ܐܢ (!condActive(cond)) {
                ܦܢܐ ܫܪܪܐ;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ܐܢ (predefined_macros.count(name)) {
                ܦܢܐ ܫܪܪܐ;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            ܦܢܐ ܫܪܪܐ;
        }

        ܐܢ (cmd == "undef") {
            ܐܢ (!condActive(cond)) {
                ܦܢܐ ܫܪܪܐ;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ܐܢ (predefined_macros.count(name)) {
                ܦܢܐ ܫܪܪܐ;
            }
            macros.erase(name);
            ܦܢܐ ܫܪܪܐ;
        }

        ܐܢ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            منطقي p = condActive(cond);
            منطقي v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ܦܢܐ ܫܪܪܐ;
        }

        ܐܢ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            منطقي p = condActive(cond);
            منطقي v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ܦܢܐ ܫܪܪܐ;
        }

        ܐܢ (cmd == "if") {
            std::string expr = trim_value(iss);
            منطقي        p    = condActive(cond);
            منطقي        v    = ܕܓܠܐ;
            ܐܢ (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ܦܢܐ ܫܪܪܐ;
        }

        ܐܢ (cmd == "elif") {
            std::string expr = trim_value(iss);

            ܐܢ (cond.empty()) {
                ارم std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ܐܢ (!c.parent_active) {
                c.active = ܕܓܠܐ;
                ܦܢܐ ܫܪܪܐ;
            }

            ܐܢ (c.taken) {
                c.active = ܕܓܠܐ;
                ܦܢܐ ܫܪܪܐ;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            منطقي                            v = ep.parse() != 0;
            c.active                          = v;
            ܐܢ (v) {
                c.taken = ܫܪܪܐ;
            }
            ܦܢܐ ܫܪܪܐ;
        }

        ܐܢ (cmd == "else") {
            ܐܢ (cond.empty()) {
                ارم std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ܐܢ (!c.parent_active) {
                c.active = ܕܓܠܐ;
                ܦܢܐ ܫܪܪܐ;
            }
            ܐܢ (c.taken) {
                c.active = ܕܓܠܐ;
            } ܐܚܪܢܐ {
                c.active = ܫܪܪܐ;
                c.taken  = ܫܪܪܐ;
            }
            ܦܢܐ ܫܪܪܐ;
        }

        ܐܢ (cmd == "endif") {
            ܐܢ (cond.empty()) {
                ارم std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ܦܢܐ ܫܪܪܐ;
        }

        // Unknown directive
        ارم std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
