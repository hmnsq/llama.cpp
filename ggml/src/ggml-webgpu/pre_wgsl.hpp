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
    طالما (a < s.size() && std::isspace((unsigned حرف) s[a])) {
        a++;
    }
    size_t b = s.size();
    طالما (b > a && std::isspace((unsigned حرف) s[b - 1])) {
        b--;
    }
    أرجع s.substr(a, b - a);
}

ساكن std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    أرجع trim(ss.str());
}

ساكن منطقي isIdentChar(حرف c) {
    أرجع std::isalnum(static_cast<unsigned حرف>(c)) || c == '_';
}

ساكن منطقي endsWithContinuation(ثابت std::string & line) {
    size_t i = line.size();
    طالما (i > 0 && std::isspace((unsigned حرف) line[i - 1])) {
        i--;
    }
    أرجع i > 0 && line[i - 1] == '\\';
}

ساكن فراغ stripContinuation(std::string & line) {
    size_t i = line.size();
    طالما (i > 0 && std::isspace((unsigned حرف) line[i - 1])) {
        i--;
    }
    𐤀𐤌 (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

ساكن std::string expandMacrosRecursiveInternal(ثابت std::string &                                  line,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

ساكن std::string expandMacroValue(ثابت std::string &                                  name,
                                    ثابت std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    𐤀𐤌 (visiting.count(name)) {
        ارم std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    𐤀𐤌 (it == macros.end()) {
        visiting.erase(name);
        أرجع name;
    }

    ثابت std::string & value = it->second;
    𐤀𐤌 (value.empty()) {
        visiting.erase(name);
        أرجع "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    أرجع expanded;
}

ساكن std::string expandMacrosRecursiveInternal(ثابت std::string &                                  line,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    طالما (i < line.size()) {
        𐤀𐤌 (isIdentChar(line[i])) {
            size_t start = i;
            طالما (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            𐤀𐤌 (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } وإلا {
                result += token;
            }
        } وإلا {
            result += line[i];
            i++;
        }
    }

    أرجع result;
}

ساكن std::string expandMacrosRecursive(ثابت std::string &                                  line,
                                         ثابت std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    أرجع expandMacrosRecursiveInternal(line, macros, visiting);
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
        𐤀𐤌 (pos >= src.size()) {
            أرجع { END, "" };
        }

        حرف c = src[pos];

        // number
        𐤀𐤌 (std::isdigit((unsigned حرف) c)) {
            size_t start = pos;
            طالما (pos < src.size() && std::isdigit((unsigned حرف) src[pos])) {
                pos++;
            }
            أرجع { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        𐤀𐤌 (std::isalpha((unsigned حرف) c) || c == '_') {
            size_t start = pos;
            طالما (pos < src.size() && (std::isalnum((unsigned حرف) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            أرجع { IDENT, std::string(src.substr(start, pos - start)) };
        }

        𐤀𐤌 (c == '(') {
            pos++;
            أرجع { LPAREN, "(" };
        }
        𐤀𐤌 (c == ')') {
            pos++;
            أرجع { RPAREN, ")" };
        }

        // multi-char operators
        ساكن ثابت حرف * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        لكل (auto op : two_ops) {
            𐤀𐤌 (src.substr(pos, 2) == op) {
                pos += 2;
                أرجع { OP, std::string(op) };
            }
        }

        // single-char operators
        𐤀𐤌 (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            أرجع { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        أرجع { END, "" };
    }

  خاص:
    std::string_view src;
    size_t           pos;

    فراغ skipWS() {
        طالما (pos < src.size() && std::isspace((unsigned حرف) src[pos])) {
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

    صحيح parse() { أرجع parseLogicalOr(); }

  خاص:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    ثابت std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    فراغ advance() { tok = lex.next(); }

    منطقي acceptOp(ثابت std::string & s) {
        𐤀𐤌 (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            أرجع true;
        }
        أرجع خطأ;
    }

    منطقي acceptKind(ExprLexer::Kind k) {
        𐤀𐤌 (tok.kind == k) {
            advance();
            أرجع true;
        }
        أرجع خطأ;
    }

    صحيح parseLogicalOr() {
        صحيح v = parseLogicalAnd();
        طالما (acceptOp("||")) {
            صحيح rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        أرجع v;
    }

    صحيح parseLogicalAnd() {
        صحيح v = parseEquality();
        طالما (acceptOp("&&")) {
            صحيح rhs = parseEquality();
            v       = (v && rhs);
        }
        أرجع v;
    }

    صحيح parseEquality() {
        صحيح v = parseRelational();
        لكل (;;) {
            𐤀𐤌 (acceptOp("==")) {
                صحيح rhs = parseRelational();
                v       = (v == rhs);
            } وإلا 𐤀𐤌 (acceptOp("!=")) {
                صحيح rhs = parseRelational();
                v       = (v != rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parseRelational() {
        صحيح v = parseShift();
        لكل (;;) {
            𐤀𐤌 (acceptOp("<")) {
                صحيح rhs = parseShift();
                v       = (v < rhs);
            } وإلا 𐤀𐤌 (acceptOp(">")) {
                صحيح rhs = parseShift();
                v       = (v > rhs);
            } وإلا 𐤀𐤌 (acceptOp("<=")) {
                صحيح rhs = parseShift();
                v       = (v <= rhs);
            } وإلا 𐤀𐤌 (acceptOp(">=")) {
                صحيح rhs = parseShift();
                v       = (v >= rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parseShift() {
        صحيح v = parseAdd();
        لكل (;;) {
            𐤀𐤌 (acceptOp("<<")) {
                صحيح rhs = parseAdd();
                v       = (v << rhs);
            } وإلا 𐤀𐤌 (acceptOp(">>")) {
                صحيح rhs = parseAdd();
                v       = (v >> rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parseAdd() {
        صحيح v = parseMult();
        لكل (;;) {
            𐤀𐤌 (acceptOp("+")) {
                صحيح rhs = parseMult();
                v       = (v + rhs);
            } وإلا 𐤀𐤌 (acceptOp("-")) {
                صحيح rhs = parseMult();
                v       = (v - rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parseMult() {
        صحيح v = parseUnary();
        لكل (;;) {
            𐤀𐤌 (acceptOp("*")) {
                صحيح rhs = parseUnary();
                v       = (v * rhs);
            } وإلا 𐤀𐤌 (acceptOp("/")) {
                صحيح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } وإلا 𐤀𐤌 (acceptOp("%")) {
                صحيح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parseUnary() {
        𐤀𐤌 (acceptOp("!")) {
            أرجع !parseUnary();
        }
        𐤀𐤌 (acceptOp("-")) {
            أرجع -parseUnary();
        }
        𐤀𐤌 (acceptOp("+")) {
            أرجع +parseUnary();
        }
        أرجع parsePrimary();
    }

    صحيح parsePrimary() {
        // '(' expr ')'
        𐤀𐤌 (acceptKind(ExprLexer::LPAREN)) {
            صحيح v = parse();
            𐤀𐤌 (!acceptKind(ExprLexer::RPAREN)) {
                ارم std::runtime_error("missing ')'");
            }
            أرجع v;
        }

        // number
        𐤀𐤌 (tok.kind == ExprLexer::NUMBER) {
            صحيح v = std::stoi(tok.text);
            advance();
            أرجع v;
        }

        // defined(identifier)
        𐤀𐤌 (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            𐤀𐤌 (acceptKind(ExprLexer::LPAREN)) {
                𐤀𐤌 (tok.kind != ExprLexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                𐤀𐤌 (!acceptKind(ExprLexer::RPAREN)) {
                    ارم std::runtime_error("missing ) in defined()");
                }
                أرجع macros.count(name) ? 1 : 0;
            } وإلا {
                // defined NAME
                𐤀𐤌 (tok.kind != ExprLexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                أرجع macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        𐤀𐤌 (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            𐤀𐤌 (it == macros.end()) {
                أرجع 0;
            }
            𐤀𐤌 (it->second.empty()) {
                أرجع 1;
            }
            أرجع evalMacroExpression(name, it->second);
        }

        // unexpected
        أرجع 0;
    }

    صحيح evalMacroExpression(ثابت std::string & name, ثابت std::string & value) {
        𐤀𐤌 (visiting.count(name)) {
            ارم std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        صحيح        v = ep.parse();
        visiting.erase(name);
        أرجع v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
صنف Preprocessor {
  عام:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        𐤀𐤌 (opts_.include_path.empty()) {
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
        أرجع result;
    }

    std::string preprocess(ثابت std::string & contents, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        أرجع result;
    }

    std::string preprocess_includes_file(ثابت std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        أرجع result;
    }

    std::string preprocess_includes(ثابت std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        أرجع result;
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
            𐤀𐤌 (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } وإلا {
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
            𐤀𐤌 (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } وإلا {
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
        𐤀𐤌 (!f.is_open()) {
            ارم std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        أرجع ss.str();
    }

    منطقي condActive(ثابت std::vector<Cond> & cond) ثابت {
        𐤀𐤌 (cond.empty()) {
            أرجع true;
        }
        أرجع cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ثابت std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ثابت std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        𐤀𐤌 (include_stack.count(name)) {
            ارم std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        أرجع out;
    }

    std::string processIncludeFile(ثابت std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ثابت std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        أرجع processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        طالما (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            𐤀𐤌 (!t.empty() && t[0] == '#') {
                طالما (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    𐤀𐤌 (!std::getline(in, line)) {
                        اكسر;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            𐤀𐤌 (!t.empty() && t[0] == '#') {
                منطقي handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                𐤀𐤌 (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } وإلا {
                𐤀𐤌 (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } وإلا 𐤀𐤌 (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        𐤀𐤌 (mode == DirectiveMode::All && !cond.empty()) {
            ارم std::runtime_error("Unclosed #if directive");
        }

        أرجع out.str();
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

        𐤀𐤌 (cmd == "include") {
            𐤀𐤌 (mode == DirectiveMode::All && !condActive(cond)) {
                أرجع true;
            }
            std::string file;
            iss >> file;
            𐤀𐤌 (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            أرجع true;
        }

        𐤀𐤌 (mode == DirectiveMode::IncludesOnly) {
            أرجع خطأ;
        }

        𐤀𐤌 (cmd == "define") {
            𐤀𐤌 (!condActive(cond)) {
                أرجع true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            𐤀𐤌 (predefined_macros.count(name)) {
                أرجع true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            أرجع true;
        }

        𐤀𐤌 (cmd == "undef") {
            𐤀𐤌 (!condActive(cond)) {
                أرجع true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            𐤀𐤌 (predefined_macros.count(name)) {
                أرجع true;
            }
            macros.erase(name);
            أرجع true;
        }

        𐤀𐤌 (cmd == "ifdef") {
            std::string name;
            iss >> name;
            منطقي p = condActive(cond);
            منطقي v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            أرجع true;
        }

        𐤀𐤌 (cmd == "ifndef") {
            std::string name;
            iss >> name;
            منطقي p = condActive(cond);
            منطقي v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            أرجع true;
        }

        𐤀𐤌 (cmd == "if") {
            std::string expr = trim_value(iss);
            منطقي        p    = condActive(cond);
            منطقي        v    = خطأ;
            𐤀𐤌 (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            أرجع true;
        }

        𐤀𐤌 (cmd == "elif") {
            std::string expr = trim_value(iss);

            𐤀𐤌 (cond.empty()) {
                ارم std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            𐤀𐤌 (!c.parent_active) {
                c.active = خطأ;
                أرجع true;
            }

            𐤀𐤌 (c.taken) {
                c.active = خطأ;
                أرجع true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            منطقي                            v = ep.parse() != 0;
            c.active                          = v;
            𐤀𐤌 (v) {
                c.taken = true;
            }
            أرجع true;
        }

        𐤀𐤌 (cmd == "else") {
            𐤀𐤌 (cond.empty()) {
                ارم std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            𐤀𐤌 (!c.parent_active) {
                c.active = خطأ;
                أرجع true;
            }
            𐤀𐤌 (c.taken) {
                c.active = خطأ;
            } وإلا {
                c.active = true;
                c.taken  = true;
            }
            أرجع true;
        }

        𐤀𐤌 (cmd == "endif") {
            𐤀𐤌 (cond.empty()) {
                ارم std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            أرجع true;
        }

        // Unknown directive
        ارم std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
