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

ساكن std::string trim_نيلاي(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    أرجع trim(ss.str());
}

ساكن منطقي isIdentChar(حرف c) {
    أرجع std::isalnum(static_cast<unsigned حرف>(c)) || c == '_';
}

ساكن منطقي endsWithContinuation(ثابت std::string & باريس) {
    size_t i = باريس.size();
    طالما (i > 0 && std::isspace((unsigned حرف) باريس[i - 1])) {
        i--;
    }
    أرجع i > 0 && باريس[i - 1] == '\\';
}

ساكن فراغ stripContinuation(std::string & باريس) {
    size_t i = باريس.size();
    طالما (i > 0 && std::isspace((unsigned حرف) باريس[i - 1])) {
        i--;
    }
    إذا (i > 0 && باريس[i - 1] == '\\') {
        باريس.erase(i - 1);
    }
}

ساكن std::string expandMacrosRecursiveInternal(ثابت std::string &                                  باريس,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

ساكن std::string expand_Macro_نيلاي(ثابت std::string &                                  name,
                                    ثابت std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    إذا (visiting.count(name)) {
        ارم std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    إذا (it == macros.end()) {
        visiting.erase(name);
        أرجع name;
    }

    ثابت std::string & value = it->second;
    إذا (value.empty()) {
        visiting.erase(name);
        أرجع "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    أرجع expanded;
}

ساكن std::string expandMacrosRecursiveInternal(ثابت std::string &                                  باريس,
                                                 ثابت std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(باريس.size());

    size_t i = 0;
    طالما (i < باريس.size()) {
        إذا (isIdentChar(باريس[i])) {
            size_t start = i;
            طالما (i < باريس.size() && isIdentChar(باريس[i])) {
                i++;
            }
            std::string token = باريس.substr(start, i - start);

            auto it = macros.find(token);
            إذا (it != macros.end()) {
                result += expand_Macro_نيلاي(token, macros, visiting);
            } وإلا {
                result += token;
            }
        } وإلا {
            result += باريس[i];
            i++;
        }
    }

    أرجع result;
}

ساكن std::string expandMacrosRecursive(ثابت std::string &                                  باريس,
                                         ثابت std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    أرجع expandMacrosRecursiveInternal(باريس, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
صنف اوڠکاڤن_Lexer {
  عام:
    تعداد Kind { END, IDENT, نومبور_2, OP, LPAREN, RPAREN };

    بنية Tok {
        Kind        kind;
        std::string text;
    };

    explicit اوڠکاڤن_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        إذا (pos >= src.size()) {
            أرجع { END, "" };
        }

        حرف c = src[pos];

        //نومبور
        إذا (std::isdigit((unsigned حرف) c)) {
            size_t start = pos;
            طالما (pos < src.size() && std::isdigit((unsigned حرف) src[pos])) {
                pos++;
            }
            أرجع { نومبور_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        إذا (std::isalpha((unsigned حرف) c) || c == '_') {
            size_t start = pos;
            طالما (pos < src.size() && (std::isalnum((unsigned حرف) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            أرجع { IDENT, std::string(src.substr(start, pos - start)) };
        }

        إذا (c == '(') {
            pos++;
            أرجع { LPAREN, "(" };
        }
        إذا (c == ')') {
            pos++;
            أرجع { RPAREN, ")" };
        }

        // multi-char operators
        ساكن ثابت حرف * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        لكل (auto op : two_ops) {
            إذا (src.substr(pos, 2) == op) {
                pos += 2;
                أرجع { OP, std::string(op) };
            }
        }

        // single-char operators
        إذا (std::string("+-*/%<>!").find(c) != std::string::npos) {
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
صنف اوڠکاڤن_Parser {
  عام:
    اوڠکاڤن_Parser(std::string_view                                     اوڠکاڤن,
               ثابت std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(اوڠکاڤن),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    صحيح parse() { أرجع parseLogicalOr(); }

  خاص:
    اوڠکاڤن_Lexer                                            lex;
    اوڠکاڤن_Lexer::Tok                                       tok;
    ثابت std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    فراغ advance() { tok = lex.next(); }

    منطقي acceptOp(ثابت std::string & s) {
        إذا (tok.kind == اوڠکاڤن_Lexer::OP && tok.text == s) {
            advance();
            أرجع true;
        }
        أرجع خطأ;
    }

    منطقي acceptKind(اوڠکاڤن_Lexer::Kind k) {
        إذا (tok.kind == k) {
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
            إذا (acceptOp("==")) {
                صحيح rhs = parseRelational();
                v       = (v == rhs);
            } وإلا إذا (acceptOp("!=")) {
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
            إذا (acceptOp("<")) {
                صحيح rhs = parseShift();
                v       = (v < rhs);
            } وإلا إذا (acceptOp(">")) {
                صحيح rhs = parseShift();
                v       = (v > rhs);
            } وإلا إذا (acceptOp("<=")) {
                صحيح rhs = parseShift();
                v       = (v <= rhs);
            } وإلا إذا (acceptOp(">=")) {
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
            إذا (acceptOp("<<")) {
                صحيح rhs = parseAdd();
                v       = (v << rhs);
            } وإلا إذا (acceptOp(">>")) {
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
            إذا (acceptOp("+")) {
                صحيح rhs = parseMult();
                v       = (v + rhs);
            } وإلا إذا (acceptOp("-")) {
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
            إذا (acceptOp("*")) {
                صحيح rhs = parseUnary();
                v       = (v * rhs);
            } وإلا إذا (acceptOp("/")) {
                صحيح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } وإلا إذا (acceptOp("%")) {
                صحيح rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } وإلا {
                اكسر;
            }
        }
        أرجع v;
    }

    صحيح parseUnary() {
        إذا (acceptOp("!")) {
            أرجع !parseUnary();
        }
        إذا (acceptOp("-")) {
            أرجع -parseUnary();
        }
        إذا (acceptOp("+")) {
            أرجع +parseUnary();
        }
        أرجع parsePrimary();
    }

    صحيح parsePrimary() {
        // '(' expr ')'
        إذا (acceptKind(اوڠکاڤن_Lexer::LPAREN)) {
            صحيح v = parse();
            إذا (!acceptKind(اوڠکاڤن_Lexer::RPAREN)) {
                ارم std::runtime_error("missing ')'");
            }
            أرجع v;
        }

        //نومبور
        إذا (tok.kind == اوڠکاڤن_Lexer::نومبور_2) {
            صحيح v = std::stoi(tok.text);
            advance();
            أرجع v;
        }

        // defined(identifier)
        إذا (tok.kind == اوڠکاڤن_Lexer::IDENT && tok.text == "defined") {
            advance();
            إذا (acceptKind(اوڠکاڤن_Lexer::LPAREN)) {
                إذا (tok.kind != اوڠکاڤن_Lexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                إذا (!acceptKind(اوڠکاڤن_Lexer::RPAREN)) {
                    ارم std::runtime_error("missing ) in defined()");
                }
                أرجع macros.count(name) ? 1 : 0;
            } وإلا {
                // defined NAME
                إذا (tok.kind != اوڠکاڤن_Lexer::IDENT) {
                    ارم std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                أرجع macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        إذا (tok.kind == اوڠکاڤن_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            إذا (it == macros.end()) {
                أرجع 0;
            }
            إذا (it->second.empty()) {
                أرجع 1;
            }
            أرجع evalMacroExpression(name, it->second);
        }

        // unexpected
        أرجع 0;
    }

    صحيح evalMacroExpression(ثابت std::string & name, ثابت std::string & value) {
        إذا (visiting.count(name)) {
            ارم std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        اوڠکاڤن_Parser ep(value, macros, visiting);
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
        إذا (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ثابت std::string & filename, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_تيندنن;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_تيندنن, DirectiveMode::All);
        أرجع result;
    }

    std::string preprocess(ثابت std::string & contents, ثابت std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_تيندنن;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_تيندنن, DirectiveMode::All);
        أرجع result;
    }

    std::string preprocess_includes_file(ثابت std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_تيندنن;
        std::string result = processFile(filename, macros, predefined, include_تيندنن, DirectiveMode::IncludesOnly);
        أرجع result;
    }

    std::string preprocess_includes(ثابت std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_تيندنن;
        std::string result = processString(contents, macros, predefined, include_تيندنن, DirectiveMode::IncludesOnly);
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
            إذا (eq_pos != std::string::npos) {
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
            إذا (eq_pos != std::string::npos) {
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
        إذا (!f.is_open()) {
            ارم std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        أرجع ss.str();
    }

    منطقي condActive(ثابت std::vector<Cond> & cond) ثابت {
        إذا (cond.empty()) {
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
                            std::unordered_set<std::string> &              include_تيندنن,
                            DirectiveMode                                  mode) {
        إذا (include_تيندنن.count(name)) {
            ارم std::runtime_error("Recursive include: " + name);
        }

        include_تيندنن.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_تيندنن, mode);
        include_تيندنن.erase(name);
        أرجع out;
    }

    std::string processIncludeFile(ثابت std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ثابت std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_تيندنن,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        أرجع processFile(full_path, macros, predefined_macros, include_تيندنن, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ثابت std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              ثابت std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_تيندنن,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        باريس;

        طالما (std::getline(in, باريس)) {
            std::string logical = باريس;
            std::string t       = trim(logical);
            إذا (!t.empty() && t[0] == '#') {
                طالما (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    إذا (!std::getline(in, باريس)) {
                        اكسر;
                    }
                    logical += "\n";
                    logical += باريس;
                }
                t = trim(logical);
            }

            إذا (!t.empty() && t[0] == '#') {
                منطقي handled = handleDirective(t, out, macros, predefined_macros, cond, include_تيندنن, mode);
                إذا (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } وإلا {
                إذا (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } وإلا إذا (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        إذا (mode == DirectiveMode::All && !cond.empty()) {
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
                         std::unordered_set<std::string> &              include_تيندنن,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        إذا (cmd == "include") {
            إذا (mode == DirectiveMode::All && !condActive(cond)) {
                أرجع true;
            }
            std::string file;
            iss >> file;
            إذا (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_تيندنن, mode);
            أرجع true;
        }

        إذا (mode == DirectiveMode::IncludesOnly) {
            أرجع خطأ;
        }

        إذا (cmd == "define") {
            إذا (!condActive(cond)) {
                أرجع true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            إذا (predefined_macros.count(name)) {
                أرجع true;
            }
            std::string value = trim_نيلاي(iss);
            macros[name]      = value;
            أرجع true;
        }

        إذا (cmd == "undef") {
            إذا (!condActive(cond)) {
                أرجع true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            إذا (predefined_macros.count(name)) {
                أرجع true;
            }
            macros.erase(name);
            أرجع true;
        }

        إذا (cmd == "ifdef") {
            std::string name;
            iss >> name;
            منطقي p = condActive(cond);
            منطقي v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            أرجع true;
        }

        إذا (cmd == "ifndef") {
            std::string name;
            iss >> name;
            منطقي p = condActive(cond);
            منطقي v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            أرجع true;
        }

        إذا (cmd == "if") {
            std::string اوڠکاڤن = trim_نيلاي(iss);
            منطقي        p    = condActive(cond);
            منطقي        v    = خطأ;
            إذا (p) {
                std::unordered_set<std::string> visiting;
                اوڠکاڤن_Parser                      ep(اوڠکاڤن, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            أرجع true;
        }

        إذا (cmd == "elif") {
            std::string اوڠکاڤن = trim_نيلاي(iss);

            إذا (cond.empty()) {
                ارم std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            إذا (!c.parent_active) {
                c.active = خطأ;
                أرجع true;
            }

            إذا (c.taken) {
                c.active = خطأ;
                أرجع true;
            }

            std::unordered_set<std::string> visiting;
            اوڠکاڤن_Parser                      ep(اوڠکاڤن, macros, visiting);
            منطقي                            v = ep.parse() != 0;
            c.active                          = v;
            إذا (v) {
                c.taken = true;
            }
            أرجع true;
        }

        إذا (cmd == "else") {
            إذا (cond.empty()) {
                ارم std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            إذا (!c.parent_active) {
                c.active = خطأ;
                أرجع true;
            }
            إذا (c.taken) {
                c.active = خطأ;
            } وإلا {
                c.active = true;
                c.taken  = true;
            }
            أرجع true;
        }

        إذا (cmd == "endif") {
            إذا (cond.empty()) {
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
