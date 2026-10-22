#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#अन्तर्भाव <cctype>
#अन्तर्भाव <fstream>
#अन्तर्भाव <sstream>
#अन्तर्भाव <stdexcept>
#अन्तर्भाव <string>
#अन्तर्भाव <string_view>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <unordered_set>
#अन्तर्भाव <vector>

नामाकाश pre_wgsl {

//==============================================================
// Options
//==============================================================
संरचना Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
स्थिर std::string trim(ध्रुव std::string & s) {
    size_t a = 0;
    यावत् (a < s.size() && std::isspace((unsigned वर्ण) s[a])) {
        a++;
    }
    size_t b = s.size();
    यावत् (b > a && std::isspace((unsigned वर्ण) s[b - 1])) {
        b--;
    }
    प्रत्यागम s.substr(a, b - a);
}

स्थिर std::string trim_मूल्य(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    प्रत्यागम trim(ss.str());
}

स्थिर तार्किक isIdentChar(वर्ण c) {
    प्रत्यागम std::isalnum(static_cast<unsigned वर्ण>(c)) || c == '_';
}

स्थिर तार्किक endsWithContinuation(ध्रुव std::string & पंक्ति) {
    size_t i = पंक्ति.size();
    यावत् (i > 0 && std::isspace((unsigned वर्ण) पंक्ति[i - 1])) {
        i--;
    }
    प्रत्यागम i > 0 && पंक्ति[i - 1] == '\\';
}

स्थिर शून्य stripContinuation(std::string & पंक्ति) {
    size_t i = पंक्ति.size();
    यावत् (i > 0 && std::isspace((unsigned वर्ण) पंक्ति[i - 1])) {
        i--;
    }
    यदि (i > 0 && पंक्ति[i - 1] == '\\') {
        पंक्ति.erase(i - 1);
    }
}

स्थिर std::string expandMacrosRecursiveInternal(ध्रुव std::string &                                  पंक्ति,
                                                 ध्रुव std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

स्थिर std::string expand_Macro_मूल्य(ध्रुव std::string &                                  name,
                                    ध्रुव std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    यदि (visiting.count(name)) {
        क्षिप std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    यदि (it == macros.end()) {
        visiting.erase(name);
        प्रत्यागम name;
    }

    ध्रुव std::string & value = it->second;
    यदि (value.empty()) {
        visiting.erase(name);
        प्रत्यागम "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    प्रत्यागम expanded;
}

स्थिर std::string expandMacrosRecursiveInternal(ध्रुव std::string &                                  पंक्ति,
                                                 ध्रुव std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(पंक्ति.size());

    size_t i = 0;
    यावत् (i < पंक्ति.size()) {
        यदि (isIdentChar(पंक्ति[i])) {
            size_t start = i;
            यावत् (i < पंक्ति.size() && isIdentChar(पंक्ति[i])) {
                i++;
            }
            std::string token = पंक्ति.substr(start, i - start);

            auto it = macros.find(token);
            यदि (it != macros.end()) {
                result += expand_Macro_मूल्य(token, macros, visiting);
            } अन्यथा {
                result += token;
            }
        } अन्यथा {
            result += पंक्ति[i];
            i++;
        }
    }

    प्रत्यागम result;
}

स्थिर std::string expandMacrosRecursive(ध्रुव std::string &                                  पंक्ति,
                                         ध्रुव std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    प्रत्यागम expandMacrosRecursiveInternal(पंक्ति, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
वर्ग व्यंजक_Lexer {
  सार्वजनिक:
    गणना Kind { END, IDENT, संख्या_2, OP, LPAREN, RPAREN };

    संरचना Tok {
        Kind        kind;
        std::string text;
    };

    explicit व्यंजक_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        यदि (pos >= src.size()) {
            प्रत्यागम { END, "" };
        }

        वर्ण c = src[pos];

        //संख्या
        यदि (std::isdigit((unsigned वर्ण) c)) {
            size_t start = pos;
            यावत् (pos < src.size() && std::isdigit((unsigned वर्ण) src[pos])) {
                pos++;
            }
            प्रत्यागम { संख्या_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        यदि (std::isalpha((unsigned वर्ण) c) || c == '_') {
            size_t start = pos;
            यावत् (pos < src.size() && (std::isalnum((unsigned वर्ण) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            प्रत्यागम { IDENT, std::string(src.substr(start, pos - start)) };
        }

        यदि (c == '(') {
            pos++;
            प्रत्यागम { LPAREN, "(" };
        }
        यदि (c == ')') {
            pos++;
            प्रत्यागम { RPAREN, ")" };
        }

        // multi-char operators
        स्थिर ध्रुव वर्ण * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        प्रति (auto op : two_ops) {
            यदि (src.substr(pos, 2) == op) {
                pos += 2;
                प्रत्यागम { OP, std::string(op) };
            }
        }

        // single-char operators
        यदि (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            प्रत्यागम { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        प्रत्यागम { END, "" };
    }

  गुप्त:
    std::string_view src;
    size_t           pos;

    शून्य skipWS() {
        यावत् (pos < src.size() && std::isspace((unsigned वर्ण) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
वर्ग व्यंजक_Parser {
  सार्वजनिक:
    व्यंजक_Parser(std::string_view                                     व्यंजक,
               ध्रुव std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(व्यंजक),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    पूर्णाङ्क parse() { प्रत्यागम parseLogicalOr(); }

  गुप्त:
    व्यंजक_Lexer                                            lex;
    व्यंजक_Lexer::Tok                                       tok;
    ध्रुव std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    शून्य advance() { tok = lex.next(); }

    तार्किक acceptOp(ध्रुव std::string & s) {
        यदि (tok.kind == व्यंजक_Lexer::OP && tok.text == s) {
            advance();
            प्रत्यागम सत्य;
        }
        प्रत्यागम असत्य;
    }

    तार्किक acceptKind(व्यंजक_Lexer::Kind k) {
        यदि (tok.kind == k) {
            advance();
            प्रत्यागम सत्य;
        }
        प्रत्यागम असत्य;
    }

    पूर्णाङ्क parseLogicalOr() {
        पूर्णाङ्क v = parseLogicalAnd();
        यावत् (acceptOp("||")) {
            पूर्णाङ्क rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        प्रत्यागम v;
    }

    पूर्णाङ्क parseLogicalAnd() {
        पूर्णाङ्क v = parseEquality();
        यावत् (acceptOp("&&")) {
            पूर्णाङ्क rhs = parseEquality();
            v       = (v && rhs);
        }
        प्रत्यागम v;
    }

    पूर्णाङ्क parseEquality() {
        पूर्णाङ्क v = parseRelational();
        प्रति (;;) {
            यदि (acceptOp("==")) {
                पूर्णाङ्क rhs = parseRelational();
                v       = (v == rhs);
            } अन्यथा यदि (acceptOp("!=")) {
                पूर्णाङ्क rhs = parseRelational();
                v       = (v != rhs);
            } अन्यथा {
                भङ्ग;
            }
        }
        प्रत्यागम v;
    }

    पूर्णाङ्क parseRelational() {
        पूर्णाङ्क v = parseShift();
        प्रति (;;) {
            यदि (acceptOp("<")) {
                पूर्णाङ्क rhs = parseShift();
                v       = (v < rhs);
            } अन्यथा यदि (acceptOp(">")) {
                पूर्णाङ्क rhs = parseShift();
                v       = (v > rhs);
            } अन्यथा यदि (acceptOp("<=")) {
                पूर्णाङ्क rhs = parseShift();
                v       = (v <= rhs);
            } अन्यथा यदि (acceptOp(">=")) {
                पूर्णाङ्क rhs = parseShift();
                v       = (v >= rhs);
            } अन्यथा {
                भङ्ग;
            }
        }
        प्रत्यागम v;
    }

    पूर्णाङ्क parseShift() {
        पूर्णाङ्क v = parseAdd();
        प्रति (;;) {
            यदि (acceptOp("<<")) {
                पूर्णाङ्क rhs = parseAdd();
                v       = (v << rhs);
            } अन्यथा यदि (acceptOp(">>")) {
                पूर्णाङ्क rhs = parseAdd();
                v       = (v >> rhs);
            } अन्यथा {
                भङ्ग;
            }
        }
        प्रत्यागम v;
    }

    पूर्णाङ्क parseAdd() {
        पूर्णाङ्क v = parseMult();
        प्रति (;;) {
            यदि (acceptOp("+")) {
                पूर्णाङ्क rhs = parseMult();
                v       = (v + rhs);
            } अन्यथा यदि (acceptOp("-")) {
                पूर्णाङ्क rhs = parseMult();
                v       = (v - rhs);
            } अन्यथा {
                भङ्ग;
            }
        }
        प्रत्यागम v;
    }

    पूर्णाङ्क parseMult() {
        पूर्णाङ्क v = parseUnary();
        प्रति (;;) {
            यदि (acceptOp("*")) {
                पूर्णाङ्क rhs = parseUnary();
                v       = (v * rhs);
            } अन्यथा यदि (acceptOp("/")) {
                पूर्णाङ्क rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } अन्यथा यदि (acceptOp("%")) {
                पूर्णाङ्क rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } अन्यथा {
                भङ्ग;
            }
        }
        प्रत्यागम v;
    }

    पूर्णाङ्क parseUnary() {
        यदि (acceptOp("!")) {
            प्रत्यागम !parseUnary();
        }
        यदि (acceptOp("-")) {
            प्रत्यागम -parseUnary();
        }
        यदि (acceptOp("+")) {
            प्रत्यागम +parseUnary();
        }
        प्रत्यागम parsePrimary();
    }

    पूर्णाङ्क parsePrimary() {
        // '(' expr ')'
        यदि (acceptKind(व्यंजक_Lexer::LPAREN)) {
            पूर्णाङ्क v = parse();
            यदि (!acceptKind(व्यंजक_Lexer::RPAREN)) {
                क्षिप std::runtime_error("missing ')'");
            }
            प्रत्यागम v;
        }

        //संख्या
        यदि (tok.kind == व्यंजक_Lexer::संख्या_2) {
            पूर्णाङ्क v = std::stoi(tok.text);
            advance();
            प्रत्यागम v;
        }

        // defined(identifier)
        यदि (tok.kind == व्यंजक_Lexer::IDENT && tok.text == "defined") {
            advance();
            यदि (acceptKind(व्यंजक_Lexer::LPAREN)) {
                यदि (tok.kind != व्यंजक_Lexer::IDENT) {
                    क्षिप std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                यदि (!acceptKind(व्यंजक_Lexer::RPAREN)) {
                    क्षिप std::runtime_error("missing ) in defined()");
                }
                प्रत्यागम macros.count(name) ? 1 : 0;
            } अन्यथा {
                // defined NAME
                यदि (tok.kind != व्यंजक_Lexer::IDENT) {
                    क्षिप std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                प्रत्यागम macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        यदि (tok.kind == व्यंजक_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            यदि (it == macros.end()) {
                प्रत्यागम 0;
            }
            यदि (it->second.empty()) {
                प्रत्यागम 1;
            }
            प्रत्यागम evalMacroExpression(name, it->second);
        }

        // unexpected
        प्रत्यागम 0;
    }

    पूर्णाङ्क evalMacroExpression(ध्रुव std::string & name, ध्रुव std::string & value) {
        यदि (visiting.count(name)) {
            क्षिप std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        व्यंजक_Parser ep(value, macros, visiting);
        पूर्णाङ्क        v = ep.parse();
        visiting.erase(name);
        प्रत्यागम v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
वर्ग Preprocessor {
  सार्वजनिक:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        यदि (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ध्रुव std::string & filename, ध्रुव std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ढेर;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_ढेर, DirectiveMode::All);
        प्रत्यागम result;
    }

    std::string preprocess(ध्रुव std::string & contents, ध्रुव std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ढेर;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_ढेर, DirectiveMode::All);
        प्रत्यागम result;
    }

    std::string preprocess_includes_file(ध्रुव std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ढेर;
        std::string result = processFile(filename, macros, predefined, include_ढेर, DirectiveMode::IncludesOnly);
        प्रत्यागम result;
    }

    std::string preprocess_includes(ध्रुव std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ढेर;
        std::string result = processString(contents, macros, predefined, include_ढेर, DirectiveMode::IncludesOnly);
        प्रत्यागम result;
    }

  गुप्त:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    गणना वर्ग DirectiveMode { All, IncludesOnly };

    संरचना Cond {
        तार्किक parent_active;
        तार्किक active;
        तार्किक taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    शून्य parseMacroDefinitions(ध्रुव std::vector<std::string> & macro_defs) {
        प्रति (ध्रुव auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            यदि (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } अन्यथा {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    शून्य buildMacros(ध्रुव std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        प्रति (ध्रुव auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        प्रति (ध्रुव auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            यदि (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } अन्यथा {
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
    std::string loadFile(ध्रुव std::string & fname) {
        std::ifstream f(fname);
        यदि (!f.is_open()) {
            क्षिप std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        प्रत्यागम ss.str();
    }

    तार्किक condActive(ध्रुव std::vector<Cond> & cond) ध्रुव {
        यदि (cond.empty()) {
            प्रत्यागम सत्य;
        }
        प्रत्यागम cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ध्रुव std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ध्रुव std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_ढेर,
                            DirectiveMode                                  mode) {
        यदि (include_ढेर.count(name)) {
            क्षिप std::runtime_error("Recursive include: " + name);
        }

        include_ढेर.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_ढेर, mode);
        include_ढेर.erase(name);
        प्रत्यागम out;
    }

    std::string processIncludeFile(ध्रुव std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ध्रुव std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_ढेर,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        प्रत्यागम processFile(full_path, macros, predefined_macros, include_ढेर, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ध्रुव std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              ध्रुव std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_ढेर,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        पंक्ति;

        यावत् (std::getline(in, पंक्ति)) {
            std::string logical = पंक्ति;
            std::string t       = trim(logical);
            यदि (!t.empty() && t[0] == '#') {
                यावत् (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    यदि (!std::getline(in, पंक्ति)) {
                        भङ्ग;
                    }
                    logical += "\n";
                    logical += पंक्ति;
                }
                t = trim(logical);
            }

            यदि (!t.empty() && t[0] == '#') {
                तार्किक handled = handleDirective(t, out, macros, predefined_macros, cond, include_ढेर, mode);
                यदि (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } अन्यथा {
                यदि (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } अन्यथा यदि (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        यदि (mode == DirectiveMode::All && !cond.empty()) {
            क्षिप std::runtime_error("Unclosed #if directive");
        }

        प्रत्यागम out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    तार्किक handleDirective(ध्रुव std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         ध्रुव std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_ढेर,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        यदि (cmd == "include") {
            यदि (mode == DirectiveMode::All && !condActive(cond)) {
                प्रत्यागम सत्य;
            }
            std::string file;
            iss >> file;
            यदि (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_ढेर, mode);
            प्रत्यागम सत्य;
        }

        यदि (mode == DirectiveMode::IncludesOnly) {
            प्रत्यागम असत्य;
        }

        यदि (cmd == "define") {
            यदि (!condActive(cond)) {
                प्रत्यागम सत्य;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            यदि (predefined_macros.count(name)) {
                प्रत्यागम सत्य;
            }
            std::string value = trim_मूल्य(iss);
            macros[name]      = value;
            प्रत्यागम सत्य;
        }

        यदि (cmd == "undef") {
            यदि (!condActive(cond)) {
                प्रत्यागम सत्य;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            यदि (predefined_macros.count(name)) {
                प्रत्यागम सत्य;
            }
            macros.erase(name);
            प्रत्यागम सत्य;
        }

        यदि (cmd == "ifdef") {
            std::string name;
            iss >> name;
            तार्किक p = condActive(cond);
            तार्किक v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            प्रत्यागम सत्य;
        }

        यदि (cmd == "ifndef") {
            std::string name;
            iss >> name;
            तार्किक p = condActive(cond);
            तार्किक v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            प्रत्यागम सत्य;
        }

        यदि (cmd == "if") {
            std::string व्यंजक = trim_मूल्य(iss);
            तार्किक        p    = condActive(cond);
            तार्किक        v    = असत्य;
            यदि (p) {
                std::unordered_set<std::string> visiting;
                व्यंजक_Parser                      ep(व्यंजक, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            प्रत्यागम सत्य;
        }

        यदि (cmd == "elif") {
            std::string व्यंजक = trim_मूल्य(iss);

            यदि (cond.empty()) {
                क्षिप std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            यदि (!c.parent_active) {
                c.active = असत्य;
                प्रत्यागम सत्य;
            }

            यदि (c.taken) {
                c.active = असत्य;
                प्रत्यागम सत्य;
            }

            std::unordered_set<std::string> visiting;
            व्यंजक_Parser                      ep(व्यंजक, macros, visiting);
            तार्किक                            v = ep.parse() != 0;
            c.active                          = v;
            यदि (v) {
                c.taken = सत्य;
            }
            प्रत्यागम सत्य;
        }

        यदि (cmd == "else") {
            यदि (cond.empty()) {
                क्षिप std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            यदि (!c.parent_active) {
                c.active = असत्य;
                प्रत्यागम सत्य;
            }
            यदि (c.taken) {
                c.active = असत्य;
            } अन्यथा {
                c.active = सत्य;
                c.taken  = सत्य;
            }
            प्रत्यागम सत्य;
        }

        यदि (cmd == "endif") {
            यदि (cond.empty()) {
                क्षिप std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            प्रत्यागम सत्य;
        }

        // Unknown directive
        क्षिप std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
