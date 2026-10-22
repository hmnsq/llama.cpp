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
    याव (a < s.size() && std::isspace((unsigned वर्ण) s[a])) {
        a++;
    }
    size_t b = s.size();
    याव (b > a && std::isspace((unsigned वर्ण) s[b - 1])) {
        b--;
    }
    पटिनिवत्त s.substr(a, b - a);
}

स्थिर std::string trim_मानम्(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    पटिनिवत्त trim(ss.str());
}

स्थिर तार्किक isIdentChar(वर्ण c) {
    पटिनिवत्त std::isalnum(static_cast<unsigned वर्ण>(c)) || c == '_';
}

स्थिर तार्किक endsWithContinuation(ध्रुव std::string & पङ्क्तिः) {
    size_t i = पङ्क्तिः.size();
    याव (i > 0 && std::isspace((unsigned वर्ण) पङ्क्तिः[i - 1])) {
        i--;
    }
    पटिनिवत्त i > 0 && पङ्क्तिः[i - 1] == '\\';
}

स्थिर शून्य stripContinuation(std::string & पङ्क्तिः) {
    size_t i = पङ्क्तिः.size();
    याव (i > 0 && std::isspace((unsigned वर्ण) पङ्क्तिः[i - 1])) {
        i--;
    }
    सचे (i > 0 && पङ्क्तिः[i - 1] == '\\') {
        पङ्क्तिः.erase(i - 1);
    }
}

स्थिर std::string expandMacrosRecursiveInternal(ध्रुव std::string &                                  पङ्क्तिः,
                                                 ध्रुव std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

स्थिर std::string expand_Macro_मानम्(ध्रुव std::string &                                  name,
                                    ध्रुव std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    सचे (visiting.count(name)) {
        क्षिप std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    सचे (it == macros.end()) {
        visiting.erase(name);
        पटिनिवत्त name;
    }

    ध्रुव std::string & value = it->second;
    सचे (value.empty()) {
        visiting.erase(name);
        पटिनिवत्त "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    पटिनिवत्त expanded;
}

स्थिर std::string expandMacrosRecursiveInternal(ध्रुव std::string &                                  पङ्क्तिः,
                                                 ध्रुव std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(पङ्क्तिः.size());

    size_t i = 0;
    याव (i < पङ्क्तिः.size()) {
        सचे (isIdentChar(पङ्क्तिः[i])) {
            size_t start = i;
            याव (i < पङ्क्तिः.size() && isIdentChar(पङ्क्तिः[i])) {
                i++;
            }
            std::string token = पङ्क्तिः.substr(start, i - start);

            auto it = macros.find(token);
            सचे (it != macros.end()) {
                result += expand_Macro_मानम्(token, macros, visiting);
            } अञ्ञथा {
                result += token;
            }
        } अञ्ञथा {
            result += पङ्क्तिः[i];
            i++;
        }
    }

    पटिनिवत्त result;
}

स्थिर std::string expandMacrosRecursive(ध्रुव std::string &                                  पङ्क्तिः,
                                         ध्रुव std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    पटिनिवत्त expandMacrosRecursiveInternal(पङ्क्तिः, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
वग्ग व्यञ्जकम्_Lexer {
  सार्वजनिक:
    गणना Kind { END, IDENT, सङ्ख्या_2, OP, LPAREN, RPAREN };

    संरचना Tok {
        Kind        kind;
        std::string text;
    };

    explicit व्यञ्जकम्_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        सचे (pos >= src.size()) {
            पटिनिवत्त { END, "" };
        }

        वर्ण c = src[pos];

        //सङ्ख्या
        सचे (std::isdigit((unsigned वर्ण) c)) {
            size_t start = pos;
            याव (pos < src.size() && std::isdigit((unsigned वर्ण) src[pos])) {
                pos++;
            }
            पटिनिवत्त { सङ्ख्या_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        सचे (std::isalpha((unsigned वर्ण) c) || c == '_') {
            size_t start = pos;
            याव (pos < src.size() && (std::isalnum((unsigned वर्ण) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            पटिनिवत्त { IDENT, std::string(src.substr(start, pos - start)) };
        }

        सचे (c == '(') {
            pos++;
            पटिनिवत्त { LPAREN, "(" };
        }
        सचे (c == ')') {
            pos++;
            पटिनिवत्त { RPAREN, ")" };
        }

        // multi-char operators
        स्थिर ध्रुव वर्ण * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        अत्थाय (auto op : two_ops) {
            सचे (src.substr(pos, 2) == op) {
                pos += 2;
                पटिनिवत्त { OP, std::string(op) };
            }
        }

        // single-char operators
        सचे (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            पटिनिवत्त { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        पटिनिवत्त { END, "" };
    }

  गुप्त:
    std::string_view src;
    size_t           pos;

    शून्य skipWS() {
        याव (pos < src.size() && std::isspace((unsigned वर्ण) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
वग्ग व्यञ्जकम्_Parser {
  सार्वजनिक:
    व्यञ्जकम्_Parser(std::string_view                                     व्यञ्जकम्,
               ध्रुव std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(व्यञ्जकम्),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    पूर्णाङ्क parse() { पटिनिवत्त parseLogicalOr(); }

  गुप्त:
    व्यञ्जकम्_Lexer                                            lex;
    व्यञ्जकम्_Lexer::Tok                                       tok;
    ध्रुव std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    शून्य advance() { tok = lex.next(); }

    तार्किक acceptOp(ध्रुव std::string & s) {
        सचे (tok.kind == व्यञ्जकम्_Lexer::OP && tok.text == s) {
            advance();
            पटिनिवत्त सच्च;
        }
        पटिनिवत्त मिच्छा;
    }

    तार्किक acceptKind(व्यञ्जकम्_Lexer::Kind k) {
        सचे (tok.kind == k) {
            advance();
            पटिनिवत्त सच्च;
        }
        पटिनिवत्त मिच्छा;
    }

    पूर्णाङ्क parseLogicalOr() {
        पूर्णाङ्क v = parseLogicalAnd();
        याव (acceptOp("||")) {
            पूर्णाङ्क rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        पटिनिवत्त v;
    }

    पूर्णाङ्क parseLogicalAnd() {
        पूर्णाङ्क v = parseEquality();
        याव (acceptOp("&&")) {
            पूर्णाङ्क rhs = parseEquality();
            v       = (v && rhs);
        }
        पटिनिवत्त v;
    }

    पूर्णाङ्क parseEquality() {
        पूर्णाङ्क v = parseRelational();
        अत्थाय (;;) {
            सचे (acceptOp("==")) {
                पूर्णाङ्क rhs = parseRelational();
                v       = (v == rhs);
            } अञ्ञथा सचे (acceptOp("!=")) {
                पूर्णाङ्क rhs = parseRelational();
                v       = (v != rhs);
            } अञ्ञथा {
                भङ्ग;
            }
        }
        पटिनिवत्त v;
    }

    पूर्णाङ्क parseRelational() {
        पूर्णाङ्क v = parseShift();
        अत्थाय (;;) {
            सचे (acceptOp("<")) {
                पूर्णाङ्क rhs = parseShift();
                v       = (v < rhs);
            } अञ्ञथा सचे (acceptOp(">")) {
                पूर्णाङ्क rhs = parseShift();
                v       = (v > rhs);
            } अञ्ञथा सचे (acceptOp("<=")) {
                पूर्णाङ्क rhs = parseShift();
                v       = (v <= rhs);
            } अञ्ञथा सचे (acceptOp(">=")) {
                पूर्णाङ्क rhs = parseShift();
                v       = (v >= rhs);
            } अञ्ञथा {
                भङ्ग;
            }
        }
        पटिनिवत्त v;
    }

    पूर्णाङ्क parseShift() {
        पूर्णाङ्क v = parseAdd();
        अत्थाय (;;) {
            सचे (acceptOp("<<")) {
                पूर्णाङ्क rhs = parseAdd();
                v       = (v << rhs);
            } अञ्ञथा सचे (acceptOp(">>")) {
                पूर्णाङ्क rhs = parseAdd();
                v       = (v >> rhs);
            } अञ्ञथा {
                भङ्ग;
            }
        }
        पटिनिवत्त v;
    }

    पूर्णाङ्क parseAdd() {
        पूर्णाङ्क v = parseMult();
        अत्थाय (;;) {
            सचे (acceptOp("+")) {
                पूर्णाङ्क rhs = parseMult();
                v       = (v + rhs);
            } अञ्ञथा सचे (acceptOp("-")) {
                पूर्णाङ्क rhs = parseMult();
                v       = (v - rhs);
            } अञ्ञथा {
                भङ्ग;
            }
        }
        पटिनिवत्त v;
    }

    पूर्णाङ्क parseMult() {
        पूर्णाङ्क v = parseUnary();
        अत्थाय (;;) {
            सचे (acceptOp("*")) {
                पूर्णाङ्क rhs = parseUnary();
                v       = (v * rhs);
            } अञ्ञथा सचे (acceptOp("/")) {
                पूर्णाङ्क rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } अञ्ञथा सचे (acceptOp("%")) {
                पूर्णाङ्क rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } अञ्ञथा {
                भङ्ग;
            }
        }
        पटिनिवत्त v;
    }

    पूर्णाङ्क parseUnary() {
        सचे (acceptOp("!")) {
            पटिनिवत्त !parseUnary();
        }
        सचे (acceptOp("-")) {
            पटिनिवत्त -parseUnary();
        }
        सचे (acceptOp("+")) {
            पटिनिवत्त +parseUnary();
        }
        पटिनिवत्त parsePrimary();
    }

    पूर्णाङ्क parsePrimary() {
        // '(' expr ')'
        सचे (acceptKind(व्यञ्जकम्_Lexer::LPAREN)) {
            पूर्णाङ्क v = parse();
            सचे (!acceptKind(व्यञ्जकम्_Lexer::RPAREN)) {
                क्षिप std::runtime_error("missing ')'");
            }
            पटिनिवत्त v;
        }

        //सङ्ख्या
        सचे (tok.kind == व्यञ्जकम्_Lexer::सङ्ख्या_2) {
            पूर्णाङ्क v = std::stoi(tok.text);
            advance();
            पटिनिवत्त v;
        }

        // defined(identifier)
        सचे (tok.kind == व्यञ्जकम्_Lexer::IDENT && tok.text == "defined") {
            advance();
            सचे (acceptKind(व्यञ्जकम्_Lexer::LPAREN)) {
                सचे (tok.kind != व्यञ्जकम्_Lexer::IDENT) {
                    क्षिप std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                सचे (!acceptKind(व्यञ्जकम्_Lexer::RPAREN)) {
                    क्षिप std::runtime_error("missing ) in defined()");
                }
                पटिनिवत्त macros.count(name) ? 1 : 0;
            } अञ्ञथा {
                // defined NAME
                सचे (tok.kind != व्यञ्जकम्_Lexer::IDENT) {
                    क्षिप std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                पटिनिवत्त macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        सचे (tok.kind == व्यञ्जकम्_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            सचे (it == macros.end()) {
                पटिनिवत्त 0;
            }
            सचे (it->second.empty()) {
                पटिनिवत्त 1;
            }
            पटिनिवत्त evalMacroExpression(name, it->second);
        }

        // unexpected
        पटिनिवत्त 0;
    }

    पूर्णाङ्क evalMacroExpression(ध्रुव std::string & name, ध्रुव std::string & value) {
        सचे (visiting.count(name)) {
            क्षिप std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        व्यञ्जकम्_Parser ep(value, macros, visiting);
        पूर्णाङ्क        v = ep.parse();
        visiting.erase(name);
        पटिनिवत्त v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
वग्ग Preprocessor {
  सार्वजनिक:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        सचे (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ध्रुव std::string & filename, ध्रुव std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_राशिः;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_राशिः, DirectiveMode::All);
        पटिनिवत्त result;
    }

    std::string preprocess(ध्रुव std::string & contents, ध्रुव std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_राशिः;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_राशिः, DirectiveMode::All);
        पटिनिवत्त result;
    }

    std::string preprocess_includes_file(ध्रुव std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_राशिः;
        std::string result = processFile(filename, macros, predefined, include_राशिः, DirectiveMode::IncludesOnly);
        पटिनिवत्त result;
    }

    std::string preprocess_includes(ध्रुव std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_राशिः;
        std::string result = processString(contents, macros, predefined, include_राशिः, DirectiveMode::IncludesOnly);
        पटिनिवत्त result;
    }

  गुप्त:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    गणना वग्ग DirectiveMode { All, IncludesOnly };

    संरचना Cond {
        तार्किक parent_active;
        तार्किक active;
        तार्किक taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    शून्य parseMacroDefinitions(ध्रुव std::vector<std::string> & macro_defs) {
        अत्थाय (ध्रुव auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            सचे (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } अञ्ञथा {
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

        अत्थाय (ध्रुव auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        अत्थाय (ध्रुव auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            सचे (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } अञ्ञथा {
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
        सचे (!f.is_open()) {
            क्षिप std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        पटिनिवत्त ss.str();
    }

    तार्किक condActive(ध्रुव std::vector<Cond> & cond) ध्रुव {
        सचे (cond.empty()) {
            पटिनिवत्त सच्च;
        }
        पटिनिवत्त cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ध्रुव std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ध्रुव std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_राशिः,
                            DirectiveMode                                  mode) {
        सचे (include_राशिः.count(name)) {
            क्षिप std::runtime_error("Recursive include: " + name);
        }

        include_राशिः.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_राशिः, mode);
        include_राशिः.erase(name);
        पटिनिवत्त out;
    }

    std::string processIncludeFile(ध्रुव std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ध्रुव std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_राशिः,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        पटिनिवत्त processFile(full_path, macros, predefined_macros, include_राशिः, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ध्रुव std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              ध्रुव std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_राशिः,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        पङ्क्तिः;

        याव (std::getline(in, पङ्क्तिः)) {
            std::string logical = पङ्क्तिः;
            std::string t       = trim(logical);
            सचे (!t.empty() && t[0] == '#') {
                याव (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    सचे (!std::getline(in, पङ्क्तिः)) {
                        भङ्ग;
                    }
                    logical += "\n";
                    logical += पङ्क्तिः;
                }
                t = trim(logical);
            }

            सचे (!t.empty() && t[0] == '#') {
                तार्किक handled = handleDirective(t, out, macros, predefined_macros, cond, include_राशिः, mode);
                सचे (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } अञ्ञथा {
                सचे (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } अञ्ञथा सचे (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        सचे (mode == DirectiveMode::All && !cond.empty()) {
            क्षिप std::runtime_error("Unclosed #if directive");
        }

        पटिनिवत्त out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    तार्किक handleDirective(ध्रुव std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         ध्रुव std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_राशिः,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        सचे (cmd == "include") {
            सचे (mode == DirectiveMode::All && !condActive(cond)) {
                पटिनिवत्त सच्च;
            }
            std::string file;
            iss >> file;
            सचे (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_राशिः, mode);
            पटिनिवत्त सच्च;
        }

        सचे (mode == DirectiveMode::IncludesOnly) {
            पटिनिवत्त मिच्छा;
        }

        सचे (cmd == "define") {
            सचे (!condActive(cond)) {
                पटिनिवत्त सच्च;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            सचे (predefined_macros.count(name)) {
                पटिनिवत्त सच्च;
            }
            std::string value = trim_मानम्(iss);
            macros[name]      = value;
            पटिनिवत्त सच्च;
        }

        सचे (cmd == "undef") {
            सचे (!condActive(cond)) {
                पटिनिवत्त सच्च;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            सचे (predefined_macros.count(name)) {
                पटिनिवत्त सच्च;
            }
            macros.erase(name);
            पटिनिवत्त सच्च;
        }

        सचे (cmd == "ifdef") {
            std::string name;
            iss >> name;
            तार्किक p = condActive(cond);
            तार्किक v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            पटिनिवत्त सच्च;
        }

        सचे (cmd == "ifndef") {
            std::string name;
            iss >> name;
            तार्किक p = condActive(cond);
            तार्किक v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            पटिनिवत्त सच्च;
        }

        सचे (cmd == "if") {
            std::string व्यञ्जकम् = trim_मानम्(iss);
            तार्किक        p    = condActive(cond);
            तार्किक        v    = मिच्छा;
            सचे (p) {
                std::unordered_set<std::string> visiting;
                व्यञ्जकम्_Parser                      ep(व्यञ्जकम्, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            पटिनिवत्त सच्च;
        }

        सचे (cmd == "elif") {
            std::string व्यञ्जकम् = trim_मानम्(iss);

            सचे (cond.empty()) {
                क्षिप std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            सचे (!c.parent_active) {
                c.active = मिच्छा;
                पटिनिवत्त सच्च;
            }

            सचे (c.taken) {
                c.active = मिच्छा;
                पटिनिवत्त सच्च;
            }

            std::unordered_set<std::string> visiting;
            व्यञ्जकम्_Parser                      ep(व्यञ्जकम्, macros, visiting);
            तार्किक                            v = ep.parse() != 0;
            c.active                          = v;
            सचे (v) {
                c.taken = सच्च;
            }
            पटिनिवत्त सच्च;
        }

        सचे (cmd == "else") {
            सचे (cond.empty()) {
                क्षिप std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            सचे (!c.parent_active) {
                c.active = मिच्छा;
                पटिनिवत्त सच्च;
            }
            सचे (c.taken) {
                c.active = मिच्छा;
            } अञ्ञथा {
                c.active = सच्च;
                c.taken  = सच्च;
            }
            पटिनिवत्त सच्च;
        }

        सचे (cmd == "endif") {
            सचे (cond.empty()) {
                क्षिप std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            पटिनिवत्त सच्च;
        }

        // Unknown directive
        क्षिप std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
