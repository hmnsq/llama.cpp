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
स्थिर std::string trim(अचर std::string & s) {
    माप_प्रकार a = 0;
    जबसम्म (a < s.size() && std::है_खाली((अचिन्हित वर्ण) s[a])) {
        a++;
    }
    माप_प्रकार b = s.size();
    जबसम्म (b > a && std::है_खाली((अचिन्हित वर्ण) s[b - 1])) {
        b--;
    }
    फर्काऊ s.substr(a, b - a);
}

स्थिर std::string trim_मान(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    फर्काऊ trim(ss.str());
}

स्थिर तार्किक isIdentChar(वर्ण c) {
    फर्काऊ std::है_अक्षर_अंक(static_cast<अचिन्हित वर्ण>(c)) || c == '_';
}

स्थिर तार्किक endsWithContinuation(अचर std::string & पङ्क्ति) {
    माप_प्रकार i = पङ्क्ति.size();
    जबसम्म (i > 0 && std::है_खाली((अचिन्हित वर्ण) पङ्क्ति[i - 1])) {
        i--;
    }
    फर्काऊ i > 0 && पङ्क्ति[i - 1] == '\\';
}

स्थिर शून्य stripContinuation(std::string & पङ्क्ति) {
    माप_प्रकार i = पङ्क्ति.size();
    जबसम्म (i > 0 && std::है_खाली((अचिन्हित वर्ण) पङ्क्ति[i - 1])) {
        i--;
    }
    यदि (i > 0 && पङ्क्ति[i - 1] == '\\') {
        पङ्क्ति.erase(i - 1);
    }
}

स्थिर std::string expandMacrosRecursiveInternal(अचर std::string &                                  पङ्क्ति,
                                                 अचर std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

स्थिर std::string expand_Macro_मान(अचर std::string &                                  name,
                                    अचर std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    यदि (visiting.count(name)) {
        क्षिप std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    स्वतः it = macros.find(name);
    यदि (it == macros.end()) {
        visiting.erase(name);
        फर्काऊ name;
    }

    अचर std::string & value = it->second;
    यदि (value.empty()) {
        visiting.erase(name);
        फर्काऊ "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    फर्काऊ expanded;
}

स्थिर std::string expandMacrosRecursiveInternal(अचर std::string &                                  पङ्क्ति,
                                                 अचर std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(पङ्क्ति.size());

    माप_प्रकार i = 0;
    जबसम्म (i < पङ्क्ति.size()) {
        यदि (isIdentChar(पङ्क्ति[i])) {
            माप_प्रकार start = i;
            जबसम्म (i < पङ्क्ति.size() && isIdentChar(पङ्क्ति[i])) {
                i++;
            }
            std::string token = पङ्क्ति.substr(start, i - start);

            स्वतः it = macros.find(token);
            यदि (it != macros.end()) {
                result += expand_Macro_मान(token, macros, visiting);
            } अन्यथा {
                result += token;
            }
        } अन्यथा {
            result += पङ्क्ति[i];
            i++;
        }
    }

    फर्काऊ result;
}

स्थिर std::string expandMacrosRecursive(अचर std::string &                                  पङ्क्ति,
                                         अचर std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    फर्काऊ expandMacrosRecursiveInternal(पङ्क्ति, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
वर्ग अभिव्यक्ति_Lexer {
  सार्वजनिक:
    गणना Kind { END, IDENT, सङ्ख्या_2, OP, LPAREN, RPAREN };

    संरचना Tok {
        Kind        kind;
        std::string text;
    };

    explicit अभिव्यक्ति_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        यदि (pos >= src.size()) {
            फर्काऊ { END, "" };
        }

        वर्ण c = src[pos];

        //सङ्ख्या
        यदि (std::है_अंक((अचिन्हित वर्ण) c)) {
            माप_प्रकार start = pos;
            जबसम्म (pos < src.size() && std::है_अंक((अचिन्हित वर्ण) src[pos])) {
                pos++;
            }
            फर्काऊ { सङ्ख्या_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        यदि (std::है_अक्षर((अचिन्हित वर्ण) c) || c == '_') {
            माप_प्रकार start = pos;
            जबसम्म (pos < src.size() && (std::है_अक्षर_अंक((अचिन्हित वर्ण) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            फर्काऊ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        यदि (c == '(') {
            pos++;
            फर्काऊ { LPAREN, "(" };
        }
        यदि (c == ')') {
            pos++;
            फर्काऊ { RPAREN, ")" };
        }

        // multi-char operators
        स्थिर अचर वर्ण * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        लागि (स्वतः op : two_ops) {
            यदि (src.substr(pos, 2) == op) {
                pos += 2;
                फर्काऊ { OP, std::string(op) };
            }
        }

        // single-char operators
        यदि (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            फर्काऊ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        फर्काऊ { END, "" };
    }

  गुप्त:
    std::string_view src;
    माप_प्रकार           pos;

    शून्य skipWS() {
        जबसम्म (pos < src.size() && std::है_खाली((अचिन्हित वर्ण) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
वर्ग अभिव्यक्ति_Parser {
  सार्वजनिक:
    अभिव्यक्ति_Parser(std::string_view                                     अभिव्यक्ति,
               अचर std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(अभिव्यक्ति),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    पूर्णाङ्क parse() { फर्काऊ parseLogicalOr(); }

  गुप्त:
    अभिव्यक्ति_Lexer                                            lex;
    अभिव्यक्ति_Lexer::Tok                                       tok;
    अचर std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    शून्य advance() { tok = lex.next(); }

    तार्किक acceptOp(अचर std::string & s) {
        यदि (tok.kind == अभिव्यक्ति_Lexer::OP && tok.text == s) {
            advance();
            फर्काऊ सत्य;
        }
        फर्काऊ असत्य;
    }

    तार्किक acceptKind(अभिव्यक्ति_Lexer::Kind k) {
        यदि (tok.kind == k) {
            advance();
            फर्काऊ सत्य;
        }
        फर्काऊ असत्य;
    }

    पूर्णाङ्क parseLogicalOr() {
        पूर्णाङ्क v = parseLogicalAnd();
        जबसम्म (acceptOp("||")) {
            पूर्णाङ्क rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        फर्काऊ v;
    }

    पूर्णाङ्क parseLogicalAnd() {
        पूर्णाङ्क v = parseEquality();
        जबसम्म (acceptOp("&&")) {
            पूर्णाङ्क rhs = parseEquality();
            v       = (v && rhs);
        }
        फर्काऊ v;
    }

    पूर्णाङ्क parseEquality() {
        पूर्णाङ्क v = parseRelational();
        लागि (;;) {
            यदि (acceptOp("==")) {
                पूर्णाङ्क rhs = parseRelational();
                v       = (v == rhs);
            } अन्यथा यदि (acceptOp("!=")) {
                पूर्णाङ्क rhs = parseRelational();
                v       = (v != rhs);
            } अन्यथा {
                रोक;
            }
        }
        फर्काऊ v;
    }

    पूर्णाङ्क parseRelational() {
        पूर्णाङ्क v = parseShift();
        लागि (;;) {
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
                रोक;
            }
        }
        फर्काऊ v;
    }

    पूर्णाङ्क parseShift() {
        पूर्णाङ्क v = parseAdd();
        लागि (;;) {
            यदि (acceptOp("<<")) {
                पूर्णाङ्क rhs = parseAdd();
                v       = (v << rhs);
            } अन्यथा यदि (acceptOp(">>")) {
                पूर्णाङ्क rhs = parseAdd();
                v       = (v >> rhs);
            } अन्यथा {
                रोक;
            }
        }
        फर्काऊ v;
    }

    पूर्णाङ्क parseAdd() {
        पूर्णाङ्क v = parseMult();
        लागि (;;) {
            यदि (acceptOp("+")) {
                पूर्णाङ्क rhs = parseMult();
                v       = (v + rhs);
            } अन्यथा यदि (acceptOp("-")) {
                पूर्णाङ्क rhs = parseMult();
                v       = (v - rhs);
            } अन्यथा {
                रोक;
            }
        }
        फर्काऊ v;
    }

    पूर्णाङ्क parseMult() {
        पूर्णाङ्क v = parseUnary();
        लागि (;;) {
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
                रोक;
            }
        }
        फर्काऊ v;
    }

    पूर्णाङ्क parseUnary() {
        यदि (acceptOp("!")) {
            फर्काऊ !parseUnary();
        }
        यदि (acceptOp("-")) {
            फर्काऊ -parseUnary();
        }
        यदि (acceptOp("+")) {
            फर्काऊ +parseUnary();
        }
        फर्काऊ parsePrimary();
    }

    पूर्णाङ्क parsePrimary() {
        // '(' expr ')'
        यदि (acceptKind(अभिव्यक्ति_Lexer::LPAREN)) {
            पूर्णाङ्क v = parse();
            यदि (!acceptKind(अभिव्यक्ति_Lexer::RPAREN)) {
                क्षिप std::runtime_error("missing ')'");
            }
            फर्काऊ v;
        }

        //सङ्ख्या
        यदि (tok.kind == अभिव्यक्ति_Lexer::सङ्ख्या_2) {
            पूर्णाङ्क v = std::stoi(tok.text);
            advance();
            फर्काऊ v;
        }

        // defined(identifier)
        यदि (tok.kind == अभिव्यक्ति_Lexer::IDENT && tok.text == "defined") {
            advance();
            यदि (acceptKind(अभिव्यक्ति_Lexer::LPAREN)) {
                यदि (tok.kind != अभिव्यक्ति_Lexer::IDENT) {
                    क्षिप std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                यदि (!acceptKind(अभिव्यक्ति_Lexer::RPAREN)) {
                    क्षिप std::runtime_error("missing ) in defined()");
                }
                फर्काऊ macros.count(name) ? 1 : 0;
            } अन्यथा {
                // defined NAME
                यदि (tok.kind != अभिव्यक्ति_Lexer::IDENT) {
                    क्षिप std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                फर्काऊ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        यदि (tok.kind == अभिव्यक्ति_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            स्वतः it = macros.find(name);
            यदि (it == macros.end()) {
                फर्काऊ 0;
            }
            यदि (it->second.empty()) {
                फर्काऊ 1;
            }
            फर्काऊ evalMacroExpression(name, it->second);
        }

        // unexpected
        फर्काऊ 0;
    }

    पूर्णाङ्क evalMacroExpression(अचर std::string & name, अचर std::string & value) {
        यदि (visiting.count(name)) {
            क्षिप std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        अभिव्यक्ति_Parser ep(value, macros, visiting);
        पूर्णाङ्क        v = ep.parse();
        visiting.erase(name);
        फर्काऊ v;
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

    std::string preprocess_file(अचर std::string & filename, अचर std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_थाक;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_थाक, DirectiveMode::All);
        फर्काऊ result;
    }

    std::string preprocess(अचर std::string & contents, अचर std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_थाक;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_थाक, DirectiveMode::All);
        फर्काऊ result;
    }

    std::string preprocess_includes_file(अचर std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_थाक;
        std::string result = processFile(filename, macros, predefined, include_थाक, DirectiveMode::IncludesOnly);
        फर्काऊ result;
    }

    std::string preprocess_includes(अचर std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_थाक;
        std::string result = processString(contents, macros, predefined, include_थाक, DirectiveMode::IncludesOnly);
        फर्काऊ result;
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
    शून्य parseMacroDefinitions(अचर std::vector<std::string> & macro_defs) {
        लागि (अचर स्वतः & def : macro_defs) {
            माप_प्रकार eq_pos = def.find('=');
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
    शून्य buildMacros(अचर std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        लागि (अचर स्वतः & [name, value] : global_macros) {
            predefined.insert(name);
        }

        लागि (अचर स्वतः & def : additional_macros) {
            माप_प्रकार      eq_pos = def.find('=');
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
    std::string loadFile(अचर std::string & fname) {
        std::ifstream f(fname);
        यदि (!f.is_open()) {
            क्षिप std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        फर्काऊ ss.str();
    }

    तार्किक condActive(अचर std::vector<Cond> & cond) अचर {
        यदि (cond.empty()) {
            फर्काऊ सत्य;
        }
        फर्काऊ cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(अचर std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            अचर std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_थाक,
                            DirectiveMode                                  mode) {
        यदि (include_थाक.count(name)) {
            क्षिप std::runtime_error("Recursive include: " + name);
        }

        include_थाक.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_थाक, mode);
        include_थाक.erase(name);
        फर्काऊ out;
    }

    std::string processIncludeFile(अचर std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   अचर std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_थाक,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        फर्काऊ processFile(full_path, macros, predefined_macros, include_थाक, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(अचर std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              अचर std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_थाक,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        पङ्क्ति;

        जबसम्म (std::getline(in, पङ्क्ति)) {
            std::string logical = पङ्क्ति;
            std::string t       = trim(logical);
            यदि (!t.empty() && t[0] == '#') {
                जबसम्म (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    यदि (!std::getline(in, पङ्क्ति)) {
                        रोक;
                    }
                    logical += "\n";
                    logical += पङ्क्ति;
                }
                t = trim(logical);
            }

            यदि (!t.empty() && t[0] == '#') {
                तार्किक handled = handleDirective(t, out, macros, predefined_macros, cond, include_थाक, mode);
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

        फर्काऊ out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    तार्किक handleDirective(अचर std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         अचर std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_थाक,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        यदि (cmd == "include") {
            यदि (mode == DirectiveMode::All && !condActive(cond)) {
                फर्काऊ सत्य;
            }
            std::string file;
            iss >> file;
            यदि (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_थाक, mode);
            फर्काऊ सत्य;
        }

        यदि (mode == DirectiveMode::IncludesOnly) {
            फर्काऊ असत्य;
        }

        यदि (cmd == "define") {
            यदि (!condActive(cond)) {
                फर्काऊ सत्य;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            यदि (predefined_macros.count(name)) {
                फर्काऊ सत्य;
            }
            std::string value = trim_मान(iss);
            macros[name]      = value;
            फर्काऊ सत्य;
        }

        यदि (cmd == "undef") {
            यदि (!condActive(cond)) {
                फर्काऊ सत्य;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            यदि (predefined_macros.count(name)) {
                फर्काऊ सत्य;
            }
            macros.erase(name);
            फर्काऊ सत्य;
        }

        यदि (cmd == "ifdef") {
            std::string name;
            iss >> name;
            तार्किक p = condActive(cond);
            तार्किक v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            फर्काऊ सत्य;
        }

        यदि (cmd == "ifndef") {
            std::string name;
            iss >> name;
            तार्किक p = condActive(cond);
            तार्किक v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            फर्काऊ सत्य;
        }

        यदि (cmd == "if") {
            std::string अभिव्यक्ति = trim_मान(iss);
            तार्किक        p    = condActive(cond);
            तार्किक        v    = असत्य;
            यदि (p) {
                std::unordered_set<std::string> visiting;
                अभिव्यक्ति_Parser                      ep(अभिव्यक्ति, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            फर्काऊ सत्य;
        }

        यदि (cmd == "elif") {
            std::string अभिव्यक्ति = trim_मान(iss);

            यदि (cond.empty()) {
                क्षिप std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            यदि (!c.parent_active) {
                c.active = असत्य;
                फर्काऊ सत्य;
            }

            यदि (c.taken) {
                c.active = असत्य;
                फर्काऊ सत्य;
            }

            std::unordered_set<std::string> visiting;
            अभिव्यक्ति_Parser                      ep(अभिव्यक्ति, macros, visiting);
            तार्किक                            v = ep.parse() != 0;
            c.active                          = v;
            यदि (v) {
                c.taken = सत्य;
            }
            फर्काऊ सत्य;
        }

        यदि (cmd == "else") {
            यदि (cond.empty()) {
                क्षिप std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            यदि (!c.parent_active) {
                c.active = असत्य;
                फर्काऊ सत्य;
            }
            यदि (c.taken) {
                c.active = असत्य;
            } अन्यथा {
                c.active = सत्य;
                c.taken  = सत्य;
            }
            फर्काऊ सत्य;
        }

        यदि (cmd == "endif") {
            यदि (cond.empty()) {
                क्षिप std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            फर्काऊ सत्य;
        }

        // Unknown directive
        क्षिप std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
