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
स्थिर std::string trim(स्थिरांक std::string & s) {
    माप_प्रकार a = 0;
    पर्यंत (a < s.size() && std::है_खाली((अचिन्हित अक्षर) s[a])) {
        a++;
    }
    माप_प्रकार b = s.size();
    पर्यंत (b > a && std::है_खाली((अचिन्हित अक्षर) s[b - 1])) {
        b--;
    }
    परत s.substr(a, b - a);
}

स्थिर std::string trim_मूल्य(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    परत trim(ss.str());
}

स्थिर तार्किक isIdentChar(अक्षर c) {
    परत std::है_अक्षर_अंक(static_cast<अचिन्हित अक्षर>(c)) || c == '_';
}

स्थिर तार्किक endsWithContinuation(स्थिरांक std::string & ओळ) {
    माप_प्रकार i = ओळ.size();
    पर्यंत (i > 0 && std::है_खाली((अचिन्हित अक्षर) ओळ[i - 1])) {
        i--;
    }
    परत i > 0 && ओळ[i - 1] == '\\';
}

स्थिर शून्य stripContinuation(std::string & ओळ) {
    माप_प्रकार i = ओळ.size();
    पर्यंत (i > 0 && std::है_खाली((अचिन्हित अक्षर) ओळ[i - 1])) {
        i--;
    }
    जर (i > 0 && ओळ[i - 1] == '\\') {
        ओळ.erase(i - 1);
    }
}

स्थिर std::string expandMacrosRecursiveInternal(स्थिरांक std::string &                                  ओळ,
                                                 स्थिरांक std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

स्थिर std::string expand_Macro_मूल्य(स्थिरांक std::string &                                  name,
                                    स्थिरांक std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    जर (visiting.count(name)) {
        फेका std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    स्वतः it = macros.find(name);
    जर (it == macros.end()) {
        visiting.erase(name);
        परत name;
    }

    स्थिरांक std::string & value = it->second;
    जर (value.empty()) {
        visiting.erase(name);
        परत "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    परत expanded;
}

स्थिर std::string expandMacrosRecursiveInternal(स्थिरांक std::string &                                  ओळ,
                                                 स्थिरांक std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(ओळ.size());

    माप_प्रकार i = 0;
    पर्यंत (i < ओळ.size()) {
        जर (isIdentChar(ओळ[i])) {
            माप_प्रकार start = i;
            पर्यंत (i < ओळ.size() && isIdentChar(ओळ[i])) {
                i++;
            }
            std::string token = ओळ.substr(start, i - start);

            स्वतः it = macros.find(token);
            जर (it != macros.end()) {
                result += expand_Macro_मूल्य(token, macros, visiting);
            } अन्यथा {
                result += token;
            }
        } अन्यथा {
            result += ओळ[i];
            i++;
        }
    }

    परत result;
}

स्थिर std::string expandMacrosRecursive(स्थिरांक std::string &                                  ओळ,
                                         स्थिरांक std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    परत expandMacrosRecursiveInternal(ओळ, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
वर्ग अभिव्यक्ती_Lexer {
  सार्वजनिक:
    गणना Kind { END, IDENT, संख्या_2, OP, LPAREN, RPAREN };

    संरचना Tok {
        Kind        kind;
        std::string text;
    };

    explicit अभिव्यक्ती_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        जर (pos >= src.size()) {
            परत { END, "" };
        }

        अक्षर c = src[pos];

        //संख्या
        जर (std::है_अंक((अचिन्हित अक्षर) c)) {
            माप_प्रकार start = pos;
            पर्यंत (pos < src.size() && std::है_अंक((अचिन्हित अक्षर) src[pos])) {
                pos++;
            }
            परत { संख्या_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        जर (std::है_अक्षर((अचिन्हित अक्षर) c) || c == '_') {
            माप_प्रकार start = pos;
            पर्यंत (pos < src.size() && (std::है_अक्षर_अंक((अचिन्हित अक्षर) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            परत { IDENT, std::string(src.substr(start, pos - start)) };
        }

        जर (c == '(') {
            pos++;
            परत { LPAREN, "(" };
        }
        जर (c == ')') {
            pos++;
            परत { RPAREN, ")" };
        }

        // multi-char operators
        स्थिर स्थिरांक अक्षर * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        साठी (स्वतः op : two_ops) {
            जर (src.substr(pos, 2) == op) {
                pos += 2;
                परत { OP, std::string(op) };
            }
        }

        // single-char operators
        जर (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            परत { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        परत { END, "" };
    }

  खाजगी:
    std::string_view src;
    माप_प्रकार           pos;

    शून्य skipWS() {
        पर्यंत (pos < src.size() && std::है_खाली((अचिन्हित अक्षर) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
वर्ग अभिव्यक्ती_Parser {
  सार्वजनिक:
    अभिव्यक्ती_Parser(std::string_view                                     अभिव्यक्ती,
               स्थिरांक std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(अभिव्यक्ती),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    पूर्णांक parse() { परत parseLogicalOr(); }

  खाजगी:
    अभिव्यक्ती_Lexer                                            lex;
    अभिव्यक्ती_Lexer::Tok                                       tok;
    स्थिरांक std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    शून्य advance() { tok = lex.next(); }

    तार्किक acceptOp(स्थिरांक std::string & s) {
        जर (tok.kind == अभिव्यक्ती_Lexer::OP && tok.text == s) {
            advance();
            परत सत्य;
        }
        परत असत्य;
    }

    तार्किक acceptKind(अभिव्यक्ती_Lexer::Kind k) {
        जर (tok.kind == k) {
            advance();
            परत सत्य;
        }
        परत असत्य;
    }

    पूर्णांक parseLogicalOr() {
        पूर्णांक v = parseLogicalAnd();
        पर्यंत (acceptOp("||")) {
            पूर्णांक rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        परत v;
    }

    पूर्णांक parseLogicalAnd() {
        पूर्णांक v = parseEquality();
        पर्यंत (acceptOp("&&")) {
            पूर्णांक rhs = parseEquality();
            v       = (v && rhs);
        }
        परत v;
    }

    पूर्णांक parseEquality() {
        पूर्णांक v = parseRelational();
        साठी (;;) {
            जर (acceptOp("==")) {
                पूर्णांक rhs = parseRelational();
                v       = (v == rhs);
            } अन्यथा जर (acceptOp("!=")) {
                पूर्णांक rhs = parseRelational();
                v       = (v != rhs);
            } अन्यथा {
                थांबा;
            }
        }
        परत v;
    }

    पूर्णांक parseRelational() {
        पूर्णांक v = parseShift();
        साठी (;;) {
            जर (acceptOp("<")) {
                पूर्णांक rhs = parseShift();
                v       = (v < rhs);
            } अन्यथा जर (acceptOp(">")) {
                पूर्णांक rhs = parseShift();
                v       = (v > rhs);
            } अन्यथा जर (acceptOp("<=")) {
                पूर्णांक rhs = parseShift();
                v       = (v <= rhs);
            } अन्यथा जर (acceptOp(">=")) {
                पूर्णांक rhs = parseShift();
                v       = (v >= rhs);
            } अन्यथा {
                थांबा;
            }
        }
        परत v;
    }

    पूर्णांक parseShift() {
        पूर्णांक v = parseAdd();
        साठी (;;) {
            जर (acceptOp("<<")) {
                पूर्णांक rhs = parseAdd();
                v       = (v << rhs);
            } अन्यथा जर (acceptOp(">>")) {
                पूर्णांक rhs = parseAdd();
                v       = (v >> rhs);
            } अन्यथा {
                थांबा;
            }
        }
        परत v;
    }

    पूर्णांक parseAdd() {
        पूर्णांक v = parseMult();
        साठी (;;) {
            जर (acceptOp("+")) {
                पूर्णांक rhs = parseMult();
                v       = (v + rhs);
            } अन्यथा जर (acceptOp("-")) {
                पूर्णांक rhs = parseMult();
                v       = (v - rhs);
            } अन्यथा {
                थांबा;
            }
        }
        परत v;
    }

    पूर्णांक parseMult() {
        पूर्णांक v = parseUnary();
        साठी (;;) {
            जर (acceptOp("*")) {
                पूर्णांक rhs = parseUnary();
                v       = (v * rhs);
            } अन्यथा जर (acceptOp("/")) {
                पूर्णांक rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } अन्यथा जर (acceptOp("%")) {
                पूर्णांक rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } अन्यथा {
                थांबा;
            }
        }
        परत v;
    }

    पूर्णांक parseUnary() {
        जर (acceptOp("!")) {
            परत !parseUnary();
        }
        जर (acceptOp("-")) {
            परत -parseUnary();
        }
        जर (acceptOp("+")) {
            परत +parseUnary();
        }
        परत parsePrimary();
    }

    पूर्णांक parsePrimary() {
        // '(' expr ')'
        जर (acceptKind(अभिव्यक्ती_Lexer::LPAREN)) {
            पूर्णांक v = parse();
            जर (!acceptKind(अभिव्यक्ती_Lexer::RPAREN)) {
                फेका std::runtime_error("missing ')'");
            }
            परत v;
        }

        //संख्या
        जर (tok.kind == अभिव्यक्ती_Lexer::संख्या_2) {
            पूर्णांक v = std::stoi(tok.text);
            advance();
            परत v;
        }

        // defined(identifier)
        जर (tok.kind == अभिव्यक्ती_Lexer::IDENT && tok.text == "defined") {
            advance();
            जर (acceptKind(अभिव्यक्ती_Lexer::LPAREN)) {
                जर (tok.kind != अभिव्यक्ती_Lexer::IDENT) {
                    फेका std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                जर (!acceptKind(अभिव्यक्ती_Lexer::RPAREN)) {
                    फेका std::runtime_error("missing ) in defined()");
                }
                परत macros.count(name) ? 1 : 0;
            } अन्यथा {
                // defined NAME
                जर (tok.kind != अभिव्यक्ती_Lexer::IDENT) {
                    फेका std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                परत macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        जर (tok.kind == अभिव्यक्ती_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            स्वतः it = macros.find(name);
            जर (it == macros.end()) {
                परत 0;
            }
            जर (it->second.empty()) {
                परत 1;
            }
            परत evalMacroExpression(name, it->second);
        }

        // unexpected
        परत 0;
    }

    पूर्णांक evalMacroExpression(स्थिरांक std::string & name, स्थिरांक std::string & value) {
        जर (visiting.count(name)) {
            फेका std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        अभिव्यक्ती_Parser ep(value, macros, visiting);
        पूर्णांक        v = ep.parse();
        visiting.erase(name);
        परत v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
वर्ग Preprocessor {
  सार्वजनिक:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        जर (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(स्थिरांक std::string & filename, स्थिरांक std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_चळत;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_चळत, DirectiveMode::All);
        परत result;
    }

    std::string preprocess(स्थिरांक std::string & contents, स्थिरांक std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_चळत;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_चळत, DirectiveMode::All);
        परत result;
    }

    std::string preprocess_includes_file(स्थिरांक std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_चळत;
        std::string result = processFile(filename, macros, predefined, include_चळत, DirectiveMode::IncludesOnly);
        परत result;
    }

    std::string preprocess_includes(स्थिरांक std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_चळत;
        std::string result = processString(contents, macros, predefined, include_चळत, DirectiveMode::IncludesOnly);
        परत result;
    }

  खाजगी:
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
    शून्य parseMacroDefinitions(स्थिरांक std::vector<std::string> & macro_defs) {
        साठी (स्थिरांक स्वतः & def : macro_defs) {
            माप_प्रकार eq_pos = def.find('=');
            जर (eq_pos != std::string::npos) {
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
    शून्य buildMacros(स्थिरांक std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        साठी (स्थिरांक स्वतः & [name, value] : global_macros) {
            predefined.insert(name);
        }

        साठी (स्थिरांक स्वतः & def : additional_macros) {
            माप_प्रकार      eq_pos = def.find('=');
            std::string name, value;
            जर (eq_pos != std::string::npos) {
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
    std::string loadFile(स्थिरांक std::string & fname) {
        std::ifstream f(fname);
        जर (!f.is_open()) {
            फेका std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        परत ss.str();
    }

    तार्किक condActive(स्थिरांक std::vector<Cond> & cond) स्थिरांक {
        जर (cond.empty()) {
            परत सत्य;
        }
        परत cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(स्थिरांक std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            स्थिरांक std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_चळत,
                            DirectiveMode                                  mode) {
        जर (include_चळत.count(name)) {
            फेका std::runtime_error("Recursive include: " + name);
        }

        include_चळत.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_चळत, mode);
        include_चळत.erase(name);
        परत out;
    }

    std::string processIncludeFile(स्थिरांक std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   स्थिरांक std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_चळत,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        परत processFile(full_path, macros, predefined_macros, include_चळत, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(स्थिरांक std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              स्थिरांक std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_चळत,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        ओळ;

        पर्यंत (std::getline(in, ओळ)) {
            std::string logical = ओळ;
            std::string t       = trim(logical);
            जर (!t.empty() && t[0] == '#') {
                पर्यंत (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    जर (!std::getline(in, ओळ)) {
                        थांबा;
                    }
                    logical += "\n";
                    logical += ओळ;
                }
                t = trim(logical);
            }

            जर (!t.empty() && t[0] == '#') {
                तार्किक handled = handleDirective(t, out, macros, predefined_macros, cond, include_चळत, mode);
                जर (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } अन्यथा {
                जर (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } अन्यथा जर (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        जर (mode == DirectiveMode::All && !cond.empty()) {
            फेका std::runtime_error("Unclosed #if directive");
        }

        परत out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    तार्किक handleDirective(स्थिरांक std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         स्थिरांक std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_चळत,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        जर (cmd == "include") {
            जर (mode == DirectiveMode::All && !condActive(cond)) {
                परत सत्य;
            }
            std::string file;
            iss >> file;
            जर (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_चळत, mode);
            परत सत्य;
        }

        जर (mode == DirectiveMode::IncludesOnly) {
            परत असत्य;
        }

        जर (cmd == "define") {
            जर (!condActive(cond)) {
                परत सत्य;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            जर (predefined_macros.count(name)) {
                परत सत्य;
            }
            std::string value = trim_मूल्य(iss);
            macros[name]      = value;
            परत सत्य;
        }

        जर (cmd == "undef") {
            जर (!condActive(cond)) {
                परत सत्य;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            जर (predefined_macros.count(name)) {
                परत सत्य;
            }
            macros.erase(name);
            परत सत्य;
        }

        जर (cmd == "ifdef") {
            std::string name;
            iss >> name;
            तार्किक p = condActive(cond);
            तार्किक v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            परत सत्य;
        }

        जर (cmd == "ifndef") {
            std::string name;
            iss >> name;
            तार्किक p = condActive(cond);
            तार्किक v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            परत सत्य;
        }

        जर (cmd == "if") {
            std::string अभिव्यक्ती = trim_मूल्य(iss);
            तार्किक        p    = condActive(cond);
            तार्किक        v    = असत्य;
            जर (p) {
                std::unordered_set<std::string> visiting;
                अभिव्यक्ती_Parser                      ep(अभिव्यक्ती, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            परत सत्य;
        }

        जर (cmd == "elif") {
            std::string अभिव्यक्ती = trim_मूल्य(iss);

            जर (cond.empty()) {
                फेका std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            जर (!c.parent_active) {
                c.active = असत्य;
                परत सत्य;
            }

            जर (c.taken) {
                c.active = असत्य;
                परत सत्य;
            }

            std::unordered_set<std::string> visiting;
            अभिव्यक्ती_Parser                      ep(अभिव्यक्ती, macros, visiting);
            तार्किक                            v = ep.parse() != 0;
            c.active                          = v;
            जर (v) {
                c.taken = सत्य;
            }
            परत सत्य;
        }

        जर (cmd == "else") {
            जर (cond.empty()) {
                फेका std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            जर (!c.parent_active) {
                c.active = असत्य;
                परत सत्य;
            }
            जर (c.taken) {
                c.active = असत्य;
            } अन्यथा {
                c.active = सत्य;
                c.taken  = सत्य;
            }
            परत सत्य;
        }

        जर (cmd == "endif") {
            जर (cond.empty()) {
                फेका std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            परत सत्य;
        }

        // Unknown directive
        फेका std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
