#यदि_अपरिभाषितम् PRE_WGSL_HPP
#परिभाषय PRE_WGSL_HPP

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
    std::string              include_मार्गः = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
स्थिर std::string trim(ध्रुव std::string & s) {
    size_t a = 0;
    यावत् (a < s.size() && std::रिक्तम्_वा((अचिह्नित वर्ण) s[a])) {
        a++;
    }
    size_t b = s.size();
    यावत् (b > a && std::रिक्तम्_वा((अचिह्नित वर्ण) s[b - 1])) {
        b--;
    }
    प्रत्यागम s.substr(a, b - a);
}

स्थिर std::string trim_मानम्(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    प्रत्यागम trim(ss.str());
}

स्थिर तार्किक isIdentChar(वर्ण c) {
    प्रत्यागम std::अक्षरम्_अङ्कः_वा(static_cast<अचिह्नित वर्ण>(c)) || c == '_';
}

स्थिर तार्किक अन्ताः_With_Continuation(ध्रुव std::string & पङ्क्तिः) {
    size_t i = पङ्क्तिः.size();
    यावत् (i > 0 && std::रिक्तम्_वा((अचिह्नित वर्ण) पङ्क्तिः[i - 1])) {
        i--;
    }
    प्रत्यागम i > 0 && पङ्क्तिः[i - 1] == '\\';
}

स्थिर शून्य stripContinuation(std::string & पङ्क्तिः) {
    size_t i = पङ्क्तिः.size();
    यावत् (i > 0 && std::रिक्तम्_वा((अचिह्नित वर्ण) पङ्क्तिः[i - 1])) {
        i--;
    }
    यदि (i > 0 && पङ्क्तिः[i - 1] == '\\') {
        पङ्क्तिः.erase(i - 1);
    }
}

स्थिर std::string expand_Macros_Recursive_आन्तरिकम्(ध्रुव std::string &                                  पङ्क्तिः,
                                                 ध्रुव std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

स्थिर std::string expand_Macro_मानम्(ध्रुव std::string &                                  name,
                                    ध्रुव std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    यदि (visiting.count(name)) {
        क्षिप std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    स्वचालित it = macros.find(name);
    यदि (it == macros.end()) {
        visiting.erase(name);
        प्रत्यागम name;
    }

    ध्रुव std::string & value = it->second;
    यदि (value.रिक्तानि()) {
        visiting.erase(name);
        प्रत्यागम "";
    }

    std::string expanded = expand_Macros_Recursive_आन्तरिकम्(value, macros, visiting);
    visiting.erase(name);
    प्रत्यागम expanded;
}

स्थिर std::string expand_Macros_Recursive_आन्तरिकम्(ध्रुव std::string &                                  पङ्क्तिः,
                                                 ध्रुव std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string परिणाम;
    परिणाम.reserve(पङ्क्तिः.size());

    size_t i = 0;
    यावत् (i < पङ्क्तिः.size()) {
        यदि (isIdentChar(पङ्क्तिः[i])) {
            size_t start = i;
            यावत् (i < पङ्क्तिः.size() && isIdentChar(पङ्क्तिः[i])) {
                i++;
            }
            std::string token = पङ्क्तिः.substr(start, i - start);

            स्वचालित it = macros.find(token);
            यदि (it != macros.end()) {
                परिणाम += expand_Macro_मानम्(token, macros, visiting);
            } अन्यथा {
                परिणाम += token;
            }
        } अन्यथा {
            परिणाम += पङ्क्तिः[i];
            i++;
        }
    }

    प्रत्यागम परिणाम;
}

स्थिर std::string expandMacrosRecursive(ध्रुव std::string &                                  पङ्क्तिः,
                                         ध्रुव std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    प्रत्यागम expand_Macros_Recursive_आन्तरिकम्(पङ्क्तिः, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
वर्ग व्यञ्जकम्_Lexer {
  सार्वजनिक:
    गणना विधा_2 { अन्तः_4, IDENT, सङ्ख्या_2, OP, LPAREN, RPAREN };

    संरचना Tok {
        विधा_2        विधा;
        std::string text;
    };

    स्पष्ट व्यञ्जकम्_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok अग्र() {
        skipWS();
        यदि (pos >= src.size()) {
            प्रत्यागम { अन्तः_4, "" };
        }

        वर्ण c = src[pos];

        //सङ्ख्या
        यदि (std::अङ्कः_वा((अचिह्नित वर्ण) c)) {
            size_t start = pos;
            यावत् (pos < src.size() && std::अङ्कः_वा((अचिह्नित वर्ण) src[pos])) {
                pos++;
            }
            प्रत्यागम { सङ्ख्या_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        यदि (std::अक्षरम्_वा((अचिह्नित वर्ण) c) || c == '_') {
            size_t start = pos;
            यावत् (pos < src.size() && (std::अक्षरम्_अङ्कः_वा((अचिह्नित वर्ण) src[pos]) || src[pos] == '_')) {
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
        प्रति (स्वचालित op : two_ops) {
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
        प्रत्यागम { अन्तः_4, "" };
    }

  गुप्त:
    std::string_view src;
    size_t           pos;

    शून्य skipWS() {
        यावत् (pos < src.size() && std::रिक्तम्_वा((अचिह्नित वर्ण) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
वर्ग व्यञ्जकम्_Parser {
  सार्वजनिक:
    व्यञ्जकम्_Parser(std::string_view                                     व्यञ्जकम्,
               ध्रुव std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(व्यञ्जकम्),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    पूर्णाङ्क parse() { प्रत्यागम parseLogicalOr(); }

  गुप्त:
    व्यञ्जकम्_Lexer                                            lex;
    व्यञ्जकम्_Lexer::Tok                                       tok;
    ध्रुव std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    शून्य advance() { tok = lex.अग्र(); }

    तार्किक acceptOp(ध्रुव std::string & s) {
        यदि (tok.विधा == व्यञ्जकम्_Lexer::OP && tok.text == s) {
            advance();
            प्रत्यागम सत्य;
        }
        प्रत्यागम असत्य;
    }

    तार्किक accept_विधा(व्यञ्जकम्_Lexer::विधा_2 k) {
        यदि (tok.विधा == k) {
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
        पूर्णाङ्क v = parse_सरणम्();
        प्रति (;;) {
            यदि (acceptOp("<")) {
                पूर्णाङ्क rhs = parse_सरणम्();
                v       = (v < rhs);
            } अन्यथा यदि (acceptOp(">")) {
                पूर्णाङ्क rhs = parse_सरणम्();
                v       = (v > rhs);
            } अन्यथा यदि (acceptOp("<=")) {
                पूर्णाङ्क rhs = parse_सरणम्();
                v       = (v <= rhs);
            } अन्यथा यदि (acceptOp(">=")) {
                पूर्णाङ्क rhs = parse_सरणम्();
                v       = (v >= rhs);
            } अन्यथा {
                भङ्ग;
            }
        }
        प्रत्यागम v;
    }

    पूर्णाङ्क parse_सरणम्() {
        पूर्णाङ्क v = parse_सङ्कलय();
        प्रति (;;) {
            यदि (acceptOp("<<")) {
                पूर्णाङ्क rhs = parse_सङ्कलय();
                v       = (v << rhs);
            } अन्यथा यदि (acceptOp(">>")) {
                पूर्णाङ्क rhs = parse_सङ्कलय();
                v       = (v >> rhs);
            } अन्यथा {
                भङ्ग;
            }
        }
        प्रत्यागम v;
    }

    पूर्णाङ्क parse_सङ्कलय() {
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
        यदि (accept_विधा(व्यञ्जकम्_Lexer::LPAREN)) {
            पूर्णाङ्क v = parse();
            यदि (!accept_विधा(व्यञ्जकम्_Lexer::RPAREN)) {
                क्षिप std::runtime_error("missing ')'");
            }
            प्रत्यागम v;
        }

        //सङ्ख्या
        यदि (tok.विधा == व्यञ्जकम्_Lexer::सङ्ख्या_2) {
            पूर्णाङ्क v = std::stoi(tok.text);
            advance();
            प्रत्यागम v;
        }

        // defined(identifier)
        यदि (tok.विधा == व्यञ्जकम्_Lexer::IDENT && tok.text == "defined") {
            advance();
            यदि (accept_विधा(व्यञ्जकम्_Lexer::LPAREN)) {
                यदि (tok.विधा != व्यञ्जकम्_Lexer::IDENT) {
                    क्षिप std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                यदि (!accept_विधा(व्यञ्जकम्_Lexer::RPAREN)) {
                    क्षिप std::runtime_error("missing ) in defined()");
                }
                प्रत्यागम macros.count(name) ? 1 : 0;
            } अन्यथा {
                // defined NAME
                यदि (tok.विधा != व्यञ्जकम्_Lexer::IDENT) {
                    क्षिप std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                प्रत्यागम macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        यदि (tok.विधा == व्यञ्जकम्_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            स्वचालित it = macros.find(name);
            यदि (it == macros.end()) {
                प्रत्यागम 0;
            }
            यदि (it->second.रिक्तानि()) {
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
        व्यञ्जकम्_Parser ep(value, macros, visiting);
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
    स्पष्ट Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        यदि (opts_.include_मार्गः.रिक्तानि()) {
            opts_.include_मार्गः = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ध्रुव std::string & filename, ध्रुव std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_राशिः;
        buildMacros(additional_macros, macros, predefined);

        std::string परिणाम = processFile(filename, macros, predefined, include_राशिः, Directive_बहुलकम्::All);
        प्रत्यागम परिणाम;
    }

    std::string preprocess(ध्रुव std::string & contents, ध्रुव std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_राशिः;
        buildMacros(additional_macros, macros, predefined);

        std::string परिणाम = processString(contents, macros, predefined, include_राशिः, Directive_बहुलकम्::All);
        प्रत्यागम परिणाम;
    }

    std::string preprocess_includes_file(ध्रुव std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_राशिः;
        std::string परिणाम = processFile(filename, macros, predefined, include_राशिः, Directive_बहुलकम्::IncludesOnly);
        प्रत्यागम परिणाम;
    }

    std::string preprocess_includes(ध्रुव std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_राशिः;
        std::string परिणाम = processString(contents, macros, predefined, include_राशिः, Directive_बहुलकम्::IncludesOnly);
        प्रत्यागम परिणाम;
    }

  गुप्त:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    गणना वर्ग Directive_बहुलकम् { All, IncludesOnly };

    संरचना Cond {
        तार्किक parent_सक्रिय;
        तार्किक सक्रिय;
        तार्किक गृहीतम्;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    शून्य parseMacroDefinitions(ध्रुव std::vector<std::string> & macro_defs) {
        प्रति (ध्रुव स्वचालित & def : macro_defs) {
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

        प्रति (ध्रुव स्वचालित & [name, value] : global_macros) {
            predefined.insert(name);
        }

        प्रति (ध्रुव स्वचालित & def : additional_macros) {
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
    std::string आरोपय_File(ध्रुव std::string & fname) {
        std::ifstream f(fname);
        यदि (!f.is_open()) {
            क्षिप std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        प्रत्यागम ss.str();
    }

    तार्किक cond_सक्रिय(ध्रुव std::vector<Cond> & cond) ध्रुव {
        यदि (cond.रिक्तानि()) {
            प्रत्यागम सत्य;
        }
        प्रत्यागम cond.back().सक्रिय;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ध्रुव std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ध्रुव std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_राशिः,
                            Directive_बहुलकम्                                  mode) {
        यदि (include_राशिः.count(name)) {
            क्षिप std::runtime_error("Recursive include: " + name);
        }

        include_राशिः.insert(name);
        std::string shader_संहिता = आरोपय_File(name);
        std::string बहिर्गमनम्         = processString(shader_संहिता, macros, predefined_macros, include_राशिः, mode);
        include_राशिः.erase(name);
        प्रत्यागम बहिर्गमनम्;
    }

    std::string processIncludeFile(ध्रुव std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ध्रुव std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_राशिः,
                                   Directive_बहुलकम्                                  mode) {
        std::string पूर्णम्_मार्गः = opts_.include_मार्गः + "/" + fname;
        प्रत्यागम processFile(पूर्णम्_मार्गः, macros, predefined_macros, include_राशिः, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ध्रुव std::string &                            shader_संहिता,
                              std::unordered_map<std::string, std::string> & macros,
                              ध्रुव std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_राशिः,
                              Directive_बहुलकम्                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  बहिर्गमनम्;
        std::istringstream अन्तर्(shader_संहिता);
        std::string        पङ्क्तिः;

        यावत् (std::getline(अन्तर्, पङ्क्तिः)) {
            std::string logical = पङ्क्तिः;
            std::string t       = trim(logical);
            यदि (!t.रिक्तानि() && t[0] == '#') {
                यावत् (अन्ताः_With_Continuation(logical)) {
                    stripContinuation(logical);
                    यदि (!std::getline(अन्तर्, पङ्क्तिः)) {
                        भङ्ग;
                    }
                    logical += "\n";
                    logical += पङ्क्तिः;
                }
                t = trim(logical);
            }

            यदि (!t.रिक्तानि() && t[0] == '#') {
                तार्किक handled = handleDirective(t, बहिर्गमनम्, macros, predefined_macros, cond, include_राशिः, mode);
                यदि (mode == Directive_बहुलकम्::IncludesOnly && !handled) {
                    बहिर्गमनम् << logical << "\n";
                }
            } अन्यथा {
                यदि (mode == Directive_बहुलकम्::IncludesOnly) {
                    बहिर्गमनम् << logical << "\n";
                } अन्यथा यदि (cond_सक्रिय(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    बहिर्गमनम् << expanded << "\n";
                }
            }
        }

        यदि (mode == Directive_बहुलकम्::All && !cond.रिक्तानि()) {
            क्षिप std::runtime_error("Unclosed #if directive");
        }

        प्रत्यागम बहिर्गमनम्.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    तार्किक handleDirective(ध्रुव std::string &                            t,
                         std::stringstream &                            बहिर्गमनम्,
                         std::unordered_map<std::string, std::string> & macros,
                         ध्रुव std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_राशिः,
                         Directive_बहुलकम्                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        यदि (cmd == "include") {
            यदि (mode == Directive_बहुलकम्::All && !cond_सक्रिय(cond)) {
                प्रत्यागम सत्य;
            }
            std::string file;
            iss >> file;
            यदि (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            बहिर्गमनम् << processIncludeFile(file, macros, predefined_macros, include_राशिः, mode);
            प्रत्यागम सत्य;
        }

        यदि (mode == Directive_बहुलकम्::IncludesOnly) {
            प्रत्यागम असत्य;
        }

        यदि (cmd == "define") {
            यदि (!cond_सक्रिय(cond)) {
                प्रत्यागम सत्य;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            यदि (predefined_macros.count(name)) {
                प्रत्यागम सत्य;
            }
            std::string value = trim_मानम्(iss);
            macros[name]      = value;
            प्रत्यागम सत्य;
        }

        यदि (cmd == "undef") {
            यदि (!cond_सक्रिय(cond)) {
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
            तार्किक p = cond_सक्रिय(cond);
            तार्किक v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            प्रत्यागम सत्य;
        }

        यदि (cmd == "ifndef") {
            std::string name;
            iss >> name;
            तार्किक p = cond_सक्रिय(cond);
            तार्किक v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            प्रत्यागम सत्य;
        }

        यदि (cmd == "if") {
            std::string व्यञ्जकम् = trim_मानम्(iss);
            तार्किक        p    = cond_सक्रिय(cond);
            तार्किक        v    = असत्य;
            यदि (p) {
                std::unordered_set<std::string> visiting;
                व्यञ्जकम्_Parser                      ep(व्यञ्जकम्, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            प्रत्यागम सत्य;
        }

        यदि (cmd == "elif") {
            std::string व्यञ्जकम् = trim_मानम्(iss);

            यदि (cond.रिक्तानि()) {
                क्षिप std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            यदि (!c.parent_सक्रिय) {
                c.सक्रिय = असत्य;
                प्रत्यागम सत्य;
            }

            यदि (c.गृहीतम्) {
                c.सक्रिय = असत्य;
                प्रत्यागम सत्य;
            }

            std::unordered_set<std::string> visiting;
            व्यञ्जकम्_Parser                      ep(व्यञ्जकम्, macros, visiting);
            तार्किक                            v = ep.parse() != 0;
            c.सक्रिय                          = v;
            यदि (v) {
                c.गृहीतम् = सत्य;
            }
            प्रत्यागम सत्य;
        }

        यदि (cmd == "else") {
            यदि (cond.रिक्तानि()) {
                क्षिप std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            यदि (!c.parent_सक्रिय) {
                c.सक्रिय = असत्य;
                प्रत्यागम सत्य;
            }
            यदि (c.गृहीतम्) {
                c.सक्रिय = असत्य;
            } अन्यथा {
                c.सक्रिय = सत्य;
                c.गृहीतम्  = सत्य;
            }
            प्रत्यागम सत्य;
        }

        यदि (cmd == "endif") {
            यदि (cond.रिक्तानि()) {
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

#यदि_अन्तः  // PRE_WGSL_HPP
