#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#അന്തര്ഭാവ <cctype>
#അന്തര്ഭാവ <fstream>
#അന്തര്ഭാവ <sstream>
#അന്തര്ഭാവ <stdexcept>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <string_view>
#അന്തര്ഭാവ <unordered_map>
#അന്തര്ഭാവ <unordered_set>
#അന്തര്ഭാവ <vector>

നാമാകാശ pre_wgsl {

//==============================================================
// Options
//==============================================================
ഘടന Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
സ്ഥിതം std::string trim(സ്ഥിരം std::string & s) {
    size_t a = 0;
    ആകുമ്പോൾ (a < s.size() && std::isspace((unsigned അക്ഷരം) s[a])) {
        a++;
    }
    size_t b = s.size();
    ആകുമ്പോൾ (b > a && std::isspace((unsigned അക്ഷരം) s[b - 1])) {
        b--;
    }
    തിരികെ s.substr(a, b - a);
}

സ്ഥിതം std::string trim_മൂല്യം(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    തിരികെ trim(ss.str());
}

സ്ഥിതം തർക്കം isIdentChar(അക്ഷരം c) {
    തിരികെ std::isalnum(static_cast<unsigned അക്ഷരം>(c)) || c == '_';
}

സ്ഥിതം തർക്കം endsWithContinuation(സ്ഥിരം std::string & വരി) {
    size_t i = വരി.size();
    ആകുമ്പോൾ (i > 0 && std::isspace((unsigned അക്ഷരം) വരി[i - 1])) {
        i--;
    }
    തിരികെ i > 0 && വരി[i - 1] == '\\';
}

സ്ഥിതം ശൂന്യം stripContinuation(std::string & വരി) {
    size_t i = വരി.size();
    ആകുമ്പോൾ (i > 0 && std::isspace((unsigned അക്ഷരം) വരി[i - 1])) {
        i--;
    }
    എങ്കിൽ (i > 0 && വരി[i - 1] == '\\') {
        വരി.erase(i - 1);
    }
}

സ്ഥിതം std::string expandMacrosRecursiveInternal(സ്ഥിരം std::string &                                  വരി,
                                                 സ്ഥിരം std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

സ്ഥിതം std::string expand_Macro_മൂല്യം(സ്ഥിരം std::string &                                  name,
                                    സ്ഥിരം std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    എങ്കിൽ (visiting.count(name)) {
        എറിയുക std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    എങ്കിൽ (it == macros.end()) {
        visiting.erase(name);
        തിരികെ name;
    }

    സ്ഥിരം std::string & value = it->second;
    എങ്കിൽ (value.empty()) {
        visiting.erase(name);
        തിരികെ "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    തിരികെ expanded;
}

സ്ഥിതം std::string expandMacrosRecursiveInternal(സ്ഥിരം std::string &                                  വരി,
                                                 സ്ഥിരം std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(വരി.size());

    size_t i = 0;
    ആകുമ്പോൾ (i < വരി.size()) {
        എങ്കിൽ (isIdentChar(വരി[i])) {
            size_t start = i;
            ആകുമ്പോൾ (i < വരി.size() && isIdentChar(വരി[i])) {
                i++;
            }
            std::string token = വരി.substr(start, i - start);

            auto it = macros.find(token);
            എങ്കിൽ (it != macros.end()) {
                result += expand_Macro_മൂല്യം(token, macros, visiting);
            } അല്ലെങ്കിൽ {
                result += token;
            }
        } അല്ലെങ്കിൽ {
            result += വരി[i];
            i++;
        }
    }

    തിരികെ result;
}

സ്ഥിതം std::string expandMacrosRecursive(സ്ഥിരം std::string &                                  വരി,
                                         സ്ഥിരം std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    തിരികെ expandMacrosRecursiveInternal(വരി, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
വർഗ്ഗം വ്യഞ്ജകം_Lexer {
  പൊതു:
    എണ്ണൽ Kind { END, IDENT, സംഖ്യ_2, OP, LPAREN, RPAREN };

    ഘടന Tok {
        Kind        kind;
        std::string text;
    };

    explicit വ്യഞ്ജകം_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        എങ്കിൽ (pos >= src.size()) {
            തിരികെ { END, "" };
        }

        അക്ഷരം c = src[pos];

        //സംഖ്യ
        എങ്കിൽ (std::isdigit((unsigned അക്ഷരം) c)) {
            size_t start = pos;
            ആകുമ്പോൾ (pos < src.size() && std::isdigit((unsigned അക്ഷരം) src[pos])) {
                pos++;
            }
            തിരികെ { സംഖ്യ_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        എങ്കിൽ (std::isalpha((unsigned അക്ഷരം) c) || c == '_') {
            size_t start = pos;
            ആകുമ്പോൾ (pos < src.size() && (std::isalnum((unsigned അക്ഷരം) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            തിരികെ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        എങ്കിൽ (c == '(') {
            pos++;
            തിരികെ { LPAREN, "(" };
        }
        എങ്കിൽ (c == ')') {
            pos++;
            തിരികെ { RPAREN, ")" };
        }

        // multi-char operators
        സ്ഥിതം സ്ഥിരം അക്ഷരം * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        വേണ്ടി (auto op : two_ops) {
            എങ്കിൽ (src.substr(pos, 2) == op) {
                pos += 2;
                തിരികെ { OP, std::string(op) };
            }
        }

        // single-char operators
        എങ്കിൽ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            തിരികെ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        തിരികെ { END, "" };
    }

  സ്വകാര്യ:
    std::string_view src;
    size_t           pos;

    ശൂന്യം skipWS() {
        ആകുമ്പോൾ (pos < src.size() && std::isspace((unsigned അക്ഷരം) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
വർഗ്ഗം വ്യഞ്ജകം_Parser {
  പൊതു:
    വ്യഞ്ജകം_Parser(std::string_view                                     വ്യഞ്ജകം,
               സ്ഥിരം std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(വ്യഞ്ജകം),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    പൂർണ്ണസംഖ്യ parse() { തിരികെ parseLogicalOr(); }

  സ്വകാര്യ:
    വ്യഞ്ജകം_Lexer                                            lex;
    വ്യഞ്ജകം_Lexer::Tok                                       tok;
    സ്ഥിരം std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    ശൂന്യം advance() { tok = lex.next(); }

    തർക്കം acceptOp(സ്ഥിരം std::string & s) {
        എങ്കിൽ (tok.kind == വ്യഞ്ജകം_Lexer::OP && tok.text == s) {
            advance();
            തിരികെ സത്യം;
        }
        തിരികെ അസത്യം;
    }

    തർക്കം acceptKind(വ്യഞ്ജകം_Lexer::Kind k) {
        എങ്കിൽ (tok.kind == k) {
            advance();
            തിരികെ സത്യം;
        }
        തിരികെ അസത്യം;
    }

    പൂർണ്ണസംഖ്യ parseLogicalOr() {
        പൂർണ്ണസംഖ്യ v = parseLogicalAnd();
        ആകുമ്പോൾ (acceptOp("||")) {
            പൂർണ്ണസംഖ്യ rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        തിരികെ v;
    }

    പൂർണ്ണസംഖ്യ parseLogicalAnd() {
        പൂർണ്ണസംഖ്യ v = parseEquality();
        ആകുമ്പോൾ (acceptOp("&&")) {
            പൂർണ്ണസംഖ്യ rhs = parseEquality();
            v       = (v && rhs);
        }
        തിരികെ v;
    }

    പൂർണ്ണസംഖ്യ parseEquality() {
        പൂർണ്ണസംഖ്യ v = parseRelational();
        വേണ്ടി (;;) {
            എങ്കിൽ (acceptOp("==")) {
                പൂർണ്ണസംഖ്യ rhs = parseRelational();
                v       = (v == rhs);
            } അല്ലെങ്കിൽ എങ്കിൽ (acceptOp("!=")) {
                പൂർണ്ണസംഖ്യ rhs = parseRelational();
                v       = (v != rhs);
            } അല്ലെങ്കിൽ {
                നിർത്തുക;
            }
        }
        തിരികെ v;
    }

    പൂർണ്ണസംഖ്യ parseRelational() {
        പൂർണ്ണസംഖ്യ v = parseShift();
        വേണ്ടി (;;) {
            എങ്കിൽ (acceptOp("<")) {
                പൂർണ്ണസംഖ്യ rhs = parseShift();
                v       = (v < rhs);
            } അല്ലെങ്കിൽ എങ്കിൽ (acceptOp(">")) {
                പൂർണ്ണസംഖ്യ rhs = parseShift();
                v       = (v > rhs);
            } അല്ലെങ്കിൽ എങ്കിൽ (acceptOp("<=")) {
                പൂർണ്ണസംഖ്യ rhs = parseShift();
                v       = (v <= rhs);
            } അല്ലെങ്കിൽ എങ്കിൽ (acceptOp(">=")) {
                പൂർണ്ണസംഖ്യ rhs = parseShift();
                v       = (v >= rhs);
            } അല്ലെങ്കിൽ {
                നിർത്തുക;
            }
        }
        തിരികെ v;
    }

    പൂർണ്ണസംഖ്യ parseShift() {
        പൂർണ്ണസംഖ്യ v = parseAdd();
        വേണ്ടി (;;) {
            എങ്കിൽ (acceptOp("<<")) {
                പൂർണ്ണസംഖ്യ rhs = parseAdd();
                v       = (v << rhs);
            } അല്ലെങ്കിൽ എങ്കിൽ (acceptOp(">>")) {
                പൂർണ്ണസംഖ്യ rhs = parseAdd();
                v       = (v >> rhs);
            } അല്ലെങ്കിൽ {
                നിർത്തുക;
            }
        }
        തിരികെ v;
    }

    പൂർണ്ണസംഖ്യ parseAdd() {
        പൂർണ്ണസംഖ്യ v = parseMult();
        വേണ്ടി (;;) {
            എങ്കിൽ (acceptOp("+")) {
                പൂർണ്ണസംഖ്യ rhs = parseMult();
                v       = (v + rhs);
            } അല്ലെങ്കിൽ എങ്കിൽ (acceptOp("-")) {
                പൂർണ്ണസംഖ്യ rhs = parseMult();
                v       = (v - rhs);
            } അല്ലെങ്കിൽ {
                നിർത്തുക;
            }
        }
        തിരികെ v;
    }

    പൂർണ്ണസംഖ്യ parseMult() {
        പൂർണ്ണസംഖ്യ v = parseUnary();
        വേണ്ടി (;;) {
            എങ്കിൽ (acceptOp("*")) {
                പൂർണ്ണസംഖ്യ rhs = parseUnary();
                v       = (v * rhs);
            } അല്ലെങ്കിൽ എങ്കിൽ (acceptOp("/")) {
                പൂർണ്ണസംഖ്യ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } അല്ലെങ്കിൽ എങ്കിൽ (acceptOp("%")) {
                പൂർണ്ണസംഖ്യ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } അല്ലെങ്കിൽ {
                നിർത്തുക;
            }
        }
        തിരികെ v;
    }

    പൂർണ്ണസംഖ്യ parseUnary() {
        എങ്കിൽ (acceptOp("!")) {
            തിരികെ !parseUnary();
        }
        എങ്കിൽ (acceptOp("-")) {
            തിരികെ -parseUnary();
        }
        എങ്കിൽ (acceptOp("+")) {
            തിരികെ +parseUnary();
        }
        തിരികെ parsePrimary();
    }

    പൂർണ്ണസംഖ്യ parsePrimary() {
        // '(' expr ')'
        എങ്കിൽ (acceptKind(വ്യഞ്ജകം_Lexer::LPAREN)) {
            പൂർണ്ണസംഖ്യ v = parse();
            എങ്കിൽ (!acceptKind(വ്യഞ്ജകം_Lexer::RPAREN)) {
                എറിയുക std::runtime_error("missing ')'");
            }
            തിരികെ v;
        }

        //സംഖ്യ
        എങ്കിൽ (tok.kind == വ്യഞ്ജകം_Lexer::സംഖ്യ_2) {
            പൂർണ്ണസംഖ്യ v = std::stoi(tok.text);
            advance();
            തിരികെ v;
        }

        // defined(identifier)
        എങ്കിൽ (tok.kind == വ്യഞ്ജകം_Lexer::IDENT && tok.text == "defined") {
            advance();
            എങ്കിൽ (acceptKind(വ്യഞ്ജകം_Lexer::LPAREN)) {
                എങ്കിൽ (tok.kind != വ്യഞ്ജകം_Lexer::IDENT) {
                    എറിയുക std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                എങ്കിൽ (!acceptKind(വ്യഞ്ജകം_Lexer::RPAREN)) {
                    എറിയുക std::runtime_error("missing ) in defined()");
                }
                തിരികെ macros.count(name) ? 1 : 0;
            } അല്ലെങ്കിൽ {
                // defined NAME
                എങ്കിൽ (tok.kind != വ്യഞ്ജകം_Lexer::IDENT) {
                    എറിയുക std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                തിരികെ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        എങ്കിൽ (tok.kind == വ്യഞ്ജകം_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            എങ്കിൽ (it == macros.end()) {
                തിരികെ 0;
            }
            എങ്കിൽ (it->second.empty()) {
                തിരികെ 1;
            }
            തിരികെ evalMacroExpression(name, it->second);
        }

        // unexpected
        തിരികെ 0;
    }

    പൂർണ്ണസംഖ്യ evalMacroExpression(സ്ഥിരം std::string & name, സ്ഥിരം std::string & value) {
        എങ്കിൽ (visiting.count(name)) {
            എറിയുക std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        വ്യഞ്ജകം_Parser ep(value, macros, visiting);
        പൂർണ്ണസംഖ്യ        v = ep.parse();
        visiting.erase(name);
        തിരികെ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
വർഗ്ഗം Preprocessor {
  പൊതു:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        എങ്കിൽ (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(സ്ഥിരം std::string & filename, സ്ഥിരം std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_അട്ടി;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_അട്ടി, DirectiveMode::All);
        തിരികെ result;
    }

    std::string preprocess(സ്ഥിരം std::string & contents, സ്ഥിരം std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_അട്ടി;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_അട്ടി, DirectiveMode::All);
        തിരികെ result;
    }

    std::string preprocess_includes_file(സ്ഥിരം std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_അട്ടി;
        std::string result = processFile(filename, macros, predefined, include_അട്ടി, DirectiveMode::IncludesOnly);
        തിരികെ result;
    }

    std::string preprocess_includes(സ്ഥിരം std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_അട്ടി;
        std::string result = processString(contents, macros, predefined, include_അട്ടി, DirectiveMode::IncludesOnly);
        തിരികെ result;
    }

  സ്വകാര്യ:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    എണ്ണൽ വർഗ്ഗം DirectiveMode { All, IncludesOnly };

    ഘടന Cond {
        തർക്കം parent_active;
        തർക്കം active;
        തർക്കം taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    ശൂന്യം parseMacroDefinitions(സ്ഥിരം std::vector<std::string> & macro_defs) {
        വേണ്ടി (സ്ഥിരം auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            എങ്കിൽ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } അല്ലെങ്കിൽ {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    ശൂന്യം buildMacros(സ്ഥിരം std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        വേണ്ടി (സ്ഥിരം auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        വേണ്ടി (സ്ഥിരം auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            എങ്കിൽ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } അല്ലെങ്കിൽ {
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
    std::string loadFile(സ്ഥിരം std::string & fname) {
        std::ifstream f(fname);
        എങ്കിൽ (!f.is_open()) {
            എറിയുക std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        തിരികെ ss.str();
    }

    തർക്കം condActive(സ്ഥിരം std::vector<Cond> & cond) സ്ഥിരം {
        എങ്കിൽ (cond.empty()) {
            തിരികെ സത്യം;
        }
        തിരികെ cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(സ്ഥിരം std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            സ്ഥിരം std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_അട്ടി,
                            DirectiveMode                                  mode) {
        എങ്കിൽ (include_അട്ടി.count(name)) {
            എറിയുക std::runtime_error("Recursive include: " + name);
        }

        include_അട്ടി.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_അട്ടി, mode);
        include_അട്ടി.erase(name);
        തിരികെ out;
    }

    std::string processIncludeFile(സ്ഥിരം std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   സ്ഥിരം std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_അട്ടി,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        തിരികെ processFile(full_path, macros, predefined_macros, include_അട്ടി, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(സ്ഥിരം std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              സ്ഥിരം std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_അട്ടി,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        വരി;

        ആകുമ്പോൾ (std::getline(in, വരി)) {
            std::string logical = വരി;
            std::string t       = trim(logical);
            എങ്കിൽ (!t.empty() && t[0] == '#') {
                ആകുമ്പോൾ (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    എങ്കിൽ (!std::getline(in, വരി)) {
                        നിർത്തുക;
                    }
                    logical += "\n";
                    logical += വരി;
                }
                t = trim(logical);
            }

            എങ്കിൽ (!t.empty() && t[0] == '#') {
                തർക്കം handled = handleDirective(t, out, macros, predefined_macros, cond, include_അട്ടി, mode);
                എങ്കിൽ (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } അല്ലെങ്കിൽ {
                എങ്കിൽ (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } അല്ലെങ്കിൽ എങ്കിൽ (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        എങ്കിൽ (mode == DirectiveMode::All && !cond.empty()) {
            എറിയുക std::runtime_error("Unclosed #if directive");
        }

        തിരികെ out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    തർക്കം handleDirective(സ്ഥിരം std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         സ്ഥിരം std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_അട്ടി,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        എങ്കിൽ (cmd == "include") {
            എങ്കിൽ (mode == DirectiveMode::All && !condActive(cond)) {
                തിരികെ സത്യം;
            }
            std::string file;
            iss >> file;
            എങ്കിൽ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_അട്ടി, mode);
            തിരികെ സത്യം;
        }

        എങ്കിൽ (mode == DirectiveMode::IncludesOnly) {
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (cmd == "define") {
            എങ്കിൽ (!condActive(cond)) {
                തിരികെ സത്യം;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            എങ്കിൽ (predefined_macros.count(name)) {
                തിരികെ സത്യം;
            }
            std::string value = trim_മൂല്യം(iss);
            macros[name]      = value;
            തിരികെ സത്യം;
        }

        എങ്കിൽ (cmd == "undef") {
            എങ്കിൽ (!condActive(cond)) {
                തിരികെ സത്യം;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            എങ്കിൽ (predefined_macros.count(name)) {
                തിരികെ സത്യം;
            }
            macros.erase(name);
            തിരികെ സത്യം;
        }

        എങ്കിൽ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            തർക്കം p = condActive(cond);
            തർക്കം v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            തിരികെ സത്യം;
        }

        എങ്കിൽ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            തർക്കം p = condActive(cond);
            തർക്കം v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            തിരികെ സത്യം;
        }

        എങ്കിൽ (cmd == "if") {
            std::string വ്യഞ്ജകം = trim_മൂല്യം(iss);
            തർക്കം        p    = condActive(cond);
            തർക്കം        v    = അസത്യം;
            എങ്കിൽ (p) {
                std::unordered_set<std::string> visiting;
                വ്യഞ്ജകം_Parser                      ep(വ്യഞ്ജകം, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            തിരികെ സത്യം;
        }

        എങ്കിൽ (cmd == "elif") {
            std::string വ്യഞ്ജകം = trim_മൂല്യം(iss);

            എങ്കിൽ (cond.empty()) {
                എറിയുക std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            എങ്കിൽ (!c.parent_active) {
                c.active = അസത്യം;
                തിരികെ സത്യം;
            }

            എങ്കിൽ (c.taken) {
                c.active = അസത്യം;
                തിരികെ സത്യം;
            }

            std::unordered_set<std::string> visiting;
            വ്യഞ്ജകം_Parser                      ep(വ്യഞ്ജകം, macros, visiting);
            തർക്കം                            v = ep.parse() != 0;
            c.active                          = v;
            എങ്കിൽ (v) {
                c.taken = സത്യം;
            }
            തിരികെ സത്യം;
        }

        എങ്കിൽ (cmd == "else") {
            എങ്കിൽ (cond.empty()) {
                എറിയുക std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            എങ്കിൽ (!c.parent_active) {
                c.active = അസത്യം;
                തിരികെ സത്യം;
            }
            എങ്കിൽ (c.taken) {
                c.active = അസത്യം;
            } അല്ലെങ്കിൽ {
                c.active = സത്യം;
                c.taken  = സത്യം;
            }
            തിരികെ സത്യം;
        }

        എങ്കിൽ (cmd == "endif") {
            എങ്കിൽ (cond.empty()) {
                എറിയുക std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            തിരികെ സത്യം;
        }

        // Unknown directive
        എറിയുക std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
