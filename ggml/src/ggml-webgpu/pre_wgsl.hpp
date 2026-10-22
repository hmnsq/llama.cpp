#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#அந்தர்பாவ <cctype>
#அந்தர்பாவ <fstream>
#அந்தர்பாவ <sstream>
#அந்தர்பாவ <stdexcept>
#அந்தர்பாவ <string>
#அந்தர்பாவ <string_view>
#அந்தர்பாவ <unordered_map>
#அந்தர்பாவ <unordered_set>
#அந்தர்பாவ <vector>

நாமாகாஶ pre_wgsl {

//==============================================================
// Options
//==============================================================
கட்டமைப்பு Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
நிலையான std::string trim(மாறிலி std::string & s) {
    size_t a = 0;
    போது (a < s.size() && std::isspace((unsigned எழுத்து) s[a])) {
        a++;
    }
    size_t b = s.size();
    போது (b > a && std::isspace((unsigned எழுத்து) s[b - 1])) {
        b--;
    }
    திரும்பு s.substr(a, b - a);
}

நிலையான std::string trim_மதிப்பு(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    திரும்பு trim(ss.str());
}

நிலையான தருக்க isIdentChar(எழுத்து c) {
    திரும்பு std::isalnum(static_cast<unsigned எழுத்து>(c)) || c == '_';
}

நிலையான தருக்க endsWithContinuation(மாறிலி std::string & வரி) {
    size_t i = வரி.size();
    போது (i > 0 && std::isspace((unsigned எழுத்து) வரி[i - 1])) {
        i--;
    }
    திரும்பு i > 0 && வரி[i - 1] == '\\';
}

நிலையான வெறுமை stripContinuation(std::string & வரி) {
    size_t i = வரி.size();
    போது (i > 0 && std::isspace((unsigned எழுத்து) வரி[i - 1])) {
        i--;
    }
    எனில் (i > 0 && வரி[i - 1] == '\\') {
        வரி.erase(i - 1);
    }
}

நிலையான std::string expandMacrosRecursiveInternal(மாறிலி std::string &                                  வரி,
                                                 மாறிலி std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

நிலையான std::string expand_Macro_மதிப்பு(மாறிலி std::string &                                  name,
                                    மாறிலி std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    எனில் (visiting.count(name)) {
        எறி std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    எனில் (it == macros.end()) {
        visiting.erase(name);
        திரும்பு name;
    }

    மாறிலி std::string & value = it->second;
    எனில் (value.empty()) {
        visiting.erase(name);
        திரும்பு "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    திரும்பு expanded;
}

நிலையான std::string expandMacrosRecursiveInternal(மாறிலி std::string &                                  வரி,
                                                 மாறிலி std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(வரி.size());

    size_t i = 0;
    போது (i < வரி.size()) {
        எனில் (isIdentChar(வரி[i])) {
            size_t start = i;
            போது (i < வரி.size() && isIdentChar(வரி[i])) {
                i++;
            }
            std::string token = வரி.substr(start, i - start);

            auto it = macros.find(token);
            எனில் (it != macros.end()) {
                result += expand_Macro_மதிப்பு(token, macros, visiting);
            } இல்லையெனில் {
                result += token;
            }
        } இல்லையெனில் {
            result += வரி[i];
            i++;
        }
    }

    திரும்பு result;
}

நிலையான std::string expandMacrosRecursive(மாறிலி std::string &                                  வரி,
                                         மாறிலி std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    திரும்பு expandMacrosRecursiveInternal(வரி, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
வகுப்பு கோவை_Lexer {
  பொது:
    எண்ணிக்கை Kind { END, IDENT, எண்_2, OP, LPAREN, RPAREN };

    கட்டமைப்பு Tok {
        Kind        kind;
        std::string text;
    };

    explicit கோவை_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        எனில் (pos >= src.size()) {
            திரும்பு { END, "" };
        }

        எழுத்து c = src[pos];

        //எண்
        எனில் (std::isdigit((unsigned எழுத்து) c)) {
            size_t start = pos;
            போது (pos < src.size() && std::isdigit((unsigned எழுத்து) src[pos])) {
                pos++;
            }
            திரும்பு { எண்_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        எனில் (std::isalpha((unsigned எழுத்து) c) || c == '_') {
            size_t start = pos;
            போது (pos < src.size() && (std::isalnum((unsigned எழுத்து) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            திரும்பு { IDENT, std::string(src.substr(start, pos - start)) };
        }

        எனில் (c == '(') {
            pos++;
            திரும்பு { LPAREN, "(" };
        }
        எனில் (c == ')') {
            pos++;
            திரும்பு { RPAREN, ")" };
        }

        // multi-char operators
        நிலையான மாறிலி எழுத்து * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        க்கு (auto op : two_ops) {
            எனில் (src.substr(pos, 2) == op) {
                pos += 2;
                திரும்பு { OP, std::string(op) };
            }
        }

        // single-char operators
        எனில் (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            திரும்பு { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        திரும்பு { END, "" };
    }

  தனிப்பட்ட:
    std::string_view src;
    size_t           pos;

    வெறுமை skipWS() {
        போது (pos < src.size() && std::isspace((unsigned எழுத்து) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
வகுப்பு கோவை_Parser {
  பொது:
    கோவை_Parser(std::string_view                                     கோவை,
               மாறிலி std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(கோவை),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    முழு parse() { திரும்பு parseLogicalOr(); }

  தனிப்பட்ட:
    கோவை_Lexer                                            lex;
    கோவை_Lexer::Tok                                       tok;
    மாறிலி std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    வெறுமை advance() { tok = lex.next(); }

    தருக்க acceptOp(மாறிலி std::string & s) {
        எனில் (tok.kind == கோவை_Lexer::OP && tok.text == s) {
            advance();
            திரும்பு உண்மை;
        }
        திரும்பு பொய்;
    }

    தருக்க acceptKind(கோவை_Lexer::Kind k) {
        எனில் (tok.kind == k) {
            advance();
            திரும்பு உண்மை;
        }
        திரும்பு பொய்;
    }

    முழு parseLogicalOr() {
        முழு v = parseLogicalAnd();
        போது (acceptOp("||")) {
            முழு rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        திரும்பு v;
    }

    முழு parseLogicalAnd() {
        முழு v = parseEquality();
        போது (acceptOp("&&")) {
            முழு rhs = parseEquality();
            v       = (v && rhs);
        }
        திரும்பு v;
    }

    முழு parseEquality() {
        முழு v = parseRelational();
        க்கு (;;) {
            எனில் (acceptOp("==")) {
                முழு rhs = parseRelational();
                v       = (v == rhs);
            } இல்லையெனில் எனில் (acceptOp("!=")) {
                முழு rhs = parseRelational();
                v       = (v != rhs);
            } இல்லையெனில் {
                நிறுத்து;
            }
        }
        திரும்பு v;
    }

    முழு parseRelational() {
        முழு v = parseShift();
        க்கு (;;) {
            எனில் (acceptOp("<")) {
                முழு rhs = parseShift();
                v       = (v < rhs);
            } இல்லையெனில் எனில் (acceptOp(">")) {
                முழு rhs = parseShift();
                v       = (v > rhs);
            } இல்லையெனில் எனில் (acceptOp("<=")) {
                முழு rhs = parseShift();
                v       = (v <= rhs);
            } இல்லையெனில் எனில் (acceptOp(">=")) {
                முழு rhs = parseShift();
                v       = (v >= rhs);
            } இல்லையெனில் {
                நிறுத்து;
            }
        }
        திரும்பு v;
    }

    முழு parseShift() {
        முழு v = parseAdd();
        க்கு (;;) {
            எனில் (acceptOp("<<")) {
                முழு rhs = parseAdd();
                v       = (v << rhs);
            } இல்லையெனில் எனில் (acceptOp(">>")) {
                முழு rhs = parseAdd();
                v       = (v >> rhs);
            } இல்லையெனில் {
                நிறுத்து;
            }
        }
        திரும்பு v;
    }

    முழு parseAdd() {
        முழு v = parseMult();
        க்கு (;;) {
            எனில் (acceptOp("+")) {
                முழு rhs = parseMult();
                v       = (v + rhs);
            } இல்லையெனில் எனில் (acceptOp("-")) {
                முழு rhs = parseMult();
                v       = (v - rhs);
            } இல்லையெனில் {
                நிறுத்து;
            }
        }
        திரும்பு v;
    }

    முழு parseMult() {
        முழு v = parseUnary();
        க்கு (;;) {
            எனில் (acceptOp("*")) {
                முழு rhs = parseUnary();
                v       = (v * rhs);
            } இல்லையெனில் எனில் (acceptOp("/")) {
                முழு rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } இல்லையெனில் எனில் (acceptOp("%")) {
                முழு rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } இல்லையெனில் {
                நிறுத்து;
            }
        }
        திரும்பு v;
    }

    முழு parseUnary() {
        எனில் (acceptOp("!")) {
            திரும்பு !parseUnary();
        }
        எனில் (acceptOp("-")) {
            திரும்பு -parseUnary();
        }
        எனில் (acceptOp("+")) {
            திரும்பு +parseUnary();
        }
        திரும்பு parsePrimary();
    }

    முழு parsePrimary() {
        // '(' expr ')'
        எனில் (acceptKind(கோவை_Lexer::LPAREN)) {
            முழு v = parse();
            எனில் (!acceptKind(கோவை_Lexer::RPAREN)) {
                எறி std::runtime_error("missing ')'");
            }
            திரும்பு v;
        }

        //எண்
        எனில் (tok.kind == கோவை_Lexer::எண்_2) {
            முழு v = std::stoi(tok.text);
            advance();
            திரும்பு v;
        }

        // defined(identifier)
        எனில் (tok.kind == கோவை_Lexer::IDENT && tok.text == "defined") {
            advance();
            எனில் (acceptKind(கோவை_Lexer::LPAREN)) {
                எனில் (tok.kind != கோவை_Lexer::IDENT) {
                    எறி std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                எனில் (!acceptKind(கோவை_Lexer::RPAREN)) {
                    எறி std::runtime_error("missing ) in defined()");
                }
                திரும்பு macros.count(name) ? 1 : 0;
            } இல்லையெனில் {
                // defined NAME
                எனில் (tok.kind != கோவை_Lexer::IDENT) {
                    எறி std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                திரும்பு macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        எனில் (tok.kind == கோவை_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            எனில் (it == macros.end()) {
                திரும்பு 0;
            }
            எனில் (it->second.empty()) {
                திரும்பு 1;
            }
            திரும்பு evalMacroExpression(name, it->second);
        }

        // unexpected
        திரும்பு 0;
    }

    முழு evalMacroExpression(மாறிலி std::string & name, மாறிலி std::string & value) {
        எனில் (visiting.count(name)) {
            எறி std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        கோவை_Parser ep(value, macros, visiting);
        முழு        v = ep.parse();
        visiting.erase(name);
        திரும்பு v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
வகுப்பு Preprocessor {
  பொது:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        எனில் (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(மாறிலி std::string & filename, மாறிலி std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_அடுக்கு;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_அடுக்கு, DirectiveMode::All);
        திரும்பு result;
    }

    std::string preprocess(மாறிலி std::string & contents, மாறிலி std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_அடுக்கு;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_அடுக்கு, DirectiveMode::All);
        திரும்பு result;
    }

    std::string preprocess_includes_file(மாறிலி std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_அடுக்கு;
        std::string result = processFile(filename, macros, predefined, include_அடுக்கு, DirectiveMode::IncludesOnly);
        திரும்பு result;
    }

    std::string preprocess_includes(மாறிலி std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_அடுக்கு;
        std::string result = processString(contents, macros, predefined, include_அடுக்கு, DirectiveMode::IncludesOnly);
        திரும்பு result;
    }

  தனிப்பட்ட:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    எண்ணிக்கை வகுப்பு DirectiveMode { All, IncludesOnly };

    கட்டமைப்பு Cond {
        தருக்க parent_active;
        தருக்க active;
        தருக்க taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    வெறுமை parseMacroDefinitions(மாறிலி std::vector<std::string> & macro_defs) {
        க்கு (மாறிலி auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            எனில் (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } இல்லையெனில் {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    வெறுமை buildMacros(மாறிலி std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        க்கு (மாறிலி auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        க்கு (மாறிலி auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            எனில் (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } இல்லையெனில் {
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
    std::string loadFile(மாறிலி std::string & fname) {
        std::ifstream f(fname);
        எனில் (!f.is_open()) {
            எறி std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        திரும்பு ss.str();
    }

    தருக்க condActive(மாறிலி std::vector<Cond> & cond) மாறிலி {
        எனில் (cond.empty()) {
            திரும்பு உண்மை;
        }
        திரும்பு cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(மாறிலி std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            மாறிலி std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_அடுக்கு,
                            DirectiveMode                                  mode) {
        எனில் (include_அடுக்கு.count(name)) {
            எறி std::runtime_error("Recursive include: " + name);
        }

        include_அடுக்கு.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_அடுக்கு, mode);
        include_அடுக்கு.erase(name);
        திரும்பு out;
    }

    std::string processIncludeFile(மாறிலி std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   மாறிலி std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_அடுக்கு,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        திரும்பு processFile(full_path, macros, predefined_macros, include_அடுக்கு, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(மாறிலி std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              மாறிலி std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_அடுக்கு,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        வரி;

        போது (std::getline(in, வரி)) {
            std::string logical = வரி;
            std::string t       = trim(logical);
            எனில் (!t.empty() && t[0] == '#') {
                போது (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    எனில் (!std::getline(in, வரி)) {
                        நிறுத்து;
                    }
                    logical += "\n";
                    logical += வரி;
                }
                t = trim(logical);
            }

            எனில் (!t.empty() && t[0] == '#') {
                தருக்க handled = handleDirective(t, out, macros, predefined_macros, cond, include_அடுக்கு, mode);
                எனில் (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } இல்லையெனில் {
                எனில் (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } இல்லையெனில் எனில் (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        எனில் (mode == DirectiveMode::All && !cond.empty()) {
            எறி std::runtime_error("Unclosed #if directive");
        }

        திரும்பு out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    தருக்க handleDirective(மாறிலி std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         மாறிலி std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_அடுக்கு,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        எனில் (cmd == "include") {
            எனில் (mode == DirectiveMode::All && !condActive(cond)) {
                திரும்பு உண்மை;
            }
            std::string file;
            iss >> file;
            எனில் (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_அடுக்கு, mode);
            திரும்பு உண்மை;
        }

        எனில் (mode == DirectiveMode::IncludesOnly) {
            திரும்பு பொய்;
        }

        எனில் (cmd == "define") {
            எனில் (!condActive(cond)) {
                திரும்பு உண்மை;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            எனில் (predefined_macros.count(name)) {
                திரும்பு உண்மை;
            }
            std::string value = trim_மதிப்பு(iss);
            macros[name]      = value;
            திரும்பு உண்மை;
        }

        எனில் (cmd == "undef") {
            எனில் (!condActive(cond)) {
                திரும்பு உண்மை;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            எனில் (predefined_macros.count(name)) {
                திரும்பு உண்மை;
            }
            macros.erase(name);
            திரும்பு உண்மை;
        }

        எனில் (cmd == "ifdef") {
            std::string name;
            iss >> name;
            தருக்க p = condActive(cond);
            தருக்க v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            திரும்பு உண்மை;
        }

        எனில் (cmd == "ifndef") {
            std::string name;
            iss >> name;
            தருக்க p = condActive(cond);
            தருக்க v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            திரும்பு உண்மை;
        }

        எனில் (cmd == "if") {
            std::string கோவை = trim_மதிப்பு(iss);
            தருக்க        p    = condActive(cond);
            தருக்க        v    = பொய்;
            எனில் (p) {
                std::unordered_set<std::string> visiting;
                கோவை_Parser                      ep(கோவை, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            திரும்பு உண்மை;
        }

        எனில் (cmd == "elif") {
            std::string கோவை = trim_மதிப்பு(iss);

            எனில் (cond.empty()) {
                எறி std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            எனில் (!c.parent_active) {
                c.active = பொய்;
                திரும்பு உண்மை;
            }

            எனில் (c.taken) {
                c.active = பொய்;
                திரும்பு உண்மை;
            }

            std::unordered_set<std::string> visiting;
            கோவை_Parser                      ep(கோவை, macros, visiting);
            தருக்க                            v = ep.parse() != 0;
            c.active                          = v;
            எனில் (v) {
                c.taken = உண்மை;
            }
            திரும்பு உண்மை;
        }

        எனில் (cmd == "else") {
            எனில் (cond.empty()) {
                எறி std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            எனில் (!c.parent_active) {
                c.active = பொய்;
                திரும்பு உண்மை;
            }
            எனில் (c.taken) {
                c.active = பொய்;
            } இல்லையெனில் {
                c.active = உண்மை;
                c.taken  = உண்மை;
            }
            திரும்பு உண்மை;
        }

        எனில் (cmd == "endif") {
            எனில் (cond.empty()) {
                எறி std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            திரும்பு உண்மை;
        }

        // Unknown directive
        எறி std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
