#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#અન્તર્ભાવ <cctype>
#અન્તર્ભાવ <fstream>
#અન્તર્ભાવ <sstream>
#અન્તર્ભાવ <stdexcept>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <string_view>
#અન્તર્ભાવ <unordered_map>
#અન્તર્ભાવ <unordered_set>
#અન્તર્ભાવ <vector>

નામાકાશ pre_wgsl {

//==============================================================
// Options
//==============================================================
સંરચના Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
સ્થિર std::string trim(અચલ std::string & s) {
    size_t a = 0;
    જ્યારે (a < s.size() && std::isspace((unsigned અક્ષર) s[a])) {
        a++;
    }
    size_t b = s.size();
    જ્યારે (b > a && std::isspace((unsigned અક્ષર) s[b - 1])) {
        b--;
    }
    પાછા s.substr(a, b - a);
}

સ્થિર std::string trim_કિંમત(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    પાછા trim(ss.str());
}

સ્થિર તાર્કિક isIdentChar(અક્ષર c) {
    પાછા std::isalnum(static_cast<unsigned અક્ષર>(c)) || c == '_';
}

સ્થિર તાર્કિક endsWithContinuation(અચલ std::string & લીટી) {
    size_t i = લીટી.size();
    જ્યારે (i > 0 && std::isspace((unsigned અક્ષર) લીટી[i - 1])) {
        i--;
    }
    પાછા i > 0 && લીટી[i - 1] == '\\';
}

સ્થિર શૂન્ય stripContinuation(std::string & લીટી) {
    size_t i = લીટી.size();
    જ્યારે (i > 0 && std::isspace((unsigned અક્ષર) લીટી[i - 1])) {
        i--;
    }
    જો (i > 0 && લીટી[i - 1] == '\\') {
        લીટી.erase(i - 1);
    }
}

સ્થિર std::string expandMacrosRecursiveInternal(અચલ std::string &                                  લીટી,
                                                 અચલ std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

સ્થિર std::string expand_Macro_કિંમત(અચલ std::string &                                  name,
                                    અચલ std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    જો (visiting.count(name)) {
        ફેંકો std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    જો (it == macros.end()) {
        visiting.erase(name);
        પાછા name;
    }

    અચલ std::string & value = it->second;
    જો (value.empty()) {
        visiting.erase(name);
        પાછા "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    પાછા expanded;
}

સ્થિર std::string expandMacrosRecursiveInternal(અચલ std::string &                                  લીટી,
                                                 અચલ std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(લીટી.size());

    size_t i = 0;
    જ્યારે (i < લીટી.size()) {
        જો (isIdentChar(લીટી[i])) {
            size_t start = i;
            જ્યારે (i < લીટી.size() && isIdentChar(લીટી[i])) {
                i++;
            }
            std::string token = લીટી.substr(start, i - start);

            auto it = macros.find(token);
            જો (it != macros.end()) {
                result += expand_Macro_કિંમત(token, macros, visiting);
            } નહીંતર {
                result += token;
            }
        } નહીંતર {
            result += લીટી[i];
            i++;
        }
    }

    પાછા result;
}

સ્થિર std::string expandMacrosRecursive(અચલ std::string &                                  લીટી,
                                         અચલ std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    પાછા expandMacrosRecursiveInternal(લીટી, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
વર્ગ પદાવલિ_Lexer {
  જાહેર:
    ગણના Kind { END, IDENT, સંખ્યા_2, OP, LPAREN, RPAREN };

    સંરચના Tok {
        Kind        kind;
        std::string text;
    };

    explicit પદાવલિ_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        જો (pos >= src.size()) {
            પાછા { END, "" };
        }

        અક્ષર c = src[pos];

        //સંખ્યા
        જો (std::isdigit((unsigned અક્ષર) c)) {
            size_t start = pos;
            જ્યારે (pos < src.size() && std::isdigit((unsigned અક્ષર) src[pos])) {
                pos++;
            }
            પાછા { સંખ્યા_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        જો (std::isalpha((unsigned અક્ષર) c) || c == '_') {
            size_t start = pos;
            જ્યારે (pos < src.size() && (std::isalnum((unsigned અક્ષર) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            પાછા { IDENT, std::string(src.substr(start, pos - start)) };
        }

        જો (c == '(') {
            pos++;
            પાછા { LPAREN, "(" };
        }
        જો (c == ')') {
            pos++;
            પાછા { RPAREN, ")" };
        }

        // multi-char operators
        સ્થિર અચલ અક્ષર * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        માટે (auto op : two_ops) {
            જો (src.substr(pos, 2) == op) {
                pos += 2;
                પાછા { OP, std::string(op) };
            }
        }

        // single-char operators
        જો (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            પાછા { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        પાછા { END, "" };
    }

  ખાનગી:
    std::string_view src;
    size_t           pos;

    શૂન્ય skipWS() {
        જ્યારે (pos < src.size() && std::isspace((unsigned અક્ષર) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
વર્ગ પદાવલિ_Parser {
  જાહેર:
    પદાવલિ_Parser(std::string_view                                     પદાવલિ,
               અચલ std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(પદાવલિ),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    પૂર્ણાંક parse() { પાછા parseLogicalOr(); }

  ખાનગી:
    પદાવલિ_Lexer                                            lex;
    પદાવલિ_Lexer::Tok                                       tok;
    અચલ std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    શૂન્ય advance() { tok = lex.next(); }

    તાર્કિક acceptOp(અચલ std::string & s) {
        જો (tok.kind == પદાવલિ_Lexer::OP && tok.text == s) {
            advance();
            પાછા સાચું;
        }
        પાછા ખોટું;
    }

    તાર્કિક acceptKind(પદાવલિ_Lexer::Kind k) {
        જો (tok.kind == k) {
            advance();
            પાછા સાચું;
        }
        પાછા ખોટું;
    }

    પૂર્ણાંક parseLogicalOr() {
        પૂર્ણાંક v = parseLogicalAnd();
        જ્યારે (acceptOp("||")) {
            પૂર્ણાંક rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        પાછા v;
    }

    પૂર્ણાંક parseLogicalAnd() {
        પૂર્ણાંક v = parseEquality();
        જ્યારે (acceptOp("&&")) {
            પૂર્ણાંક rhs = parseEquality();
            v       = (v && rhs);
        }
        પાછા v;
    }

    પૂર્ણાંક parseEquality() {
        પૂર્ણાંક v = parseRelational();
        માટે (;;) {
            જો (acceptOp("==")) {
                પૂર્ણાંક rhs = parseRelational();
                v       = (v == rhs);
            } નહીંતર જો (acceptOp("!=")) {
                પૂર્ણાંક rhs = parseRelational();
                v       = (v != rhs);
            } નહીંતર {
                તોડો;
            }
        }
        પાછા v;
    }

    પૂર્ણાંક parseRelational() {
        પૂર્ણાંક v = parseShift();
        માટે (;;) {
            જો (acceptOp("<")) {
                પૂર્ણાંક rhs = parseShift();
                v       = (v < rhs);
            } નહીંતર જો (acceptOp(">")) {
                પૂર્ણાંક rhs = parseShift();
                v       = (v > rhs);
            } નહીંતર જો (acceptOp("<=")) {
                પૂર્ણાંક rhs = parseShift();
                v       = (v <= rhs);
            } નહીંતર જો (acceptOp(">=")) {
                પૂર્ણાંક rhs = parseShift();
                v       = (v >= rhs);
            } નહીંતર {
                તોડો;
            }
        }
        પાછા v;
    }

    પૂર્ણાંક parseShift() {
        પૂર્ણાંક v = parseAdd();
        માટે (;;) {
            જો (acceptOp("<<")) {
                પૂર્ણાંક rhs = parseAdd();
                v       = (v << rhs);
            } નહીંતર જો (acceptOp(">>")) {
                પૂર્ણાંક rhs = parseAdd();
                v       = (v >> rhs);
            } નહીંતર {
                તોડો;
            }
        }
        પાછા v;
    }

    પૂર્ણાંક parseAdd() {
        પૂર્ણાંક v = parseMult();
        માટે (;;) {
            જો (acceptOp("+")) {
                પૂર્ણાંક rhs = parseMult();
                v       = (v + rhs);
            } નહીંતર જો (acceptOp("-")) {
                પૂર્ણાંક rhs = parseMult();
                v       = (v - rhs);
            } નહીંતર {
                તોડો;
            }
        }
        પાછા v;
    }

    પૂર્ણાંક parseMult() {
        પૂર્ણાંક v = parseUnary();
        માટે (;;) {
            જો (acceptOp("*")) {
                પૂર્ણાંક rhs = parseUnary();
                v       = (v * rhs);
            } નહીંતર જો (acceptOp("/")) {
                પૂર્ણાંક rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } નહીંતર જો (acceptOp("%")) {
                પૂર્ણાંક rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } નહીંતર {
                તોડો;
            }
        }
        પાછા v;
    }

    પૂર્ણાંક parseUnary() {
        જો (acceptOp("!")) {
            પાછા !parseUnary();
        }
        જો (acceptOp("-")) {
            પાછા -parseUnary();
        }
        જો (acceptOp("+")) {
            પાછા +parseUnary();
        }
        પાછા parsePrimary();
    }

    પૂર્ણાંક parsePrimary() {
        // '(' expr ')'
        જો (acceptKind(પદાવલિ_Lexer::LPAREN)) {
            પૂર્ણાંક v = parse();
            જો (!acceptKind(પદાવલિ_Lexer::RPAREN)) {
                ફેંકો std::runtime_error("missing ')'");
            }
            પાછા v;
        }

        //સંખ્યા
        જો (tok.kind == પદાવલિ_Lexer::સંખ્યા_2) {
            પૂર્ણાંક v = std::stoi(tok.text);
            advance();
            પાછા v;
        }

        // defined(identifier)
        જો (tok.kind == પદાવલિ_Lexer::IDENT && tok.text == "defined") {
            advance();
            જો (acceptKind(પદાવલિ_Lexer::LPAREN)) {
                જો (tok.kind != પદાવલિ_Lexer::IDENT) {
                    ફેંકો std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                જો (!acceptKind(પદાવલિ_Lexer::RPAREN)) {
                    ફેંકો std::runtime_error("missing ) in defined()");
                }
                પાછા macros.count(name) ? 1 : 0;
            } નહીંતર {
                // defined NAME
                જો (tok.kind != પદાવલિ_Lexer::IDENT) {
                    ફેંકો std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                પાછા macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        જો (tok.kind == પદાવલિ_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            જો (it == macros.end()) {
                પાછા 0;
            }
            જો (it->second.empty()) {
                પાછા 1;
            }
            પાછા evalMacroExpression(name, it->second);
        }

        // unexpected
        પાછા 0;
    }

    પૂર્ણાંક evalMacroExpression(અચલ std::string & name, અચલ std::string & value) {
        જો (visiting.count(name)) {
            ફેંકો std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        પદાવલિ_Parser ep(value, macros, visiting);
        પૂર્ણાંક        v = ep.parse();
        visiting.erase(name);
        પાછા v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
વર્ગ Preprocessor {
  જાહેર:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        જો (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(અચલ std::string & filename, અચલ std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ઢગલો;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_ઢગલો, DirectiveMode::All);
        પાછા result;
    }

    std::string preprocess(અચલ std::string & contents, અચલ std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ઢગલો;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_ઢગલો, DirectiveMode::All);
        પાછા result;
    }

    std::string preprocess_includes_file(અચલ std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ઢગલો;
        std::string result = processFile(filename, macros, predefined, include_ઢગલો, DirectiveMode::IncludesOnly);
        પાછા result;
    }

    std::string preprocess_includes(અચલ std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ઢગલો;
        std::string result = processString(contents, macros, predefined, include_ઢગલો, DirectiveMode::IncludesOnly);
        પાછા result;
    }

  ખાનગી:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    ગણના વર્ગ DirectiveMode { All, IncludesOnly };

    સંરચના Cond {
        તાર્કિક parent_active;
        તાર્કિક active;
        તાર્કિક taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    શૂન્ય parseMacroDefinitions(અચલ std::vector<std::string> & macro_defs) {
        માટે (અચલ auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            જો (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } નહીંતર {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    શૂન્ય buildMacros(અચલ std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        માટે (અચલ auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        માટે (અચલ auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            જો (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } નહીંતર {
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
    std::string loadFile(અચલ std::string & fname) {
        std::ifstream f(fname);
        જો (!f.is_open()) {
            ફેંકો std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        પાછા ss.str();
    }

    તાર્કિક condActive(અચલ std::vector<Cond> & cond) અચલ {
        જો (cond.empty()) {
            પાછા સાચું;
        }
        પાછા cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(અચલ std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            અચલ std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_ઢગલો,
                            DirectiveMode                                  mode) {
        જો (include_ઢગલો.count(name)) {
            ફેંકો std::runtime_error("Recursive include: " + name);
        }

        include_ઢગલો.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_ઢગલો, mode);
        include_ઢગલો.erase(name);
        પાછા out;
    }

    std::string processIncludeFile(અચલ std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   અચલ std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_ઢગલો,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        પાછા processFile(full_path, macros, predefined_macros, include_ઢગલો, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(અચલ std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              અચલ std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_ઢગલો,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        લીટી;

        જ્યારે (std::getline(in, લીટી)) {
            std::string logical = લીટી;
            std::string t       = trim(logical);
            જો (!t.empty() && t[0] == '#') {
                જ્યારે (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    જો (!std::getline(in, લીટી)) {
                        તોડો;
                    }
                    logical += "\n";
                    logical += લીટી;
                }
                t = trim(logical);
            }

            જો (!t.empty() && t[0] == '#') {
                તાર્કિક handled = handleDirective(t, out, macros, predefined_macros, cond, include_ઢગલો, mode);
                જો (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } નહીંતર {
                જો (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } નહીંતર જો (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        જો (mode == DirectiveMode::All && !cond.empty()) {
            ફેંકો std::runtime_error("Unclosed #if directive");
        }

        પાછા out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    તાર્કિક handleDirective(અચલ std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         અચલ std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_ઢગલો,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        જો (cmd == "include") {
            જો (mode == DirectiveMode::All && !condActive(cond)) {
                પાછા સાચું;
            }
            std::string file;
            iss >> file;
            જો (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_ઢગલો, mode);
            પાછા સાચું;
        }

        જો (mode == DirectiveMode::IncludesOnly) {
            પાછા ખોટું;
        }

        જો (cmd == "define") {
            જો (!condActive(cond)) {
                પાછા સાચું;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            જો (predefined_macros.count(name)) {
                પાછા સાચું;
            }
            std::string value = trim_કિંમત(iss);
            macros[name]      = value;
            પાછા સાચું;
        }

        જો (cmd == "undef") {
            જો (!condActive(cond)) {
                પાછા સાચું;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            જો (predefined_macros.count(name)) {
                પાછા સાચું;
            }
            macros.erase(name);
            પાછા સાચું;
        }

        જો (cmd == "ifdef") {
            std::string name;
            iss >> name;
            તાર્કિક p = condActive(cond);
            તાર્કિક v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            પાછા સાચું;
        }

        જો (cmd == "ifndef") {
            std::string name;
            iss >> name;
            તાર્કિક p = condActive(cond);
            તાર્કિક v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            પાછા સાચું;
        }

        જો (cmd == "if") {
            std::string પદાવલિ = trim_કિંમત(iss);
            તાર્કિક        p    = condActive(cond);
            તાર્કિક        v    = ખોટું;
            જો (p) {
                std::unordered_set<std::string> visiting;
                પદાવલિ_Parser                      ep(પદાવલિ, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            પાછા સાચું;
        }

        જો (cmd == "elif") {
            std::string પદાવલિ = trim_કિંમત(iss);

            જો (cond.empty()) {
                ફેંકો std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            જો (!c.parent_active) {
                c.active = ખોટું;
                પાછા સાચું;
            }

            જો (c.taken) {
                c.active = ખોટું;
                પાછા સાચું;
            }

            std::unordered_set<std::string> visiting;
            પદાવલિ_Parser                      ep(પદાવલિ, macros, visiting);
            તાર્કિક                            v = ep.parse() != 0;
            c.active                          = v;
            જો (v) {
                c.taken = સાચું;
            }
            પાછા સાચું;
        }

        જો (cmd == "else") {
            જો (cond.empty()) {
                ફેંકો std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            જો (!c.parent_active) {
                c.active = ખોટું;
                પાછા સાચું;
            }
            જો (c.taken) {
                c.active = ખોટું;
            } નહીંતર {
                c.active = સાચું;
                c.taken  = સાચું;
            }
            પાછા સાચું;
        }

        જો (cmd == "endif") {
            જો (cond.empty()) {
                ફેંકો std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            પાછા સાચું;
        }

        // Unknown directive
        ફેંકો std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
