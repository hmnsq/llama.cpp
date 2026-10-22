#నిర్వచితం_కాకపోతే PRE_WGSL_HPP
#నిర్వచించు PRE_WGSL_HPP

#అన్తర్భావ <cctype>
#అన్తర్భావ <fstream>
#అన్తర్భావ <sstream>
#అన్తర్భావ <stdexcept>
#అన్తర్భావ <string>
#అన్తర్భావ <string_view>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <unordered_set>
#అన్తర్భావ <vector>

నామాకాశ pre_wgsl {

//==============================================================
// Options
//==============================================================
నిర్మాణం Options {
    std::string              include_మార్గం = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
స్థిర std::string trim(స్థిరరాశి std::string & s) {
    size_t a = 0;
    అయితే (a < s.size() && std::ఖాళీనా((గుర్తులేని అక్షరం) s[a])) {
        a++;
    }
    size_t b = s.size();
    అయితే (b > a && std::ఖాళీనా((గుర్తులేని అక్షరం) s[b - 1])) {
        b--;
    }
    తిరిగి s.substr(a, b - a);
}

స్థిర std::string trim_విలువ(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    తిరిగి trim(ss.str());
}

స్థిర తార్కిక isIdentChar(అక్షరం c) {
    తిరిగి std::అక్షరమా_అంకెనా(static_cast<గుర్తులేని అక్షరం>(c)) || c == '_';
}

స్థిర తార్కిక చివర్లు_With_Continuation(స్థిరరాశి std::string & పంక్తి) {
    size_t i = పంక్తి.size();
    అయితే (i > 0 && std::ఖాళీనా((గుర్తులేని అక్షరం) పంక్తి[i - 1])) {
        i--;
    }
    తిరిగి i > 0 && పంక్తి[i - 1] == '\\';
}

స్థిర శూన్య stripContinuation(std::string & పంక్తి) {
    size_t i = పంక్తి.size();
    అయితే (i > 0 && std::ఖాళీనా((గుర్తులేని అక్షరం) పంక్తి[i - 1])) {
        i--;
    }
    ఒకవేళ (i > 0 && పంక్తి[i - 1] == '\\') {
        పంక్తి.erase(i - 1);
    }
}

స్థిర std::string expand_Macros_Recursive_అంతర్గత(స్థిరరాశి std::string &                                  పంక్తి,
                                                 స్థిరరాశి std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

స్థిర std::string expand_Macro_విలువ(స్థిరరాశి std::string &                                  name,
                                    స్థిరరాశి std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ఒకవేళ (visiting.count(name)) {
        విసురు std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    స్వయంచాలక it = macros.find(name);
    ఒకవేళ (it == macros.end()) {
        visiting.erase(name);
        తిరిగి name;
    }

    స్థిరరాశి std::string & value = it->second;
    ఒకవేళ (value.ఖాళీలు()) {
        visiting.erase(name);
        తిరిగి "";
    }

    std::string expanded = expand_Macros_Recursive_అంతర్గత(value, macros, visiting);
    visiting.erase(name);
    తిరిగి expanded;
}

స్థిర std::string expand_Macros_Recursive_అంతర్గత(స్థిరరాశి std::string &                                  పంక్తి,
                                                 స్థిరరాశి std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string ఫలితం;
    ఫలితం.reserve(పంక్తి.size());

    size_t i = 0;
    అయితే (i < పంక్తి.size()) {
        ఒకవేళ (isIdentChar(పంక్తి[i])) {
            size_t start = i;
            అయితే (i < పంక్తి.size() && isIdentChar(పంక్తి[i])) {
                i++;
            }
            std::string token = పంక్తి.substr(start, i - start);

            స్వయంచాలక it = macros.find(token);
            ఒకవేళ (it != macros.end()) {
                ఫలితం += expand_Macro_విలువ(token, macros, visiting);
            } లేకపోతే {
                ఫలితం += token;
            }
        } లేకపోతే {
            ఫలితం += పంక్తి[i];
            i++;
        }
    }

    తిరిగి ఫలితం;
}

స్థిర std::string expandMacrosRecursive(స్థిరరాశి std::string &                                  పంక్తి,
                                         స్థిరరాశి std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    తిరిగి expand_Macros_Recursive_అంతర్గత(పంక్తి, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
తరగతి సమాసం_Lexer {
  బహిరంగ:
    గణన రకం_3 { చివర_3, IDENT, సంఖ్య_2, OP, LPAREN, RPAREN };

    నిర్మాణం Tok {
        రకం_3        రకం_2;
        std::string పాఠ్యం_2;
    };

    స్పష్టమైన సమాసం_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok తదుపరి() {
        skipWS();
        ఒకవేళ (pos >= src.size()) {
            తిరిగి { చివర_3, "" };
        }

        అక్షరం c = src[pos];

        //సంఖ్య
        ఒకవేళ (std::అంకెనా((గుర్తులేని అక్షరం) c)) {
            size_t start = pos;
            అయితే (pos < src.size() && std::అంకెనా((గుర్తులేని అక్షరం) src[pos])) {
                pos++;
            }
            తిరిగి { సంఖ్య_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ఒకవేళ (std::అక్షరమా((గుర్తులేని అక్షరం) c) || c == '_') {
            size_t start = pos;
            అయితే (pos < src.size() && (std::అక్షరమా_అంకెనా((గుర్తులేని అక్షరం) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            తిరిగి { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ఒకవేళ (c == '(') {
            pos++;
            తిరిగి { LPAREN, "(" };
        }
        ఒకవేళ (c == ')') {
            pos++;
            తిరిగి { RPAREN, ")" };
        }

        // multi-char operators
        స్థిర స్థిరరాశి అక్షరం * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        కొరకు (స్వయంచాలక op : two_ops) {
            ఒకవేళ (src.substr(pos, 2) == op) {
                pos += 2;
                తిరిగి { OP, std::string(op) };
            }
        }

        // single-char operators
        ఒకవేళ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            తిరిగి { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        తిరిగి { చివర_3, "" };
    }

  అంతరంగ:
    std::string_view src;
    size_t           pos;

    శూన్య skipWS() {
        అయితే (pos < src.size() && std::ఖాళీనా((గుర్తులేని అక్షరం) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
తరగతి సమాసం_Parser {
  బహిరంగ:
    సమాసం_Parser(std::string_view                                     సమాసం,
               స్థిరరాశి std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(సమాసం),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    పూర్ణాంకం parse() { తిరిగి parseLogicalOr(); }

  అంతరంగ:
    సమాసం_Lexer                                            lex;
    సమాసం_Lexer::Tok                                       tok;
    స్థిరరాశి std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    శూన్య advance() { tok = lex.తదుపరి(); }

    తార్కిక acceptOp(స్థిరరాశి std::string & s) {
        ఒకవేళ (tok.రకం_2 == సమాసం_Lexer::OP && tok.పాఠ్యం_2 == s) {
            advance();
            తిరిగి నిజం;
        }
        తిరిగి అబద్ధం;
    }

    తార్కిక accept_రకం(సమాసం_Lexer::రకం_3 k) {
        ఒకవేళ (tok.రకం_2 == k) {
            advance();
            తిరిగి నిజం;
        }
        తిరిగి అబద్ధం;
    }

    పూర్ణాంకం parseLogicalOr() {
        పూర్ణాంకం v = parseLogicalAnd();
        అయితే (acceptOp("||")) {
            పూర్ణాంకం rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        తిరిగి v;
    }

    పూర్ణాంకం parseLogicalAnd() {
        పూర్ణాంకం v = parseEquality();
        అయితే (acceptOp("&&")) {
            పూర్ణాంకం rhs = parseEquality();
            v       = (v && rhs);
        }
        తిరిగి v;
    }

    పూర్ణాంకం parseEquality() {
        పూర్ణాంకం v = parseRelational();
        కొరకు (;;) {
            ఒకవేళ (acceptOp("==")) {
                పూర్ణాంకం rhs = parseRelational();
                v       = (v == rhs);
            } లేకపోతే ఒకవేళ (acceptOp("!=")) {
                పూర్ణాంకం rhs = parseRelational();
                v       = (v != rhs);
            } లేకపోతే {
                ఆపు;
            }
        }
        తిరిగి v;
    }

    పూర్ణాంకం parseRelational() {
        పూర్ణాంకం v = parse_జరుపు();
        కొరకు (;;) {
            ఒకవేళ (acceptOp("<")) {
                పూర్ణాంకం rhs = parse_జరుపు();
                v       = (v < rhs);
            } లేకపోతే ఒకవేళ (acceptOp(">")) {
                పూర్ణాంకం rhs = parse_జరుపు();
                v       = (v > rhs);
            } లేకపోతే ఒకవేళ (acceptOp("<=")) {
                పూర్ణాంకం rhs = parse_జరుపు();
                v       = (v <= rhs);
            } లేకపోతే ఒకవేళ (acceptOp(">=")) {
                పూర్ణాంకం rhs = parse_జరుపు();
                v       = (v >= rhs);
            } లేకపోతే {
                ఆపు;
            }
        }
        తిరిగి v;
    }

    పూర్ణాంకం parse_జరుపు() {
        పూర్ణాంకం v = parse_కూడు();
        కొరకు (;;) {
            ఒకవేళ (acceptOp("<<")) {
                పూర్ణాంకం rhs = parse_కూడు();
                v       = (v << rhs);
            } లేకపోతే ఒకవేళ (acceptOp(">>")) {
                పూర్ణాంకం rhs = parse_కూడు();
                v       = (v >> rhs);
            } లేకపోతే {
                ఆపు;
            }
        }
        తిరిగి v;
    }

    పూర్ణాంకం parse_కూడు() {
        పూర్ణాంకం v = parseMult();
        కొరకు (;;) {
            ఒకవేళ (acceptOp("+")) {
                పూర్ణాంకం rhs = parseMult();
                v       = (v + rhs);
            } లేకపోతే ఒకవేళ (acceptOp("-")) {
                పూర్ణాంకం rhs = parseMult();
                v       = (v - rhs);
            } లేకపోతే {
                ఆపు;
            }
        }
        తిరిగి v;
    }

    పూర్ణాంకం parseMult() {
        పూర్ణాంకం v = parseUnary();
        కొరకు (;;) {
            ఒకవేళ (acceptOp("*")) {
                పూర్ణాంకం rhs = parseUnary();
                v       = (v * rhs);
            } లేకపోతే ఒకవేళ (acceptOp("/")) {
                పూర్ణాంకం rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } లేకపోతే ఒకవేళ (acceptOp("%")) {
                పూర్ణాంకం rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } లేకపోతే {
                ఆపు;
            }
        }
        తిరిగి v;
    }

    పూర్ణాంకం parseUnary() {
        ఒకవేళ (acceptOp("!")) {
            తిరిగి !parseUnary();
        }
        ఒకవేళ (acceptOp("-")) {
            తిరిగి -parseUnary();
        }
        ఒకవేళ (acceptOp("+")) {
            తిరిగి +parseUnary();
        }
        తిరిగి parsePrimary();
    }

    పూర్ణాంకం parsePrimary() {
        // '(' expr ')'
        ఒకవేళ (accept_రకం(సమాసం_Lexer::LPAREN)) {
            పూర్ణాంకం v = parse();
            ఒకవేళ (!accept_రకం(సమాసం_Lexer::RPAREN)) {
                విసురు std::runtime_error("missing ')'");
            }
            తిరిగి v;
        }

        //సంఖ్య
        ఒకవేళ (tok.రకం_2 == సమాసం_Lexer::సంఖ్య_2) {
            పూర్ణాంకం v = std::stoi(tok.పాఠ్యం_2);
            advance();
            తిరిగి v;
        }

        // defined(identifier)
        ఒకవేళ (tok.రకం_2 == సమాసం_Lexer::IDENT && tok.పాఠ్యం_2 == "defined") {
            advance();
            ఒకవేళ (accept_రకం(సమాసం_Lexer::LPAREN)) {
                ఒకవేళ (tok.రకం_2 != సమాసం_Lexer::IDENT) {
                    విసురు std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.పాఠ్యం_2;
                advance();
                ఒకవేళ (!accept_రకం(సమాసం_Lexer::RPAREN)) {
                    విసురు std::runtime_error("missing ) in defined()");
                }
                తిరిగి macros.count(name) ? 1 : 0;
            } లేకపోతే {
                // defined NAME
                ఒకవేళ (tok.రకం_2 != సమాసం_Lexer::IDENT) {
                    విసురు std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.పాఠ్యం_2;
                advance();
                తిరిగి macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ఒకవేళ (tok.రకం_2 == సమాసం_Lexer::IDENT) {
            std::string name = tok.పాఠ్యం_2;
            advance();
            స్వయంచాలక it = macros.find(name);
            ఒకవేళ (it == macros.end()) {
                తిరిగి 0;
            }
            ఒకవేళ (it->second.ఖాళీలు()) {
                తిరిగి 1;
            }
            తిరిగి eval_Macro_వ్యక్తీకరణ(name, it->second);
        }

        // unexpected
        తిరిగి 0;
    }

    పూర్ణాంకం eval_Macro_వ్యక్తీకరణ(స్థిరరాశి std::string & name, స్థిరరాశి std::string & value) {
        ఒకవేళ (visiting.count(name)) {
            విసురు std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        సమాసం_Parser ep(value, macros, visiting);
        పూర్ణాంకం        v = ep.parse();
        visiting.erase(name);
        తిరిగి v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
తరగతి Preprocessor {
  బహిరంగ:
    స్పష్టమైన Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ఒకవేళ (opts_.include_మార్గం.ఖాళీలు()) {
            opts_.include_మార్గం = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(స్థిరరాశి std::string & filename, స్థిరరాశి std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_దొంతర;
        buildMacros(additional_macros, macros, predefined);

        std::string ఫలితం = processFile(filename, macros, predefined, include_దొంతర, Directive_బహుళకం::All);
        తిరిగి ఫలితం;
    }

    std::string preprocess(స్థిరరాశి std::string & contents, స్థిరరాశి std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_దొంతర;
        buildMacros(additional_macros, macros, predefined);

        std::string ఫలితం = processString(contents, macros, predefined, include_దొంతర, Directive_బహుళకం::All);
        తిరిగి ఫలితం;
    }

    std::string preprocess_includes_file(స్థిరరాశి std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_దొంతర;
        std::string ఫలితం = processFile(filename, macros, predefined, include_దొంతర, Directive_బహుళకం::IncludesOnly);
        తిరిగి ఫలితం;
    }

    std::string preprocess_includes(స్థిరరాశి std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_దొంతర;
        std::string ఫలితం = processString(contents, macros, predefined, include_దొంతర, Directive_బహుళకం::IncludesOnly);
        తిరిగి ఫలితం;
    }

  అంతరంగ:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    గణన తరగతి Directive_బహుళకం { All, IncludesOnly };

    నిర్మాణం Cond {
        తార్కిక మాతృ_చురుకైన;
        తార్కిక చురుకైన;
        తార్కిక తీసుకున్న;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    శూన్య parseMacroDefinitions(స్థిరరాశి std::vector<std::string> & macro_defs) {
        కొరకు (స్థిరరాశి స్వయంచాలక & def : macro_defs) {
            size_t eq_pos = def.find('=');
            ఒకవేళ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } లేకపోతే {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    శూన్య buildMacros(స్థిరరాశి std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        కొరకు (స్థిరరాశి స్వయంచాలక & [name, value] : global_macros) {
            predefined.insert(name);
        }

        కొరకు (స్థిరరాశి స్వయంచాలక & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            ఒకవేళ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } లేకపోతే {
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
    std::string ఎక్కించు_File(స్థిరరాశి std::string & fname) {
        std::ifstream f(fname);
        ఒకవేళ (!f.is_open()) {
            విసురు std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        తిరిగి ss.str();
    }

    తార్కిక cond_చురుకైన(స్థిరరాశి std::vector<Cond> & cond) స్థిరరాశి {
        ఒకవేళ (cond.ఖాళీలు()) {
            తిరిగి నిజం;
        }
        తిరిగి cond.back().చురుకైన;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(స్థిరరాశి std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            స్థిరరాశి std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_దొంతర,
                            Directive_బహుళకం                                  mode) {
        ఒకవేళ (include_దొంతర.count(name)) {
            విసురు std::runtime_error("Recursive include: " + name);
        }

        include_దొంతర.insert(name);
        std::string shader_సంకేతం = ఎక్కించు_File(name);
        std::string బహిర్గతం         = processString(shader_సంకేతం, macros, predefined_macros, include_దొంతర, mode);
        include_దొంతర.erase(name);
        తిరిగి బహిర్గతం;
    }

    std::string processIncludeFile(స్థిరరాశి std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   స్థిరరాశి std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_దొంతర,
                                   Directive_బహుళకం                                  mode) {
        std::string నిండిన_మార్గం = opts_.include_మార్గం + "/" + fname;
        తిరిగి processFile(నిండిన_మార్గం, macros, predefined_macros, include_దొంతర, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(స్థిరరాశి std::string &                            shader_సంకేతం,
                              std::unordered_map<std::string, std::string> & macros,
                              స్థిరరాశి std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_దొంతర,
                              Directive_బహుళకం                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  బహిర్గతం;
        std::istringstream లోపల(shader_సంకేతం);
        std::string        పంక్తి;

        అయితే (std::getline(లోపల, పంక్తి)) {
            std::string logical = పంక్తి;
            std::string t       = trim(logical);
            ఒకవేళ (!t.ఖాళీలు() && t[0] == '#') {
                అయితే (చివర్లు_With_Continuation(logical)) {
                    stripContinuation(logical);
                    ఒకవేళ (!std::getline(లోపల, పంక్తి)) {
                        ఆపు;
                    }
                    logical += "\n";
                    logical += పంక్తి;
                }
                t = trim(logical);
            }

            ఒకవేళ (!t.ఖాళీలు() && t[0] == '#') {
                తార్కిక handled = handleDirective(t, బహిర్గతం, macros, predefined_macros, cond, include_దొంతర, mode);
                ఒకవేళ (mode == Directive_బహుళకం::IncludesOnly && !handled) {
                    బహిర్గతం << logical << "\n";
                }
            } లేకపోతే {
                ఒకవేళ (mode == Directive_బహుళకం::IncludesOnly) {
                    బహిర్గతం << logical << "\n";
                } లేకపోతే ఒకవేళ (cond_చురుకైన(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    బహిర్గతం << expanded << "\n";
                }
            }
        }

        ఒకవేళ (mode == Directive_బహుళకం::All && !cond.ఖాళీలు()) {
            విసురు std::runtime_error("Unclosed #if directive");
        }

        తిరిగి బహిర్గతం.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    తార్కిక handleDirective(స్థిరరాశి std::string &                            t,
                         std::stringstream &                            బహిర్గతం,
                         std::unordered_map<std::string, std::string> & macros,
                         స్థిరరాశి std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_దొంతర,
                         Directive_బహుళకం                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        ఒకవేళ (cmd == "include") {
            ఒకవేళ (mode == Directive_బహుళకం::All && !cond_చురుకైన(cond)) {
                తిరిగి నిజం;
            }
            std::string file;
            iss >> file;
            ఒకవేళ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            బహిర్గతం << processIncludeFile(file, macros, predefined_macros, include_దొంతర, mode);
            తిరిగి నిజం;
        }

        ఒకవేళ (mode == Directive_బహుళకం::IncludesOnly) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (cmd == "define") {
            ఒకవేళ (!cond_చురుకైన(cond)) {
                తిరిగి నిజం;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ఒకవేళ (predefined_macros.count(name)) {
                తిరిగి నిజం;
            }
            std::string value = trim_విలువ(iss);
            macros[name]      = value;
            తిరిగి నిజం;
        }

        ఒకవేళ (cmd == "undef") {
            ఒకవేళ (!cond_చురుకైన(cond)) {
                తిరిగి నిజం;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ఒకవేళ (predefined_macros.count(name)) {
                తిరిగి నిజం;
            }
            macros.erase(name);
            తిరిగి నిజం;
        }

        ఒకవేళ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            తార్కిక p = cond_చురుకైన(cond);
            తార్కిక v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            తిరిగి నిజం;
        }

        ఒకవేళ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            తార్కిక p = cond_చురుకైన(cond);
            తార్కిక v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            తిరిగి నిజం;
        }

        ఒకవేళ (cmd == "if") {
            std::string సమాసం = trim_విలువ(iss);
            తార్కిక        p    = cond_చురుకైన(cond);
            తార్కిక        v    = అబద్ధం;
            ఒకవేళ (p) {
                std::unordered_set<std::string> visiting;
                సమాసం_Parser                      ep(సమాసం, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            తిరిగి నిజం;
        }

        ఒకవేళ (cmd == "elif") {
            std::string సమాసం = trim_విలువ(iss);

            ఒకవేళ (cond.ఖాళీలు()) {
                విసురు std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ఒకవేళ (!c.మాతృ_చురుకైన) {
                c.చురుకైన = అబద్ధం;
                తిరిగి నిజం;
            }

            ఒకవేళ (c.తీసుకున్న) {
                c.చురుకైన = అబద్ధం;
                తిరిగి నిజం;
            }

            std::unordered_set<std::string> visiting;
            సమాసం_Parser                      ep(సమాసం, macros, visiting);
            తార్కిక                            v = ep.parse() != 0;
            c.చురుకైన                          = v;
            ఒకవేళ (v) {
                c.తీసుకున్న = నిజం;
            }
            తిరిగి నిజం;
        }

        ఒకవేళ (cmd == "else") {
            ఒకవేళ (cond.ఖాళీలు()) {
                విసురు std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ఒకవేళ (!c.మాతృ_చురుకైన) {
                c.చురుకైన = అబద్ధం;
                తిరిగి నిజం;
            }
            ఒకవేళ (c.తీసుకున్న) {
                c.చురుకైన = అబద్ధం;
            } లేకపోతే {
                c.చురుకైన = నిజం;
                c.తీసుకున్న  = నిజం;
            }
            తిరిగి నిజం;
        }

        ఒకవేళ (cmd == "endif") {
            ఒకవేళ (cond.ఖాళీలు()) {
                విసురు std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            తిరిగి నిజం;
        }

        // Unknown directive
        విసురు std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#అయితే_ముగింపు  // PRE_WGSL_HPP
