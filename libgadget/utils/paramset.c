#include <fstream>
#include "paramset.h"
#include "endrun.h"

#define INT 1
#define DOUBLE 3
#define STRING 5
#define ENUM 10

static int parse_enum(ParameterEnum& table, const std::string strchoices, const std::string name) {
    int outvalue = 0;
    bool valid = false;
    const std::string delim = "\",;&| \t";
    std::string remaining = strchoices;
    auto startpos = remaining.find_first_not_of(delim);
    while (startpos != std::string::npos) {
        /*Trim extra chars from left*/
        remaining = remaining.substr(startpos);
        auto sz = remaining.find_first_of(delim);
        auto token = remaining.substr(0, sz);
        if(table.contains(token)) {
            outvalue |= table[token];
            valid = true;
        }
        /* We are done*/
        if(sz == std::string::npos)
            break;
        /* Trim processed chars*/
        remaining = remaining.substr(sz);
        startpos = remaining.find_first_not_of(delim);
    };
//    message(0, "enum parse input %s out %s\n", strchoices.c_str(), format_enum(table, outvalue).c_str());
    if(!valid)
        endrun(6, "Parameter %s set with string %s had no valid entries\n", name.c_str(), strchoices.c_str());
    return outvalue;
}

int param_validate(ParameterSet * ps)
{
    int flag = 0;
    /* copy over the default values */
    for(auto pp : ps->p) {
        if(pp.second.required == REQUIRED && pp.second.lineno < 0) {
            message(0, "Parameter `%s` is required, but not set.\n", pp.first.c_str());
            flag = 1;
        }
    }
    return flag;
}

static void
param_set_from_string(ParameterSet * ps, const std::string name, std::string value, int lineno)
{
    auto& pp = ps->p[name];
    pp.lineno = lineno;
    switch(pp.type) {
        case INT:
            pp.value = std::stoi(value);
            break;
        case DOUBLE:
            pp.value = std::stod(value);
            break;
        case STRING:
            {
                /* Trim whitespace from end of string*/
                auto endvalue = value.find_last_not_of(" \t");
                if(endvalue != std::string::npos)
                    endvalue += 1;
                pp.value = value.substr(0, endvalue);
            }
            break;
        case ENUM:
            pp.value = parse_enum(pp.enumtable, value, name);
            break;
        default:
            endrun(4, "Unexpected type for parameter %s: %d\n", name.c_str(), pp.type);
    }
}

static int param_emit(ParameterSet * ps, std::string token, int lineno)
{
    std::string limits(" \t=");
    std::string comments("%#");
    /* Remove comments*/
    for(auto cc : comments) {
        auto comment = token.find(cc);
        if(comment != std::string::npos) {
            token.erase(comment);
        }
    }
    if(token.size() == 0)
        return 0;
    /* Parse a line: find the key-value pair*/
    auto key = token.find_first_not_of(limits);
    if(key == std::string::npos) {
        return 0;
    }
    auto sep = token.substr(key).find_first_of(limits);
    if(sep == std::string::npos) {
        message(0, "line %d : `%s` is malformed.\n", lineno, token.c_str());
        return 1;
    }
    std::string name = token.substr(key, sep);
    std::string valuestr = token.substr(sep+key);
    auto value = valuestr.find_first_not_of(limits);
    if(value == std::string::npos) {
        message(0, "line %d : `%s` is malformed.\n", lineno, token.c_str());
        return 1;
    }
    valuestr = valuestr.substr(value);
    if(ps->p.contains(name))
        param_set_from_string(ps, name, valuestr, lineno);
    else
        message(0, "Line %d: Parameter `%s` is unknown.\n", lineno, name.c_str());
    return 0;
}

int param_parse_file (ParameterSet * ps, const std::string filename)
{
    std::ifstream content(filename);
    if(!content.is_open()) {
        endrun(1, "Could not read file: %s\n", filename.c_str());
    }
    std::string line;
    int flag = 0;
    int lineno = 0;
    while(std::getline(content, line)) {
        flag |= param_emit(ps, line, lineno);
        lineno ++;
    }
    return flag;
}

template <typename T> void
param_declare(ParameterSet * ps, const std::string name, const int type, const enum ParameterFlag required, const T defvalue, const std::string help)
{
    ParameterSchema pp;
    pp.type = type;
    pp.required = required;
    pp.lineno = -1;
    pp.help = help;
    pp.defvalue = defvalue;
    pp.value = defvalue;
    ps->p[name] = pp;
}

void
param_declare_int(ParameterSet * ps, const std::string name, const enum ParameterFlag required, const int defvalue, const std::string help)
{
    param_declare<int>(ps, name, INT, required, defvalue, help);
}
void
param_declare_double(ParameterSet * ps, const std::string name, const enum ParameterFlag required, const double defvalue, const std::string help)
{
    param_declare<double>(ps, name, DOUBLE, required, defvalue, help);
}

void
param_declare_string(ParameterSet * ps, const std::string name, const enum ParameterFlag required, const std::string defvalue, const std::string help)
{
    param_declare<std::string>(ps, name, STRING, required, defvalue, help);
}

void
param_declare_enum(ParameterSet * ps, const std::string name, ParameterEnum& enumtable, const enum ParameterFlag required, const std::string defvalue, const std::string help)
{
    param_declare<int>(ps, name, ENUM, required, parse_enum(enumtable, defvalue, name), help);
    ps->p[name].enumtable = enumtable;
}

double
param_get_double(ParameterSet * ps, const std::string name)
{
    return std::get<double>(ps->p[name].value);
}

std::string&
param_get_string(ParameterSet * ps, const std::string name)
{
    return std::get<std::string>(ps->p[name].value);
}

int
param_get_int(ParameterSet * ps, const std::string name)
{
    return std::get<int>(ps->p[name].value);
}

int
param_get_enum(ParameterSet * ps, const std::string name)
{
    return std::get<int>(ps->p[name].value);
}

static std::string format_enum(ParameterEnum& table, int value) {
    std::string formatted;
    for(auto it = table.begin(); it != table.end(); ++it) {
        if((value & it->second) == it->second) {
            if(formatted.size() > 0)
                formatted += " | ";
            formatted += it->first;
        }
    }
    return formatted;
}

static std::string
param_format_value(ParameterSchema& p)
{
    auto value = p.value;
    if(p.lineno < 0)
        value = p.defvalue;
    if(p.type == ENUM) {
        return format_enum(p.enumtable, std::get<int>(value));
    }
    else if(p.type == INT)
        return std::to_string(std::get<int>(value));
    else if(p.type == DOUBLE)
        return std::to_string(std::get<double>(value));
    return std::get<std::string>(value);
}

void param_dump(ParameterSet * ps, FILE * stream)
{
    for(auto it = ps->p.begin(); it != ps->p.end(); ++it) {
        std::string v = param_format_value(it->second);
        if(it->second.lineno >= 0) {
            fprintf(stream, "%-31s %-20s # Line %03d # %s \n", it->first.c_str(), v.c_str(), it->second.lineno, it->second.help.c_str());
        } else {
            fprintf(stream, "%-31s %-20s # Default  # %s \n", it->first.c_str(), v.c_str(), it->second.help.c_str());
        }
    }
    fflush(stream);
}
