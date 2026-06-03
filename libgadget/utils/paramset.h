#ifndef PARAMSET_H
#define PARAMSET_H

#include <string>
#include <map>
#include <variant>

enum ParameterFlag {
    REQUIRED = 0,
    OPTIONAL = 1,
};

typedef std::map<std::string, int> ParameterEnum;

typedef std::variant<int, double, std::string> ParameterValue;

typedef struct ParameterSchema {
    int type;
    int lineno;
    ParameterValue defvalue;
    ParameterValue value;
    std::string help;
    enum ParameterFlag required;
    ParameterEnum enumtable;
} ParameterSchema;

struct ParameterSet {
    /* Raw string content of the parameter file*/
    std::string content;
    /* Map from parameter schema to value*/
    std::map<std::string, ParameterSchema> p;
};

void
param_declare_int(ParameterSet * ps, const std::string name, const enum ParameterFlag required, const int defvalue, const std::string help);

void
param_declare_double(ParameterSet * ps, const std::string name, const enum ParameterFlag required, const double defvalue, const std::string help);

void
param_declare_string(ParameterSet * ps, const std::string name, const enum ParameterFlag required, const std::string defvalue, const std::string help);

void
param_declare_enum(ParameterSet * ps, const std::string name, ParameterEnum& enumtable, const enum ParameterFlag required, const std::string defvalue, const std::string help);

double
param_get_double(ParameterSet * ps, const std::string name);

std::string&
param_get_string(ParameterSet * ps, const std::string name);

int
param_get_int(ParameterSet * ps, const std::string name);

int
param_get_enum(ParameterSet * ps, const std::string name);

/* returns 0 on no error; 1 on error */
int param_parse_file (ParameterSet * ps, const std::string filename);
/* returns 0 on no error; 1 on error */
int param_validate(ParameterSet * ps);
void param_dump(ParameterSet * ps, FILE * stream);

#endif
