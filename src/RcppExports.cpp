// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// ExtendedRunMC
List ExtendedRunMC(arma::cube grid_, float isovalue_, int size_x_, int size_y_, int size_z_, float s_x, float s_y, float s_z);
RcppExport SEXP _tcie_ExtendedRunMC(SEXP grid_SEXP, SEXP isovalue_SEXP, SEXP size_x_SEXP, SEXP size_y_SEXP, SEXP size_z_SEXP, SEXP s_xSEXP, SEXP s_ySEXP, SEXP s_zSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type grid_(grid_SEXP);
    Rcpp::traits::input_parameter< float >::type isovalue_(isovalue_SEXP);
    Rcpp::traits::input_parameter< int >::type size_x_(size_x_SEXP);
    Rcpp::traits::input_parameter< int >::type size_y_(size_y_SEXP);
    Rcpp::traits::input_parameter< int >::type size_z_(size_z_SEXP);
    Rcpp::traits::input_parameter< float >::type s_x(s_xSEXP);
    Rcpp::traits::input_parameter< float >::type s_y(s_ySEXP);
    Rcpp::traits::input_parameter< float >::type s_z(s_zSEXP);
    rcpp_result_gen = Rcpp::wrap(ExtendedRunMC(grid_, isovalue_, size_x_, size_y_, size_z_, s_x, s_y, s_z));
    return rcpp_result_gen;
END_RCPP
}
// AnalizeGrid
List AnalizeGrid(arma::cube grid, int size_x_, int size_y_, int size_z_, float iso);
RcppExport SEXP _tcie_AnalizeGrid(SEXP gridSEXP, SEXP size_x_SEXP, SEXP size_y_SEXP, SEXP size_z_SEXP, SEXP isoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type grid(gridSEXP);
    Rcpp::traits::input_parameter< int >::type size_x_(size_x_SEXP);
    Rcpp::traits::input_parameter< int >::type size_y_(size_y_SEXP);
    Rcpp::traits::input_parameter< int >::type size_z_(size_z_SEXP);
    Rcpp::traits::input_parameter< float >::type iso(isoSEXP);
    rcpp_result_gen = Rcpp::wrap(AnalizeGrid(grid, size_x_, size_y_, size_z_, iso));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_tcie_ExtendedRunMC", (DL_FUNC) &_tcie_ExtendedRunMC, 8},
    {"_tcie_AnalizeGrid", (DL_FUNC) &_tcie_AnalizeGrid, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_tcie(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}