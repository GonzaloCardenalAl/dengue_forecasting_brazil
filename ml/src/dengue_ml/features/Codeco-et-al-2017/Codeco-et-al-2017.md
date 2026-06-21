---
title: 'Estimating the effective reproduction number of dengue considering temperature-dependent generation intervals'
author: Claudia Codeço, Daniel Villela, Flavio Coelho
output: html_document
---

## Description

Scripts and datasets analyzed by Codeço et al. Paper submitted to Epidemics (2017). 


## Summary

The study proposes a method that allows dengue's R_t estimates in the presence of temperature-mediated seasonality. This method is compared to the standard approach using both simulation and real datasets.

## Files

**Data**

The .csv files contain data for Foz do Iguacu (FI) and Rio de Janeiro (RJ)

- FIdata.csv
- RJdata.csv

Columns are:

- SE : epidemiological week
- temp_med: mean week temperature
- casos: dengue cases
- pop: estimated population

**Simulated data**

File SimuRt.RData contains the simulated data used in the paper.

**(td)Rt estimation**

Two files contain the functions required to calculate (td)Rt and (ti)Rt:

- sumgamma_v2.R
- EstRtGt_v4.R

To illustrate their usage, please check the file **Rt_calc-example.rmd**

## Contact

claudia.codeco@fiocruz.br




