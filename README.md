# drogonapp

Full project for fast automated calibration predictions at GSI FAIR.

Now works with Epics and trigger data, predicts ionization losses corrections for FEB22 data.

Backend is based on the Drogon framework: https://github.com/drogonframework/drogon


# Some guidance

Creating MDC calibration dataset: 

1. In directory /lustre/hades/user/vkladov/sub/testBatchFarm/ with file makeList.cc create list of raw hld files for each day of experiment (given the directory of raw files), then merge some of them which are interesting for you
2. with analysisDST.cc, Makefile and sendScript_raw_SL.sh ---- create an executable to analyse raw hld files, send jobs to batch farm and get resulting files in a specified directory 
3. Created files contain directories (corresponding to run numbers) and a number of histograms in each of them (corresponding to spacial parts of detector or cells sec-mod).
4. "hadd file.root /.../***.root" to merge these files into one and then with /u/vkladov/testBatchFarm/analyseHists.cc and makeCalibTable() get the offline target calibration table.

