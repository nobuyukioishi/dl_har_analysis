# DL-HAR - Analysis Submodule

This is the submodule repository of the [dl_har_public repository](https://github.com/STRCSussex-UbiCompSiegen/dl_har_public).

## Contributing to this repository

If you want to contribute to this repository **make sure to fork and clone the main repository [dl_har_public repository](https://github.com/STRCSussex-UbiCompSiegen/dl_har_public) with all its submodules**. Please run:

```
git clone --recurse-submodules -j8 git@github.com:STRCSussex-UbiCompSiegen/dl_har_public.git
```
If you want to have your modification be merged into the repository, please issue a **pull request**. If you don't know how to do so, please check out [this guide](https://jarv.is/notes/how-to-pull-request-fork-github/).

## Repository structure
In the following each of the main components will be briefly summarised. Output of the analysis 

### ```analysis.py```
Contains all relevant methods to analyse (previously saved) training and testing predictions. Depending on the type of validation method employed during training results are also printed out aggregated subject-wise.

The ```analysis.py``` script can also be run on its own making it possible to rerun the analysis of a previously run experiment. To do so run:

```python analysis.py -d [timestamp of the experiment]```

Note that the timestamp of the experiment is to be written in the format ```YYYYMMDD/hhmms``` (which is equivalent to the way the log directory is structured)
