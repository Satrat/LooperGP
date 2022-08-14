# ReRe:GP
Sound and Music Computing Masters Project, generating loopable symbolic music using TransformerXL and DadaGP dataset

## Data Parse Folder Layout

#### calc_loop_stats.py
Script for analyzing the number of loops in a dataset without saving the output to file

#### dadacompile.py
Transforms dataset folder into .npz file for training

#### extract_ex.ipynb
Generates a folder of inferences attempts and runs the loop extraction algorithm on each of them. Also reports loop density statistics

#### make_loops.py
Implementation of the loop extraction algorithm

#### save_loops.py
Script for building the ReRe:GP loop dataset from a DadaGP dataset folder
