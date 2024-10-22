# Hamiltonian Snippets
Hamiltonian Snippets code for the upcoming paper

## How to use this repository
1. Make sure you have `git` installed on your machine as well as `Python 3.12`
2. Click on the <span style='color:green'>green</span> button `<> Code` and copy the URL `https://github.com/MauroCE/HamiltonianSnippets.git`
3. Open a terminal where you want to clone the repository (typically in `Documents/`) and run `git clone https://github.com/MauroCE/HamiltonianSnippets.git`
4. Navigate into the directory `cd HamiltonianSnippets`
5. Open your favourite Python IDE, personally I like PyCharm but VScode is a great choice too.
6. Run the scripts in  `experiments/` to reproduce the experiments or create your own file and import `from HamiltonianSnippets.sampler import hamiltonian_snippet`

## What HamiltonianSnippets can do
This package can run a Hamiltonian Snippet and perform three types of adaptation:

1. Step size adaptation, requiring the user to specify a distribution for the step sizes. We currently support inverse gaussian (recommended) and discrete uniform distribution.
2. Number of leapfrog steps adaptation, based on contractivity arguments
3. Tempering parameter adaptation using the ESS

The method allows the user to select a fixed mass matrix (full or diagonal), a schedule of mass matrices (requiring the user to specify a function that given a gamma returns a mass matrix, either full or diagonal) and an adaptive mass matrix. The adaptive mass matrix function is not recommended for use, I have never tested it.

