# CAS: Customized Annealing Schedules

This is a repo for constructing customized annealing schedules for flux qubits.
The paper at [arXvi:2103.06461](https://arxiv.org/abs/2103.06461) details the theory and methods that were used in this codebase.

## Installation
First clone the repo using `git clone` , which copies all the contents of this package into a directory on your system.
Next go to the repo directory, e.g., using `cd CAS`.
Then run `pip install .` to install this package. You can also use `pip install -e .` to install the package in the 'editable' mode, which allows you to edit the source code and makes the changes immediately available for your use.

Before using the `pip install`, it is strongly recommended to create an isolated virtual environment and install the package in that environment. 
For instructions on creating python virtual environments, please see [this article](https://docs.python-guide.org/dev/virtualenvs/)
## Capabilities
It contains modules that can simulate the circuit model for Capacitively Shunted FLux Qubits (CSFQ) and tunable inductive couplers. 
It can then construct a circuit containing multiple CSFQs and tunable couplers that are coupled together.

For these circuits, the code can calculate the Pauli coefficients of the effective qubit models given a set of circuit magnetic fluxes.
It can also calculate circuit magnetic biases that yield a given customized annealing schedule.
For small circuits (depending on computer memory, typically 4 qubits and 4 couplers) the code can calculate the above using the full Schrieffer–Wolff (SW) method. 
For larger circuits, the code uses a pairwise-SW approximation to find the circuit biases and Pauli schedules.

## Examples
The [examples](https://github.com/USCqserver/CAS/tree/master/docs/examples) directory contains jupyter example notebooks that explain how different methods of the module are used.
The [examples/cas_paper](https://github.com/USCqserver/CAS/tree/master/docs/examples/cas_paper) directory includes the notebooks that were used to make the figures and plots of the paper at [arXvi:2103.06461](https://arxiv.org/abs/2103.06461).


## Contributions
Fork the repo for yourself, create a branch associated to your specific issue on your forked repo, push your changes to that branch, and then create a pull request to be merged onto the main repo.

## Citation
If you use this code in your research, please cite its paper as _citation goes here_.
