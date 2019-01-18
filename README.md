# Final project for Reinforcement Learning course. MVA-2018

## Acknowledgments
Please note that some of the functions to communicate with the SUMO agent come from the code this repository: wingsweihua/IntelliLight.
The rest of the implementation (agents, training, etc) is original.

## Code Usage
Please find the project report in the `report.pdf` file.

You can find the requirements for the project in the `requirements.txt` file. You should also install the SUMO library: http://sumo.dlr.de/wiki/Installing.

The script `main_multiagent.py` will train and evaluate 4 agents on the same grid. This produces the results presented in the report. (to switch to the "flow A", please replace the file `data/four_agents/flow.xml` with the `data/four_agents/flow_a.xml`). You can always visualize results by setting the argument use_gui to True.
