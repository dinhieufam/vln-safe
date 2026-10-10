"""Action providers: the objects rollout asks for the next primitive. One contract
(interface.py), three implementations (in-process vlnce_policy, subprocess remote,
scripted), and a registry that builds the right one from an agent config.
"""
