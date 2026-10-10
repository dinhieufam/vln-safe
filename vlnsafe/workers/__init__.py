"""Model workers. Each script here runs in ITS OWN conda env as a subprocess and imports
`vlnsafe.geometry` and nothing else from vlnsafe/. They talk to the harness over the
JSON-lines protocol in protocol.py. Kept a package only so `vlnsafe.geometry` resolves on
their sys.path; the harness never imports these modules.
"""
