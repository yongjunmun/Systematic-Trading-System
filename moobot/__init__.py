"""moobot - a paper-trading bot for moomoo / Futu OpenAPI.

Safety model: every code path that can send an order goes through
``moobot.broker.Broker``, which refuses to construct itself unless the
configured trading environment is ``SIMULATE`` (paper money).
"""

__version__ = "1.0.0"
