Crypto MarketMaking
===================

This module contains the High-Frequency Trading (HFT) backtesting pipeline
and seed strategies used in the LLM-Guided Evolution system.

Overview
--------

The MarketMaking module allows you to:

- Run HFT backtests on crypto order book data  
- Evaluate trading strategies using fitness scores  
- Use seed strategies as starting points for evolution  

Running a Backtest
------------------

Run the default model:

.. code-block:: bash

   python eval.py

Run a specific seed:

.. code-block:: bash

   python eval.py --model seeds.seedHOST

Key Components
--------------

- ``eval.py`` → main backtesting entry point  
- ``seeds/`` → built-in trading strategies  
- ``dataExploration/`` → data analysis notebook  

Adding a Strategy
-----------------

Create a new file in ``seeds/`` and implement:

.. code-block:: python

   class Model:
       def run(self, backtest, recorder):
           ...

Then run:

.. code-block:: bash

   python eval.py --model seeds.seedMyStrategy

Troubleshooting
---------------

- **Data file not found** → check data path in ``eval.py``  
- **Import errors** → verify module paths  
- **Recorder overflow** → increase buffer size  