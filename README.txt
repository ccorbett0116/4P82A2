To execute this code:
Tested using Python 3.9

Install required packages with:
pip install -r requirements.txt

Replace the DEAP algorithms.py file with the one included in newDeapFiles\algorithms.py, in pycharm, the algorithms file can be found in:

Windows:
4P82A2Submission\venv\Lib\site-packages\deap\algorithms.py

Macos:
4P82A2Submission\venv\lib\python3.9\site-packages\deap\algorithms.py

Replace the DEAP migration.py file with the one included in newDeapFiles\migration.py, in pycharm, the migration file can be found in:

Windows:
4P82A2Submission\venv\Lib\site-packages\deap\tools\migration.py

Macos:


run main\main.py

Note: this code will run on all available cores, to limit the number of cores used,
change "os.cpu_count()" on line 182 of main\main.py to the desired number of cores.