Titanic API
===========

Model
-----

.. autosummary::
    :toctree: generated
    
    model

Evaluation (eval.py)
--------------------

Module path: ``sota.Titanic.eval``

.. rubric:: Overview
This script loads the Titanic training data, splits it into training/testing sets, trains a machine-learning model, makes predictions, and then computes a confusion matrix to show how many cases the model got wrong.

Preprocessing (preprocess.py)
--------------------------------

Module path: ``sota.Titanic.preprocess``

Reading and Dropping Unneeded Columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    train_df = pd.read_csv("data/train.csv")
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

- Loads the dataset.
- Drops ``Ticket`` and ``Cabin`` because they provide little value
  (``Cabin`` has too many missing values, and ``Ticket`` isn't predictive).

Extracting the Passenger's Title
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset['Title'] = dataset.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)

- Extracts titles such as ``Mr``, ``Mrs``, ``Miss``, ``Master`` from the Name column.

Grouping and Standardizing Titles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major',
         'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

- Rare titles are grouped into ``Rare``.
- French titles are replaced with their English equivalents.

Mapping Titles to Numbers
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping).fillna(0)

- Converts text titles to numbers for machine learning.

Dropping Name and Passenger ID
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset = dataset.drop(['Name', 'PassengerId'], axis=1)

- ``Name`` is no longer needed after extracting the title.
- ``PassengerId`` is only an identifier and not useful for prediction.

Encoding Sex as Numeric
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

- Converts ``female/male`` into binary numeric values.

Estimating and Filling Missing Ages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script estimates missing ages based on median values for each sex and
passenger class combination.

- There are 2 sexes and 3 passenger classes → a 2x3 matrix of median ages.
- Missing ages for a group are filled using that group’s median age.

Converting Age to Categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After filling missing ages, the script converts ages into discrete groups:

+------------------------+-------+
| Age Range (years)      | Code  |
+========================+=======+
| 0–16                   | 0     |
| 17–32                  | 1     |
| 33–48                  | 2     |
| 49–64                  | 3     |
| 65+                    | 4     |
+------------------------+-------+

This turns continuous ages into categories to help the model find patterns.

Family Features (Family Size and IsAlone)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

- ``FamilySize`` counts how many family members a passenger traveled with.
- ``IsAlone`` becomes 1 if the person travelled alone.

The script then removes the original family columns since the new feature
replaces them.

Creating Age-Class Interaction Feature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset['Age*Class'] = dataset.Age * dataset.Pclass

- Combines age and passenger class into a single feature.
- Helps capture relationships such as: *younger children in 1st class vs.
older passengers in 3rd class*.

Processing the Embarked Feature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fills missing values with the most common port of embarkation.
- Maps ports to numbers:

  - ``S`` → 0
  - ``C`` → 1
  - ``Q`` → 2

Categorizing Fare
~~~~~~~~~~~~~~~~~

Fare values are grouped into four categories (0–3) to simplify them for the
model.

Saving the Processed Data
~~~~~~~~~~~~~~~~~~~~~~~~~

The transformed dataset is printed and saved to:

.. code-block:: text

    data/processed_train.csv
