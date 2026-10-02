# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/DDonnyy/GenPlanner-lib/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                    |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/genplanner/\_\_init\_\_.py                          |        5 |        0 |        0 |        0 |    100% |           |
| src/genplanner/\_config.py                              |       15 |        0 |        0 |        0 |    100% |           |
| src/genplanner/errors/\_\_init\_\_.py                   |        1 |        0 |        0 |        0 |    100% |           |
| src/genplanner/errors/errors.py                         |       19 |        9 |        2 |        0 |     48% |3-5, 8-10, 15-17 |
| src/genplanner/main/\_\_init\_\_.py                     |        1 |        0 |        0 |        0 |    100% |           |
| src/genplanner/main/genplanner.py                       |      192 |       46 |       66 |       18 |     73% |169-\>171, 171-\>173, 173-\>176, 197-198, 203, 208, 225, 231-232, 235-237, 242-246, 249, 269-270, 284, 376, 454, 472, 475-476, 485, 524, 547-571, 602-603 |
| src/genplanner/main/init\_validation.py                 |      160 |      102 |       62 |       11 |     31% |32-43, 50, 56-\>61, 85-88, 102-129, 140-180, 190, 196-215, 229-232, 237-243, 249, 255-257, 262-284 |
| src/genplanner/tasks/\_\_init\_\_.py                    |        4 |        0 |        0 |        0 |    100% |           |
| src/genplanner/tasks/feat2blocks.py                     |       75 |       49 |       30 |        6 |     30% |13-71, 80, 82, 84, 89-90, 93-95, 130-137 |
| src/genplanner/tasks/feat2func.py                       |        2 |        1 |        0 |        0 |     50% |         2 |
| src/genplanner/tasks/feat2parts.py                      |        2 |        1 |        0 |        0 |     50% |         2 |
| src/genplanner/tasks/feat2terr.py                       |      202 |      185 |       76 |        0 |      6% |23-34, 39-260, 264-344 |
| src/genplanner/tasks/polygon\_splitter.py               |      287 |      101 |      124 |       40 |     56% |46, 52, 56, 62, 68, 73, 81-110, 115, 124-140, 149, 154, 158, 183-184, 190-191, 197-198, 221, 223, 236, 244-247, 270-271, 276, 282, 284, 293-309, 312, 346, 364-\>369, 379-383, 403, 406-407, 416, 431-433, 437-448, 463, 468-\>471, 490, 500-\>503, 509, 515, 519-\>526, 534-549 |
| src/genplanner/utils/\_\_init\_\_.py                    |        1 |        0 |        0 |        0 |    100% |           |
| src/genplanner/utils/geom\_utils.py                     |      145 |       29 |       28 |        7 |     76% |14-16, 20-31, 53, 88-96, 109, 114, 117, 139, 174-\>177, 180, 185 |
| src/genplanner/zone\_relations/\_\_init\_\_.py          |        2 |        0 |        0 |        0 |    100% |           |
| src/genplanner/zone\_relations/forbidden\_terr\_kind.py |        2 |        0 |        0 |        0 |    100% |           |
| src/genplanner/zone\_relations/relation\_matrix.py      |      201 |      143 |       92 |        4 |     23% |84-98, 118-176, 191, 201, 204, 207, 210, 213-228, 242-260, 271-279, 284, 286, 289, 293, 296, 308-340, 349-377 |
| src/genplanner/zones/\_\_init\_\_.py                    |        7 |        0 |        0 |        0 |    100% |           |
| src/genplanner/zones/\_basic\_func\_zones.py            |        1 |        0 |        0 |        0 |    100% |           |
| src/genplanner/zones/\_basic\_terr\_zones.py            |        1 |        0 |        0 |        0 |    100% |           |
| src/genplanner/zones/abc\_zone.py                       |       22 |        7 |        4 |        0 |     58% |56, 71-73, 88-90 |
| src/genplanner/zones/basic\_zone.py                     |        9 |        2 |        2 |        1 |     73% |    31, 40 |
| src/genplanner/zones/functional\_zones.py               |       50 |        7 |       14 |        6 |     80% |23, 107, 110, 114, 116, 118, 136 |
| src/genplanner/zones/genplan\_zone.py                   |       14 |        0 |        0 |        0 |    100% |           |
| src/genplanner/zones/territory\_zones.py                |       40 |        7 |        6 |        3 |     78% |87-90, 95, 98, 107 |
| **TOTAL**                                               | **1460** |  **689** |  **506** |   **96** | **46%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/DDonnyy/GenPlanner-lib/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/DDonnyy/GenPlanner-lib/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/DDonnyy/GenPlanner-lib/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/DDonnyy/GenPlanner-lib/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FDDonnyy%2FGenPlanner-lib%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/DDonnyy/GenPlanner-lib/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.