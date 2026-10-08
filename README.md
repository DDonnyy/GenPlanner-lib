# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/DDonnyy/GenPlanner-lib/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                    |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/genplanner/\_\_init\_\_.py                          |        5 |        0 |        0 |        0 |    100% |           |
| src/genplanner/\_config.py                              |       15 |        0 |        0 |        0 |    100% |           |
| src/genplanner/errors/\_\_init\_\_.py                   |        1 |        0 |        0 |        0 |    100% |           |
| src/genplanner/errors/errors.py                         |       19 |        0 |        2 |        0 |    100% |           |
| src/genplanner/main/\_\_init\_\_.py                     |        1 |        0 |        0 |        0 |    100% |           |
| src/genplanner/main/genplanner.py                       |      205 |       25 |       72 |       17 |     85% |175-\>177, 177-\>179, 179-\>182, 203-204, 209, 214, 231, 237-238, 241-243, 248-252, 255, 276-277, 291, 463, 481, 484-485, 563-564, 572-\>575, 618-619 |
| src/genplanner/main/init\_validation.py                 |      164 |       13 |       66 |       14 |     87% |50, 103, 115, 141, 146, 190, 197, 211-215, 230, 233-\>238, 243-\>246, 247, 287-\>290 |
| src/genplanner/tasks/\_\_init\_\_.py                    |        4 |        0 |        0 |        0 |    100% |           |
| src/genplanner/tasks/feat2blocks.py                     |       75 |       49 |       30 |        6 |     30% |13-71, 80, 82, 84, 89-90, 93-95, 130-137 |
| src/genplanner/tasks/feat2func.py                       |        2 |        1 |        0 |        0 |     50% |         2 |
| src/genplanner/tasks/feat2parts.py                      |        2 |        1 |        0 |        0 |     50% |         2 |
| src/genplanner/tasks/feat2terr.py                       |      202 |       54 |       76 |       20 |     68% |31-33, 59-75, 81, 84-89, 113-119, 128, 132, 143, 146, 179-185, 190-193, 255, 258-260, 262-\>264, 289-291, 297, 306, 328-330, 338-\>341, 344-348 |
| src/genplanner/tasks/polygon\_splitter.py               |      304 |       43 |      128 |       40 |     80% |68, 86, 96, 102, 108, 113, 123-\>exit, 125, 132, 155, 166, 173, 189, 194, 198, 223-224, 230-231, 238, 261, 263, 276, 310-311, 316, 322, 324, 344-\>348, 345-\>344, 386, 404-\>409, 420-\>422, 445, 448-449, 462-\>465, 471, 492-503, 518, 523-\>526, 541, 547, 552, 572, 576-577 |
| src/genplanner/utils/\_\_init\_\_.py                    |        1 |        0 |        0 |        0 |    100% |           |
| src/genplanner/utils/geom\_utils.py                     |      150 |       19 |       28 |        7 |     84% |15-17, 35, 61, 96-104, 117, 122, 125, 148, 183-\>186, 194 |
| src/genplanner/zone\_relations/\_\_init\_\_.py          |        2 |        0 |        0 |        0 |    100% |           |
| src/genplanner/zone\_relations/forbidden\_terr\_kind.py |        2 |        0 |        0 |        0 |    100% |           |
| src/genplanner/zone\_relations/relation\_matrix.py      |      201 |       63 |       92 |       15 |     67% |119, 121-\>125, 129, 133, 136, 140-142, 152-\>167, 160, 172, 191, 201, 204, 246, 256, 284, 286, 289, 293, 296, 308-340, 349-377 |
| src/genplanner/zones/\_\_init\_\_.py                    |        7 |        0 |        0 |        0 |    100% |           |
| src/genplanner/zones/\_basic\_func\_zones.py            |        1 |        0 |        0 |        0 |    100% |           |
| src/genplanner/zones/\_basic\_terr\_zones.py            |        1 |        0 |        0 |        0 |    100% |           |
| src/genplanner/zones/abc\_zone.py                       |       22 |        4 |        4 |        1 |     73% | 72, 88-90 |
| src/genplanner/zones/basic\_zone.py                     |        9 |        2 |        2 |        1 |     73% |    31, 40 |
| src/genplanner/zones/functional\_zones.py               |       50 |        7 |       14 |        6 |     80% |23, 107, 110, 114, 116, 118, 136 |
| src/genplanner/zones/genplan\_zone.py                   |       14 |        0 |        0 |        0 |    100% |           |
| src/genplanner/zones/territory\_zones.py                |       40 |        6 |        6 |        3 |     80% |87-90, 95, 98 |
| **TOTAL**                                               | **1499** |  **287** |  **520** |  **130** | **76%** |           |


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