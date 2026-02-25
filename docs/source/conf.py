import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

from pygments.lexers import get_lexer_by_name
from sphinx.highlighting import lexers

sys.path.insert(0, os.path.abspath("../../src"))

project = "GenPlanner"
author = "Donny"

try:
    release = pkg_version("genplanner")
except PackageNotFoundError:
    release = "0.0.0"

version = ".".join(release.split(".")[:2])
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

autodoc_mock_imports = ["genplanner._rust"]

html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "_static/favicon.png"

myst_enable_extensions = ["colon_fence", "deflist", "substitution"]
nb_execution_mode = "off"

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_preprocess_types = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
}
napoleon_type_aliases = {
    "gpd.GeoDataFrame": "geopandas.GeoDataFrame",
    "GeoDataFrame": "geopandas.GeoDataFrame",
    "nx.Graph": "networkx.Graph",
    "Graph": "networkx.Graph",
    "Series": "pandas.Series",
    "DataFrame": "pandas.DataFrame",
    "LineString": "shapely.geometry.LineString",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
lexers["ipython2"] = get_lexer_by_name("ipython3")
