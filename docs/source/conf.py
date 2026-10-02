import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

from pygments.lexers import get_lexer_by_name
from sphinx.highlighting import lexers

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

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
    "sphinx_copybutton",
    "sphinx_design",
]

autodoc_mock_imports = ["genplanner._rust"]

html_theme = "furo"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_favicon = "_static/favicon.png"

myst_enable_extensions = ["colon_fence", "deflist", "substitution"]
nb_execution_mode = "off"

autosummary_generate = True
autodoc_typehints = "none"
autodoc_member_order = "bysource"
nitpick_ignore = [("py:class", "gpd.GeoDataFrame")]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_preprocess_types = True
napoleon_use_ivar = True

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
