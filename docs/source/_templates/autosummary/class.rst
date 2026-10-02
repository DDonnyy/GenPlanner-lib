{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   {% if objname == "GenPlanner" %}
   :exclude-members: original_territory, features2terr_zones, features2terr_zones2blocks
   {% endif %}

   {% block methods %}
   {% if objname == "GenPlanner" %}
   {% set public_methods = methods | reject("equalto", "__init__") | list %}
   {% if public_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree: generated
      :nosignatures:
   {% for item in public_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endif %}
   {% endblock %}
